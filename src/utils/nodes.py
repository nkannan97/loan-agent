from __future__ import annotations

import json
import logging

from functools import lru_cache
from typing import Any, Dict, List, Literal, Union

import tiktoken

from langchain.tools.base import ToolException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import ToolNode
from openevals.llm import create_llm_as_judge

from src.agents.utils.prompts import context_summarization_prompt, query_enrichment_prompt, reflection_agent_prompt, response_improvement_prompt
from src.agents.utils.reflection import Reflection
from src.agents.utils.tools import tools
from src.agents.config import AgentModelSettings, LlmSettings, ReflectionSettings, ShortTermMemorySettings
from src.agents.llm_manager import BedrockModels, LlmManager


logger = logging.getLogger(__name__)

# Initialize models
llm_manager = LlmManager()
bedrock_models = BedrockModels()
agent_model_settings = AgentModelSettings()
llm_settings = LlmSettings()
short_term_memory_settings = ShortTermMemorySettings()


@lru_cache(maxsize=1)
def _get_token_counter():
    """Get a token counter function using tiktoken.

    Returns:
        Callable: A function that takes a list of messages and returns the token count
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Using Claude's encoding

    def count_tokens(messages: List[Union[Dict[str, Any], BaseMessage]]) -> int:
        """Count the number of tokens in a list of messages.

        Args:
            messages: List of messages to count tokens for. Can be either dictionaries
                     or LangChain message objects.

        Returns:
            int: Total number of tokens
        """
        num_tokens = 0
        for message in messages:
            # Handle LangChain message objects
            if isinstance(message, BaseMessage):
                # Count role tokens based on message type
                if isinstance(message, SystemMessage):
                    num_tokens += len(encoding.encode("system"))
                elif isinstance(message, HumanMessage):
                    num_tokens += len(encoding.encode("user"))
                elif isinstance(message, AIMessage):
                    num_tokens += len(encoding.encode("assistant"))
                # Count content tokens
                if message.content:
                    num_tokens += len(encoding.encode(message.content))
            # Handle dictionary messages
            elif isinstance(message, dict):
                # Count message role tokens
                if "role" in message:
                    num_tokens += len(encoding.encode(message["role"]))
                # Count content tokens
                if "content" in message:
                    num_tokens += len(encoding.encode(message["content"]))
            # Add overhead for each message
            num_tokens += 3
        return num_tokens

    return count_tokens


@lru_cache(maxsize=4)
def _get_model(model_name: str, max_tokens: int | None = None) -> Any:
    """Get the specified language model with tools bound.

    Args:
        model_name (str): Name of the model to use ("azure_openai" or "aws_bedrock_sonnet" or "aws_bedrock_haiku")

    Returns:
        Any: The configured language model with tools bound

    Raises:
        ValueError: If an unsupported model type is specified
    """

    if max_tokens:
        llm_manager.set_llm_max_tokens(max_tokens)
        bedrock_models.set_llm_max_tokens(max_tokens)

    if model_name == "azure_openai":
        model = llm_manager.get_llm()
    elif model_name == "aws_bedrock_sonnet":
        model = bedrock_models.get_sonnet_llm()
    elif model_name == "aws_bedrock_haiku":
        model = bedrock_models.get_haiku_llm()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model


def should_continue(state: Dict[str, Any]) -> str:
    """Determine whether to continue the conversation or end it.

    Args:
        state (Dict[str, Any]): Current conversation state

    Returns:
        str: "continue" if there are tool calls, "end" otherwise
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "reflect"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful Assistant.

CONTEXT:
- You have access to the current conversation messages.

TONE AND VOICE:
- Professional and knowledgeable, but approachable and friendly
- Clear and concise in explanations, avoiding technical jargon unless necessary
- Patient and understanding when clarification is needed
- Proactive in suggesting best practices and potential improvements
- Maintains a positive and solution-oriented approach
- Explain the "why" behind suggestions

RESPONSE GUIDELINES:
- Provide clear, structured, and actionable responses
- Break down complex topics into digestible parts
- Include relevant examples when helpful
- Acknowledge uncertainties and ask for clarification when needed
- Use markdown formatting for better readability
- Always validate inputs and suggest best practices

Remember to maintain a helpful, professional, and consistent tone throughout all interactions.
"""


def trimming_messages(messages: Any) -> List[Any]:
    original_message_count = len(messages)
    # Get token counter
    token_counter = _get_token_counter()

    # Determine trimming strategy based on configuration
    if not short_term_memory_settings.short_term_memory_message_based_trimming:
        # Use actual token counting
        max_tokens_limit = llm_settings.llm_max_context_window - llm_settings.llm_response_buffer
        logger.info(f"Using token limit: {max_tokens_limit}")

        # Trim messages using actual token counting
        trimmed_messages = trim_messages(
            messages,
            strategy="last",  # Keep most recent messages
            max_tokens=max_tokens_limit,  # Use actual token limit
            include_system=True,  # Keep system message
            token_counter=token_counter,  # Use actual token counter
            start_on="human",
        )

        # Log if messages were trimmed with token information
        if len(trimmed_messages) < original_message_count:
            logging.info(
                f"Trimmed conversation history from {original_message_count} to {len(trimmed_messages)} messages "
                f"(max_tokens={llm_settings.llm_max_context_window}, buffer={llm_settings.llm_response_buffer})"
            )
    else:
        # Use message count-based trimming
        max_messages = short_term_memory_settings.short_term_memory_chat_history_buffer_size
        logger.info(f"Using message count-based trimming: max_messages={max_messages}")

        # Trim messages using message count
        trimmed_messages = trim_messages(
            messages,
            strategy="last",  # Keep most recent messages
            max_tokens=max_messages,  # This becomes max_messages when token_counter=len
            include_system=True,  # Keep system message
            token_counter=len,  # Count each message as 1 token
            start_on="human",
        )

        # Log if messages were trimmed
        if len(trimmed_messages) < original_message_count:
            trimmed_tokens = token_counter(trimmed_messages)
            logging.info(
                f"Trimmed conversation history from {original_message_count} to {len(trimmed_messages)} messages "
                f"(trimmed_tokens={trimmed_tokens})"
            )

    return trimmed_messages


def context_enrichment(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context Enrichment node that implements progressive context summarization.

    1. Compresses older conversation parts into summaries
    2. Preserves recent detailed context
    3. Creates a condensed but complete context for the agent

    Args:
        state: Current agent state containing messages
        config: Configuration options

    Returns:
        dict: Updated state with compressed context and condensed query
    """
    logger.info("Begin context enrichment...")

    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages found in state")
        return state

    # Get current user query (last human message)
    current_query = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            current_query = message.content
            break

    if not current_query:
        logger.warning("No human message found")
        return state

    # Apply context summarization on all prior conversations
    try:
        # Get all conversation history (everything except current query)
        conversation_history = messages[:-1] if len(messages) > 1 else []

        # If no prior conversation, no compression needed
        if not conversation_history:
            logger.info("No prior conversation, using current query as-is")
            state["condensed_query"] = current_query
            state["context_summary"] = "First message in conversation"
            return state

        logger.info(f"conversation_history: {conversation_history}")

        # Compress ALL prior conversation into summary
        context_summary = _compress_conversation_history(conversation_history)
        logger.info(f"context_summary: {context_summary}")
        logger.info(f"Compressed {len(conversation_history)} prior messages into summary")

        # Create condensed query combining compressed history + current query
        condensed_query = _create_condensed_query_simple(current_query, context_summary)

        # Store compressed context in state
        state["condensed_query"] = condensed_query
        state["context_summary"] = context_summary

        logger.info(f"Context compressed successfully. Summary: {context_summary}")
        logger.info(f"Condensed query: {condensed_query}")

        for i in range(len(state["messages"])):
            if isinstance(state["messages"][i], HumanMessage):
                logger.info(f"Replacing original query with condensed query: {condensed_query}")
                state["messages"][i] = HumanMessage(content=condensed_query)
                break

        return state

    except Exception as e:
        logger.error(f"Error in context compression: {e}")
        # Fallback to simple approach
        state["condensed_query"] = current_query
        state["context_summary"] = "Error in context compression"
        return state


def _compress_conversation_history(conversation_history: List[BaseMessage]) -> str:
    """
    Compress entire conversation history into a concise summary.
    """

    # Format all conversation messages for compression
    conversation_text = ""
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            conversation_text += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            # Include brief mention of AI responses but focus on user context
            conversation_text += f"ADA: {message.content}\n"

    context_summarization = context_summarization_prompt.format(conversation_history=conversation_text.strip())

    try:
        model = _get_model(agent_model_settings.agent_model)
        logger.info(f"model type for context compression...: {type(model)}")
        response = model.invoke([SystemMessage(content=context_summarization)])
        summary = response.content.strip()
        return summary

    except Exception as e:
        logger.error(f"Error compressing conversation: {e}")


def _create_condensed_query_simple(current_query: str, context_summary: str) -> str:
    """
    Create condensed query by combining compressed context with current query.

    Focuses on the user's actual current question plus relevant problem context.
    """
    context_enrichment_prompt = query_enrichment_prompt.format(context_summary=context_summary, current_query=current_query)

    try:
        model = _get_model(agent_model_settings.agent_model)
        response = model.invoke([SystemMessage(content=context_enrichment_prompt)])
        condensed_query = response.content.strip()
        return condensed_query

    except Exception as e:
        logger.error(f"Error creating condensed query: {e}")
        return f"{context_summary}. Current question: {current_query}"


def call_model(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Call the language model with the current state and configuration.

    Args:
        state (Dict[str, Any]): Current conversation state
        config (Dict[str, Any]): Configuration options

    Returns:
        Dict[str, Any]: Updated state with model response
    """
    messages = state["messages"]
    surface_info = state.get("surface_info")  # Get surface_info from state

    # Use condensed query logic
    logger.info(f"messages in call_model() function...: {messages}")
    # Add system prompt first
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = agent_model_settings.agent_model

    messages = trimming_messages(messages)

    # Handle both dict and SurfaceInfo object cases for surface check
   
    model = _get_model(model_name)

    model = model.bind_tools(tools)
    response = model.invoke(messages)

    return {"messages": [response]}


def evaluate_agent_response(state: Dict[str, Any]) -> Dict[str, Any]:
    reflection_settings = ReflectionSettings()
    logger.info(f"reflection_enabled: {reflection_settings.reflection_enabled}")
    logger.info(f"reflection enabled on the first query: {reflection_settings.reflection_enabled_for_first_query_only}")

    if not reflection_settings.reflection_enabled:
        logger.info(f"Reflection is disabled. Skipping reflection node execution....")
        state["reflection_complete"] = True
        return state

    messages = state.get("messages", [])
    if len(messages) < 2:
        logger.warning("Not enough messages for reflection evaluation")
        return state


    user_query = state["condensed_query"]
    last_message = messages[-1].content if messages else ""

    logger.info(f"user_query_reflection: {user_query}")

    model = _get_model(agent_model_settings.agent_model)

    try:
        if agent_model_settings.agent_model == "azure_openai":
            # For Azure OpenAI, use text-based evaluation
            reflection_prompt_text = reflection_agent_prompt.format(inputs=user_query, outputs=last_message)
            messages = [HumanMessage(content=reflection_prompt_text)]

            response = model.invoke(messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Azure OpenAI response: {response_text}")

            # Parse response and create Reflection object
            reflection_passed = "true" in response_text.lower()
            # Extract comment after "reflection_comment:" or use entire response
            if "reflection_comment:" in response_text.lower():
                reflection_comment = response_text.split("reflection_comment:", 1)[1].strip()
            else:
                reflection_comment = response_text.strip()

        else:
            # For other models, use structured output
            evaluator = create_llm_as_judge(prompt=reflection_agent_prompt, judge=model, output_schema=Reflection)

            result = evaluator(inputs=user_query, outputs=last_message)
            logger.info(f"Structured evaluator response: {result}")

            reflection_passed = result.get("reflection_score", False)
            reflection_comment = result.get("reflection_comment", "No feedback provided")

        # Now work with the clean Reflection object
        if reflection_passed:
            logger.info("Response approved by reflection judge")
            state["reflection_complete"] = True

        else:
            logger.info(f"Response was not approved by the reflection judge... regenerating the response")
            # Increment reflection iterations and provide feedback for improvement
            state["reflection_iterations"] = state.get("reflection_iterations", 0) + 1

            # Add the reflection feedback as a new user message for the agent to improve
            if state["reflection_iterations"] <= 3:
                feedback_message = HumanMessage(
                    content=response_improvement_prompt.format(reflection_comment=reflection_comment)
                )
                state["messages"] = feedback_message

        return state

    except Exception as e:
        logger.error(f"Error in reflection evaluation: {str(e)}")
        logger.info("Reflection evaluation failed, approving response by default")
        state["reflection_complete"] = True
        return state


def should_continue_reflection(state: Dict[str, Any]) -> Literal["end", "continue"]:
    if state["reflection_iterations"] > 3:
        return "end"
    if len(state["messages"]) == 0:
        return "end"

    if isinstance(state["messages"][-1], HumanMessage):
        return "continue"

    logger.info(f'final response from the reflection node: {state["messages"][-1]}')
    return "end"


# Define the function to execute tools
tool_node = ToolNode(tools)
