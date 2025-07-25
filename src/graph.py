from __future__ import annotations

import logging

from typing import Any, AsyncGenerator, Dict, List, Literal, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from src.agents.utils.nodes import (
    call_model,
    context_enrichment,
    evaluate_agent_response,
    should_continue,
    should_continue_reflection,
    tool_node,
)

from src.agents.utils.state import AgentState


# Initialize logger
LOGGER = load_logger(__name__)


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["azure_openai", "aws_bedrock_sonnet", "aws_bedrock_haiku"]


__all__ = ["graph", "run_core_agent", "process_non_streaming"]


def create_checkpointer() -> MemorySaver:
    """
    Create and configure the checkpointer for the agent graph.

    Returns:
        Union[MongoDBSaver, MemorySaver, None]: The configured checkpointer instance, or None.
    """
    return MemorySaver()



def create_agent_graph() -> CompiledStateGraph:
    """
    Create and configure the agent graph.

    Returns:
        CompiledStateGraph: The configured and compiled state graph
    """
    checkpointer = create_checkpointer()

    # Define a new graph
    workflow = StateGraph(AgentState, config_schema=GraphConfig)

    # Define the nodes
    workflow.add_node("context_enrichment", context_enrichment)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_node("reflection", evaluate_agent_response)
    # Set the entrypoint as context_enrichment
    workflow.set_entry_point("context_enrichment")

    # Add edges
    workflow.add_edge("context_enrichment", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "reflect": "reflection",
        },
    )
    workflow.add_conditional_edges("reflection", should_continue_reflection, {"continue": "agent", "end": END})
    workflow.add_edge("action", "agent")

    return workflow.compile(checkpointer=checkpointer)


# Create the compiled graph
graph: CompiledStateGraph = create_agent_graph()


def run_core_agent(agent_input: Dict[str, Any], session_id: str = None) -> AsyncGenerator[str, None]:
    """
    Run the core agent with the given input that matches v1 format.
   
    str: Final response and tool invocation announcements, without raw tool outputs.
    """
    user_task = agent_input.get("message", "")
    user_name = agent_input.get("user name", "unknown")

    # Use session_id as thread_id for Slack channel threads
    config = {"configurable": {"user_id": user_name}}

    # If session_id is provided, merge it into the metadata
    if user_name:
        config["metadata"] = {"user_id": user_name}

    # Get the async generator from astream
    initial_state = {
        "messages": [HumanMessage(content=user_task)],
        "user_name": user_name,
        "reflection_iterations": 0,
        "reflection_complete": False,
    }
    events = graph.stream(initial_state, config, stream_mode="values")

    # Process each event from the generator
    for event in events:
        if isinstance(event, dict) and "messages" in event and event["messages"]:
            last_message = event["messages"][-1].content

            if (
                last_message.strip()
                and isinstance(event["messages"][-1], AIMessage)
                and (
                    event.get("reflection_iterations", 0) > MAX_REFLECTION_ITERATIONS
                    or event.get("reflection_complete")
                )
            ):
                LOGGER.info("Sending final response to the user.......")
                LOGGER.info(f"last_message: {last_message}")
                yield last_message.strip()
                break


def process_non_streaming(agent_input: Dict[str, Any], session_id: str = None) -> str:
    """
    Process the input in non-streaming mode and return complete response.

    Args:
        agent_input (Dict[str, Any]): Dictionary containing:
            - user name (str): The name of the user
            - message (str): The user's message
            - token (str): Authentication token
            - surface_info (Dict[str, Any]): Information about the surface type containing:
                - surface (Surface): The surface enum value
                - type (SurfaceType): The surface type enum value
                - source (str): The source of the request
        session_id (str, optional): The session ID for the request.

    Returns:
        str: The complete response including tool usage and final answer.
    """
    # Initialize empty list to store response chunks
    response_chunks = []

    # Collect chunks from the generator
    for chunk in run_core_agent(agent_input, session_id):
        # Only append non-empty chunks after stripping whitespace
        if chunk and chunk.strip():
            response_chunks.append(str(chunk))

    # Join chunks with newlines and return
    return "\n".join(response_chunks)
