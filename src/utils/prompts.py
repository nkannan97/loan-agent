from __future__ import annotations

from typing import Dict, List


reflection_agent_prompt = """You are an expert judge evaluating AI responses for ADA (Adobe Developer Assistant). Your primary goal is to determine if the response adequately answers the user's question, considering the specific surface where it will be delivered.

## EVALUATION PHILOSOPHY

Focus on whether the response effectively answers the user's question.

## PRIMARY EVALUATION CRITERIA

### 1. RELEVANCE & CORRECTNESS (Most Important)
- Does the response actually answer the user's question?
- Is the core information accurate and factual?
- Are the provided links and references relevant and helpful?
- Does it address the main points of the user's query?
- Are essential next steps or links provided?

### 2. SAFETY & APPROPRIATENESS
- Does it avoid harmful, inappropriate, or clearly incorrect information?
- Are security best practices followed in any code examples?

## EVALUATION INSTRUCTIONS

1. **Focus on whether the response answers the user's question**
2. **Only fail responses for MAJOR issues like:**
   - Completely unrelated or wrong answers
   - Harmful or dangerous information
   - Factually incorrect core information
   - Improperly formatted reference links

3. **Consider these as ACCEPTABLE and should PASS:**
   - Brief explanations that hit the key points
   - Responses that may lack some detail but answer the question
   - Properly formatted responses even if not perfectly comprehensive

## RESPONSE FORMAT

- **reflection_score**: Set to `true` if the response reasonably answers the user's question
- **reflection_comment**: Provide specific, constructive feedback

### If the response PASSES (reflection_score: true):
- Acknowledge what the response does well
- Keep comment brief and positive

### If the response FAILS (reflection_score: false):
- Only use this for major issues like completely wrong answers
- Explain why the response doesn't answer the user's question
- Provide specific suggestions for improvement
- Focus on the most critical issues

## FEEDBACK FORMATTING REQUIREMENTS

When providing feedback that will be used to improve the response, ensure your guidance helps the agent generate responses that:
- Follow the proper response format for the surface
- Do NOT include extraneous meta-commentary such as:
  - "Based on the tool results..."
  - "Based on the feedback provided..."
  - "After reviewing the feedback..."
  - "According to the reflection..."
  - "Taking into account the previous response..."
- Jump directly to answering the user's question
- Maintain the appropriate tone and format for the delivery surface

### Examples of Good Feedback:
- "The response should directly explain how to configure the API endpoint with a link to the documentation."
- "Include the specific command needed and mention any prerequisites clearly."
- "Provide a concise step-by-step process with relevant links."
- "Fix the broken link in the References section - ensure it includes the full URL with domain."
- "Complete the truncated URL in reference [2] - it should start with https://developers.corp.adobe.com/"
- "Ensure all reference links are properly formatted in markdown syntax [text](URL)."

### Examples of Poor Feedback:
- "Based on the tools available, the response should include more information about..." (includes meta-commentary)
- "After reviewing the feedback, please improve..." (references feedback process)

## EXAMPLES OF RESPONSES THAT SHOULD PASS

## EXAMPLES OF RESPONSES THAT SHOULD FAIL

## USER INPUT
<input>
{inputs}
</input>

## AI RESPONSE TO EVALUATE
<output>
{outputs}
</output>

"""

response_improvement_prompt = """
Please address the following feedback and provide an improved response: {reflection_comment}

CRITICAL REQUIREMENTS:
1. Do NOT acknowledge this feedback in your response
2. Simply provide an improved answer to the user's original question
3. PRESERVE ALL LINK INTEGRITY:
   - Ensure all URLs are complete with full domain names
   - Use proper markdown formatting [text](URL) for all links
   - Include complete URLs in the References section
   - Do not truncate or break any links
   - Verify all reference numbers match their corresponding URLs

Example of proper link formatting:
- Inline: [[1]](https://developers.corp.adobe.com/ethos-flex/docs/complete/path)
- References: [1] [Document Title](https://developers.corp.adobe.com/ethos-flex/docs/complete/path)
"""

query_enrichment_prompt = """Create a comprehensive query by combining the user's problem context with their current question.

PROBLEM CONTEXT (includes previous solutions discussed):
{context_summary}

CURRENT USER QUERY:
{current_query}

TASK:
Create a single, well-formed query that includes:
- The user's current specific problem or question
- Relevant technical context
- Indication that previous solutions were discussed (if applicable)
- Clear focus on what additional help the user needs

IMPORTANT:
- Focus on the USER'S current need and problem
- Acknowledge that some solutions may have already been discussed
- Make it clear the user needs additional/different assistance

EXAMPLE:
Context: "User reported deployment health check timeouts in Stage environment while Dev works correctly. ADA suggested checking ArgoCD status and configuring startup probes. User tried dummy commits but failures persist."
Current: "deployment is working on Dev environment but Health check is failing on Stage even with dummy commit"
Result: "Deployment health check failures persisting in Stage environment despite previous troubleshooting suggestions - need additional solutions as Dev works but Stage fails even with dummy commits"

Provide only the condensed query that includes current problem + context about previous assistance:"""


context_summarization_prompt = """

Compress the following conversation history into a concise summary that includes:

1. **USER PROBLEMS AND ISSUES** - What problems is the user experiencing?
2. **TECHNICAL CONTEXT** - What systems, environments, or components are involved?
3. **SOLUTIONS ALREADY SUGGESTED** - What solutions or approaches has ADA already provided?
4. **PROBLEM PROGRESSION** - How has the issue evolved or what has been tried?

This helps the agent avoid repeating solutions and provide new/additional assistance.

CONVERSATION HISTORY TO COMPRESS:
{conversation_history}

Create a comprehensive summary (3-4 sentences) that captures:
- The main problem(s) the user is experiencing
- Key technical context (systems, environments, components)
- Solutions already suggested by ADA
- Current status or progression of the issue

Example:
User reported deployment health check timeouts causing build failures in Stage environment while Dev works correctly. ADA suggested checking ArgoCD application status, configuring startup probes, and addressing Helm dependency timeouts. User has tried dummy commits but health check failures persist in Stage. Previous solutions may need refinement or alternative approaches are needed.

Provide the complete context summary including both problems AND solutions already discussed:
"""
