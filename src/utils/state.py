from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from src.common.dataclasses import SurfaceInfo


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_name: str | None = None
    reflection_iterations: int
    reflection_complete: bool = False
    condensed_query: str | None = None
    context_summary: str | None = None
