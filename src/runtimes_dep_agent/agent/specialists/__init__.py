"""Factories for specialist LangChain agents used by the supervisor."""

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable


@dataclass
class SpecialistSpec:
    """Container describing a specialist agent and the tool exposed to the supervisor."""
 
    name: str
    agent: Runnable
    tool: Any


__all__ = ["SpecialistSpec"]
