"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """
    Minimal interface for LLM backends.
    Subclasses translate between this common interface and provider SDKs.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> dict:
        """
        Send messages to the LLM and return a response dict with keys:
          - content (str): final text response (may be empty if tool_calls present)
          - tool_calls (list[dict]): list of {name, arguments} dicts (may be empty)
          - stop_reason (str): "end_turn" | "tool_use" | "max_tokens"
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
