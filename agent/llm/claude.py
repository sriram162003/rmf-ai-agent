"""Anthropic Claude backend."""

from __future__ import annotations

import os
from typing import Any

from agent.llm.base import LLMBackend


class ClaudeBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = "") -> None:
        import anthropic

        self._model = model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )

    @property
    def name(self) -> str:
        return f"claude/{self._model}"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> dict:
        # Convert generic tool format → Anthropic format
        anthropic_tools = []
        if tools:
            for t in tools:
                anthropic_tools.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input, "id": block.id})

        stop_map = {"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens"}
        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
            "stop_reason": stop_map.get(response.stop_reason, "end_turn"),
        }
