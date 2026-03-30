"""OpenAI backend."""

from __future__ import annotations

import json
import os

from agent.llm.base import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o", api_key: str = "") -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> dict:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # Convert generic tool format → OpenAI function format
        openai_tools = []
        if tools:
            for t in tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )

        kwargs = {"model": self._model, "messages": all_messages}
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments or "{}"),
                        "id": tc.id,
                    }
                )

        finish = response.choices[0].finish_reason
        stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
        return {
            "content": msg.content or "",
            "tool_calls": tool_calls,
            "stop_reason": stop_map.get(finish, "end_turn"),
        }
