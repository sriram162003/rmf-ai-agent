"""Ollama (local LLM) backend."""

from __future__ import annotations

import json

import httpx

from agent.llm.base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(
        self, model: str = "llama3", base_url: str = "http://localhost:11434"
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> dict:
        all_messages = list(messages)
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        payload: dict = {
            "model": self._model,
            "messages": all_messages,
            "stream": False,
        }

        # Inject tool descriptions into system prompt (Ollama JSON tool support varies by model)
        if tools:
            tool_desc = json.dumps(tools, indent=2)
            tool_instruction = (
                f"\n\nYou have access to the following tools. "
                f"To call a tool respond with a JSON object: "
                f'{{\"tool\": \"<name>\", \"arguments\": {{...}}}}.\n\n{tool_desc}'
            )
            if all_messages and all_messages[0]["role"] == "system":
                all_messages[0]["content"] += tool_instruction
            else:
                all_messages.insert(0, {"role": "system", "content": tool_instruction})
            payload["messages"] = all_messages
            payload["format"] = "json"

        resp = await self._http.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        tool_calls = []

        # Try to parse tool call from JSON response
        if tools and content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_calls.append(
                        {
                            "name": parsed["tool"],
                            "arguments": parsed.get("arguments", {}),
                            "id": "ollama-0",
                        }
                    )
                    content = ""
            except (json.JSONDecodeError, KeyError):
                pass

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return {"content": content, "tool_calls": tool_calls, "stop_reason": stop_reason}

    async def close(self) -> None:
        await self._http.aclose()
