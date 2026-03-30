"""
Natural language → structured RMF task request planner.

Uses the LLM to convert a free-text command into a valid
rmf_api_msgs DispatchTaskRequest JSON payload.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.llm.base import LLMBackend
    from rmf.client import RMFClient

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """
You are a task planning assistant for a robot fleet managed by Open-RMF.

Your job is to convert a natural language command into a valid JSON task dispatch
request for the Open-RMF API.

The response MUST be a single JSON object with this structure:
{
  "type": "dispatch_task_request",
  "request": {
    "unix_millis_earliest_start_time": 0,
    "priority": {"type": "binary", "value": 0},
    "category": "<task_category>",
    "description": {<task-specific fields>},
    "labels": ["<optional label>"],
    "requester": "rmf-ai-agent"
  }
}

Common categories and their description schemas:
- "patrol": {"places": ["<waypoint1>", "<waypoint2>"], "rounds": 1}
- "delivery": {"pickup": {"place": "<place>", "handler": "<handler>", "payload": [{}]},
               "dropoff": {"place": "<place>", "handler": "<handler>", "payload": [{}]}}
- "clean": {"zone": "<zone_name>"}
- "go_to_place": {"place": "<waypoint_name>"}

Use the available_task_types context if provided to pick the right category.
Respond ONLY with the JSON object, no explanation.
"""


class TaskPlanner:
    def __init__(self, llm: "LLMBackend", rmf: "RMFClient") -> None:
        self._llm = llm
        self._rmf = rmf
        self._task_types: list[str] = []

    async def refresh_task_types(self) -> None:
        try:
            result = await self._rmf.discover_task_types()
            self._task_types = result.get("available_tasks", [])
        except Exception as exc:
            logger.warning("Could not discover task types: %s", exc)

    async def plan(self, command: str, fleet_context: str = "") -> dict | None:
        """
        Convert a natural language command to an RMF task dispatch request.
        Returns the parsed dict or None if planning fails.
        """
        context_parts = []
        if self._task_types:
            context_parts.append(f"Available task types: {', '.join(self._task_types)}")
        if fleet_context:
            context_parts.append(f"Fleet context:\n{fleet_context}")

        user_content = command
        if context_parts:
            user_content = "\n\n".join(context_parts) + f"\n\nCommand: {command}"

        messages = [{"role": "user", "content": user_content}]
        response = await self._llm.chat(messages, system=_PLANNER_SYSTEM)
        text = response.get("content", "").strip()

        if not text:
            logger.warning("Planner returned empty response")
            return None

        try:
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Planner JSON parse error: %s\nRaw: %s", exc, text)
            return None
