"""
AI Agent Core — ReAct loop (Reason → Act → Observe).

Receives natural language commands, uses tools to interact with Open-RMF,
and streams responses back to the dashboard via an async generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.hitl import HITLGate
    from agent.llm.base import LLMBackend
    from agent.planner import TaskPlanner
    from agent.replanner import TaskMonitor
    from rmf.client import RMFClient
    from rmf.fleet_scanner import FleetScanner

logger = logging.getLogger(__name__)

MAX_TURNS = 10  # Maximum ReAct iterations before stopping

_SYSTEM_PROMPT = """
You are an AI agent controlling a multi-robot fleet via the Open-RMF platform.

You can dispatch tasks, monitor progress, control building infrastructure (doors, lifts),
and manage fleets. Always reason step-by-step before taking actions.

When the user gives a command:
1. List available fleets/robots/doors/lifts if context is needed.
2. For robot tasks: plan and dispatch using the appropriate task category.
3. For doors: use control_door(open/close).
4. For lifts: use call_lift with the destination floor name (e.g. L1, L2, B1).
5. For fleet management: decommission/recommission requires human approval.
6. Always report task IDs, door names, and lift states clearly.

Be concise. If an action affects safety or is irreversible, request human approval first.
"""

# -----------------------------------------------------------------------
# Tool definitions (generic format understood by all LLM backends)
# -----------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_fleets",
        "description": "List all available robot fleets and their current robot states.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "scan_fleets",
        "description": "Trigger a fresh fleet discovery scan (RMF API + WiFi + BLE).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "dispatch_task",
        "description": "Dispatch a task to the robot fleet via Open-RMF.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Natural language description of the task to perform.",
                },
                "fleet": {
                    "type": "string",
                    "description": "Target fleet name (optional; omit to let RMF choose).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "get_task_status",
        "description": "Get the current status of a dispatched task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The task ID to query."}
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "cancel_task",
        "description": "Cancel a running or queued task (requires human approval).",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "request_human_approval",
        "description": "Request human approval before executing a sensitive action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Short action name."},
                "description": {"type": "string", "description": "Why approval is needed."},
            },
            "required": ["action", "description"],
        },
    },
    {
        "name": "list_active_tasks",
        "description": "List all currently active/queued tasks.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_doors",
        "description": "List all doors in the facility and their current states.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "control_door",
        "description": "Open or close a specific door.",
        "parameters": {
            "type": "object",
            "properties": {
                "door_name": {"type": "string", "description": "Name of the door to control."},
                "mode": {
                    "type": "string",
                    "enum": ["open", "close"],
                    "description": "Desired door state.",
                },
            },
            "required": ["door_name", "mode"],
        },
    },
    {
        "name": "list_lifts",
        "description": "List all lifts/elevators in the facility and their current states.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "call_lift",
        "description": "Call a lift to a specific floor.",
        "parameters": {
            "type": "object",
            "properties": {
                "lift_name": {"type": "string", "description": "Name of the lift."},
                "destination_floor": {
                    "type": "string",
                    "description": "Target floor name (e.g. 'L1', 'L2', 'B1').",
                },
            },
            "required": ["lift_name", "destination_floor"],
        },
    },
    {
        "name": "manage_fleet",
        "description": "Decommission (stop accepting tasks) or recommission a fleet.",
        "parameters": {
            "type": "object",
            "properties": {
                "fleet_name": {"type": "string"},
                "operation": {
                    "type": "string",
                    "enum": ["decommission", "recommission"],
                    "description": "decommission=stop tasks, recommission=re-enable.",
                },
            },
            "required": ["fleet_name", "operation"],
        },
    },
]


class AgentCore:
    def __init__(
        self,
        llm: "LLMBackend",
        rmf: "RMFClient",
        fleet_scanner: "FleetScanner",
        task_monitor: "TaskMonitor",
        hitl: "HITLGate",
        planner: "TaskPlanner",
        require_approval_for: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._rmf = rmf
        self._scanner = fleet_scanner
        self._monitor = task_monitor
        self._hitl = hitl
        self._planner = planner
        self._require_approval_for: set[str] = set(require_approval_for or [])

        # Register replanner callback
        self._monitor.on_failure(self._on_task_failure)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def chat(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Process a user message via the ReAct loop.
        Yields streamed text chunks to the caller.
        """
        messages: list[dict] = [{"role": "user", "content": user_message}]

        for turn in range(MAX_TURNS):
            response = await self._llm.chat(messages, tools=TOOLS, system=_SYSTEM_PROMPT)

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])
            stop_reason = response.get("stop_reason", "end_turn")

            if content:
                yield content

            if stop_reason == "end_turn" or not tool_calls:
                break

            # Add assistant response to message history
            messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

            # Execute each tool call
            tool_results = []
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc.get("arguments", {})
                tc_id = tc.get("id", tool_name)

                yield f"\n> Calling `{tool_name}`...\n"

                try:
                    result = await self._execute_tool(tool_name, args)
                except Exception as exc:
                    result = {"error": str(exc)}
                    logger.exception("Tool %s failed", tool_name)

                tool_results.append(
                    {
                        "tool_call_id": tc_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )

            # Feed tool results back into the conversation
            messages.extend(tool_results)

        else:
            yield "\n[Agent reached maximum reasoning steps]"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "list_fleets":
            return await self._tool_list_fleets()
        elif name == "scan_fleets":
            return await self._tool_scan_fleets()
        elif name == "dispatch_task":
            return await self._tool_dispatch_task(args)
        elif name == "get_task_status":
            return await self._tool_get_task_status(args)
        elif name == "cancel_task":
            return await self._tool_cancel_task(args)
        elif name == "request_human_approval":
            return await self._tool_request_approval(args)
        elif name == "list_active_tasks":
            return {"active_tasks": self._monitor.active_tasks()}
        elif name == "list_doors":
            return await self._tool_list_doors()
        elif name == "control_door":
            return await self._tool_control_door(args)
        elif name == "list_lifts":
            return await self._tool_list_lifts()
        elif name == "call_lift":
            return await self._tool_call_lift(args)
        elif name == "manage_fleet":
            return await self._tool_manage_fleet(args)
        else:
            return {"error": f"Unknown tool: {name}"}

    async def _tool_list_fleets(self) -> dict:
        return await self._scanner.registry.snapshot()

    async def _tool_scan_fleets(self) -> dict:
        snapshot = await self._scanner.scan_once()
        return {"fleets": snapshot, "count": len(snapshot)}

    async def _tool_dispatch_task(self, args: dict) -> dict:
        command = args.get("command", "")
        fleet = args.get("fleet", "")

        # Build fleet context for the planner
        snapshot = await self._scanner.registry.snapshot()
        fleet_context = json.dumps(snapshot, indent=2) if snapshot else ""

        # Check if dispatch requires approval
        if "dispatch_task" in self._require_approval_for:
            approved = await self._hitl.request_approval(
                "dispatch_task",
                f"Dispatch task: {command}",
                {"command": command, "fleet": fleet},
            )
            if not approved:
                return {"status": "rejected", "reason": "Human approval denied or timed out"}

        # Refresh task types then plan
        await self._planner.refresh_task_types()
        task_request = await self._planner.plan(command, fleet_context)

        if not task_request:
            return {"error": "Could not convert command to a valid task request"}

        result = await self._rmf.dispatch_task(task_request)
        task_id = result.get("state", {}).get("booking", {}).get("id", "")

        if task_id:
            self._monitor.track(task_id, {"command": command, "fleet": fleet, "request": task_request})

        return {"task_id": task_id, "status": "dispatched", "detail": result}

    async def _tool_get_task_status(self, args: dict) -> dict:
        task_id = args.get("task_id", "")
        state = await self._rmf.get_task_state(task_id)
        return state

    async def _tool_cancel_task(self, args: dict) -> dict:
        task_id = args.get("task_id", "")
        reason = args.get("reason", "")

        approved = await self._hitl.request_approval(
            "cancel_task",
            f"Cancel task {task_id}: {reason}",
            {"task_id": task_id},
        )
        if not approved:
            return {"status": "rejected", "reason": "Cancellation not approved"}

        result = await self._rmf.cancel_task(task_id)
        return {"status": "canceled", "detail": result}

    async def _tool_request_approval(self, args: dict) -> dict:
        action = args.get("action", "")
        description = args.get("description", "")
        approved = await self._hitl.request_approval(action, description)
        return {"approved": approved}

    async def _tool_list_doors(self) -> dict:
        doors = await self._rmf.get_doors()
        result = {}
        for d in doors:
            name = d.get("door", {}).get("name", "") or d.get("name", "")
            if name:
                try:
                    state = await self._rmf.get_door_state(name)
                    d["state"] = state
                except Exception:
                    pass
                result[name] = d
        return result

    async def _tool_control_door(self, args: dict) -> dict:
        door_name = args.get("door_name", "")
        mode_str = args.get("mode", "open")
        mode = 2 if mode_str == "open" else 0
        result = await self._rmf.request_door(door_name, mode)
        return {"door": door_name, "requested_mode": mode_str, "detail": result}

    async def _tool_list_lifts(self) -> dict:
        lifts = await self._rmf.get_lifts()
        result = {}
        for lft in lifts:
            name = lft.get("name", "")
            if name:
                try:
                    state = await self._rmf.get_lift_state(name)
                    lft["state"] = state
                except Exception:
                    pass
                result[name] = lft
        return result

    async def _tool_call_lift(self, args: dict) -> dict:
        lift_name = args.get("lift_name", "")
        floor = args.get("destination_floor", "")
        result = await self._rmf.request_lift(lift_name, floor)
        return {"lift": lift_name, "requested_floor": floor, "detail": result}

    async def _tool_manage_fleet(self, args: dict) -> dict:
        fleet_name = args.get("fleet_name", "")
        operation = args.get("operation", "")

        if operation == "decommission":
            approved = await self._hitl.request_approval(
                "decommission_fleet",
                f"Decommission fleet '{fleet_name}' — it will stop accepting tasks.",
                {"fleet": fleet_name},
            )
            if not approved:
                return {"status": "rejected"}
            result = await self._rmf.decommission_fleet(fleet_name)
        elif operation == "recommission":
            result = await self._rmf.recommission_fleet(fleet_name)
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {"fleet": fleet_name, "operation": operation, "detail": result}

    # ------------------------------------------------------------------
    # Replanner callback
    # ------------------------------------------------------------------

    async def _on_task_failure(self, task_id: str, state: str, metadata: dict) -> None:
        logger.warning("Replanning triggered for task %s (state=%s)", task_id, state)
        command = metadata.get("command", "")
        if not command:
            return

        if metadata.get("_escalate_to_hitl"):
            await self._hitl.request_approval(
                "replan_task",
                f"Task {task_id} failed {state} repeatedly. Manual intervention needed.",
                metadata,
            )
            return

        # Re-dispatch the same command automatically
        try:
            await self._tool_dispatch_task({"command": command, "fleet": metadata.get("fleet", "")})
            logger.info("Replan successful for original command: %s", command)
        except Exception as exc:
            logger.error("Replan failed: %s", exc)
