"""
Async client for the Open-RMF API server.

Wraps REST endpoints (httpx) and real-time Socket.IO subscriptions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

import httpx
import socketio

logger = logging.getLogger(__name__)


class RMFClient:
    """Async HTTP + Socket.IO client for Open-RMF API server."""

    def __init__(self, api_url: str = "http://localhost:8000", jwt_token: str = ""):
        self.api_url = api_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if jwt_token:
            self._headers["Authorization"] = f"Bearer {jwt_token}"

        self._http = httpx.AsyncClient(
            base_url=self.api_url,
            headers=self._headers,
            timeout=15.0,
        )

        # Socket.IO for real-time updates
        self._sio = socketio.AsyncClient(logger=False, engineio_logger=False)
        self._sio_connected = False
        self._task_callbacks: list[Callable[[dict], Coroutine]] = []
        self._fleet_callbacks: list[Callable[[dict], Coroutine]] = []

        @self._sio.event
        async def connect():
            self._sio_connected = True
            logger.info("Socket.IO connected to RMF API server")

        @self._sio.event
        async def disconnect():
            self._sio_connected = False
            logger.warning("Socket.IO disconnected from RMF API server")

        @self._sio.on("task_state_update")
        async def on_task_update(data):
            for cb in self._task_callbacks:
                await cb(data)

        @self._sio.on("fleet_state_update")
        async def on_fleet_update(data):
            for cb in self._fleet_callbacks:
                await cb(data)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect Socket.IO to the RMF API server."""
        if self._sio_connected:
            return
        sio_url = self.api_url
        auth = {"token": self._headers.get("Authorization", "").replace("Bearer ", "")}
        try:
            await self._sio.connect(sio_url, auth=auth, wait_timeout=10)
        except Exception as exc:
            logger.warning("Socket.IO connection failed: %s (real-time updates disabled)", exc)

    async def close(self) -> None:
        await self._http.aclose()
        if self._sio_connected:
            await self._sio.disconnect()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_task_update(self, callback: Callable[[dict], Coroutine]) -> None:
        self._task_callbacks.append(callback)

    def on_fleet_update(self, callback: Callable[[dict], Coroutine]) -> None:
        self._fleet_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Fleet endpoints
    # ------------------------------------------------------------------

    async def get_fleets(self) -> list[dict]:
        """GET /fleets — list all registered fleets."""
        resp = await self._http.get("/fleets")
        resp.raise_for_status()
        return resp.json()

    async def get_fleet_state(self, fleet_name: str) -> dict:
        """GET /fleets/{name}/state — full fleet + robot state."""
        resp = await self._http.get(f"/fleets/{fleet_name}/state")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Task endpoints
    # ------------------------------------------------------------------

    async def dispatch_task(self, request: dict) -> dict:
        """POST /tasks/dispatch_task — submit a new task."""
        resp = await self._http.post("/tasks/dispatch_task", json=request)
        resp.raise_for_status()
        return resp.json()

    async def cancel_task(self, task_id: str, requester: str = "rmf-ai-agent") -> dict:
        """POST /tasks/cancel_task."""
        resp = await self._http.post(
            "/tasks/cancel_task",
            json={"type": "cancel_task_request", "task_id": task_id, "requester": requester},
        )
        resp.raise_for_status()
        return resp.json()

    async def interrupt_task(self, task_id: str, labels: list[str] | None = None) -> dict:
        resp = await self._http.post(
            "/tasks/interrupt_task",
            json={"task_id": task_id, "labels": labels or []},
        )
        resp.raise_for_status()
        return resp.json()

    async def resume_task(self, task_id: str) -> dict:
        resp = await self._http.post("/tasks/resume_task", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    async def get_task_state(self, task_id: str) -> dict:
        """GET /tasks/{task_id}/state."""
        resp = await self._http.get(f"/tasks/{task_id}/state")
        resp.raise_for_status()
        return resp.json()

    async def get_tasks(self, **filters: Any) -> list[dict]:
        """GET /tasks — query task list with optional filters."""
        resp = await self._http.get("/tasks", params=filters)
        resp.raise_for_status()
        return resp.json()

    async def discover_task_types(self) -> dict:
        """POST /tasks/task_discovery — get available task categories."""
        resp = await self._http.post("/tasks/task_discovery", json={})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Infrastructure — Doors
    # ------------------------------------------------------------------

    async def get_doors(self) -> list[dict]:
        resp = await self._http.get("/doors")
        resp.raise_for_status()
        return resp.json()

    async def get_door_state(self, door_name: str) -> dict:
        """GET /doors/{door_name}/state"""
        resp = await self._http.get(f"/doors/{door_name}/state")
        resp.raise_for_status()
        return resp.json()

    async def request_door(self, door_name: str, mode: int) -> dict:
        """
        POST /doors/{door_name}/request
        mode: 0 = CLOSED, 2 = OPEN
        """
        resp = await self._http.post(
            f"/doors/{door_name}/request",
            json={"mode": mode},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Infrastructure — Lifts
    # ------------------------------------------------------------------

    async def get_lifts(self) -> list[dict]:
        resp = await self._http.get("/lifts")
        resp.raise_for_status()
        return resp.json()

    async def get_lift_state(self, lift_name: str) -> dict:
        """GET /lifts/{lift_name}/state"""
        resp = await self._http.get(f"/lifts/{lift_name}/state")
        resp.raise_for_status()
        return resp.json()

    async def request_lift(
        self,
        lift_name: str,
        destination_floor: str,
        door_state: int = 2,
        request_type: int = 1,
    ) -> dict:
        """
        POST /lifts/{lift_name}/request
        door_state: 0=CLOSED, 1=AGV_ZONE, 2=HUMAN_ZONE
        request_type: 1=AGV_MODE, 2=HUMAN_MODE
        """
        resp = await self._http.post(
            f"/lifts/{lift_name}/request",
            json={
                "destination_floor": destination_floor,
                "door_state": door_state,
                "request_type": request_type,
            },
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Infrastructure — Fleet-level lane management
    # ------------------------------------------------------------------

    async def decommission_fleet(self, fleet_name: str) -> dict:
        """POST /fleets/{name}/decommission — stop fleet from accepting tasks."""
        resp = await self._http.post(f"/fleets/{fleet_name}/decommission")
        resp.raise_for_status()
        return resp.json()

    async def recommission_fleet(self, fleet_name: str) -> dict:
        """POST /fleets/{name}/recommission — re-enable fleet for tasks."""
        resp = await self._http.post(f"/fleets/{fleet_name}/recommission")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Building map
    # ------------------------------------------------------------------

    async def get_building(self) -> dict:
        resp = await self._http.get("/building")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return True if the RMF API server is reachable."""
        try:
            resp = await self._http.get("/time", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
