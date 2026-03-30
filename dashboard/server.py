"""
Dashboard server — FastAPI on port 18789.

Provides:
  - Single-page HTML dashboard (GET /)
  - REST API for fleets, tasks, HITL approvals
  - WebSocket /ws/chat  — bidirectional NL agent chat
  - WebSocket /ws/status — real-time fleet/task status feed
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from agent.core import AgentCore
from agent.hitl import HITLGate
from agent.planner import TaskPlanner
from agent.replanner import TaskMonitor
from rmf.client import RMFClient
from rmf.fleet_scanner import FleetRegistry, FleetScanner
from dashboard.setup import router as setup_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_llm(cfg: dict):
    backend = cfg.get("llm", {}).get("backend", "claude")
    model = cfg.get("llm", {}).get("model", "claude-sonnet-4-6")
    api_key = cfg.get("llm", {}).get("api_key", "") or ""

    if backend == "claude":
        from agent.llm.claude import ClaudeBackend
        return ClaudeBackend(model=model, api_key=api_key)
    elif backend == "openai":
        from agent.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model, api_key=api_key)
    elif backend == "ollama":
        from agent.llm.ollama import OllamaBackend
        ollama_url = cfg.get("llm", {}).get("ollama_url", "http://localhost:11434")
        return OllamaBackend(model=model, base_url=ollama_url)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    rmf: RMFClient
    registry: FleetRegistry
    scanner: FleetScanner
    hitl: HITLGate
    monitor: TaskMonitor
    planner: TaskPlanner
    agent: AgentCore
    status_connections: list[WebSocket]

    def __init__(self):
        self.status_connections = []


state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    rmf_cfg = cfg.get("rmf", {})
    scanner_cfg = cfg.get("fleet_scanner", {})
    hitl_cfg = cfg.get("hitl", {})

    # RMF client
    state.rmf = RMFClient(
        api_url=os.environ.get("RMF_API_URL", rmf_cfg.get("api_url", "http://localhost:8000")),
        jwt_token=rmf_cfg.get("jwt_token", ""),
    )
    await state.rmf.connect()

    # Fleet registry + scanner
    state.registry = FleetRegistry()
    state.scanner = FleetScanner(
        rmf_client=state.rmf,
        registry=state.registry,
        rmf_poll_interval=scanner_cfg.get("rmf_poll_interval", 10),
        wifi_mdns=scanner_cfg.get("wifi_mdns", True),
        bluetooth=scanner_cfg.get("bluetooth", True),
        ble_service_uuid=scanner_cfg.get("ble_service_uuid", "12345678-1234-1234-1234-123456789abc"),
        ble_scan_duration=scanner_cfg.get("ble_scan_duration", 5.0),
    )
    await state.scanner.start()

    # HITL gate
    state.hitl = HITLGate(auto_approve_timeout=hitl_cfg.get("auto_approve_timeout", 60))

    # Task monitor
    state.monitor = TaskMonitor()
    state.rmf.on_task_update(state.monitor.handle_task_update)

    # Forward fleet updates to status WebSocket subscribers
    async def broadcast_fleet(data: dict):
        await _broadcast_status({"type": "fleet_update", "data": data})

    state.rmf.on_fleet_update(broadcast_fleet)

    # LLM + planner + agent
    llm = build_llm(cfg)
    state.planner = TaskPlanner(llm=llm, rmf=state.rmf)
    state.agent = AgentCore(
        llm=llm,
        rmf=state.rmf,
        fleet_scanner=state.scanner,
        task_monitor=state.monitor,
        hitl=state.hitl,
        planner=state.planner,
        require_approval_for=hitl_cfg.get("require_approval_for", []),
    )

    logger.info("rmf-ai-agent started")
    yield

    await state.scanner.stop()
    await state.rmf.close()
    logger.info("rmf-ai-agent stopped")


# ---------------------------------------------------------------------------
# Live connection re-initialisation (called by setup wizard after save)
# ---------------------------------------------------------------------------

async def reinit_connections(cfg: dict) -> None:
    """Tear down and rebuild RMF client + fleet scanner from updated config."""
    rmf_cfg = cfg.get("rmf", {})
    scanner_cfg = cfg.get("fleet_scanner", {})

    # Stop existing scanner and close RMF connection
    try:
        await state.scanner.stop()
    except Exception:
        pass
    try:
        await state.rmf.close()
    except Exception:
        pass

    # Re-create RMF client
    state.rmf = RMFClient(
        api_url=os.environ.get("RMF_API_URL", rmf_cfg.get("api_url", "http://localhost:8000")),
        jwt_token=rmf_cfg.get("jwt_token", ""),
    )
    await state.rmf.connect()

    # Re-wire fleet update broadcast
    async def broadcast_fleet(data: dict):
        await _broadcast_status({"type": "fleet_update", "data": data})

    state.rmf.on_fleet_update(broadcast_fleet)
    state.rmf.on_task_update(state.monitor.handle_task_update)

    # Re-create fleet scanner
    state.scanner = FleetScanner(
        rmf_client=state.rmf,
        registry=state.registry,
        rmf_poll_interval=scanner_cfg.get("rmf_poll_interval", 10),
        wifi_mdns=scanner_cfg.get("wifi_mdns", True),
        bluetooth=scanner_cfg.get("bluetooth", True),
        ble_service_uuid=scanner_cfg.get("ble_service_uuid", "12345678-1234-1234-1234-123456789abc"),
        ble_scan_duration=scanner_cfg.get("ble_scan_duration", 5.0),
    )
    await state.scanner.start()

    logger.info("rmf-ai-agent connections re-initialised")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RMF AI Agent", version="1.0.0", lifespan=lifespan)
app.include_router(setup_router)

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _TEMPLATES.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# Fleet API
# ---------------------------------------------------------------------------

@app.get("/api/fleets")
async def get_fleets():
    return await state.registry.snapshot()


@app.post("/api/fleets/scan")
async def trigger_scan():
    result = await state.scanner.scan_once()
    return {"fleets": result, "count": len(result)}


# ---------------------------------------------------------------------------
# Task API (proxied from RMF)
# ---------------------------------------------------------------------------

@app.get("/api/tasks")
async def get_tasks(status: str | None = None):
    params = {}
    if status:
        params["status"] = status
    try:
        return await state.rmf.get_tasks(**params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/tasks/active")
async def get_active_tasks():
    return {"task_ids": state.monitor.active_tasks()}


# ---------------------------------------------------------------------------
# HITL Approval API
# ---------------------------------------------------------------------------

@app.get("/api/approvals")
async def get_approvals():
    return state.hitl.pending_requests()


@app.get("/api/approvals/all")
async def get_all_approvals():
    return state.hitl.all_requests()


@app.post("/api/approvals/{approval_id}/approve")
async def approve(approval_id: str, reason: str = ""):
    ok = state.hitl.approve(approval_id, reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Approval request not found or already resolved")
    await _broadcast_status({"type": "approval_resolved", "id": approval_id, "approved": True})
    return {"status": "approved"}


@app.post("/api/approvals/{approval_id}/reject")
async def reject(approval_id: str, reason: str = ""):
    ok = state.hitl.reject(approval_id, reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Approval request not found or already resolved")
    await _broadcast_status({"type": "approval_resolved", "id": approval_id, "approved": False})
    return {"status": "rejected"}


# ---------------------------------------------------------------------------
# Door API
# ---------------------------------------------------------------------------

@app.get("/api/doors")
async def get_doors():
    try:
        return await state.rmf.get_doors()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/doors/{door_name}/state")
async def get_door_state(door_name: str):
    try:
        return await state.rmf.get_door_state(door_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/doors/{door_name}/open")
async def open_door(door_name: str):
    try:
        return await state.rmf.request_door(door_name, mode=2)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/doors/{door_name}/close")
async def close_door(door_name: str):
    try:
        return await state.rmf.request_door(door_name, mode=0)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Lift API
# ---------------------------------------------------------------------------

@app.get("/api/lifts")
async def get_lifts():
    try:
        return await state.rmf.get_lifts()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/lifts/{lift_name}/state")
async def get_lift_state(lift_name: str):
    try:
        return await state.rmf.get_lift_state(lift_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/lifts/{lift_name}/request")
async def request_lift(lift_name: str, destination_floor: str):
    try:
        return await state.rmf.request_lift(lift_name, destination_floor)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Building map + Gazebo bridge config
# ---------------------------------------------------------------------------

@app.get("/api/building")
async def get_building():
    try:
        return await state.rmf.get_building()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/bridge-config")
async def get_bridge_config():
    """Return rosbridge WebSocket URL so the frontend can connect to ROS 2."""
    cfg = load_config()
    gazebo_cfg = cfg.get("gazebo", {})
    return {
        "rosbridge_url": gazebo_cfg.get("rosbridge_url", "ws://172.28.144.1:9090"),
        "default_level": gazebo_cfg.get("default_level", "L1"),
    }


# ---------------------------------------------------------------------------
# WebSocket: Chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            message = json.loads(data) if data.startswith("{") else {"message": data}
            user_text = message.get("message", data)

            async for chunk in state.agent.chat(user_text):
                await ws.send_text(json.dumps({"type": "chunk", "content": chunk}))

            await ws.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Chat WebSocket error")
        try:
            await ws.send_text(json.dumps({"type": "error", "content": str(exc)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: Status feed
# ---------------------------------------------------------------------------

@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
    await ws.accept()
    state.status_connections.append(ws)
    try:
        # Send initial fleet snapshot
        snapshot = await state.registry.snapshot()
        await ws.send_text(json.dumps({"type": "fleet_snapshot", "data": snapshot}))

        while True:
            # Keep connection alive, real updates come via broadcasts
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        state.status_connections = [c for c in state.status_connections if c != ws]


async def _broadcast_status(payload: dict) -> None:
    dead = []
    for ws in state.status_connections:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.status_connections = [c for c in state.status_connections if c != ws]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    dash = cfg.get("dashboard", {})
    uvicorn.run(
        "dashboard.server:app",
        host=dash.get("host", "0.0.0.0"),
        port=dash.get("port", 18789),
        reload=False,
    )
