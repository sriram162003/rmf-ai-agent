"""
Setup wizard API router — prefix /api/setup

Provides endpoints for the first-run onboarding wizard embedded in the
dashboard.  Import this module from server.py and include the router.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/setup", tags=["setup"])

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _write_config(data: dict) -> None:
    """Atomic write: write to .tmp then rename."""
    tmp = _CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    tmp.replace(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# GET /api/setup/status
# ---------------------------------------------------------------------------

@router.get("/status")
async def setup_status():
    """Return {complete: bool, missing: [str]} based on current config."""
    cfg = _read_config()
    missing: list[str] = []

    if not cfg.get("_setup_complete", False):
        missing.append("_setup_complete")

    rmf_url = cfg.get("rmf", {}).get("api_url", "http://localhost:8000")
    if rmf_url == "http://localhost:8000" and not os.environ.get("RMF_API_URL"):
        missing.append("rmf.api_url")

    api_key = cfg.get("llm", {}).get("api_key", "") or ""
    if not api_key and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        missing.append("llm.api_key")

    return {"complete": len(missing) == 0, "missing": missing}


# ---------------------------------------------------------------------------
# POST /api/setup/detect-wsl
# ---------------------------------------------------------------------------

@router.post("/detect-wsl")
async def detect_wsl():
    """Detect WSL2 IP address (Windows only).  Graceful on Linux."""
    if platform.system() == "Windows" and shutil.which("wsl"):
        try:
            # Find the Ubuntu-22.04 distro name (user may have multiple distros)
            list_r = subprocess.run(
                ["wsl", "--list", "--verbose"],
                capture_output=True, timeout=5, encoding="utf-16-le", errors="replace",
            )
            distro_name = "Ubuntu-22.04"
            for line in (list_r.stdout + list_r.stderr).splitlines():
                clean = line.strip().lstrip("*").strip()
                if clean and "22" in clean and "ubuntu" in clean.lower():
                    name = clean.split()[0]
                    if name:
                        distro_name = name
                    break

            result = subprocess.run(
                ["wsl", "-d", distro_name, "hostname", "-I"],
                capture_output=True, text=True, timeout=5,
            )
            raw = result.stdout.strip()
            if raw:
                ip = raw.split()[0]
                return {"ok": True, "ip": ip, "distro": distro_name}
        except Exception as exc:
            return {"ok": False, "ip": None, "error": str(exc)}
    # On Linux the agent IS inside WSL2 — return the host's IP via /etc/resolv.conf
    try:
        resolv = Path("/etc/resolv.conf").read_text()
        for line in resolv.splitlines():
            if line.startswith("nameserver"):
                ip = line.split()[1]
                return {"ok": True, "ip": ip, "note": "linux_host_ip"}
    except Exception:
        pass
    return {"ok": False, "ip": None, "error": "Could not detect WSL2 host IP"}


# ---------------------------------------------------------------------------
# POST /api/setup/test-rmf
# ---------------------------------------------------------------------------

class RmfTestRequest(BaseModel):
    url: str


@router.post("/test-rmf")
async def test_rmf(body: RmfTestRequest):
    """Attempt GET {url}/time and return {ok, latency_ms}."""
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{body.url.rstrip('/')}/time", timeout=5.0)
        latency_ms = round((time.monotonic() - t0) * 1000)
        return {"ok": r.status_code == 200, "latency_ms": latency_ms, "status_code": r.status_code}
    except Exception as exc:
        return {"ok": False, "latency_ms": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# POST /api/setup/test-llm
# ---------------------------------------------------------------------------

class LlmTestRequest(BaseModel):
    backend: str
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    ollama_url: Optional[str] = "http://localhost:11434"


@router.post("/test-llm")
async def test_llm(body: LlmTestRequest):
    """Perform a minimal LLM call to verify credentials/connectivity."""
    try:
        if body.backend == "claude":
            import anthropic
            key = body.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return {"ok": False, "error": "No API key provided"}
            client = anthropic.Anthropic(api_key=key)
            # Use sync call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            def _call():
                return client.messages.create(
                    model=body.model or "claude-haiku-4-5-20251001",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Reply with just: ok"}],
                )
            await loop.run_in_executor(None, _call)
            return {"ok": True}

        elif body.backend == "openai":
            from openai import OpenAI
            key = body.api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return {"ok": False, "error": "No API key provided"}
            client = OpenAI(api_key=key)
            loop = asyncio.get_event_loop()
            def _call():
                return client.chat.completions.create(
                    model=body.model or "gpt-4o-mini",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Reply with just: ok"}],
                )
            await loop.run_in_executor(None, _call)
            return {"ok": True}

        elif body.backend == "ollama":
            url = (body.ollama_url or "http://localhost:11434").rstrip("/")
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                models = [m.get("name","") for m in r.json().get("models", [])]
                return {"ok": True, "models": models}
            return {"ok": False, "error": f"Ollama returned {r.status_code}"}

        else:
            return {"ok": False, "error": f"Unknown backend: {body.backend}"}

    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# POST /api/setup/test-rosbridge
# ---------------------------------------------------------------------------

class RosbridgeTestRequest(BaseModel):
    url: str


@router.post("/test-rosbridge")
async def test_rosbridge(body: RosbridgeTestRequest):
    """Attempt a WebSocket connect to rosbridge and immediately close."""
    try:
        import websockets
        async with websockets.connect(body.url, open_timeout=5):
            pass
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# GET /api/setup/wsl-commands
# ---------------------------------------------------------------------------

_WSL_COMMANDS: dict[str, dict[str, list[str]]] = {
    "humble": {
        # Ubuntu 22.04 (Jammy)
        "install_ros": [
            "# ── 1. Locale ──────────────────────────────────────",
            "sudo apt update && sudo apt install -y locales",
            "sudo locale-gen en_US en_US.UTF-8",
            "sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8",
            "export LANG=en_US.UTF-8",
            "",
            "# ── 2. Add ROS 2 apt repository ────────────────────",
            "sudo apt install -y software-properties-common curl",
            "sudo add-apt-repository universe",
            "sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key "
            "-o /usr/share/keyrings/ros-archive-keyring.gpg",
            'echo "deb [arch=$(dpkg --print-architecture) '
            "signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] "
            'http://packages.ros.org/ros2/ubuntu jammy main" '
            "| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null",
            "",
            "# ── 3. Install ROS 2 Humble ────────────────────────",
            "sudo apt update && sudo apt upgrade -y",
            "sudo apt install -y ros-humble-desktop",
            'echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc',
            "source ~/.bashrc",
        ],
        "install_rmf": [
            "# NOTE: rmf-demos and rmf-api-server are NOT in the standard apt repo.",
            "# Build Open-RMF from source using vcstool + colcon (~20-30 min).",
            "",
            "# ── 1. Install build tools ──────────────────────────",
            "sudo apt install -y python3-vcstool python3-colcon-common-extensions python3-rosdep",
            "",
            "# ── 2. Create workspace & fetch sources ─────────────",
            "mkdir -p ~/rmf_ws/src && cd ~/rmf_ws",
            "curl https://raw.githubusercontent.com/open-rmf/rmf/main/rmf.repos | vcs import src/",
            "",
            "# ── 3. Install all ROS dependencies ─────────────────",
            "sudo rosdep init 2>/dev/null || true",
            "rosdep update",
            "rosdep install --from-paths src --ignore-src -y --rosdistro humble \\",
            '  --skip-keys "gz_transport_vendor gz_fuel_tools_vendor gz_sim_vendor gz_plugin_vendor gz_math_vendor gz_msgs_vendor gz_gui_vendor gz_utils_vendor"',
            "",
            "# ── 4. Build — skip Gazebo sim plugins (not needed for RMF API) ────",
            "source /opt/ros/humble/setup.bash",
            "colcon build --symlink-install \\",
            '  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-nonnull" \\',
            "  --packages-skip rmf_robot_sim_gz_plugins rmf_building_sim_gz_plugins rmf_demos_gz rmf_robot_sim_common \\",
            "  --continue-on-error",
            "",
            "# ── 5. Source the workspace ─────────────────────────",
            "echo 'source ~/rmf_ws/install/setup.bash' >> ~/.bashrc",
            "source ~/rmf_ws/install/setup.bash",
        ],
        "install_rosbridge": [
            "# ── Install rosbridge suite ─────────────────────────",
            "sudo apt update",
            "sudo apt install -y ros-humble-rosbridge-suite",
        ],
        "rmf": [
            "source /opt/ros/humble/setup.bash",
            "ros2 launch rmf_demos office.launch.xml",
        ],
        "rosbridge": [
            "source /opt/ros/humble/setup.bash",
            "ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
        ],
    },
    "jazzy": {
        # Ubuntu 24.04 (Noble)
        "install_ros": [
            "# ── 1. Locale ──────────────────────────────────────",
            "sudo apt update && sudo apt install -y locales",
            "sudo locale-gen en_US en_US.UTF-8",
            "sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8",
            "export LANG=en_US.UTF-8",
            "",
            "# ── 2. Add ROS 2 apt repository ────────────────────",
            "sudo apt install -y software-properties-common curl",
            "sudo add-apt-repository universe",
            "sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key "
            "-o /usr/share/keyrings/ros-archive-keyring.gpg",
            'echo "deb [arch=$(dpkg --print-architecture) '
            "signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] "
            'http://packages.ros.org/ros2/ubuntu noble main" '
            "| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null",
            "",
            "# ── 3. Install ROS 2 Jazzy ─────────────────────────",
            "sudo apt update && sudo apt upgrade -y",
            "sudo apt install -y ros-jazzy-desktop",
            'echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc',
            "source ~/.bashrc",
        ],
        "install_rmf": [
            "# NOTE: Open-RMF is NOT in the standard apt repo.",
            "# Build from source using vcstool + colcon (~20-30 min).",
            "",
            "# ── 1. Install build tools ──────────────────────────",
            "sudo apt install -y python3-vcstool python3-colcon-common-extensions python3-rosdep",
            "",
            "# ── 2. Create workspace & fetch sources ─────────────",
            "mkdir -p ~/rmf_ws/src && cd ~/rmf_ws",
            "curl https://raw.githubusercontent.com/open-rmf/rmf/main/rmf.repos | vcs import src/",
            "",
            "# ── 3. Install all ROS dependencies ─────────────────",
            "sudo rosdep init 2>/dev/null || true",
            "rosdep update",
            "rosdep install --from-paths src --ignore-src -y --rosdistro jazzy \\",
            '  --skip-keys "gz_transport_vendor gz_fuel_tools_vendor gz_sim_vendor gz_plugin_vendor gz_math_vendor gz_msgs_vendor gz_gui_vendor gz_utils_vendor"',
            "",
            "# ── 4. Build — skip Gazebo sim plugins ──────────────",
            "source /opt/ros/jazzy/setup.bash",
            "colcon build --symlink-install \\",
            '  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-nonnull" \\',
            "  --packages-skip rmf_robot_sim_gz_plugins rmf_building_sim_gz_plugins rmf_demos_gz rmf_robot_sim_common \\",
            "  --continue-on-error",
            "",
            "# ── 5. Source the workspace ─────────────────────────",
            "echo 'source ~/rmf_ws/install/setup.bash' >> ~/.bashrc",
            "source ~/rmf_ws/install/setup.bash",
        ],
        "install_rosbridge": [
            "# ── Install rosbridge suite ─────────────────────────",
            "sudo apt update",
            "sudo apt install -y ros-jazzy-rosbridge-suite",
        ],
        "rmf": [
            "source /opt/ros/jazzy/setup.bash",
            "ros2 launch rmf_demos office.launch.xml",
        ],
        "rosbridge": [
            "source /opt/ros/jazzy/setup.bash",
            "ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
        ],
    },
    "rolling": {
        # Ubuntu 24.04 (Noble)
        "install_ros": [
            "# ── 1. Locale ──────────────────────────────────────",
            "sudo apt update && sudo apt install -y locales",
            "sudo locale-gen en_US en_US.UTF-8",
            "sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8",
            "export LANG=en_US.UTF-8",
            "",
            "# ── 2. Add ROS 2 apt repository ────────────────────",
            "sudo apt install -y software-properties-common curl",
            "sudo add-apt-repository universe",
            "sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key "
            "-o /usr/share/keyrings/ros-archive-keyring.gpg",
            'echo "deb [arch=$(dpkg --print-architecture) '
            "signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] "
            'http://packages.ros.org/ros2/ubuntu noble main" '
            "| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null",
            "",
            "# ── 3. Install ROS 2 Rolling ───────────────────────",
            "sudo apt update && sudo apt upgrade -y",
            "sudo apt install -y ros-rolling-desktop",
            'echo "source /opt/ros/rolling/setup.bash" >> ~/.bashrc',
            "source ~/.bashrc",
        ],
        "install_rmf": [
            "# NOTE: Open-RMF is NOT in the standard apt repo.",
            "# Build from source using vcstool + colcon (~20-30 min).",
            "",
            "# ── 1. Install build tools ──────────────────────────",
            "sudo apt install -y python3-vcstool python3-colcon-common-extensions python3-rosdep",
            "",
            "# ── 2. Create workspace & fetch sources ─────────────",
            "mkdir -p ~/rmf_ws/src && cd ~/rmf_ws",
            "curl https://raw.githubusercontent.com/open-rmf/rmf/main/rmf.repos | vcs import src/",
            "",
            "# ── 3. Install all ROS dependencies ─────────────────",
            "sudo rosdep init 2>/dev/null || true",
            "rosdep update",
            "rosdep install --from-paths src --ignore-src -y --rosdistro rolling \\",
            '  --skip-keys "gz_transport_vendor gz_fuel_tools_vendor gz_sim_vendor gz_plugin_vendor gz_math_vendor gz_msgs_vendor gz_gui_vendor gz_utils_vendor"',
            "",
            "# ── 4. Build — skip Gazebo sim plugins ──────────────",
            "source /opt/ros/rolling/setup.bash",
            "colcon build --symlink-install \\",
            '  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-nonnull" \\',
            "  --packages-skip rmf_robot_sim_gz_plugins rmf_building_sim_gz_plugins rmf_demos_gz rmf_robot_sim_common \\",
            "  --continue-on-error",
            "",
            "# ── 5. Source the workspace ─────────────────────────",
            "echo 'source ~/rmf_ws/install/setup.bash' >> ~/.bashrc",
            "source ~/rmf_ws/install/setup.bash",
        ],
        "install_rosbridge": [
            "# ── Install rosbridge suite ─────────────────────────",
            "sudo apt update",
            "sudo apt install -y ros-rolling-rosbridge-suite",
        ],
        "rmf": [
            "source /opt/ros/rolling/setup.bash",
            "ros2 launch rmf_demos office.launch.xml",
        ],
        "rosbridge": [
            "source /opt/ros/rolling/setup.bash",
            "ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
        ],
    },
}


@router.get("/wsl-commands")
async def wsl_commands(ros_distro: str = "humble"):
    """Return copy-ready bash commands for WSL2 setup."""
    distro = ros_distro.lower()
    cmds = _WSL_COMMANDS.get(distro, _WSL_COMMANDS["humble"])
    return {
        "ros_distro": distro,
        "install_ros_commands": cmds["install_ros"],
        "install_rmf_commands": cmds["install_rmf"],
        "install_rosbridge_commands": cmds["install_rosbridge"],
        "rmf_commands": cmds["rmf"],
        "rosbridge_commands": cmds["rosbridge"],
    }


# ---------------------------------------------------------------------------
# POST /api/setup/save
# ---------------------------------------------------------------------------

class SaveRequest(BaseModel):
    rmf_api_url: Optional[str] = None
    rmf_jwt_token: Optional[str] = None
    llm_backend: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_ollama_url: Optional[str] = None
    rosbridge_url: Optional[str] = None
    default_level: Optional[str] = None
    fleet_scanner_wifi_mdns: Optional[bool] = None
    fleet_scanner_bluetooth: Optional[bool] = None
    fleet_scanner_rmf_poll_interval: Optional[int] = None


@router.post("/save")
async def save_config(body: SaveRequest):
    """
    Merge wizard payload into config.yaml, set _setup_complete: true,
    then re-initialise live connections via the server module's reinit helper.
    """
    cfg = _read_config()

    # Merge fields
    if body.rmf_api_url is not None:
        cfg.setdefault("rmf", {})["api_url"] = body.rmf_api_url
    if body.rmf_jwt_token is not None:
        cfg.setdefault("rmf", {})["jwt_token"] = body.rmf_jwt_token
    if body.llm_backend is not None:
        cfg.setdefault("llm", {})["backend"] = body.llm_backend
    if body.llm_model is not None:
        cfg.setdefault("llm", {})["model"] = body.llm_model
    if body.llm_api_key is not None:
        cfg.setdefault("llm", {})["api_key"] = body.llm_api_key
    if body.llm_ollama_url is not None:
        cfg.setdefault("llm", {})["ollama_url"] = body.llm_ollama_url
    if body.rosbridge_url is not None:
        cfg.setdefault("gazebo", {})["rosbridge_url"] = body.rosbridge_url
    if body.default_level is not None:
        cfg.setdefault("gazebo", {})["default_level"] = body.default_level
    if body.fleet_scanner_wifi_mdns is not None:
        cfg.setdefault("fleet_scanner", {})["wifi_mdns"] = body.fleet_scanner_wifi_mdns
    if body.fleet_scanner_bluetooth is not None:
        cfg.setdefault("fleet_scanner", {})["bluetooth"] = body.fleet_scanner_bluetooth
    if body.fleet_scanner_rmf_poll_interval is not None:
        cfg.setdefault("fleet_scanner", {})["rmf_poll_interval"] = body.fleet_scanner_rmf_poll_interval

    cfg["_setup_complete"] = True
    _write_config(cfg)

    # Re-initialise live connections
    try:
        from dashboard import server as _srv
        await _srv.reinit_connections(cfg)
    except Exception as exc:
        # Non-fatal: config is saved, connections will reconnect on next request
        return {"ok": True, "warning": f"Config saved but reinit failed: {exc}"}

    return {"ok": True}
