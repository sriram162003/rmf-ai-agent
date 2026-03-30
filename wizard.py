#!/usr/bin/env python3
"""
RMF AI Agent — CLI Setup Wizard
Run from the project root:  python wizard.py

Covers:
  1. Python dependency check & install
  2. WSL2 presence & Ubuntu 22.04 check   (Windows only)
  3. ROS 2 Humble install inside WSL2
  4. Open-RMF packages install inside WSL2
  5. rosbridge_suite install inside WSL2
  6. RMF API URL configuration + connection test
  7. LLM backend configuration + connection test
  8. rosbridge URL configuration + connection test
  9. Fleet scanner settings
 10. Review & save config.yaml
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional

import yaml

# ── Colour helpers ─────────────────────────────────────────────────────────────
_IS_WIN = platform.system() == "Windows"
_WSL_DISTRO = "Ubuntu-22.04"  # set by step_wsl2() once the 22.04 distro name is confirmed
_NO_COLOR = (
    os.environ.get("NO_COLOR")
    or (_IS_WIN and "WT_SESSION" not in os.environ and "TERM_PROGRAM" not in os.environ)
)

def _c(code: str, t: str) -> str:
    return t if _NO_COLOR else f"\033[{code}m{t}\033[0m"

def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1",  t)
def red(t):    return _c("31", t)
def dim(t):    return _c("2",  t)
def blue(t):   return _c("34", t)

def ok(m):   print(f"  {green('✓')} {m}")
def warn(m): print(f"  {yellow('!')} {m}")
def err(m):  print(f"  {red('✗')} {m}")
def info(m): print(f"  {dim('·')} {m}")

def sep(title: str = "") -> None:
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{cyan('─' * pad)} {bold(title)} {cyan('─' * pad)}")
    else:
        print(cyan("─" * w))

# ── Config ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_REQ_PATH    = _ROOT / "requirements.txt"

def _read_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}

def _write_cfg(data: dict) -> None:
    tmp = _CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    tmp.replace(_CONFIG_PATH)

# ── Prompt helpers ─────────────────────────────────────────────────────────────
def ask(prompt: str, default: str = "") -> str:
    display = f"{prompt} {dim(f'[{default}]')}: " if default else f"{prompt}: "
    try:
        v = input(display).strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    return v if v else default

def ask_bool(prompt: str, default: bool = True) -> bool:
    v = ask(f"{prompt} ({'Y/n' if default else 'y/N'})", "y" if default else "n").lower()
    return v in ("y", "yes", "1", "true")

def ask_choice(prompt: str, choices: list[str], default: str = "") -> str:
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        mark = green("▶") if c == default else " "
        print(f"  {mark} {i}. {c}")
    while True:
        idx = ask(f"Choice (1–{len(choices)})",
                  str(choices.index(default) + 1) if default in choices else "1")
        if idx.isdigit() and 1 <= int(idx) <= len(choices):
            return choices[int(idx) - 1]
        print(red("  Invalid, try again."))

# ── WSL command helpers ────────────────────────────────────────────────────────
def _is_wsl2_host() -> bool:
    """True when running FROM Windows with wsl.exe available."""
    return _IS_WIN and bool(shutil.which("wsl"))

def _is_inside_wsl() -> bool:
    """True when this script itself is running INSIDE WSL2."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False

def wsl_run(bash_cmd: str, stream: bool = False) -> subprocess.CompletedProcess:
    """
    Run a bash command inside WSL2 (from Windows) or directly (from Linux/WSL2).
    Always targets _WSL_DISTRO (Ubuntu-22.04) so we don't accidentally run
    commands in the wrong default distro.
    If stream=True the output goes to the terminal live (for installs).
    """
    if _is_wsl2_host():
        cmd = ["wsl", "-d", _WSL_DISTRO, "--", "bash", "-c", bash_cmd]
    else:
        cmd = ["bash", "-c", bash_cmd]

    if stream:
        return subprocess.run(cmd)
    return subprocess.run(cmd, capture_output=True, text=True)

def wsl_check(bash_cmd: str) -> bool:
    return wsl_run(bash_cmd).returncode == 0

def wsl_out(bash_cmd: str) -> str:
    r = wsl_run(bash_cmd)
    return r.stdout.strip() if r.returncode == 0 else ""

# ── Step 1 — Python dependencies ──────────────────────────────────────────────
def step_python_deps():
    sep("Step 1 · Python Dependencies")

    if not _REQ_PATH.exists():
        warn("requirements.txt not found — skipping.")
        return

    packages = [
        line.split(">=")[0].split("==")[0].strip()
        for line in _REQ_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    missing = []
    for pkg in packages:
        # importlib name differs from pip name for some packages
        import_name = pkg.replace("-", "_").lower()
        special = {
            "pyyaml": "yaml",
            "python_socketio": "socketio",
            "python_multipart": "multipart",
            "python_dotenv": "dotenv",
            "pillow": "PIL",
        }
        import_name = special.get(import_name, import_name)
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg)

    if not missing:
        ok("All Python packages already installed.")
        return

    warn(f"Missing packages: {', '.join(missing)}")
    if ask_bool("Install missing packages now via pip?", default=True):
        print()
        ret = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(_REQ_PATH)]
        ).returncode
        if ret == 0:
            ok("Packages installed successfully.")
        else:
            err("pip install failed — fix manually then re-run the wizard.")
            sys.exit(1)
    else:
        warn("Skipped — run  pip install -r requirements.txt  before starting the server.")

# ── Step 2 — WSL2 & Ubuntu 22.04 ──────────────────────────────────────────────
def step_wsl2() -> Optional[str]:
    """Returns detected WSL2 IP (or None)."""
    if _is_inside_wsl():
        sep("Step 2 · WSL2 / Host IP")
        info("Running inside WSL2 — reading host IP from /etc/resolv.conf")
        ip = None
        try:
            for line in Path("/etc/resolv.conf").read_text().splitlines():
                if line.startswith("nameserver"):
                    ip = line.split()[1]
                    break
        except Exception:
            pass
        if ip:
            ok(f"Host IP: {bold(ip)}")
        else:
            warn("Could not read host IP — enter URLs manually below.")
        return ip

    if not _IS_WIN:
        sep("Step 2 · Host Detection")
        info("Not on Windows and not inside WSL2 — skipping WSL2 checks.")
        return None

    sep("Step 2 · WSL2 & Ubuntu 22.04")

    # Check wsl.exe
    if not shutil.which("wsl"):
        err("WSL2 not found on this machine.")
        info("Enable it in PowerShell (run as Administrator):")
        print(f"\n    {cyan('wsl --install')}\n")
        info("Then reboot and re-run this wizard.")
        if not ask_bool("Continue anyway (skip WSL2 checks)?", default=False):
            sys.exit(0)
        return None

    # List distros
    r = subprocess.run(["wsl", "--list", "--verbose"], capture_output=True, text=True, encoding="utf-16-le", errors="replace")
    distro_output = r.stdout + r.stderr

    has_22 = "Ubuntu-22.04" in distro_output or "Ubuntu 22.04" in distro_output
    has_20 = "Ubuntu-20.04" in distro_output or "Ubuntu 20.04" in distro_output or "Ubuntu" in distro_output

    if has_22:
        ok("Ubuntu 22.04 found in WSL2.")
        # Discover the exact distro name so every subsequent wsl_run targets it.
        global _WSL_DISTRO
        for line in distro_output.splitlines():
            clean = line.strip().lstrip("*").strip()
            if clean and "22" in clean and "ubuntu" in clean.lower():
                name = clean.split()[0]
                if name:
                    _WSL_DISTRO = name
                break
        info(f"Targeting WSL2 distro: {bold(_WSL_DISTRO)}")
    elif has_20:
        warn("Found Ubuntu 20.04 — ROS 2 Humble requires Ubuntu 22.04.")
        info("Install Ubuntu 22.04 alongside it (your 20.04 is untouched):")
        print(f"\n    {cyan('wsl --install -d Ubuntu-22.04')}\n")
        if ask_bool("Open a new PowerShell to run that command now?", default=False):
            subprocess.Popen(["powershell", "-Command",
                              "Start-Process powershell -ArgumentList '-NoExit','-Command','wsl --install -d Ubuntu-22.04'"])
            input(dim("\nPress Enter once Ubuntu 22.04 has finished installing…"))
    else:
        warn("No Ubuntu WSL2 distro found.")
        info("Run this in PowerShell to install Ubuntu 22.04:")
        print(f"\n    {cyan('wsl --install -d Ubuntu-22.04')}\n")
        if ask_bool("Open a new PowerShell to run that command now?", default=False):
            subprocess.Popen(["powershell", "-Command",
                              "Start-Process powershell -ArgumentList '-NoExit','-Command','wsl --install -d Ubuntu-22.04'"])
            input(dim("\nPress Enter once Ubuntu 22.04 has finished installing…"))

    # Detect IP (use the resolved distro so we get the right network address)
    ip = None
    try:
        r2 = subprocess.run(["wsl", "-d", _WSL_DISTRO, "hostname", "-I"], capture_output=True, text=True, timeout=5)
        raw = r2.stdout.strip()
        if raw:
            ip = raw.split()[0]
    except Exception:
        pass

    if ip:
        ok(f"WSL2 IP detected: {bold(ip)}")
    else:
        warn("Could not detect WSL2 IP — enter URLs manually below.")

    return ip

# ── Step 3 — ROS 2 Humble ─────────────────────────────────────────────────────
_ROS2_INSTALL_CMDS = """\
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-humble-desktop
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc"""

def step_ros2():
    sep("Step 3 · ROS 2 Humble")

    if not (_is_wsl2_host() or _is_inside_wsl()):
        info("Not on Windows/WSL2 — skipping ROS 2 install check.")
        return

    # Check Ubuntu version
    ubuntu_ver = wsl_out("lsb_release -rs 2>/dev/null")
    if ubuntu_ver:
        info(f"WSL2 Ubuntu version: {ubuntu_ver}")
        if not ubuntu_ver.startswith("22"):
            warn(f"Ubuntu {ubuntu_ver} detected — Humble requires 22.04.")
            warn("Please install Ubuntu 22.04 (Step 2) before proceeding.")
            if not ask_bool("Continue anyway?", default=False):
                sys.exit(0)

    # Check if ros2 already installed — test for setup.bash, not `ros2` in PATH,
    # because non-interactive WSL shells don't source ~/.bashrc automatically.
    if wsl_check("test -f /opt/ros/humble/setup.bash"):
        ros_ver = wsl_out("bash -c 'source /opt/ros/humble/setup.bash 2>/dev/null && ros2 --version 2>/dev/null'")
        ok(f"ROS 2 Humble already installed. {dim(ros_ver)}")
        return

    warn("ROS 2 not found in WSL2.")
    print()
    print(f"  {bold('Installation commands:')}")
    for line in _ROS2_INSTALL_CMDS.strip().splitlines():
        print(f"    {cyan(line)}")
    print()

    if ask_bool("Run ROS 2 install now inside WSL2? (takes ~10 min, requires sudo)", default=True):
        print(yellow("\n  WSL2 terminal output will appear below. Enter your WSL2 sudo password if prompted.\n"))
        wsl_run(_ROS2_INSTALL_CMDS, stream=True)
        if wsl_check("test -f /opt/ros/humble/setup.bash"):
            ok("ROS 2 Humble installed successfully.")
        else:
            err("ROS 2 install may have failed — check output above.")
            warn("You can re-run the wizard or install manually.")
    else:
        info("Skipped — copy the commands above and run them in your WSL2 terminal.")

# ── Step 4 — Open-RMF packages ────────────────────────────────────────────────
# Open-RMF (rmf-demos, rmf-api-server, etc.) is NOT in the standard ROS 2 apt
# repo — it must be built from source via vcstool + colcon.
# Gazebo simulation vendor packages are not in the Humble rosdep index.
# We skip them — they're only needed for Gazebo visualisation, not for
# the RMF API server / fleet adapter / traffic scheduler used by this agent.
_GZ_SKIP_KEYS = (
    "gz_transport_vendor gz_fuel_tools_vendor gz_sim_vendor gz_plugin_vendor "
    "gz_math_vendor gz_msgs_vendor gz_rendering_vendor gz_sensors_vendor "
    "gz_gui_vendor gz_utils_vendor gz_common_vendor gz_image_vendor"
)
_GZ_SKIP_PKGS = (
    "rmf_robot_sim_gz_plugins rmf_building_sim_gz_plugins rmf_demos_gz "
    "rmf_robot_sim_common"
)

_RMF_SOURCE_BUILD_CMD = f"""\
sudo apt install -y python3-vcstool python3-colcon-common-extensions python3-rosdep && \
mkdir -p ~/rmf_ws/src && cd ~/rmf_ws && \
curl -fsSL https://raw.githubusercontent.com/open-rmf/rmf/main/rmf.repos | vcs import src/ && \
sudo rosdep init 2>/dev/null || true && rosdep update && \
rosdep install --from-paths src --ignore-src -y --rosdistro humble --skip-keys "{_GZ_SKIP_KEYS}" && \
source /opt/ros/humble/setup.bash && \
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-nonnull" \
  --packages-skip {_GZ_SKIP_PKGS} --continue-on-error && \
echo 'source ~/rmf_ws/install/setup.bash' >> ~/.bashrc"""

_RMF_BUILD_STEPS = [
    "sudo apt install -y python3-vcstool python3-colcon-common-extensions python3-rosdep",
    "mkdir -p ~/rmf_ws/src && cd ~/rmf_ws",
    "curl -fsSL https://raw.githubusercontent.com/open-rmf/rmf/main/rmf.repos | vcs import src/",
    "sudo rosdep init 2>/dev/null || true && rosdep update",
    f'rosdep install --from-paths src --ignore-src -y --rosdistro humble --skip-keys "{_GZ_SKIP_KEYS}"',
    "source /opt/ros/humble/setup.bash",
    f'colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wno-nonnull" --packages-skip {_GZ_SKIP_PKGS} --continue-on-error',
    "echo 'source ~/rmf_ws/install/setup.bash' >> ~/.bashrc",
    "source ~/rmf_ws/install/setup.bash",
]

def step_rmf_install():
    sep("Step 4 · Open-RMF Packages")

    if not (_is_wsl2_host() or _is_inside_wsl()):
        info("Not on Windows/WSL2 — skipping.")
        return

    # Check if core packages are built (rmf_traffic + rmf_fleet_adapter both required)
    if wsl_check("test -d ~/rmf_ws/install/rmf_traffic && test -d ~/rmf_ws/install/rmf_fleet_adapter && test -d ~/rmf_ws/install/rmf_demos && test -d ~/rmf_ws/install/rmf_demos_maps"):
        ok("Open-RMF workspace already built.")
        return

    warn("Open-RMF workspace not found.")
    info("rmf-demos and rmf-api-server are NOT in the standard apt repo.")
    info("Must be built from source using vcstool + colcon (~20-30 min).")
    print()
    print(f"  {bold('Build steps:')}")
    for line in _RMF_BUILD_STEPS:
        print(f"    {cyan(line)}")
    print()

    if ask_bool("Build Open-RMF from source now inside WSL2? (requires sudo, ~20-30 min)", default=True):
        print(yellow("\n  WSL2 output below. Enter sudo password if prompted.\n"))
        wsl_run(_RMF_SOURCE_BUILD_CMD, stream=True)
        if wsl_check("test -d ~/rmf_ws/install/rmf_traffic && test -d ~/rmf_ws/install/rmf_fleet_adapter && test -d ~/rmf_ws/install/rmf_demos && test -d ~/rmf_ws/install/rmf_demos_maps"):
            ok("Open-RMF built and installed successfully.")
        else:
            err("Build may have failed — check output above.")
            warn("You can re-run the wizard or build manually using the steps above.")
    else:
        info("Skipped — run the build steps above in your WSL2 terminal.")

# ── Step 5 — rosbridge ────────────────────────────────────────────────────────
_ROSBRIDGE_PKG = "ros-humble-rosbridge-suite"

def step_rosbridge_install():
    sep("Step 5 · rosbridge_suite")

    if not (_is_wsl2_host() or _is_inside_wsl()):
        info("Not on Windows/WSL2 — skipping.")
        return

    if wsl_check(f"dpkg -l {_ROSBRIDGE_PKG} 2>/dev/null | grep -q '^ii'"):
        ok("rosbridge_suite already installed.")
        return

    warn(f"{_ROSBRIDGE_PKG} not found.")
    install_cmd = f"sudo apt update && sudo apt install -y {_ROSBRIDGE_PKG}"
    print(f"\n    {cyan(install_cmd)}\n")

    if ask_bool("Install rosbridge_suite now inside WSL2?", default=True):
        print(yellow("\n  WSL2 output below. Enter sudo password if prompted.\n"))
        wsl_run(install_cmd, stream=True)
        if wsl_check(f"dpkg -l {_ROSBRIDGE_PKG} 2>/dev/null | grep -q '^ii'"):
            ok("rosbridge_suite installed.")
        else:
            err("Install may have failed — check output above.")
    else:
        info("Skipped — run the command above in your WSL2 terminal.")

# ── Connection testers ─────────────────────────────────────────────────────────
def _test_rmf(url: str) -> tuple[bool, str]:
    try:
        import httpx
        t0 = time.monotonic()
        r = httpx.get(f"{url.rstrip('/')}/time", timeout=5.0)
        ms = round((time.monotonic() - t0) * 1000)
        return (True, f"{ms}ms") if r.status_code == 200 else (False, f"HTTP {r.status_code}")
    except Exception as e:
        return False, str(e)

def _test_rosbridge(url: str) -> tuple[bool, str]:
    async def _inner():
        import websockets
        async with websockets.connect(url, open_timeout=5):
            pass
    try:
        asyncio.run(_inner())
        return True, "connected"
    except Exception as e:
        return False, str(e)

def _test_llm(backend: str, api_key: str, model: str, ollama_url: str) -> tuple[bool, str]:
    try:
        if backend == "claude":
            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return False, "No API key — set ANTHROPIC_API_KEY env or enter key"
            c = anthropic.Anthropic(api_key=key)
            c.messages.create(model=model or "claude-haiku-4-5-20251001",
                              max_tokens=10,
                              messages=[{"role": "user", "content": "Reply: ok"}])
            return True, "OK"
        elif backend == "openai":
            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return False, "No API key — set OPENAI_API_KEY env or enter key"
            c = OpenAI(api_key=key)
            c.chat.completions.create(model=model or "gpt-4o-mini", max_tokens=10,
                                      messages=[{"role": "user", "content": "Reply: ok"}])
            return True, "OK"
        elif backend == "ollama":
            import httpx
            r = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5.0)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return True, "models: " + (", ".join(models[:3]) or "none")
            return False, f"HTTP {r.status_code}"
        return False, f"Unknown backend: {backend}"
    except Exception as e:
        return False, str(e)

# ── Step 6 — RMF API config ────────────────────────────────────────────────────
def step_rmf_config(cfg: dict, wsl_ip: Optional[str]) -> dict:
    sep("Step 6 · RMF API")
    rmf = cfg.get("rmf", {})
    suggested = f"http://{wsl_ip}:8000" if wsl_ip else rmf.get("api_url", "http://localhost:8000")

    url = ask("RMF API URL", rmf.get("api_url") or suggested)
    print(f"  {dim('Testing…')}", end="", flush=True)
    success, detail = _test_rmf(url)
    print(f"\r  {green('✓') if success else yellow('!')} RMF API: {detail}")
    if not success:
        warn("Not reachable — start RMF in WSL2 first:")
        print(f"    {cyan('source /opt/ros/humble/setup.bash')}")
        print(f"    {cyan('ros2 launch rmf_demos office.launch.xml')}\n")

    jwt = ask("JWT token (blank = dev/open mode)", rmf.get("jwt_token", "") or "")
    return {"api_url": url, "jwt_token": jwt, "_ok": success}

# ── Step 7 — LLM ──────────────────────────────────────────────────────────────
def step_llm(cfg: dict) -> dict:
    sep("Step 7 · LLM Backend")
    llm = cfg.get("llm", {})
    backend = ask_choice("Which LLM backend?",
                         ["claude", "openai", "ollama"],
                         default=llm.get("backend", "claude"))

    api_key = model = ""
    ollama_url = llm.get("ollama_url", "http://localhost:11434")

    if backend == "claude":
        if os.environ.get("ANTHROPIC_API_KEY"):
            info("ANTHROPIC_API_KEY env var found — leave blank to use it.")
        api_key = ask("Anthropic API key", llm.get("api_key", "") or "")
        model   = ask("Model", llm.get("model") or "claude-sonnet-4-6")
    elif backend == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            info("OPENAI_API_KEY env var found — leave blank to use it.")
        api_key = ask("OpenAI API key", llm.get("api_key", "") or "")
        model   = ask("Model", llm.get("model") or "gpt-4o")
    elif backend == "ollama":
        ollama_url = ask("Ollama URL", ollama_url)
        model      = ask("Model", llm.get("model") or "llama3")

    print(f"  {dim('Testing…')}", end="", flush=True)
    success, detail = _test_llm(backend, api_key, model, ollama_url)
    print(f"\r  {green('✓') if success else yellow('!')} LLM: {detail}")
    if not success:
        warn("Check your API key / model name and re-run if needed.")

    return {"backend": backend, "api_key": api_key, "model": model,
            "ollama_url": ollama_url, "_ok": success}

# ── Step 8 — rosbridge config ─────────────────────────────────────────────────
def step_rosbridge_config(cfg: dict, wsl_ip: Optional[str]) -> dict:
    sep("Step 8 · rosbridge WebSocket")
    gazebo = cfg.get("gazebo", {})
    suggested = f"ws://{wsl_ip}:9090" if wsl_ip else gazebo.get("rosbridge_url", "ws://localhost:9090")

    url = ask("rosbridge WebSocket URL", gazebo.get("rosbridge_url") or suggested)
    print(f"  {dim('Testing…')}", end="", flush=True)
    success, detail = _test_rosbridge(url)
    print(f"\r  {green('✓') if success else yellow('!')} rosbridge: {detail}")
    if not success:
        warn("Start rosbridge in WSL2:")
        print(f"    {cyan('source /opt/ros/humble/setup.bash')}")
        print(f"    {cyan('ros2 launch rosbridge_server rosbridge_websocket_launch.xml')}\n")

    level = ask("Default map level", gazebo.get("default_level", "L1"))
    return {"rosbridge_url": url, "default_level": level, "_ok": success}

# ── Step 9 — Fleet scanner ────────────────────────────────────────────────────
def step_scanner(cfg: dict) -> dict:
    sep("Step 9 · Fleet Scanner")
    sc = cfg.get("fleet_scanner", {})
    wifi  = ask_bool("Enable WiFi mDNS discovery?",    sc.get("wifi_mdns", True))
    ble   = ask_bool("Enable Bluetooth BLE discovery?", sc.get("bluetooth", True))
    poll  = ask("RMF API poll interval (seconds)", str(sc.get("rmf_poll_interval", 10)))
    try:
        poll = max(5, int(poll))
    except ValueError:
        poll = 10
    return {"wifi_mdns": wifi, "bluetooth": ble, "rmf_poll_interval": poll}

# ── Step 10 — Review & Save ───────────────────────────────────────────────────
def step_save(cfg: dict, rmf: dict, llm: dict, ros: dict, scanner: dict):
    sep("Step 10 · Review & Save")

    rows = [
        ("RMF API URL",    rmf["api_url"],                    rmf["_ok"]),
        ("JWT Token",      rmf["jwt_token"] or "(open mode)", None),
        ("LLM Backend",    llm["backend"],                    llm["_ok"]),
        ("LLM Model",      llm["model"] or "—",               llm["_ok"]),
        ("rosbridge URL",  ros["rosbridge_url"],               ros["_ok"]),
        ("Default Level",  ros["default_level"],               None),
        ("WiFi mDNS",      str(scanner["wifi_mdns"]),          None),
        ("BLE",            str(scanner["bluetooth"]),          None),
        ("Poll Interval",  f"{scanner['rmf_poll_interval']}s", None),
    ]

    col = max(len(r[0]) for r in rows) + 2
    for label, value, tested in rows:
        status = green("✓ tested") if tested is True else (yellow("! failed") if tested is False else dim("—"))
        print(f"  {label:<{col}}{cyan(value):<45} {status}")

    print()
    if not ask_bool("Save to config/config.yaml and apply?", default=True):
        print(yellow("\nAborted — nothing written.\n"))
        sys.exit(0)

    cfg.setdefault("rmf", {}).update({"api_url": rmf["api_url"], "jwt_token": rmf["jwt_token"]})
    cfg.setdefault("llm", {}).update({
        "backend":    llm["backend"],
        "model":      llm["model"],
        "api_key":    llm["api_key"],
        "ollama_url": llm["ollama_url"],
    })
    cfg.setdefault("gazebo", {}).update({
        "rosbridge_url": ros["rosbridge_url"],
        "default_level": ros["default_level"],
    })
    cfg.setdefault("fleet_scanner", {}).update({
        "wifi_mdns":         scanner["wifi_mdns"],
        "bluetooth":         scanner["bluetooth"],
        "rmf_poll_interval": scanner["rmf_poll_interval"],
    })
    cfg["_setup_complete"] = True
    _write_cfg(cfg)

    print()
    ok(bold("config/config.yaml saved!"))
    print()
    print(bold("  Next steps:"))
    print(f"  1. In WSL2 terminal A:  {cyan('source /opt/ros/humble/setup.bash && ros2 launch rmf_demos office.launch.xml')}")
    print(f"  2. In WSL2 terminal B:  {cyan('source /opt/ros/humble/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml')}")
    print(f"  3. In PowerShell:       {cyan('python -m dashboard.server')}")
    print(f"  4. Open browser:        {cyan('http://localhost:18789')}")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print()
    print(bold(cyan("╔════════════════════════════════════════════════════╗")))
    print(bold(cyan("║     RMF AI Agent — Setup Wizard  (10 steps)        ║")))
    print(bold(cyan("╚════════════════════════════════════════════════════╝")))
    print()
    print("Configures Python deps, WSL2, ROS 2, Open-RMF, and config.yaml.")
    print(f"Press {bold('Enter')} to accept defaults  ·  {bold('Ctrl+C')} to abort anytime.\n")
    input(dim("Press Enter to begin…"))

    cfg = _read_cfg()

    step_python_deps()
    wsl_ip = step_wsl2()
    step_ros2()
    step_rmf_install()
    step_rosbridge_install()

    rmf     = step_rmf_config(cfg, wsl_ip)
    llm     = step_llm(cfg)
    ros     = step_rosbridge_config(cfg, wsl_ip)
    scanner = step_scanner(cfg)

    step_save(cfg, rmf, llm, ros, scanner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{yellow('Wizard aborted — nothing was saved.')}\n")
        sys.exit(0)
