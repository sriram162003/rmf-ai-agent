"""Entry point: python -m rmf_ai_agent"""
import uvicorn
import yaml
from pathlib import Path

cfg_path = Path(__file__).parent / "config" / "config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

dash = cfg.get("dashboard", {})

uvicorn.run(
    "dashboard.server:app",
    host=dash.get("host", "0.0.0.0"),
    port=dash.get("port", 18789),
    reload=False,
)
