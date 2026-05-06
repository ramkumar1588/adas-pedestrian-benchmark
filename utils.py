"""
utils.py  —  Shared config loader used by every pipeline script.

Usage in any script (root or subdirectory):
    from utils import load_config, REPO_ROOT
    cfg   = load_config()
    paths = cfg["paths"]

Paths in config.yaml are resolved relative to the repo root.
Absolute paths (e.g. "D:/dataset/...") also work unchanged.
"""

from pathlib import Path
import yaml


def _find_repo_root() -> Path:
    """Walk up from this file to find the directory containing config.yaml."""
    current = Path(__file__).resolve().parent
    for _ in range(6):                      # search up to 6 levels
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "config.yaml not found in any parent directory.\n"
        "Make sure config.yaml is at the repo root."
    )


REPO_ROOT = _find_repo_root()


def load_config(config_file: str = "config.yaml") -> dict:
    """
    Load config.yaml and return a dict with all paths resolved
    as absolute Path objects. Works regardless of which directory
    the calling script lives in.
    """
    cfg_path = REPO_ROOT / config_file
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    for key, raw in cfg.get("paths", {}).items():
        p = Path(raw)
        cfg["paths"][key] = p if p.is_absolute() else (REPO_ROOT / p).resolve()

    return cfg
