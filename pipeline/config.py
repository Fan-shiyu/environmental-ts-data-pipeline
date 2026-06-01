"""Config loader: reads config/sites.yaml."""

from pathlib import Path

import yaml

# Repo root = the directory containing pipeline/ (this file is pipeline/config.py).
# Used to resolve relative paths in the config so the pipeline works regardless of
# the current working directory (local dev, GitHub Actions Linux runner, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = 'config/sites.yaml') -> dict:
    """Read and return the YAML config as a dict.

    Relative paths (the config file itself and each AoI's `path`) are resolved
    against the repo root, so callers work from any working directory and on
    any OS. Absolute paths are left untouched.
    """
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative AoI GeoJSON paths against the repo root.
    for aoi in config.get("aois", {}).values():
        aoi_path = Path(aoi["path"])
        if not aoi_path.is_absolute():
            aoi["path"] = str(_REPO_ROOT / aoi_path)

    return config
