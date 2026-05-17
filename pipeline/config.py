"""Config loader: reads config/sites.yaml."""

import yaml


def load_config(path: str = 'config/sites.yaml') -> dict:
    """Read and return the YAML config as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
