from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

from Retrieval.common.io import read_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_PATH = repo_root() / "Retrieval/config.yaml"


def get_config_path(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    env_path = os.environ.get("RUBRIC_REACT_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


@lru_cache(maxsize=8)
def load_react_config(explicit_path: str | None = None) -> Dict:
    path = get_config_path(explicit_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return read_yaml(path)


def resolve_repo_path(path_value: str) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_value)))
    if path.is_absolute():
        return path
    return repo_root() / path


def ensure_required_path(path_value: str, *, label: str) -> Path:
    path = resolve_repo_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path
