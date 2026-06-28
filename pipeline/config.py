"""Configuration loading and CRDS pinning.

Every path and tunable parameter in the pipeline comes from a single YAML file
(see ``config/pipeline.example.yaml``). There are intentionally no hardcoded
machine paths anywhere else in the package.

Reproducibility note: JWST calibration is only deterministic if the CRDS
reference context is pinned. ``apply_crds_env()`` sets the relevant environment
variables from the config and MUST be called *before* importing any ``jwst``
pipeline module, because the CRDS context is resolved at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def _expand(value):
    """Recursively expand ${VARS} and ~ in any strings within the structure."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | os.PathLike = "config/pipeline.yaml") -> dict:
    """Load the pipeline YAML config, expanding environment variables and ``~``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy config/pipeline.example.yaml to "
            f"config/pipeline.yaml and edit the paths + crds.context."
        )
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return _expand(cfg)


def apply_crds_env(cfg: dict) -> str:
    """Set CRDS environment variables from the config and return the pinned context.

    Call this before importing ``jwst.pipeline``. Raises if the context is still
    the unedited placeholder, to prevent accidental non-reproducible runs.
    """
    crds = cfg.get("crds", {})
    context = crds.get("context", "")
    if not context or "XXXX" in context:
        raise ValueError(
            "crds.context is not set in the config. Pin it to the pmap used for "
            "the published reduction (e.g. jwst_1322.pmap) — runs are not "
            "reproducible with a floating context."
        )
    os.environ["CRDS_CONTEXT"] = context
    if crds.get("path"):
        os.environ["CRDS_PATH"] = crds["path"]
    if crds.get("server_url"):
        os.environ["CRDS_SERVER_URL"] = crds["server_url"]
    return context


def ensure_dirs(cfg: dict) -> None:
    """Create the output directories named in cfg['paths'] if missing."""
    for key, val in cfg.get("paths", {}).items():
        if key.endswith("_dir") or key in ("refs_dir", "logs_dir"):
            Path(val).mkdir(parents=True, exist_ok=True)


def segments_for(cfg: dict, target: str) -> list[str]:
    return cfg["targets"][target]["segments"]
