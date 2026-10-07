"""Resolve project-local recipes with relocation-safe disposable dataset caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

from flow_factory.hparams import Arguments


def load_config(path: Path) -> Arguments:
    root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("The recipe must be an ordinary file inside this project")
    config = Arguments.load_from_yaml(str(resolved))
    # Arrow caches can contain absolute image paths; never reuse them after relocation.
    namespace = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    config.data_args.cache_dir = str(root / "cache/flow_factory" / namespace)
    return config
