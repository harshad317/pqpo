"""Artifact + manifest writer (Sec 3.8 / 10).

Every run writes a manifest with provenance (git commit, model, decoding params,
pricing snapshot) so results are replayable and auditable.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(out_dir: str, model_cfg, pricing_snapshot: dict, extra: dict = None):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "run_dir": out_dir,
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_provider": model_cfg.provider,
        "model_name": model_cfg.model_name,
        "model_version": model_cfg.model_version,
        "decoding_parameters": {
            "temperature": model_cfg.default_temperature,
            "max_output_tokens": model_cfg.default_max_output_tokens,
        },
        "pricing_snapshot": pricing_snapshot,
        **(extra or {}),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def write_json(out_dir: str, name: str, obj) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
