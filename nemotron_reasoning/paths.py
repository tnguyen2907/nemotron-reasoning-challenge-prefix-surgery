from __future__ import annotations

from pathlib import Path

VOLUME_NAME = "nemotron-prefix-surgery"
MOUNT_ROOT = Path("/mnt/nemotron-prefix-surgery")
BASE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

REQUIRED_VOLUME_DIRS = [
    "data",
    "hf-cache",
    "hf-cache/hub",
    "hf-cache/datasets",
    "runs",
]


def volume_path(*parts: str) -> Path:
    return MOUNT_ROOT.joinpath(*parts)


def run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return volume_path("runs", run_id)


def ensure_run_layout(run_id: str) -> Path:
    root = run_dir(run_id)
    for child in [
        "config",
        "data",
        "logs",
        "checkpoints",
        "adapter",
        "eval",
        "predictions",
        "submissions",
    ]:
        root.joinpath(child).mkdir(parents=True, exist_ok=True)
    return root
