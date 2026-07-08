"""
Central configuration for the ISLiM SLT pipeline.

All paths are resolved from environment variables with sensible defaults.
To override, set before running:

    export ISIGN_DATA_DIR=/path/to/iSign
    export ISIGN_KEYPOINTS_DIR=/path/to/keypoints
    export ISIGN_VIDEOS_DIR=/path/to/videos
    export SPAMOOF_CSV_DIR=/path/to/spamof/csvs
    export POSE_STITCH_CSV_DIR=/path/to/splits
    export GERMAN_KEYPOINTS_DIR=/path/to/phoenix/keypoints
    export GERMAN_CSV_DIR=/path/to/phoenix/csv
    export CHECKPOINT_DIR=/path/to/checkpoints
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.resolve()


def _env_dir(key: str, default: str) -> Path:
    val = os.environ.get(key, default)
    p = Path(os.path.expanduser(val))
    if not p.is_absolute():
        p = Path.home() / p
    return p


# ── iSign data ──
ISIGN_DATA_DIR = _env_dir("ISIGN_DATA_DIR", "~/Datasets/iSign")
ISIGN_KEYPOINTS_DIR = _env_dir(
    "ISIGN_KEYPOINTS_DIR",
    str(_PROJECT_ROOT / "data" / "keypoints"))
ISIGN_VIDEOS_DIR = _env_dir(
    "ISIGN_VIDEOS_DIR",
    str(_PROJECT_ROOT / "data" / "videos"))
ISIGN_METADATA_CSV = str(ISIGN_DATA_DIR / "iSign_v1.1.csv")

# ── SpaMo-OF subset ──
SPAMOOF_CSV_DIR = _env_dir(
    "SPAMOOF_CSV_DIR",
    str(_PROJECT_ROOT / "data" / "spamof"))

# ── PoseStitch split (fixed 90/5/5, seed=42) ──
POSE_STITCH_CSV_DIR = _env_dir(
    "POSE_STITCH_CSV_DIR",
    str(_PROJECT_ROOT / "data" / "splits"))

# ── Phoenix-2014-T (German DGS) ──
GERMAN_KEYPOINTS_DIR = _env_dir(
    "GERMAN_KEYPOINTS_DIR", "~/Datasets/phoenix2014T/keypoints")
GERMAN_CSV_DIR = _env_dir(
    "GERMAN_CSV_DIR", "~/Datasets/phoenix2014T/splits")

# ── Checkpoints ──
CHECKPOINT_DIR = _env_dir(
    "CHECKPOINT_DIR", str(_PROJECT_ROOT / "checkpoints"))

# ── Experiment output ──
EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"

# ── Pretrained T5 model ──
T5_MODEL_NAME = os.environ.get("T5_MODEL_NAME", "t5-small")


def verify():
    """Raise FileNotFoundError with helpful message if key directories are missing."""
    missing = []
    for name, path in [
        ("iSign keypoints", ISIGN_KEYPOINTS_DIR),
        ("PoseStitch splits", POSE_STITCH_CSV_DIR),
    ]:
        if not Path(path).exists():
            missing.append(f"  {name}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required data.\n"
            "Set the corresponding environment variable or place data in:\n\n"
            + "\n".join(missing) + "\n\n"
            "See README.md for setup instructions.\n"
        )
