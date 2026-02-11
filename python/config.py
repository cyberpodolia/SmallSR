from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _project_root()
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", PROJECT_ROOT / "weights"))

DEFAULT_ENGINE_NAME = "student_sr_1280x960_fp16.engine"
ENGINE_PATH = Path(os.getenv("ENGINE_PATH", WEIGHTS_DIR / DEFAULT_ENGINE_NAME))

STUDENT_WEIGHTS = Path(os.getenv("STUDENT_WEIGHTS", WEIGHTS_DIR / "student_sr.pth"))

DEFAULT_TARGET_SIZE = os.getenv("TARGET_SIZE", "1280x960")


def parse_size(size_str: str) -> tuple[int, int]:
    if "x" not in size_str:
        raise ValueError('Size must be "WxH", e.g. "1280x960"')
    w_s, h_s = size_str.lower().split("x", 1)
    return int(w_s), int(h_s)


def resolve_path(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()
