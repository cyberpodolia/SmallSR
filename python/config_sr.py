from __future__ import annotations

# Canonical SR contract for the project:
# LR input is 640x480 (4:3), final output is 1280x960 (exact 2x, same aspect).
LR_W = 640
LR_H = 480
TARGET_W = 1280
TARGET_H = 960

# Optional post-process display pad to 1280x1024 if required by a viewer.
DISPLAY_PAD_TO_1024 = False


def target_size_wh() -> tuple[int, int]:
    return TARGET_W, TARGET_H


def target_size_hw() -> tuple[int, int]:
    return TARGET_H, TARGET_W

