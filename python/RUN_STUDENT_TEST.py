import argparse

import cv2
import numpy as np
import torch

from config import (
    DEFAULT_TARGET_SIZE,
    PROJECT_ROOT,
    STUDENT_WEIGHTS,
    parse_size,
    resolve_path,
)
from config_sr import TARGET_H, TARGET_W
from train.train_student_sr import StudentSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_student(
    model: StudentSR, frame_bgr: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    target_w, target_h = target_size
    frame_bgr = cv2.resize(
        frame_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # NCHW

    x_t = torch.from_numpy(x).to(device)

    out = model(x_t).clamp(0, 1)
    out_np = (out[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "test.png"),
        help="Input image path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "test_student.png"),
        help="Output image path",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=str(STUDENT_WEIGHTS),
        help="Path to student_sr.pth",
    )
    ap.add_argument(
        "--target-size",
        type=str,
        default=DEFAULT_TARGET_SIZE,
        help='Target size WxH, e.g. "1280x960"',
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_size = parse_size(args.target_size)
    if target_size != (TARGET_W, TARGET_H):
        raise RuntimeError(
            f"Resolution contract violated: expected target size {(TARGET_W, TARGET_H)}, got {target_size}"
        )

    weights_path = resolve_path(args.weights)
    model = StudentSR(num_blocks=8, ch=64).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = cv2.imread(args.input)
    if img is None:
        raise RuntimeError(f"Failed to read input image: {args.input}")
    out = run_student(model, img, target_size)
    if out.shape[:2] != (TARGET_H, TARGET_W):
        raise RuntimeError(
            f"Resolution contract violated: expected output {(TARGET_H, TARGET_W)}, got {out.shape[:2]}"
        )
    print("Input shape:", img.shape)
    print("Output shape:", out.shape)
    cv2.imwrite(args.output, out)
