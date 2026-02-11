import argparse

import torch

from config import PROJECT_ROOT, resolve_path
from config_sr import TARGET_H, TARGET_W
from train.train_student_sr import StudentSR


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "weights" / "student_sr.pth"),
        help="Path to student checkpoint (.pth)",
    )
    ap.add_argument(
        "--onnx-out",
        type=str,
        default=str(PROJECT_ROOT / "weights" / "student_sr_1280x960.onnx"),
        help="Output ONNX file path",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StudentSR(num_blocks=8, ch=64).to(device)
    ckpt = torch.load(resolve_path(args.weights), map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, TARGET_H, TARGET_W, device=device)

    torch.onnx.export(
        model,
        dummy,
        resolve_path(args.onnx_out),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX exported: {resolve_path(args.onnx_out)}")
    print(f"Output shape contract: (N,3,{TARGET_H},{TARGET_W})")


if __name__ == "__main__":
    main()

