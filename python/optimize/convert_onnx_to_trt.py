import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT, resolve_path

DEFAULT_ONNX = PROJECT_ROOT / "weights" / "student_sr_1280x960.onnx"
DEFAULT_ENGINE = PROJECT_ROOT / "weights" / "student_sr_1280x960_fp16.engine"
DEFAULT_TRTEXEC = os.getenv("TRTEXEC_PATH", "trtexec")
DEFAULT_WORKSPACE_GB = 4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX))
    ap.add_argument("--engine-path", type=str, default=str(DEFAULT_ENGINE))
    ap.add_argument("--trtexec-path", type=str, default=DEFAULT_TRTEXEC)
    ap.add_argument("--workspace-gb", type=int, default=DEFAULT_WORKSPACE_GB)
    return ap.parse_args()


def main():
    print("Converting ONNX to TensorRT engine...")

    args = parse_args()
    onnx_path = resolve_path(Path(args.onnx_path))
    engine_path = resolve_path(Path(args.engine_path))
    trtexec_path = args.trtexec_path
    workspace_gb = args.workspace_gb

    if not os.path.exists(trtexec_path):
        print(f"Error: trtexec not found at {trtexec_path}")
        print("Please specify the correct path to trtexec.exe")
        return

    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        return

    # trtexec command
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--memPoolSize=workspace:{workspace_gb * 1024}",
        "--minShapes=input:1x3x960x1280",
        "--optShapes=input:1x3x960x1280",
        "--maxShapes=input:1x3x960x1280",
        "--verbose",
    ]

    print("Running command:")
    print(" ".join(command))

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"\nTensorRT engine saved to: {engine_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during conversion: {e}")
    except FileNotFoundError:
        print(
            f"\nError: Could not find {trtexec_path}. Please ensure it is installed and the path is correct."
        )


if __name__ == "__main__":
    main()
