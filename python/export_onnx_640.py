import torch
import sys
import os

# Assuming the script is run from project_clean/python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.train_student_sr import StudentSR

# Define paths relative to the project_clean directory, assuming script is run from project_clean/python
CKPT_PATH = "../weights/student_sr.pth"
ONNX_PATH = "../weights/student_sr_640x480.onnx"

# Fixed resolution for this export
H, W = 480, 640

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained StudentSR model
    model = StudentSR(num_blocks=8, ch=64).to(device)
    
    print(f"Loading checkpoint from: {CKPT_PATH}")
    # Handle both dictionary and direct state_dict checkpoints
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure the student_sr.pth file is in the 'weights' directory.")
        return
        
    model.eval()

    # 2. Create a dummy input tensor with the desired fixed size
    dummy_input = torch.randn(1, 3, H, W, device=device)
    print(f"Creating ONNX model with fixed input size (H, W): ({H}, {W})")

    # 3. Export the model to ONNX format with a fixed size
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11, # Using opset 11 for better compatibility with older TRT
        do_constant_folding=True,
        dynamic_axes=None  # Ensure fixed size by setting dynamic_axes to None
    )

    print(f"Successfully exported ONNX model to: {ONNX_PATH}")


if __name__ == "__main__":
    main()
