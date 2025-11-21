import onnxruntime as ort
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Verify an ONNX model.')
    parser.add_argument('--onnx-path', type=str, required=True, help='Path to the ONNX model.')
    args = parser.parse_args()

    ONNX_PATH = args.onnx_path
    H, W = 960, 1280

    print(f"Verifying ONNX model: {ONNX_PATH}")

    # 1. Load session
    session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("ONNX model loaded successfully.")

    # 2. Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    print(f"Input Name: {input_name}, Shape: {input_shape}")
    print(f"Output Name: {output_name}, Shape: {output_shape}")

    # 3. Create dummy input
    # Use static shape from export script for verification
    dummy_input = np.random.randn(1, 3, H, W).astype(np.float32)

    # 4. Run inference
    result = session.run([output_name], {input_name: dummy_input})
    output_tensor = result[0]

    # 5. Print output info
    print(f"Output Tensor Shape: {output_tensor.shape}")
    print(f"Output Tensor Min: {output_tensor.min()}")
    print(f"Output Tensor Max: {output_tensor.max()}")
    print(f"Output Tensor Mean: {output_tensor.mean()}")

    if output_tensor.shape == (1, 3, H, W):
        print("\nVerification PASSED: Output shape is correct.")
    else:
        print(f"\nVerification FAILED: Output shape is {output_tensor.shape}, expected {(1, 3, H, W)}")

if __name__ == "__main__":
    main()
