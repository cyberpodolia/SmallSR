import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run TensorRT inference on a single image.")
parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
parser.add_argument("--output_image", type=str, default="output_sr.png", help="Path to save the output SR image.")
parser.add_argument("--engine_path", type=str, 
                    default="weights/student_sr_1280x960_fp16.engine", # Path relative to project_clean
                    help="Path to the TensorRT engine file.")
args = parser.parse_args()

# Target size for SR
OUT_W = 1280
OUT_H = 960

# -----------------------------
# TensorRT Loader
# -----------------------------
class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.INFO)
        # Ensure engine path is absolute if not already, or relative to current script
        # On Jetson, usually run from docker_db/project_clean/python, engine is in docker_db/project_clean/weights
        # So, it's ../weights/student_sr_1280x960_fp16.engine
        abs_engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.engine_path))
        print(f"Attempting to load engine from: {abs_engine_path}")

        with open(abs_engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not find input/output tensors in engine")

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        img_chw = np.ascontiguousarray(img_chw.astype(np.float32))

        engine_in_shape = self.engine.get_tensor_shape(self.input_name)
        if any(d == -1 for d in engine_in_shape):
            self.context.set_input_shape(self.input_name, img_chw.shape)

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_size = int(np.prod(out_shape))

        d_input = cuda.mem_alloc(img_chw.nbytes)
        out_host = cuda.pagelocked_empty(out_size, dtype=np.float32)
        d_output = cuda.mem_alloc(out_host.nbytes)

        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))

        cuda.memcpy_htod_async(d_input, img_chw, self.stream)
        
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 failed")
            
        cuda.memcpy_dtoh_async(out_host, d_output, self.stream)
        self.stream.synchronize()

        return out_host.reshape(out_shape)


def main():
    # Adjust engine_path if necessary based on execution context
    # Assuming the script is run from project_clean/python
    # and engine is in project_clean/weights
    engine_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.engine_path)
    
    print(f"Loading engine from: {engine_file_path}")
    trt_model = TRTModule(engine_file_path)
    
    print(f"Loading input image: {args.input_image}")
    frame = cv2.imread(args.input_image)
    if frame is None:
        raise FileNotFoundError(f"Input image not found at {args.input_image}")

    start_time = time.time()
    
    # --- Pre-processing ---
    up = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    chw = x.transpose(2, 0, 1)[None]

    # --- Inference ---
    print("Running inference...")
    out = trt_model.infer(chw)

    # --- Post-processing ---
    out_img = (out[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    
    end_time = time.time()
    print(f"Inference time: {(end_time - start_time):.4f} seconds")

    # --- Save Output ---
    cv2.imwrite(args.output_image, out_bgr)
    print(f"Output SR image saved to: {args.output_image}")

if __name__ == "__main__":
    main()
