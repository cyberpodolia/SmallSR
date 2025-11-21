import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import argparse
import os

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run real-time TensorRT inference for Super Resolution.")
parser.add_argument("--engine_path", type=str, 
                    default="../weights/student_sr_1280x960_fp16.engine",
                    help="Path to the TensorRT engine file, relative to the script location.")
parser.add_argument("--camera_index", type=int, default=0, help="Index of the camera to use.")
parser.add_argument("--no_display", action="store_true", help="Disable OpenCV window display (for headless mode).")
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
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found at: {engine_path}")
            
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
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


# -----------------------------
# Camera helpers
# -----------------------------
def open_camera(camera_index):
    # On Jetson, let OpenCV choose the backend. GStreamer is often the default in L4T containers.
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    print(f"Camera opened at index {camera_index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
    return cap


# -----------------------------
# Real-time pipeline
# -----------------------------
def main():
    print(f"Loading engine from: {args.engine_path}")
    trt_model = TRTModule(args.engine_path)
    
    cap = open_camera(args.camera_index)
    
    if not args.no_display:
        print("Display enabled. Press ESC in the window to exit.")
    else:
        print("Display disabled (headless mode). Press Ctrl+C to exit.")

    frame_count = 0
    fps = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("No frames from camera, stopping.")
                break
            
            # --- Pre-processing ---
            up = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            chw = x.transpose(2, 0, 1)[None]

            # --- Inference ---
            out = trt_model.infer(chw)

            # --- FPS Calculation ---
            frame_count += 1
            current_time = time.time()
            if (current_time - start_time) >= 1.0:
                fps = frame_count / (current_time - start_time)
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time

            if not args.no_display:
                # --- Post-processing for display ---
                out_img = (out[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                
                cv2.putText(out_bgr, f"SR FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Student SR TensorRT (Jetson)", out_bgr)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Script finished.")


if __name__ == "__main__":
    main()
