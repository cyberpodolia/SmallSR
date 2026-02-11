import argparse
import os
import time
import urllib.request

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from config import DEFAULT_TARGET_SIZE, ENGINE_PATH, parse_size, resolve_path
from config_sr import TARGET_H, TARGET_W

# Target size for SR (default 1280x960)
OUT_W, OUT_H = parse_size(DEFAULT_TARGET_SIZE)


class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.INFO)
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

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        img_chw = np.ascontiguousarray(img_chw.astype(np.float32))

        engine_in_shape = self.engine.get_tensor_shape(self.input_name)
        if any(d == -1 for d in engine_in_shape):
            self.context.set_input_shape(self.input_name, img_chw.shape)

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_size = int(np.prod(out_shape))

        d_input = cuda.mem_alloc(img_chw.nbytes)
        out_host = np.empty(out_size, dtype=np.float32)
        d_output = cuda.mem_alloc(out_host.nbytes)

        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))

        cuda.memcpy_htod_async(d_input, img_chw, self.stream)
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        cuda.memcpy_dtoh_async(out_host, d_output, self.stream)
        self.stream.synchronize()

        return out_host.reshape(out_shape)


def open_ps3_eye():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        for idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"PS3 Eye detected: index {idx}, backend {backend}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 60)

                try:
                    print("Backend name:", cap.getBackendName())
                except Exception:
                    pass
                print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
                return cap
            cap.release()

    raise RuntimeError("PS3 Eye camera not found on available indexes/backends")


class HttpJpegSource:
    def __init__(self, url: str, timeout: float = 2.0, max_fail_s: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.max_fail_s = max_fail_s
        self.last_ok = time.time()

    def read(self):
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return False, None
            self.last_ok = time.time()
            return True, frame
        except Exception:
            if time.time() - self.last_ok > self.max_fail_s:
                return False, None
            time.sleep(0.02)
            return True, None

    def release(self):
        pass


class ReconnectableMjpegCapture:
    def __init__(self, url: str, retry_s: float = 1.0, max_fail_s: float = 5.0):
        self.url = url
        self.retry_s = retry_s
        self.max_fail_s = max_fail_s
        self.last_ok = time.time()
        self.cap = None
        self._open()

    def _open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.url)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            time.sleep(self.retry_s)
            self._open()
            return True, None

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.last_ok = time.time()
            return True, frame

        if time.time() - self.last_ok > self.max_fail_s:
            return False, None

        time.sleep(self.retry_s)
        self._open()
        return True, None

    def release(self):
        if self.cap is not None:
            self.cap.release()


def open_http_source(url: str):
    url_l = url.lower()
    if url_l.endswith(".mjpeg") or "mjpeg" in url_l or "stream" in url_l:
        return ReconnectableMjpegCapture(url)
    return HttpJpegSource(url)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-path", type=str, default=str(ENGINE_PATH), help="Path to TensorRT engine file")
    ap.add_argument("--target-size", type=str, default=DEFAULT_TARGET_SIZE, help='Target size WxH, e.g. "1280x960"')
    ap.add_argument("--source", choices=["camera", "http"], default="camera", help="Frame source type")
    ap.add_argument("--cam-url", type=str, default=os.getenv("CAM_URL", "http://host.docker.internal:8001/stream.mjpeg"), help="HTTP camera URL for --source http")
    ap.add_argument("--no-display", action="store_true", help="Disable OpenCV windows (required in headless containers)")
    return ap.parse_args()


def main():
    args = parse_args()

    global OUT_W, OUT_H
    OUT_W, OUT_H = parse_size(args.target_size)
    if (OUT_W, OUT_H) != (TARGET_W, TARGET_H):
        raise RuntimeError(
            f"Resolution contract violated: expected {(TARGET_W, TARGET_H)}, got {(OUT_W, OUT_H)}"
        )

    engine_path = resolve_path(args.engine_path)
    trt_model = TRTModule(str(engine_path))
    gpu_name = pycuda.autoinit.device.name()
    device_info = f"TensorRT | {gpu_name}"

    if args.source == "http":
        cap = open_http_source(args.cam_url)
        print(f"HTTP source: {args.cam_url}")
    else:
        cap = open_ps3_eye()
    print("ESC = exit")

    frame_count = 0
    fps = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No frames from camera source (ret=False or frame=None)")
            continue

        up = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        chw = x.transpose(2, 0, 1)[None]

        out = trt_model.infer(chw)
        out_img = (out[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        if out_img.shape[:2] != (TARGET_H, TARGET_W):
            raise RuntimeError(
                f"Resolution contract violated: expected output {(TARGET_H, TARGET_W)}, got {out_img.shape[:2]}"
            )

        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        cv2.putText(out_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out_bgr, device_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.no_display:
            if frame_count == 0:
                print(f"FPS: {fps:.2f} | {device_info}")
            continue

        cv2.imshow("Original 640x480", frame)
        cv2.imshow("Student SR TensorRT (1280x960)", out_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
