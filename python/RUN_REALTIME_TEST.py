import argparse
import os
import time
import urllib.request

import torch
import cv2
import numpy as np
from train.train_student_sr import StudentSR
from config import STUDENT_WEIGHTS, resolve_path
from PIL import Image

# -------------------------------
#        Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

SCALE = 2  # 640x480 -> 1280x960


@torch.no_grad()
def run_student(frame_bgr: np.ndarray) -> np.ndarray:
    # 1) upscale original frame x2 (4:3 -> 4:3)
    h, w = frame_bgr.shape[:2]
    target_w, target_h = w * SCALE, h * SCALE

    frame_bgr_up = cv2.resize(
        frame_bgr,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    # 2) run through student model
    rgb = cv2.cvtColor(frame_bgr_up, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # NCHW

    x_t = torch.from_numpy(x).to(device)
    out = model(x_t).clamp(0, 1)

    out_np = (out[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# -------------------------------
#      Real-Time Pipeline
# -------------------------------
def open_ps3_eye():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"PS3 Eye found at index {idx}")
            return cap
    raise RuntimeError("PS3 Eye not found. Plug the camera and try again.")


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

    def set(self, prop_id, value):
        if self.cap is not None:
            self.cap.set(prop_id, value)

    def release(self):
        if self.cap is not None:
            self.cap.release()


def open_http_source(url: str):
    url_l = url.lower()
    if url_l.endswith(".mjpeg") or "mjpeg" in url_l or "stream" in url_l:
        return ReconnectableMjpegCapture(url)
    return HttpJpegSource(url)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["camera", "http"], default="camera")
    parser.add_argument(
        "--cam-url",
        default=os.getenv("CAM_URL", "http://host.docker.internal:8000/stream.mjpeg"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(STUDENT_WEIGHTS),
        help="Path to student_sr.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global model
    weights_path = resolve_path(args.weights)
    model = StudentSR(num_blocks=8, ch=64).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.source == "http":
        cap = open_http_source(args.cam_url)
        print(f"Reading video over HTTP: {args.cam_url}")
    else:
        cap = open_ps3_eye()

    if hasattr(cap, "set"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera stopped returning frames.")
            break
        if frame is None:
            continue

        out_student = run_student(frame)

        # --- Blend original and restored ---
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        student_pil = Image.fromarray(cv2.cvtColor(out_student, cv2.COLOR_BGR2RGB))

        frame_pil_resized = frame_pil.resize(student_pil.size)

        blended_pil = Image.blend(frame_pil_resized, student_pil, alpha=0.6)
        out = cv2.cvtColor(np.array(blended_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("PS3 Eye -> Student SR x2 (1280x960)", out)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
