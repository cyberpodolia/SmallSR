import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class FrameStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def open_ps3_eye():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"PS3 Eye found at index {idx}")
            return cap
    raise RuntimeError("PS3 Eye not found. Plug the camera and try again.")


def capture_loop(store: FrameStore, width: int, height: int, fps: int):
    cap = open_ps3_eye()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            store.set(frame)
        else:
            time.sleep(0.01)


def make_handler(store: FrameStore, jpeg_quality: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ["/", "/health"]:
                frame = store.get()
                if frame is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            if self.path == "/frame.jpg":
                frame = store.get()
                if frame is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                )
                if not ok:
                    self.send_response(500)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(buf.tobytes())
                return

            if self.path == "/stream.mjpeg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()

                while True:
                    frame = store.get()
                    if frame is None:
                        time.sleep(0.02)
                        continue
                    ok, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    )
                    if not ok:
                        time.sleep(0.02)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(buf)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(buf.tobytes())
                        self.wfile.write(b"\r\n")
                    except BrokenPipeError:
                        break
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def main():
    args = parse_args()
    store = FrameStore()
    t = threading.Thread(
        target=capture_loop,
        args=(store, args.width, args.height, args.fps),
        daemon=True,
    )
    t.start()

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(store, args.jpeg_quality)
    )
    print(f"Cam server on http://{args.host}:{args.port}")
    print("Endpoints: /frame.jpg , /stream.mjpeg")
    server.serve_forever()


if __name__ == "__main__":
    main()
