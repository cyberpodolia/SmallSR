import argparse
import os

import cv2

from config import PROJECT_ROOT


def capture_frames(camera_index, output_dir, num_frames):
    """
    Captures a specified number of frames from a single camera and saves them to a directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    print(f"Starting capture from camera {camera_index}. Press 'q' to stop early.")

    for i in range(num_frames):
        is_reading, frame = cap.read()

        if not is_reading:
            print(f"Error: Could not read frame {i + 1}/{num_frames}.")
            break

        image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(image_path, frame)

        if (i + 1) % 100 == 0:
            print(f"Captured {i + 1}/{num_frames} frames.")

        # Allow for early exit by pressing 'q' in a display window
        # cv2.imshow("Capture", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print(f"Capture complete. {num_frames} frames saved to {output_dir}")

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ps3_captures_vol2"),
    )
    ap.add_argument("--num-frames", type=int, default=5000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture_frames(args.camera_index, args.output_dir, args.num_frames)
