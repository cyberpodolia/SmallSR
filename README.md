# smallSR - Lightweight Super-Resolution and Denoising Pipeline

`smallSR` is a compact teacher-student pipeline for image restoration and real-time super-resolution.
It is designed for low-resolution camera streams (for example PS3 Eye) and deployment on resource-limited NVIDIA devices.

## Core Features

- Super-resolution with canonical final output `1280x960` from `640x480` input (exact 2x, 4:3 preserved).
- Teacher pipeline can perform 4x internal upscale, then downscale to `1280x960` for detail stabilization.
- Denoise and deblur teacher stages for high-quality pseudo-ground-truth generation.
- Lightweight student model for fast real-time inference.
- ONNX and TensorRT export/inference support.
- Batch image processing via `python/image_pipeline` scripts.

## Resolution Contract

Canonical contract used across student/ONNX/TRT paths:

- LR input: `640x480`
- Final output: `1280x960`

Do not use `1280x1024` in the core pipeline.
If a display requires `1280x1024`, pad as a post-process step only.

## Repository Layout

```text
project_clean/
  python/
    capture_ps3_camera.py
    RUN_STUDENT_TEST.py
    RUN_REALTIME_TEST.py
    realtime_trt_student.py
    convert_to_onnx.py
    image_pipeline/
    train/
  weights/
  scripts/
  README.md
```

## Host Setup (.venv, Windows)

```powershell
cd Y:\gemini\project_clean
.\.venv\Scripts\Activate.ps1
```

Run host camera bridge:

```powershell
python python\cam_server\run_cam_server.py --port 8001
```

Camera endpoints:

- `http://127.0.0.1:8001/frame.jpg`
- `http://127.0.0.1:8001/stream.mjpeg`

## Student Inference (PyTorch)

```powershell
python python\RUN_STUDENT_TEST.py --input path\to\input.png --output out\result.png --target-size 1280x960
```

## ONNX Export

```powershell
python python\convert_to_onnx.py --weights weights\student_sr.pth --onnx-out weights\student_sr_1280x960.onnx
```

## Docker TensorRT Runtime

Build (if needed):

```powershell
docker build -t project_clean:gpu .
```

Headless TRT run over HTTP camera:

```powershell
docker run --rm -it --gpus all `
  -e CAM_URL=http://host.docker.internal:8001/stream.mjpeg `
  -v "Y:\gemini\project_clean:/app" `
  -w /app/python `
  project_clean:ngc-trt `
  python realtime_trt_student.py `
    --engine-path /app/weights/student_sr_1280x960_fp16_trt107_linux.engine `
    --source http `
    --cam-url http://host.docker.internal:8001/stream.mjpeg `
    --no-display
```

Visible TRT window on Windows:

- Run `XLaunch`/`VcXsrv` with `Disable access control`.
- Keep host camera bridge running.
- Run:

```powershell
.\scripts\run_trt_http_visible.ps1 -Port 8001
```

## Engine Compatibility

TensorRT plan files are runtime/platform-specific.
If you see `platform tag mismatch`, rebuild the engine inside the same TensorRT runtime where you execute inference.

## Training

Example training command:

```powershell
python -m train.train_student_sr `
  --raw-dir "Y:\gemini\project_clean\data\ps3_captures_vol2" `
  --hires-dir "Y:\gemini\project_clean\data\ps3_captures_processed_vol2" `
  --raw-val-dir "Y:\gemini\project_clean\data\validation_lr" `
  --hires-val-dir "Y:\gemini\project_clean\data\validation_hr" `
  --out "Y:\gemini\project_clean\weights\student_sr.pth" `
  --epochs 20 --batch-size 8 --patch-size 128 --num-workers 4
```

Training logs are stored in `training_logs/`.
