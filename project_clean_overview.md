# Project Overview: smallSR (project_clean)

This document provides a summary of the `smallSR` project contained in the `project_clean` directory.

## 1. Project Goal

The project's main objective is to create a lightweight, real-time super-resolution (SR) and denoising pipeline. It is designed to enhance low-resolution video streams (e.g., from a PS3 Eye camera) and to be deployable on low-power devices like the Jetson Nano.

## 2. Methodology: Teacher-Student Knowledge Distillation

The project uses a "Teacher-Student" knowledge distillation approach:

*   **Teacher Pipeline**: A powerful but slow and computationally expensive offline pipeline. It uses multiple state-of-the-art models (NAFNet, NAFSSR) to perform 4x super-resolution, deblurring, and denoising, producing very high-quality images from low-quality source frames.
*   **Student Model**: A small, fast, and lightweight ResNet-style model (`StudentSR`). This model is trained to mimic the output of the Teacher pipeline. The goal is for the student to achieve similar visual results but in real-time.

## 3. Core Workflow

1.  **Data Acquisition**: Capture raw, low-resolution frames (e.g., 640x480) from a source like a PS3 camera.
2.  **Teacher Processing**: Process these raw frames with the Teacher pipeline to generate corresponding high-resolution, clean "ground truth" images (e.g., 1280x1024).
3.  **Student Training**: Train the `StudentSR` model on the `raw / teacher` image pairs. The student learns to transform the low-quality input into the high-quality output.
4.  **Optimization**: Convert the trained PyTorch student model (`.pth`) into more efficient formats for deployment:
    *   **ONNX**: A cross-platform format.
    *   **TensorRT Engine**: A highly optimized format for NVIDIA GPUs (like the one in the Jetson Nano) to maximize throughput and minimize latency.
5.  **Inference**: Use the final optimized student model for real-time video enhancement.

## 4. Key Files and Scripts

| Path                                                       | Description                                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `PROJECT_DOCUMENTATION.md`                                 | Detailed, step-by-step explanation of the entire project pipeline, architecture, and rationale.         |
| `README.md`                                                | High-level overview, setup instructions, and basic usage commands.                                      |
| `requirements.txt`                                         | Lists all the Python dependencies required for the project.                                             |
|                                                            |                                                                                                         |
| **Data Acquisition**                                       |                                                                                                         |
| `python/capture_ps3_camera.py`                             | Script to capture and save frames from a PS3 Eye camera.                                                |
|                                                            |                                                                                                         |
| **Teacher Pipeline**                                       |                                                                                                         |
| `python/image_pipeline/run_pipeline_custom_V2_mtread.py`   | **The main Teacher pipeline script.** Takes a directory of raw images and processes them.               |
| `python/image_pipeline/nafssr_arch.py`                     | Defines the architecture for the NAFSSR super-resolution model.                                         |
| `python/image_pipeline/nafnet_arch.py`                     | Defines the architecture for the NAFNet deblurring/denoising models.                                    |
|                                                            |                                                                                                         |
| **Student Training**                                       |                                                                                                         |
| `python/train/train_student_sr.py`                         | **The main Student training script.** Trains the `StudentSR` model using raw and teacher image pairs.    |
| `python/plot_metrics.py`                                   | Utility to plot loss and PSNR from a training log file (`training_log.csv`).                             |
|                                                            |                                                                                                         |
| **Optimization**                                           |                                                                                                         |
| `python/optimize/export_student_to_onnx.py`                | Converts the trained PyTorch student model (`.pth`) to the ONNX format (`.onnx`).                       |
| `python/optimize/convert_onnx_to_trt.py`                   | Converts the `.onnx` model into a `.engine` file using NVIDIA's TensorRT for maximum performance.       |
|                                                            |                                                                                                         |
| **Inference / Testing**                                    |                                                                                                         |
| `python/RUN_STUDENT_TEST.py`                               | Runs inference on an input image/folder using the PyTorch student model.                                |
| `python/realtime_trt_student.py`                           | **The main real-time script.** Captures from PS3 camera and performs inference using the TensorRT engine. |
|                                                            |                                                                                                         |
| **Weights & Models**                                       |                                                                                                         |
| `weights/student_sr.pth`                                   | The trained weights for the PyTorch student model.                                                      |
| `weights/student_sr_1280x960.onnx`                         | The student model in ONNX format.                                                                       |
| `weights/student_sr_1280x960_fp16.engine`                  | The final, optimized TensorRT engine for real-time deployment.                                          |
