import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import os
import cv2
from torchvision import transforms

# --- Local Imports ---
# Ensure the local libraries can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from image_pipeline.nafssr_arch import NAFNetSR
try:
    from nafnetlib.core import DeblurProcessor, DenoiseProcessor
except ImportError:
    print("Error: nafnetlib is not installed. Please run 'pip install nafnetlib'")
    sys.exit(1)


def run_custom_pipeline(image_paths, output_dir_path, batch_size=1):
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_dir = Path(__file__).resolve().parent.parent.parent
    model_dir = project_dir / "weights"
    print("model_dir", model_dir, "project_dir", project_dir)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"No images provided for processing.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- Initialize Models (once) ---
    print("Initializing models...")
    denoise_processor = DenoiseProcessor(model_id='sidd_width64', model_dir=str(model_dir), device=device)
    deblur_processor = DeblurProcessor(model_id='reds_width64', model_dir=str(model_dir), device=device)

    # SR Model
    weights_path_sr = model_dir / "NAFSSR-L_4x.pth"
    sr_model = NAFNetSR(
        up_scale=4,
        width=128,
        num_blks=128,
        drop_path_rate=0.3,
        drop_out_rate=0.,
        dual=False
    ).to(device)
    state_dict_sr = torch.load(str(weights_path_sr), map_location=device)["params"]
    sr_model.load_state_dict(state_dict_sr, strict=False)
    sr_model.eval()
    print("All models loaded!")

    # Трансформації, щоб не створювати їх кожного разу
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    total = len(image_paths)
    batch_size = max(1, int(batch_size))
    print(f"Batch size for SR: {batch_size}")

    # --- Основний цикл батчами ---
    for batch_start in range(0, total, batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        current_batch_size = len(batch_paths)

        print(f"\n=== Batch {batch_start // batch_size + 1}: "
              f"images {batch_start + 1}–{batch_start + current_batch_size} / {total} ===")

        # 1) Готуємо батч для SR: LR deblur + blend → tensor stack
        mixed_images = []
        for local_idx, input_path in enumerate(batch_paths):
            global_idx = batch_start + local_idx
            print(f"\n--- Preparing image {global_idx + 1}/{total}: {os.path.basename(input_path)} ---")

            # Step 1: load
            print(f"[{global_idx+1}/{total}] Step 1: Loading...")
            input_image = Image.open(input_path).convert("RGB")

            # Step 1.1: Deblur LR + blend
            print(f"[{global_idx+1}/{total}] Step 1.1: Deblur LR + blend (alpha=0.25)...")
            deblurred_lr = deblur_processor.process(input_image)
            mixed = Image.blend(input_image, deblurred_lr, alpha=0.25)

            mixed_images.append(mixed)

        # 2) Batch SR на GPU
        print(f"\n[Batch SR] Running super-resolution for {current_batch_size} images...")
        batch_tensors = [to_tensor(img) for img in mixed_images]  # list of CxHxW
        batch_tensor = torch.stack(batch_tensors, dim=0).to(device)  # BxCxHxW

        with torch.no_grad():
            output_batch = sr_model(batch_tensor)  # BxCxH'xW'

        output_batch = output_batch.clamp(0, 1).cpu()

        # 3) Post-processing для кожного елементу батчу (CPU + nafnetlib)
        for local_idx, input_path in enumerate(batch_paths):
            global_idx = batch_start + local_idx
            print(f"\n--- Post-processing image {global_idx+1}/{total}: {os.path.basename(input_path)} ---")

            # Витягуємо відповідний SR-тензор
            sr_tensor = output_batch[local_idx]
            upscaled_image = to_pil(sr_tensor)

            # Step 3: Downscale
            print(f"[{global_idx+1}/{total}] Step 3: Downscaling to 1920 x 1440...")
            downscaled_image = upscaled_image.resize((1920, 1440), Image.Resampling.LANCZOS)

            # Step 4: Deblur HR
            print(f"[{global_idx+1}/{total}] Step 4: Deblurring (REDS) on HR...")
            deblurred_image = deblur_processor.process(downscaled_image)

            # Step 5: Denoise HR
            print(f"[{global_idx+1}/{total}] Step 5: Denoising (SIDD)...")
            denoised_image = denoise_processor.process(deblurred_image)

            # Step 6: Sharpen (залишено вимкненим)
            print(f"[{global_idx+1}/{total}] Step 6: Sharpening (disabled, passing denoised)...")
            final_image = denoised_image

            # Step 7: Save
            input_filename = os.path.basename(input_path)
            output_filename = f"{os.path.splitext(input_filename)[0]}_processed.png"
            output_path = output_dir / output_filename

            final_image.save(str(output_path), "PNG", quality=98)
            print(f"[{global_idx+1}/{total}] Final result SAVED -> {output_path}")

    print("\nAll processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a full image processing pipeline on a directory of images.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the directory containing input images.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='The directory where processed images will be saved.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for SR model (1 = no batching).')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    image_files = []
    if input_path.is_dir():
        # Add more extensions if needed
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        print(f"Searching for images in {input_path}...")
        found_files = set()
        for f in input_path.glob('*'):
            if f.suffix.lower() in supported_extensions:
                found_files.add(f)

        # Sort files to ensure consistent order
        image_files = sorted(list(found_files), key=lambda f: f.name)
    elif input_path.is_file():
        image_files.append(input_path)

    if not image_files:
        print(f"No image files found in {args.input_dir}")
        sys.exit(1)

    # Можна додати start-index / index, якщо треба — поки прибрав для простоти,
    # але легко повернути назад.

    run_custom_pipeline(image_files, args.output_dir, args.batch_size)
