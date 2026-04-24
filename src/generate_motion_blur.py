import os
import cv2
import argparse
import albumentations as A
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Standalone tool to generate a Motion Blurred dataset.")
    # We use direct paths here so the user is forced to be explicit
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Path to the clean image directory (e.g., data/coco-dataset/val2017)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to save blurred images (e.g., data/ablation-datasets/coco_blur_k15)")
    parser.add_argument('--kernel_size', type=int, default=15, 
                        help="Magnitude of the blur (streak length). Must be an odd integer >= 3.")
    args = parser.parse_args()

    # 1. Validation
    if args.kernel_size < 3 or args.kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be an odd integer >= 3. You provided: {args.kernel_size}")

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Setup Transform
    # blur_limit=(k, k) ensures constant magnitude; allow_shifted=True enables random angles
    transform = A.Compose([
        A.MotionBlur(blur_limit=(args.kernel_size, args.kernel_size), allow_shifted=True, p=1.0)
    ])

    # 3. Process
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Generating blur (K={args.kernel_size}) for {len(image_files)} images...")

    for filename in tqdm(image_files, desc="Blurring", unit="img"):
        img = cv2.imread(os.path.join(args.input_dir, filename))
        if img is None: continue

        blurred = transform(image=img)["image"]

        # Save at 100 quality to ensure the only 'noise' is the blur we intended
        cv2.imwrite(os.path.join(args.output_dir, filename), blurred, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"\nDone! Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()