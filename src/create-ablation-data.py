import os
import cv2
import argparse
import random
import numpy as np
import albumentations as A
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, stddev=25):
    """
    Applies Gaussian noise to each pixel based on a normal distribution.
    """
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Consolidated Ablation Dataset Generator")
    
    # Path Arguments
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Path to clean images (e.g., data/coco-dataset/val2017)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to save processed images")
    
    # Ablation Selection
    parser.add_argument('--type', type=str, choices=['motion_blur', 'gaussian_noise'], 
                        default='motion_blur', help="Type of ablation to apply")
    
    # Parameter Arguments
    parser.add_argument('--kernel_size', type=int, default=15, 
                        help="Motion blur kernel size (odd integer >= 3)")
    parser.add_argument('--noise_mean', type=float, default=0.0, 
                        help="Mean for Gaussian noise")
    parser.add_argument('--noise_std', type=float, default=25.0, 
                        help="Standard deviation for Gaussian noise")

    # Sampling Argument for testing 
    parser.add_argument('--num_images', type=int, default=None,
                        help="Number of images to randomly sample and process. If not set, processes all.")

    args = parser.parse_args()

    # 1. Validation & Setup
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if args.type == 'motion_blur' and (args.kernel_size < 3 or args.kernel_size % 2 == 0):
        raise ValueError(f"Kernel size must be an odd integer >= 3. Provided: {args.kernel_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Setup Motion Blur Transform (Random Angle)
    # Uses albumentations to randomize the angle (0-360) for every image.
    blur_transform = A.Compose([
        A.MotionBlur(blur_limit=(args.kernel_size, args.kernel_size), allow_shifted=True, p=1.0)
    ])

    # 3. Identify and Sample Images
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Apply Sampling Logic
    if args.num_images is not None:
        if args.num_images >= len(image_files):
            print(f"Note: --num_images ({args.num_images}) is >= available images ({len(image_files)}). Using all.")
        else:
            print(f"Sampling {args.num_images} images randomly from {len(image_files)}...")
            image_files = random.sample(image_files, args.num_images)

    # 4. Process Images
    print(f"Applying {args.type} to {len(image_files)} images...")

    for filename in tqdm(image_files, desc="Processing", unit="img"):
        img = cv2.imread(os.path.join(args.input_dir, filename))
        if img is None: continue

        if args.type == 'motion_blur':
            # Randomized angle blur
            processed = blur_transform(image=img)["image"]
        else:
            # Gaussian shift per pixel
            processed = add_gaussian_noise(img, mean=args.noise_mean, stddev=args.noise_std)

        # Save at 100 quality for high-fidelity ablation results
        cv2.imwrite(os.path.join(args.output_dir, filename), processed, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"\nDone! Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()