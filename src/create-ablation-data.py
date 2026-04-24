import cv2 as cv
import numpy as np
import os
from model_configs import DATASET_PATH

# ----- Configs ----- #
ABLATION_TYPE = "motion_blur" # or "gaussian_noise"
OUTPUT_DIR = "../data/coco/ablation-datasets/" + ABLATION_TYPE # Path to the output directory for ablation images

# Motion blur parameters
MOTION_BLUR_KERNEL_SIZE = 15
MOTION_BLUR_ANGLE = 45

# Gaussian noise parameters
GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_STDDEV = 25

# ----- Motion Blur ----- #
def apply_motion_blur(image, kernel_size=15, angle=0):
    # Create the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel_size

    # Rotate the kernel to the specified angle
    M = cv.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv.warpAffine(kernel, M, (kernel_size, kernel_size))

    # Apply the motion blur to the image
    blurred = cv.filter2D(image, -1, kernel)

    return blurred

# ----- Noise Addition ----- #
def add_gaussian_noise(image, mean=0, stddev=25):
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

    # Add the noise to the image
    noisy_image = cv.add(image.astype(np.float32), noise)

    # Clip the values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# ----- Create Ablation Datasets ----- #
def create_ablation_data(input_dir, output_dir, ablation_type="motion_blur"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = cv.imread(image_path)

            if ablation_type == "motion_blur":
                processed_image = apply_motion_blur(image, kernel_size=MOTION_BLUR_KERNEL_SIZE, angle=MOTION_BLUR_ANGLE)
                output_filename = f"{os.path.splitext(filename)[0]}_blurred.jpg"
            elif ablation_type == "gaussian_noise":
                processed_image = add_gaussian_noise(image, mean=GAUSSIAN_NOISE_MEAN, stddev=GAUSSIAN_NOISE_STDDEV)
                output_filename = f"{os.path.splitext(filename)[0]}_noisy.jpg"
            else:
                continue

            # Save the processed images
            cv.imwrite(os.path.join(output_dir, output_filename), processed_image)

if __name__ == "__main__":
    create_ablation_data(DATASET_PATH, OUTPUT_DIR, ABLATION_TYPE)