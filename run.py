import argparse
import cv2
import numpy as np
from compare import compare_image


def run_patch_comparison(baseline_path: str, test_path: str, patch_size: int):
    image_a = cv2.imread(baseline_path)
    image_b = cv2.imread(test_path)

    if image_a is None:
        print(f"Error: Could not read image at {baseline_path}")
        return

    if image_b is None:
        print(f"Error: Could not read image at {test_path}")
        return

    ssim_matrix = compare_image(image_a, image_b, patch_size)

    if ssim_matrix.size == 0:
        print("Error: Comparison failed. Images might have different shapes or are too small.")
        return

    print("SSIM Matrix:")
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    print(ssim_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patch-based image comparison.")
    parser.add_argument("baseline", help="Path to the baseline image")
    parser.add_argument("test", help="Path to the test image")
    parser.add_argument("--patch-size", type=int, default=128, help="Size of the patches")
    args = parser.parse_args()

    run_patch_comparison(args.baseline, args.test, args.patch_size)
