import argparse

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compare_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    if image_a.shape != image_b.shape:
        return 0.0

    min_side = min(image_a.shape[:2])
    if min_side < 7:
        return 0.0

    channel_axis = 2 if len(image_a.shape) == 3 else None

    # NOTE: range of score
    # 1.0 perfectly match
    # 0 no structural similarity
    # -1.0 perfectly anti structural
    return float(ssim(image_a, image_b, channel_axis=channel_axis))


def run_comparison(baseline_path: str, test_path: str):
    image_a = cv2.imread(baseline_path)
    image_b = cv2.imread(test_path)

    if image_a is None:
        print(f"Error: Could not read image at {baseline_path}")
        return

    if image_b is None:
        print(f"Error: Could not read image at {test_path}")
        return

    score = compare_ssim(image_a, image_b)
    print(f"SSIM: {score:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two images using SSIM.")
    parser.add_argument("baseline", help="Path to the baseline image")
    parser.add_argument("test", help="Path to the test image")
    args = parser.parse_args()

    run_comparison(args.baseline, args.test)
