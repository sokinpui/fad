import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops

sys.path.append(str(Path(__file__).parent.parent))

from compare import PixelComparetor


def compare_images(path1: str, path2: str, output_path: str):
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")

    if img1.size != img2.size:
        print(f"Error: Image dimensions do not match: {img1.size} vs {img2.size}")
        return

    diff = ImageChops.difference(img1, img2)
    diff.save(output_path)

    bbox = diff.getbbox()
    if bbox is None:
        print("Images are identical!")
        return

    print(f"Images are different. Bounding box of differences: {bbox}")


def compare_images_cv(path1: str, path2: str):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    diff = cv2.absdiff(img1, img2)

    non_zero_count = np.count_nonzero(np.any(diff != 0, axis=2))

    total_pixels = img1.shape[0] * img1.shape[1]
    percentage = (non_zero_count / total_pixels) * 100

    print(f"Percentage difference: {percentage:.4f}%")


def compare_images_cv2(path1: str, path2: str, threshold=30):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, pixels_above_threshold = cv2.threshold(
        gray_diff, threshold, 255, cv2.THRESH_BINARY
    )

    non_zero_count = np.count_nonzero(pixels_above_threshold)

    total_pixels = img1.shape[0] * img1.shape[1]
    percentage = (non_zero_count / total_pixels) * 100

    print(f"Percentage difference: {percentage:.4f}%")

    cv2.imwrite("visible_diff.png", pixels_above_threshold)


def compare_using_pixel_comparator(
    path1: str, path2: str, threshold: int = 30, output_path: str = "pixel_diff.png"
):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        return

    comparator = PixelComparetor(threshold=threshold)
    percentage = comparator.compare(img1, img2)
    print(
        f"PixelComparetor (threshold={threshold}): {percentage * 100:.4f}% difference"
    )

    mask = comparator.get_diff_mask(img1, img2)
    if mask.size > 0:
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description="Compare two images and save the difference."
    )
    parser.add_argument("path1", help="Path to the first image")
    parser.add_argument("path2", help="Path to the second image")
    parser.add_argument(
        "--output", "-o", default="output.png", help="Path to save the difference image"
    )
    args = parser.parse_args()

    threshold = 10
    # compare_images(args.path1, args.path2, args.output)
    # compare_images_cv(args.path1, args.path2)
    compare_images_cv2(args.path1, args.path2, threshold)
    compare_using_pixel_comparator(args.path1, args.path2, threshold)


if __name__ == "__main__":
    main()
