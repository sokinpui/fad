import cv2
import numpy as np
from typing import List


def divide_into_patches(image: np.ndarray, patch_size: int) -> List[np.ndarray]:
    if image is None:
        return []

    if patch_size <= 0:
        return []

    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        return []

    patches = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    return patches


def convert_to_oklab(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.array([])

    if len(image.shape) != 3 or image.shape[2] != 3:
        return np.array([])

    rgb = image.astype(np.float32) / 255.0
    rgb = rgb[..., ::-1]

    linear_rgb = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )

    rgb_to_lms = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])
    lms = np.dot(linear_rgb, rgb_to_lms.T)

    lms_non_linear = np.cbrt(lms)

    lms_to_oklab = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])

    return np.dot(lms_non_linear, lms_to_oklab.T)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    if image is None or image.size == 0:
        return np.array([])

    if kernel_size <= 0:
        return image

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
