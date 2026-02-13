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
