from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity as ssim

from image import divide_into_patches


class Comparetor(ABC):
    @abstractmethod
    def compare(self, image_a: Any, image_b: Any) -> Any:
        pass


class SSIMComparetor(Comparetor):
    def compare(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        if image_a.shape != image_b.shape:
            return 0.0

        min_side = min(image_a.shape[:2])
        if min_side < 7:
            return 0.0

        # For an RGB image with shape (256, 256, 3)
        channel_axis = -1

        # range of score
        # 1.0 perfectly match
        # 0 no structural similarity
        # -1.0 perfectly anti structural
        return float(ssim(image_a, image_b, channel_axis=channel_axis))


class PixelComparetor(Comparetor):
    def __init__(self, threshold=30):
        self.threshold = threshold

    def get_diff_mask(self, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        if image_a.shape != image_b.shape:
            return np.array([])

        diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
        return np.any(diff > self.threshold, axis=-1)

    def compare(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        mask = self.get_diff_mask(image_a, image_b)

        if mask.size == 0:
            return 0.0

        return float(np.sum(mask) / mask.size)


def compare_image(
    image_a: np.ndarray, image_b: np.ndarray, patch_size: int = 128
) -> np.ndarray:
    if image_a is None or image_b is None:
        return np.array([])

    if image_a.shape != image_b.shape:
        return np.array([])

    patches_a = divide_into_patches(image_a, patch_size)
    patches_b = divide_into_patches(image_b, patch_size)

    if not patches_a:
        return np.array([])

    comparator = SSIMComparetor()

    with ProcessPoolExecutor() as executor:
        scores = list(executor.map(comparator.compare, patches_a, patches_b))

    rows, cols = image_a.shape[0] // patch_size, image_a.shape[1] // patch_size
    return np.array(scores).reshape((rows, cols))
