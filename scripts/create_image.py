import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List


class ImageCreator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def create_solid(self, rgb_color: List[int]) -> np.ndarray:
        if len(rgb_color) != 3:
            raise ValueError("Color must contain exactly 3 components (R,G,B).")

        bgr_color = rgb_color[::-1]
        return np.full((self.height, self.width, 3), bgr_color, dtype=np.uint8)


def parse_rgb(color_str: str) -> List[int]:
    components = color_str.split(",")
    if len(components) != 3:
        raise ValueError("Invalid color format. Use R,G,B (e.g., 255,0,0).")

    return [int(c.strip()) for c in components]


def main():
    parser = argparse.ArgumentParser(description="Create a solid color image.")
    parser.add_argument("color", help="Fill color in R,G,B format (e.g., 255,0,0)")
    parser.add_argument("--output", "-o", default="solid_image.png", help="Output file path")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    args = parser.parse_args()

    try:
        rgb = parse_rgb(args.color)
    except ValueError as e:
        print(f"Error: {e}")
        return

    creator = ImageCreator(args.width, args.height)
    image = creator.create_solid(rgb)

    output_path = Path(args.output)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    main()
