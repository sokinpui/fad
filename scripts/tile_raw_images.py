import argparse
import cv2
from pathlib import Path
from typing import Tuple

class ImageTiler:
    def __init__(
        self, 
        input_dir: str = "raw_images", 
        output_dir: str = "tiled_images", 
        tile_size: int = 512,
        margins: Tuple[int, int] = (0, 0)
    ):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.tile_size = tile_size
        self.margins = margins
        self.supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def run(self):
        if not self.input_path.exists():
            print(f"Error: Input path '{self.input_path}' not found.")
            return

        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.input_path.is_file():
            self._process_image(self.input_path)
            return

        for file_path in self.input_path.iterdir():
            self._process_image(file_path)

    def _process_image(self, file_path: Path):
        if file_path.suffix.lower() not in self.supported_extensions:
            return

        image = cv2.imread(str(file_path))
        if image is None:
            return

        h, w = image.shape[:2]
        print(f"Tiling {file_path.name} ({w}x{h})...")

        margin_w, margin_h = self.margins
        
        y_start, y_end = margin_h, h - margin_h
        x_start, x_end = margin_w, w - margin_w

        for y in range(y_start, y_end - self.tile_size + 1, self.tile_size):
            for x in range(x_start, x_end - self.tile_size + 1, self.tile_size):
                self._extract_tile(image, x, y, file_path.stem)

    def _extract_tile(self, image: cv2.Mat, x: int, y: int, base_name: str):
        tile = image[y : y + self.tile_size, x : x + self.tile_size]
        
        if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
            return

        tile_filename = f"{base_name}_y{y}_x{x}.bmp"
        save_path = self.output_path / tile_filename
        cv2.imwrite(str(save_path), tile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile large images into smaller patches.")
    parser.add_argument("--input", default="raw_images", help="Input image file or directory")
    parser.add_argument("--output", "--out", default="tiled_images", help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Tile size")
    parser.add_argument(
        "--margin", type=int, nargs=2, default=[0, 0], metavar=("W", "H"),
        help="Exclude W pixels from left/right and H pixels from top/bottom"
    )
    args = parser.parse_args()

    tiler = ImageTiler(args.input, args.output, args.size, tuple(args.margin))
    tiler.run()
