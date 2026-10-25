from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}

def list_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")