"""Datasets used in the project."""

import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_wine(path: str = os.path.join(DATA, "wine.csv")):
    """Wine dataset (UCI): 178 wines from 3 cultivars, 13 chemical measurements.

    Returns (X, cultivar, feature_names).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader]
    arr = np.array(rows)
    return arr[:, 1:], arr[:, 0].astype(int), header[1:]


def load_image(path: str = os.path.join(DATA, "astronaut.png"), gray: bool = True) -> np.ndarray:
    """Image as floats in [0, 1]; gray=True converts to luminance (0.299 R + 0.587 G + 0.114 B)."""
    import matplotlib.image as mpimg

    img = mpimg.imread(path)[..., :3].astype(float)
    if img.max() > 1:
        img /= 255.0
    return img @ np.array([0.299, 0.587, 0.114]) if gray else img
