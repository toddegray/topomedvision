"""Image I/O, preprocessing, and overlay utilities.

The pipeline expects 2D grayscale arrays in [0, 1] with shape (H, W).  We try
to keep the conversions explicit so a reader can follow exactly what the
topology layer sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage import exposure, filters

# Use a fixed working size so persistent homology on user uploads stays fast.
# Cubical persistence cost grows with pixel count; 192x192 is a good demo
# trade-off between latency (~1s) and visible structure.
DEFAULT_SIZE = (192, 192)


def load_image(path: str | Path, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Load an image from disk as a normalized grayscale array.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``size`` with values in [0, 1].
    """
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    return _to_float01(np.asarray(img))


def from_pil(img: Image.Image, size: Tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Convert a PIL image (any mode) to a normalized grayscale array."""
    img = img.convert("L").resize(size, Image.BILINEAR)
    return _to_float01(np.asarray(img))


def _to_float01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def preprocess(image: np.ndarray, denoise: bool = True, equalize: bool = False) -> np.ndarray:
    """Light preprocessing for MRI-like grayscale slices.

    - Optional gaussian denoise (sigma=1) to suppress single-pixel speckle that
      would otherwise create many short-lived topological features.
    - Optional CLAHE-style local contrast equalization.  **Off by default**
      because CLAHE rescales contrast locally, which can flatten exactly the
      global "tumor is brighter than parenchyma" signal we want the topology
      layer to see.  Turn it on for slices acquired with very different gain
      settings, where global intensity scales aren't comparable.
    """
    out = image.astype(np.float32, copy=True)
    if denoise:
        out = filters.gaussian(out, sigma=1.0, preserve_range=True)
    if equalize:
        out = exposure.equalize_adapthist(np.clip(out, 0, 1), clip_limit=0.02)
    return _to_float01(out)


def otsu_mask(image: np.ndarray) -> np.ndarray:
    """Baseline tumor mask using Otsu thresholding on the bright tail.

    The classical "non-topological" baseline we compare against in the demo.
    """
    thresh = filters.threshold_otsu(image)
    return (image > thresh).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (220, 30, 30),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask onto a grayscale image as an RGB overlay.

    Returns uint8 RGB suitable for ``st.image`` / matplotlib.
    """
    base = np.clip(image, 0, 1)
    rgb = np.stack([base, base, base], axis=-1) * 255.0
    rgb = rgb.astype(np.float32)

    mask_bool = mask.astype(bool)
    if mask_bool.any():
        tint = np.array(color, dtype=np.float32)
        rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * tint
    return rgb.clip(0, 255).astype(np.uint8)


def list_sample_images(samples_dir: str | Path) -> list[Path]:
    """Return PNG files in the samples directory, sorted for stable UI order."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        return []
    return sorted(samples_dir.glob("*.png"))
