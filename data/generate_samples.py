"""Synthetic MRI-like sample generator (legacy — not used by the demo).

The repo now ships six real public-domain MRI slices in ``data/samples/``
(see ``data/README_data.md`` for sources), so this script is no longer on
the setup path.  It is kept around as a fallback for environments where
real medical images can't be shipped: running it will overwrite
``data/samples/`` and ``labels.json`` with cartoon brain renderings.
The output is NOT clinical data and must not be used for medical purposes.

Each sample is a 192x192 grayscale PNG that mimics a coarse axial brain
slice:

- An elliptical "skull" boundary
- A roughly elliptical "brain" region with low-frequency intensity texture
- Optional "tumor" blob(s): bright dense mass, possibly with a darker
  necrotic core (creating a 1D loop in the persistence diagram)
- Gaussian noise on top so the topology layer has something to denoise

Run::

    python data/generate_samples.py

Outputs PNGs to ``data/samples/`` and writes a labels manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


RNG_SEED = 42
SIZE = 192
SAMPLES_DIR = Path(__file__).parent / "samples"


def _ellipse_mask(shape, center, radii, angle_deg=0.0):
    h, w = shape
    cy, cx = center
    ry, rx = radii
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.deg2rad(angle_deg)
    xr = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    yr = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    return (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0


def _smooth_noise(shape, scale: int, rng: np.random.Generator) -> np.ndarray:
    """Generate low-frequency texture by upsampling small random fields."""
    small = rng.standard_normal(size=(max(1, shape[0] // scale), max(1, shape[1] // scale)))
    small = small.astype(np.float32)
    img = Image.fromarray(small).resize((shape[1], shape[0]), Image.BICUBIC)
    out = np.asarray(img, dtype=np.float32)
    out = (out - out.mean()) / (out.std() + 1e-6)
    return out


def _make_brain(rng: np.random.Generator) -> np.ndarray:
    """Skull-stripped axial brain cartoon (matches BraTS preprocessing).

    Real BraTS volumes are already skull-stripped, so we don't include the
    skull here.  Including it would make the skull the dominant topological
    feature in every slice and drown out the tumor signal we care about.
    """
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    cy, cx = SIZE // 2, SIZE // 2

    # Brain parenchyma (mid-grey textured) — soft elliptical blob.
    brain_mask = _ellipse_mask((SIZE, SIZE), (cy, cx), (SIZE * 0.38, SIZE * 0.33))
    texture = _smooth_noise(canvas.shape, scale=18, rng=rng) * 0.08
    canvas[brain_mask] = 0.40 + texture[brain_mask]

    # Ventricles (two darker oval regions roughly at the centre).
    for offset in (-0.05, 0.05):
        v = _ellipse_mask(
            (SIZE, SIZE),
            (cy - 4, int(cx + offset * SIZE)),
            (SIZE * 0.07, SIZE * 0.04),
        )
        canvas[v] = 0.15

    return canvas


def _add_tumor(canvas: np.ndarray, rng: np.random.Generator, *, with_core: bool) -> np.ndarray:
    """Drop a bright tumor lesion into the slice.

    Two flavours:

    - **solid** (``with_core=False``): 3–5 isolated bright "hotspots"
      placed in the brain.  Each hotspot yields its own H₀ persistence pair
      that lives from hotspot intensity down to the parenchyma intensity —
      so the topological signature is multiple long bars in H₀.
    - **ring** (``with_core=True``): a bright torus around a dark core.
      Geometrically a loop — produces a persistent H₁ feature.

    Intensities are calibrated so per-image min-max normalization (applied
    in :func:`_finalize`) puts hotspots at 1.0 and parenchyma around 0.4 —
    the long persistence bars then sit at ≈ 0.6, far above the texture
    floor (≈ 0.05) of healthy slices.
    """
    h, w = canvas.shape
    tumor_cy = rng.integers(int(h * 0.30), int(h * 0.65))
    tumor_cx = rng.integers(int(w * 0.30), int(w * 0.70))

    yy, xx = np.mgrid[0:h, 0:w]

    if with_core:
        # Bright rim + dark core.  Rim from difference-of-Gaussians; clip at
        # zero so the centre is genuinely darker than the rim → H₁ loop.
        radius = rng.integers(int(h * 0.09), int(h * 0.13))
        dist = np.sqrt((yy - tumor_cy) ** 2 + (xx - tumor_cx) ** 2)
        outer = np.exp(-(dist / (radius * 0.80)) ** 2).astype(np.float32)
        inner = np.exp(-(dist / (radius * 0.40)) ** 2).astype(np.float32)
        ring = np.clip(outer - inner * 1.4, 0, None)
        canvas = canvas + ring * 0.55
        return canvas

    # --- Solid: isolated bright hotspots scattered inside a small region. ---
    spread = rng.integers(int(h * 0.08), int(h * 0.12))
    n_peaks = int(rng.integers(3, 6))
    for _ in range(n_peaks):
        for _attempt in range(20):
            py = rng.integers(tumor_cy - spread, tumor_cy + spread + 1)
            px = rng.integers(tumor_cx - spread, tumor_cx + spread + 1)
            if (py - tumor_cy) ** 2 + (px - tumor_cx) ** 2 <= spread ** 2:
                break
        sub_dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        sub_radius = rng.uniform(2.5, 4.0)
        # Use np.maximum (not add) so overlapping hotspots don't stack to
        # saturation — each peak stays at its true height.
        peak_amp = float(rng.uniform(0.40, 0.55))
        canvas = np.maximum(canvas, peak_amp * np.exp(-(sub_dist / sub_radius) ** 2))
    return canvas


def _finalize(canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    canvas = canvas + rng.normal(0, 0.02, size=canvas.shape).astype(np.float32)
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min() + 1e-8)
    return (canvas * 255).astype(np.uint8)


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    manifest = []
    plan = [
        ("01_healthy_axial.png", "healthy", False),
        ("02_healthy_axial.png", "healthy", False),
        ("03_tumor_solid.png",   "tumor",   False),
        ("04_tumor_solid.png",   "tumor",   False),
        ("05_tumor_ring.png",    "tumor",   True),
        ("06_tumor_ring.png",    "tumor",   True),
    ]

    for fname, label, with_core in plan:
        canvas = _make_brain(rng)
        if label == "tumor":
            canvas = _add_tumor(canvas, rng, with_core=with_core)
        img = _finalize(canvas, rng)

        out_path = SAMPLES_DIR / fname
        Image.fromarray(img).save(out_path)
        manifest.append({"file": fname, "label": label, "ring_enhanced": bool(with_core)})

    with open(SAMPLES_DIR / "labels.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
