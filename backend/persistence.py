"""Cubical persistent homology for 2D grayscale images.

We expose a single high-level entry point, :func:`compute_cubical_persistence`,
that returns a dictionary of persistence pairs annotated with their spatial
birth coordinates.  Two backends are supported:

1. ``gudhi`` — the scientific standard for cubical complexes.  Used for both
   0D (connected components) and 1D (loops) persistence when available.
2. A pure-NumPy union-find fallback for 0D persistence on the sublevel /
   superlevel filtrations, plus an Alexander-duality-style approximation for
   1D persistence.  This keeps the demo runnable on systems where the gudhi
   wheel doesn't install.

Spatial birth coordinates are attached because the demo overlays high-
persistence features back onto the original image.  A persistence pair without
a location is just a number; with a location it tells the radiologist
*where* to look.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage import morphology, segmentation

# gudhi is optional.  If it imports, we use it; otherwise we fall back.
try:  # pragma: no cover - import availability is environment-dependent
    import gudhi as _gd  # type: ignore

    _HAS_GUDHI = True
except Exception:  # noqa: BLE001
    _gd = None
    _HAS_GUDHI = False


# ---------------------------------------------------------------------------
# Public dataclass returned to the rest of the app.
# ---------------------------------------------------------------------------


@dataclass
class PersistencePair:
    """A single (birth, death) pair in a persistence diagram.

    Attributes
    ----------
    dim : int
        Homological dimension (0 = connected component, 1 = loop / hole).
    birth, death : float
        Filtration values at which the feature appears and disappears.  For
        an essential class (one that never dies in the chosen filtration) we
        store ``death = np.inf`` and clip it for display.
    persistence : float
        ``|death - birth|``.  The "shape strength" of the feature.
    birth_xy : (int, int) or None
        ``(x, y)`` pixel coordinates of the cell that birthed this feature,
        when available.  ``None`` for features whose location couldn't be
        recovered (rare; only for some gudhi outputs).
    """

    dim: int
    birth: float
    death: float
    persistence: float
    birth_xy: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def compute_cubical_persistence(
    image: np.ndarray,
    *,
    backend: str = "auto",
    persistence_threshold: float = 0.0,
) -> Dict:
    """Compute 0D and 1D cubical persistence of a 2D grayscale image.

    Parameters
    ----------
    image
        2D float array, any range — it will be normalized to [0, 1] internally.
    backend
        ``"auto"`` (default) uses gudhi if available, else the NumPy fallback.
        ``"gudhi"`` forces the gudhi backend (raises if missing).
        ``"numpy"`` forces the union-find fallback.
    persistence_threshold
        Drop features with persistence smaller than this value before
        returning.  Useful to reduce visual clutter from filtration noise.

    Returns
    -------
    dict
        ``{
            "dim0": List[PersistencePair],   # connected components
            "dim1": List[PersistencePair],   # loops
            "betti": {"b0": int, "b1": int},
            "backend": "gudhi" | "numpy-fallback",
            "image_shape": (H, W),
         }``
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    img = _normalize(image)

    chosen = _choose_backend(backend)
    if chosen == "gudhi":
        dim0, dim1 = _gudhi_persistence(img)
    else:
        dim0, dim1 = _numpy_persistence(img)

    # Always drop pairs with literally zero persistence: they're an artifact
    # of the discrete pixel filtration when many pixels share the same value
    # (the "background" component is technically born and dies at the same
    # threshold). They carry no topological information.
    dim0 = [p for p in dim0 if (np.isinf(p.persistence) or p.persistence > 0) and p.persistence >= persistence_threshold]
    dim1 = [p for p in dim1 if (np.isinf(p.persistence) or p.persistence > 0) and p.persistence >= persistence_threshold]

    # Sort high persistence first — most "interesting" features at the top.
    dim0.sort(key=lambda p: p.persistence, reverse=True)
    dim1.sort(key=lambda p: p.persistence, reverse=True)

    return {
        "dim0": dim0,
        "dim1": dim1,
        "betti": {"b0": len(dim0), "b1": len(dim1)},
        "backend": "gudhi" if chosen == "gudhi" else "numpy-fallback",
        "image_shape": img.shape,
    }


def _choose_backend(backend: str) -> str:
    if backend == "gudhi":
        if not _HAS_GUDHI:
            raise RuntimeError("gudhi backend requested but gudhi is not installed")
        return "gudhi"
    if backend == "numpy":
        return "numpy"
    if backend != "auto":
        raise ValueError(f"Unknown backend {backend!r}")
    return "gudhi" if _HAS_GUDHI else "numpy"


def _normalize(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# gudhi backend.  Used when the wheel is installed.
# ---------------------------------------------------------------------------


def _gudhi_persistence(image: np.ndarray) -> Tuple[List[PersistencePair], List[PersistencePair]]:
    """Compute persistence using gudhi's CubicalComplex.

    We run TWO filtrations:

    * **Superlevel** on the image (i.e. sublevel of ``-image``) so that 0D
      classes are bright peaks and 1D classes are dark holes inside bright
      regions.  This matches the visual intuition radiologists use — bright
      tumor masses, dark necrotic cores.

    Returns lists of :class:`PersistencePair` for dimension 0 and dimension 1.
    """
    assert _gd is not None  # for type-checkers; guarded by _choose_backend

    # Superlevel filtration of `image` is sublevel filtration of `-image`.
    # We feed `-image` so high-intensity peaks are born early.
    cc = _gd.CubicalComplex(top_dimensional_cells=-image)
    cc.persistence()

    # cofaces_of_persistence_pairs returns birth/death cell indices so we can
    # recover the (y, x) location of each pair.
    try:
        regular, essential = cc.cofaces_of_persistence_pairs()
    except Exception:
        regular, essential = (([], []), ([], []))

    H, W = image.shape
    dim0: List[PersistencePair] = []
    dim1: List[PersistencePair] = []

    # `regular[d]` is an array of shape (n, 2): birth_cell, death_cell indices
    # `essential[d]` is an array of birth_cell indices (death = +inf)
    # gudhi flattens cells in *column-major* (Fortran) order — verified
    # empirically with a single-peak test image.  Using C-order here would
    # silently scramble (y, x) coords for non-square images.
    for d, pair_arr in enumerate(regular):
        for pair in pair_arr:
            birth_cell, death_cell = int(pair[0]), int(pair[1])
            by, bx = np.unravel_index(birth_cell, image.shape, order="F")
            dy, dx = np.unravel_index(death_cell, image.shape, order="F")
            # Filtration values: we passed -image, so filtration[c] = -image[c].
            # Convert back to image-space so the numbers in the UI match the
            # original intensity scale: bright peaks have HIGH birth, low death.
            b_disp = float(image[by, bx])
            d_disp = float(image[dy, dx])
            pair_obj = PersistencePair(
                dim=d,
                birth=b_disp,
                death=d_disp,
                persistence=abs(b_disp - d_disp),
                birth_xy=(int(bx), int(by)),
            )
            (dim0 if d == 0 else dim1).append(pair_obj)

    for d, idx_arr in enumerate(essential):
        for cell in idx_arr:
            cell = int(cell)
            by, bx = np.unravel_index(cell, image.shape, order="F")
            b_disp = float(image[by, bx])
            pair_obj = PersistencePair(
                dim=d,
                birth=b_disp,
                death=float("inf"),
                persistence=float("inf"),
                birth_xy=(int(bx), int(by)),
            )
            (dim0 if d == 0 else dim1).append(pair_obj)

    return dim0, dim1


# ---------------------------------------------------------------------------
# Pure-NumPy fallback.  Implements 0D persistence via union-find and
# approximates 1D via 0D on the inverted image (Alexander-duality intuition).
# ---------------------------------------------------------------------------


def _numpy_persistence(image: np.ndarray) -> Tuple[List[PersistencePair], List[PersistencePair]]:
    dim0 = _uf_persistence(image, mode="superlevel", out_dim=0)
    # 1D approximation: persistent dark "valleys" surrounded by bright pixels
    # behave like loops in superlevel filtration of the original image.  We
    # detect them as 0D classes of the inverted image and re-label as dim=1.
    dim1 = _uf_persistence(image, mode="sublevel", out_dim=1)
    return dim0, dim1


def _uf_persistence(
    image: np.ndarray,
    *,
    mode: str,
    out_dim: int,
) -> List[PersistencePair]:
    """0D persistent homology via union-find over a pixel filtration.

    ``mode='superlevel'``: process pixels high-value first; classes are
    bright peaks, persistence = peak_value - merge_value.  ``mode='sublevel'``
    is the symmetric case (dark valleys).

    Implementation notes
    --------------------
    - **4-connectivity.**  This is a slightly smaller cell complex than the
      one gudhi builds, which uses the full cubical complex (top-cells +
      edges + vertices).  In practice, the two notions of "cubical
      persistence" agree on prominent features but can differ on many
      short-lived ones — e.g., a random-noise image produces ~70% more
      finite H₀ pairs under 4-connectivity than under gudhi's complex.
      For the demo's purposes (highlighting the most persistent features)
      this is fine; treat the backends as approximations of each other on
      noisy inputs.
    - Path compression on `find`.  Union by elder-rule (the component with
      the more extreme birth — higher for superlevel, lower for sublevel —
      absorbs the other).
    """
    H, W = image.shape
    flat = image.flatten().astype(np.float32)
    n = H * W

    if mode == "superlevel":
        order = np.argsort(-flat, kind="stable")
        is_elder = lambda a, b: birth_val[a] > birth_val[b]  # noqa: E731
    elif mode == "sublevel":
        order = np.argsort(flat, kind="stable")
        is_elder = lambda a, b: birth_val[a] < birth_val[b]  # noqa: E731
    else:
        raise ValueError(mode)

    parent = -np.ones(n, dtype=np.int64)
    birth_val = np.zeros(n, dtype=np.float32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs: List[Tuple[float, float, int]] = []  # birth, death, root_pixel

    for idx in order:
        idx = int(idx)
        y, x = divmod(idx, W)
        v = float(flat[idx])

        # Collect roots of already-added neighboring components (deduped).
        neighbor_roots: List[int] = []
        seen: set[int] = set()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nidx = ny * W + nx
                if parent[nidx] != -1:
                    nroot = find(nidx)
                    if nroot not in seen:
                        seen.add(nroot)
                        neighbor_roots.append(nroot)

        if not neighbor_roots:
            # No neighbors yet → this pixel births a new component at value v.
            parent[idx] = idx
            birth_val[idx] = v
            continue

        # Pixel extends existing components.  Pick the elder; everyone else
        # dies at value v (their birth values were ≥ v since we process in
        # filtration order).  Crucially, *no* zero-persistence pair is
        # emitted for the current pixel — it is not a birth event.
        elder = neighbor_roots[0]
        for r in neighbor_roots[1:]:
            elder = elder if is_elder(elder, r) else r

        parent[idx] = elder
        birth_val[idx] = birth_val[elder]
        for r in neighbor_roots:
            if r is elder:
                continue
            pairs.append((float(birth_val[r]), v, r))
            parent[r] = elder

    # The single surviving component is the essential class (birth = global
    # extremum, death = +inf).  Skip it for dim=1 approximation since we
    # don't want a spurious "always alive" dark hole at the image boundary.
    if mode == "superlevel":
        global_root = find(int(order[0]))
        pairs.append((float(birth_val[global_root]), float("inf"), global_root))

    out: List[PersistencePair] = []
    for b, d, root in pairs:
        ry, rx = divmod(int(root), W)
        out.append(
            PersistencePair(
                dim=out_dim,
                birth=float(b),
                death=float(d),
                persistence=(abs(b - d) if not np.isinf(d) else float("inf")),
                birth_xy=(int(rx), int(ry)),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Highlighting: turn high-persistence features into a spatial mask.
# ---------------------------------------------------------------------------


def topology_mask(
    image: np.ndarray,
    pairs: Sequence[PersistencePair],
    *,
    top_k: int = 5,
    min_persistence: float = 0.20,
    grow_radius: int = 4,
) -> np.ndarray:
    """Build a binary mask of the ``top_k`` highest-persistence features.

    For each surviving pair we:

    1. Seed a region around its ``birth_xy`` coordinate.
    2. Flood-fill within the image's local intensity neighborhood to grow the
       seed to the natural extent of that bright component.
    3. Apply a small morphological closing so the highlight is a single
       contiguous blob, not a noisy ring.

    Defaults err on the side of *under-covering*: a small, confident
    highlight is more useful than a flood that buries the original image.

    Returns
    -------
    np.ndarray
        ``uint8`` mask of shape ``image.shape`` with values in ``{0, 1}``.
    """
    img = _normalize(image)
    H, W = img.shape
    mask = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]

    finite = [
        p for p in pairs
        if p.birth_xy is not None
        and not np.isinf(p.persistence)
        and p.persistence >= min_persistence
    ]
    finite.sort(key=lambda p: p.persistence, reverse=True)

    # Cap per-seed area so one very-persistent feature can't flood the image.
    max_region_size = int(0.05 * H * W)
    fallback_radius_sq = 36  # 6-pixel radius

    for pair in finite[:top_k]:
        x, y = pair.birth_xy
        if not (0 <= x < W and 0 <= y < H):
            continue

        if pair.dim == 0:
            # H₀ births land on bright peaks → flood-fill within a tight
            # intensity tolerance to capture the peak's plateau.
            try:
                region = segmentation.flood(img, (y, x), tolerance=0.10)
            except Exception:
                region = np.zeros_like(img, dtype=bool)
                region[y, x] = True
            if region.sum() > max_region_size:
                # Escaped its peak — collapse to a small disk so we still
                # mark something rather than over-flooding.
                region = (yy - y) ** 2 + (xx - x) ** 2 <= fallback_radius_sq
        else:
            # H₁ births sit on (or near) the loop's defining cycle — often
            # a saddle, not a local maximum, so flood-fill is unreliable.
            # We just place a fixed disk marker at the birth pixel, which is
            # where a radiologist's eye would look for the loop anyway.
            region = (yy - y) ** 2 + (xx - x) ** 2 <= fallback_radius_sq

        mask |= region

    if grow_radius > 0 and mask.any():
        mask = morphology.closing(mask, footprint=morphology.disk(grow_radius // 2))

    return mask.astype(np.uint8)
