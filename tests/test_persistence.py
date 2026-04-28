"""Unit tests for backend.persistence.

These verify the *topology math*, not the demo behaviour.  Each test uses a
synthetic image whose persistence diagram can be read off by hand, then
asserts the computed diagram matches.

Tests run against both backends (gudhi when available, plus the pure-NumPy
fallback) so a regression in either is caught.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from backend import persistence
from backend.persistence import PersistencePair


# Backends to exercise.  We always include the NumPy fallback; gudhi is added
# if the import succeeded, so the suite still runs (with one path skipped) on
# environments where gudhi is missing.
BACKENDS = ["numpy"] + (["gudhi"] if persistence._HAS_GUDHI else [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite_dim0(result: dict) -> List[PersistencePair]:
    return [p for p in result["dim0"] if not np.isinf(p.persistence)]


def _approx_birth_death(pair: PersistencePair, birth: float, death: float, tol: float = 1e-6) -> bool:
    return abs(pair.birth - birth) <= tol and abs(pair.death - death) <= tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_constant_image_has_one_essential_class(backend):
    """A flat image has exactly one connected component, born and never dying."""
    img = np.full((10, 10), 0.5, dtype=np.float32)
    result = persistence.compute_cubical_persistence(img, backend=backend)

    essential = [p for p in result["dim0"] if np.isinf(p.persistence)]
    assert len(essential) == 1, f"expected 1 essential H0 class, got {len(essential)}"
    # No finite H0 features (no merges happened).
    assert _finite_dim0(result) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_peak_has_one_essential_class_and_no_finite(backend):
    """One bright pixel on a dark background → single essential H0, no finite pairs."""
    img = np.zeros((9, 9), dtype=np.float32)
    img[4, 4] = 1.0
    result = persistence.compute_cubical_persistence(img, backend=backend)

    essential = [p for p in result["dim0"] if np.isinf(p.persistence)]
    assert len(essential) == 1
    # A single peak that just grows — never merges with another component.
    assert _finite_dim0(result) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_separated_peaks_produce_one_finite_pair(backend):
    """Two equal-height peaks separated by a valley → 1 essential + 1 finite H0.

    The finite pair has birth = peak height, death = valley height (the
    threshold at which the two components first connect).
    """
    img = np.zeros((5, 11), dtype=np.float32)
    img[2, 2] = 1.0   # left peak
    img[2, 8] = 1.0   # right peak
    # Valley: a row of 0.3 connecting them when threshold drops to 0.3
    img[2, 3:8] = 0.3

    result = persistence.compute_cubical_persistence(img, backend=backend)
    finite = _finite_dim0(result)
    essential = [p for p in result["dim0"] if np.isinf(p.persistence)]

    assert len(essential) == 1
    assert len(finite) == 1, f"expected exactly one finite H0 pair, got {len(finite)}"
    pair = finite[0]
    # Birth is at the younger peak's height (1.0); death is when the two
    # components merge (valley value 0.3).
    assert _approx_birth_death(pair, birth=1.0, death=0.3, tol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_persistence_threshold_filters_short_bars(backend):
    """Pairs below the threshold should be removed before returning."""
    img = np.zeros((5, 11), dtype=np.float32)
    img[2, 2] = 1.0
    img[2, 8] = 1.0
    img[2, 3:8] = 0.95  # very shallow valley → very short finite bar

    full = persistence.compute_cubical_persistence(img, backend=backend, persistence_threshold=0.0)
    pruned = persistence.compute_cubical_persistence(img, backend=backend, persistence_threshold=0.20)

    assert len(_finite_dim0(full)) == 1
    assert len(_finite_dim0(pruned)) == 0, "pair with persistence ~0.05 should be filtered"


@pytest.mark.parametrize("backend", BACKENDS)
def test_birth_coordinate_is_at_an_actual_peak(backend):
    """The birth_xy of a finite H0 pair should land on a high-intensity pixel.

    We don't assert the *exact* coordinate (in case of ties / orientation
    conventions) — only that the cell at birth_xy is at peak intensity.
    """
    img = np.zeros((9, 19), dtype=np.float32)
    img[4, 3] = 1.0
    img[4, 15] = 1.0
    img[4, 4:15] = 0.2

    result = persistence.compute_cubical_persistence(img, backend=backend)
    finite = _finite_dim0(result)
    assert finite, "expected at least one finite pair"
    p = finite[0]
    assert p.birth_xy is not None
    bx, by = p.birth_xy
    assert img[by, bx] == pytest.approx(1.0), \
        f"birth_xy=({bx},{by}) lands on intensity {img[by, bx]}, not the peak"


@pytest.mark.parametrize("backend", BACKENDS)
def test_persistence_pairs_are_sorted_by_persistence_descending(backend):
    """Both dim0 and dim1 lists should be sorted with the longest bar first."""
    rng = np.random.default_rng(0)
    img = rng.random((24, 24)).astype(np.float32)

    result = persistence.compute_cubical_persistence(img, backend=backend)
    for pairs in (result["dim0"], result["dim1"]):
        finite = [p.persistence for p in pairs if not np.isinf(p.persistence)]
        assert finite == sorted(finite, reverse=True), \
            f"pairs returned unsorted: {finite[:5]}…"


def test_invalid_image_shape_raises():
    img3d = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2D"):
        persistence.compute_cubical_persistence(img3d)


def test_invalid_backend_name_raises():
    img = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown backend"):
        persistence.compute_cubical_persistence(img, backend="invented")


@pytest.mark.parametrize("backend", BACKENDS)
def test_topology_mask_is_binary_and_correct_shape(backend):
    img = np.zeros((20, 20), dtype=np.float32)
    img[5, 5] = 1.0
    img[15, 15] = 1.0
    img[5:16, 5:16] = np.maximum(img[5:16, 5:16], 0.2)

    result = persistence.compute_cubical_persistence(img, backend=backend)
    pairs = list(result["dim0"]) + list(result["dim1"])
    mask = persistence.topology_mask(img, pairs, top_k=2, min_persistence=0.1)

    assert mask.shape == img.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    # At least one of the two peaks should have been highlighted.
    assert mask.sum() > 0, "expected some pixels to be highlighted"
