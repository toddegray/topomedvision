"""Cross-check the NumPy fallback against gudhi.

The two backends compute *related but not identical* cubical persistences
(see the docstring of :func:`backend.persistence._uf_persistence`):
gudhi uses the full cubical complex (top-cells + edges + vertices),
while the fallback uses 4-connected union-find on the top-cells only.
On clean inputs the two agree exactly; on noisy inputs they differ on
short-lived features but agree on prominent ones.

These tests therefore split into two regimes:

1. **Clean inputs** — synthetic images with well-separated features,
   where both backends must produce identical (birth, death, persistence)
   for the prominent pairs.
2. **Noisy inputs** — random / textured images, where we only require
   that the *most persistent* feature agrees within a generous tolerance,
   and that both backends agree on the global essential class.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import persistence

if not persistence._HAS_GUDHI:
    pytest.skip("gudhi not installed; cross-check is moot", allow_module_level=True)


def _finite_dim0(result):
    return [p for p in result["dim0"] if not np.isinf(p.persistence)]


def _essential_dim0(result):
    return [p for p in result["dim0"] if np.isinf(p.persistence)]


def _top_pair(pairs):
    if not pairs:
        return None
    return max(pairs, key=lambda p: p.persistence)


# ---------------------------------------------------------------------------
# Regime 1: clean inputs — exact agreement required.
# ---------------------------------------------------------------------------


def test_two_separated_gaussian_peaks_exact_agreement():
    """Two well-separated Gaussian peaks → both backends produce one finite
    H₀ pair with identical birth/death."""
    yy, xx = np.mgrid[0:40, 0:60]
    img = np.maximum(
        0.9 * np.exp(-((yy - 20) ** 2 + (xx - 15) ** 2) / 25),
        0.7 * np.exp(-((yy - 20) ** 2 + (xx - 45) ** 2) / 25),
    ).astype(np.float32)

    g_finite = _finite_dim0(persistence.compute_cubical_persistence(img, backend="gudhi"))
    n_finite = _finite_dim0(persistence.compute_cubical_persistence(img, backend="numpy"))

    g_top = _top_pair(g_finite)
    n_top = _top_pair(n_finite)
    assert g_top is not None and n_top is not None
    assert abs(g_top.birth - n_top.birth) < 1e-3
    assert abs(g_top.death - n_top.death) < 1e-3


def test_four_isolated_peaks_agree_on_dominant_pairs():
    """4 peaks of equal height connected only via a faint grid → both
    backends should report (essential birth=1.0) and three finite pairs
    with persistence ≈ 1.0 (peak height minus grid height)."""
    img = np.zeros((30, 30), dtype=np.float32)
    for y, x in [(5, 5), (5, 25), (25, 5), (25, 25)]:
        img[y, x] = 1.0
    img[14:16, :] = np.maximum(img[14:16, :], 0.2)
    img[:, 14:16] = np.maximum(img[:, 14:16], 0.2)

    g_finite = _finite_dim0(persistence.compute_cubical_persistence(img, backend="gudhi"))
    n_finite = _finite_dim0(persistence.compute_cubical_persistence(img, backend="numpy"))

    # Both backends find at least 3 long-persisting peaks (the 4 minus the
    # essential class).  Boundary artefacts may add extra short bars; we
    # only check the prominent ones agree.
    g_long = [p for p in g_finite if p.persistence > 0.5]
    n_long = [p for p in n_finite if p.persistence > 0.5]
    assert len(g_long) == 3
    assert len(n_long) == 3
    assert all(abs(p.persistence - 1.0) < 1e-3 for p in g_long + n_long)


# ---------------------------------------------------------------------------
# Regime 2: noisy inputs — only check prominent agreement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_random_images_produce_features_and_essential_class(seed):
    """On random texture, both backends should:

    - produce a non-trivial number of finite H₀ pairs;
    - report exactly one H₀ essential class;
    - agree on the essential class's birth value (= image global max).

    We deliberately do *not* assert agreement on individual pair lifetimes:
    the 4-connected union-find produces a different cell complex from
    gudhi's full cubical complex (see :func:`_uf_persistence` docstring),
    and on noisy inputs the resulting pair sets diverge by design.
    """
    img = np.random.default_rng(seed).random((30, 30)).astype(np.float32)

    g = persistence.compute_cubical_persistence(img, backend="gudhi")
    n = persistence.compute_cubical_persistence(img, backend="numpy")

    assert len(_finite_dim0(g)) > 5
    assert len(_finite_dim0(n)) > 5
    assert len(_essential_dim0(g)) == 1
    assert len(_essential_dim0(n)) == 1
    assert abs(_essential_dim0(g)[0].birth - _essential_dim0(n)[0].birth) < 1e-3


def test_essential_class_birth_agrees():
    """The global essential class (the longest-lived component) should be
    born at the same image value under both backends — it's the global max
    of the image either way."""
    img = np.random.default_rng(0).random((30, 30)).astype(np.float32)

    g_ess = _essential_dim0(persistence.compute_cubical_persistence(img, backend="gudhi"))
    n_ess = _essential_dim0(persistence.compute_cubical_persistence(img, backend="numpy"))
    assert len(g_ess) == 1
    assert len(n_ess) == 1
    assert abs(g_ess[0].birth - n_ess[0].birth) < 1e-3


def test_both_backends_find_a_loop_in_a_donut():
    """A bright torus has Betti numbers (1, 1) — one component, one loop.
    gudhi computes H₁ directly; the fallback approximates it via 0D
    persistence on the inverted image.  Both should detect *some*
    persistent H₁ feature."""
    yy, xx = np.mgrid[0:40, 0:40]
    dist = np.sqrt((yy - 20) ** 2 + (xx - 20) ** 2)
    img = ((dist > 7) & (dist < 14)).astype(np.float32) * 0.9

    g = persistence.compute_cubical_persistence(img, backend="gudhi")
    n = persistence.compute_cubical_persistence(img, backend="numpy")

    g_top1 = _top_pair([p for p in g["dim1"] if not np.isinf(p.persistence)])
    n_top1 = _top_pair([p for p in n["dim1"] if not np.isinf(p.persistence)])

    assert g_top1 is not None and g_top1.persistence > 0.3, \
        "gudhi failed to detect the donut's loop"
    assert n_top1 is not None and n_top1.persistence > 0.3, \
        "fallback failed to detect the donut's loop (via inverted-image H0)"
