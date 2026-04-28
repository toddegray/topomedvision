"""Generate static PNG screenshots for the README.

These are NOT screenshots of the running Streamlit app — capturing those
requires a headless browser (playwright/puppeteer) and is out of scope for
the demo's hard dependencies.  Instead, this script produces composite
matplotlib figures showing the *content* of each Streamlit tab.  They
render to ``assets/`` and are referenced from the README.

For the prediction tab we render a custom gauge / table view because the
Streamlit ``st.metric`` widget has no matplotlib equivalent — we build it
from primitives so the visual style stays consistent with the other
figures.

Run::

    python scripts/generate_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import hybrid_model, persistence, utils, visualization  # noqa: E402

ASSETS_DIR = REPO_ROOT / "assets"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MODEL_PATH = REPO_ROOT / "models" / "topo_classifier.joblib"

# Sample chosen for screenshots: ring-enhanced tumor — interesting in every
# tab (multiple persistent H₀, a clear H₁ loop, distinct from healthy).
DEMO_SAMPLE = "05_tumor_ring_axial.png"
HEALTHY_SAMPLE = "01_healthy_axial_t1.png"

DPI = 140


def _load_and_preprocess(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(Image.open(SAMPLES_DIR / name).convert("L").resize((192, 192)))
    raw = raw.astype(np.float32) / 255.0
    arr = utils.preprocess(raw, denoise=True, equalize=False)
    result = persistence.compute_cubical_persistence(arr, persistence_threshold=0.05)
    result["image"] = arr
    return raw, arr, result


# ---------------------------------------------------------------------------
# Tab 1: Original (raw vs preprocessed)
# ---------------------------------------------------------------------------


def _screenshot_original(raw: np.ndarray, arr: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("As uploaded")
    axes[0].axis("off")
    axes[1].imshow(arr, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("After preprocessing\n(denoise; CLAHE off)")
    axes[1].axis("off")
    fig.suptitle("Tab 1 · Original", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2: Persistence (diagram + barcode + histogram)
# ---------------------------------------------------------------------------


def _screenshot_persistence(result: dict, out: Path) -> None:
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    # Diagram (top-left)
    ax_diag = fig.add_subplot(gs[0, 0])
    diag = np.linspace(0, 1, 50)
    ax_diag.plot(diag, diag, "--", color="gray", linewidth=1)
    for dim, color, label in ((0, "#1f77b4", "H₀"), (1, "#d62728", "H₁")):
        pairs = result[f"dim{dim}"]
        if pairs:
            xs = [p.birth for p in pairs]
            ys = [min(p.death, 1.05) if not np.isinf(p.death) else 1.05 for p in pairs]
            ax_diag.scatter(xs, ys, s=24, alpha=0.75, color=color, label=label)
    ax_diag.set_xlim(0, 1.05)
    ax_diag.set_ylim(0, 1.1)
    ax_diag.set_xlabel("Birth")
    ax_diag.set_ylabel("Death")
    ax_diag.set_title("Persistence diagram")
    ax_diag.grid(alpha=0.3)
    ax_diag.legend(loc="lower right")

    # Barcode (top-right)
    ax_bar = fig.add_subplot(gs[0, 1])
    grouped = []
    for p in sorted(result["dim0"], key=lambda q: q.persistence, reverse=True):
        grouped.append((0, p))
    for p in sorted(result["dim1"], key=lambda q: q.persistence, reverse=True):
        grouped.append((1, p))
    grouped = grouped[:30]
    for i, (dim, p) in enumerate(grouped):
        b = p.birth
        d = min(p.death, 1.05) if not np.isinf(p.death) else 1.05
        color = "#1f77b4" if dim == 0 else "#d62728"
        ax_bar.hlines(i, b, d, color=color, linewidth=2)
    ax_bar.set_xlim(0, 1.1)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Filtration value")
    ax_bar.set_title(f"Barcode (top {len(grouped)})")

    # Lifetime histogram (bottom, full-width)
    ax_hist = fig.add_subplot(gs[1, :])
    finite0 = [p.persistence for p in result["dim0"] if not np.isinf(p.persistence)]
    finite1 = [p.persistence for p in result["dim1"] if not np.isinf(p.persistence)]
    if finite0:
        ax_hist.hist(finite0, bins=25, alpha=0.6, color="#1f77b4", label="H₀")
    if finite1:
        ax_hist.hist(finite1, bins=25, alpha=0.6, color="#d62728", label="H₁")
    ax_hist.set_xlabel("Persistence (lifetime)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of feature lifetimes")
    if finite0 or finite1:
        ax_hist.legend()

    fig.suptitle("Tab 2 · Persistence", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 3: Highlight (birth points + overlay)
# ---------------------------------------------------------------------------


def _screenshot_highlight(arr: np.ndarray, result: dict, out: Path) -> None:
    pairs = list(result["dim0"]) + list(result["dim1"])
    mask = persistence.topology_mask(arr, pairs, top_k=5, min_persistence=0.20)
    overlay = utils.overlay_mask(arr, mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(arr, cmap="gray", vmin=0, vmax=1)
    axes[0].axis("off")
    axes[0].set_title("Top-5 birth pixels (H₀ blue, H₁ red)")
    finite = sorted(
        [p for p in pairs if p.birth_xy is not None and not np.isinf(p.persistence)],
        key=lambda p: p.persistence, reverse=True,
    )
    dim_colors = {0: "#1f77b4", 1: "#d62728"}
    for rank, p in enumerate(finite[:5]):
        x, y = p.birth_xy
        c = dim_colors.get(p.dim, "#ff3333")
        axes[0].scatter([x], [y], s=160, facecolors="none", edgecolors=c, linewidths=2.4)
        axes[0].text(x + 5, y + 5, f"#{rank+1} H{p.dim}", color=c, fontsize=10, fontweight="bold")

    axes[1].imshow(overlay)
    axes[1].axis("off")
    axes[1].set_title(f"Topology mask ({100*mask.mean():.1f}% coverage)")

    fig.suptitle("Tab 3 · Highlight", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 4: Prediction (score gauge + top driver bars)
# ---------------------------------------------------------------------------


def _screenshot_prediction(result: dict, arr: np.ndarray, out: Path) -> None:
    classifier = hybrid_model.load_classifier(MODEL_PATH)
    score = hybrid_model.score_tumor_likelihood(
        result["dim0"], result["dim1"], image=arr, classifier=classifier
    )

    fig, (ax_gauge, ax_bars) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]}
    )

    # Gauge (semicircular bar)
    theta = np.linspace(np.pi, 0, 100)
    ax_gauge.plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=14, solid_capstyle="round")
    color = "#d62728" if score.score > 0.7 else "#ff9800" if score.score > 0.4 else "#2ca02c"
    n = max(2, int(100 * score.score))
    ax_gauge.plot(np.cos(theta[:n]), np.sin(theta[:n]), color=color, linewidth=14, solid_capstyle="round")
    ax_gauge.text(0, -0.18, f"{score.score*100:.0f}%", ha="center", va="center",
                  fontsize=28, fontweight="bold", color=color)
    ax_gauge.text(0, -0.45, "tumor likelihood", ha="center", va="center", fontsize=10, color="#666")
    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-0.6, 1.2)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")

    # Top contributions bars
    contribs = sorted(score.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
    names = [k for k, _ in contribs][::-1]
    vals = [v for _, v in contribs][::-1]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    ax_bars.barh(names, vals, color=colors, alpha=0.75)
    ax_bars.axvline(0, color="black", linewidth=0.6)
    ax_bars.set_xlabel("Weighted contribution to score")
    ax_bars.set_title("Top feature drivers")
    ax_bars.grid(axis="x", alpha=0.3)

    fig.suptitle("Tab 4 · Prediction", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 5: Baseline (original | otsu | topology)
# ---------------------------------------------------------------------------


def _screenshot_baseline(arr: np.ndarray, result: dict, out: Path) -> None:
    pairs = list(result["dim0"]) + list(result["dim1"])
    topo_mask = persistence.topology_mask(arr, pairs, top_k=5, min_persistence=0.20)
    otsu = utils.otsu_mask(arr)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(utils.overlay_mask(arr, otsu, color=(40, 130, 220)))
    axes[1].set_title(f"Otsu threshold\n({100*otsu.mean():.0f}% coverage)")
    axes[1].axis("off")

    axes[2].imshow(utils.overlay_mask(arr, topo_mask))
    axes[2].set_title(f"Topology highlight\n({100*topo_mask.mean():.1f}% coverage)")
    axes[2].axis("off")

    fig.suptitle("Tab 5 · Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Healthy vs tumor side-by-side (hero shot for the README)
# ---------------------------------------------------------------------------


def _screenshot_hero(out: Path) -> None:
    """Hero figure: healthy vs tumor side-by-side, original + barcode."""
    raw_h, arr_h, res_h = _load_and_preprocess(HEALTHY_SAMPLE)
    raw_t, arr_t, res_t = _load_and_preprocess(DEMO_SAMPLE)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    for col, (label, arr, res) in enumerate(
        [("Healthy", arr_h, res_h), ("Tumor (ring-enhanced)", arr_t, res_t)]
    ):
        axes[0, col].imshow(arr, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontweight="bold")
        axes[0, col].axis("off")

        ax = axes[1, col]
        grouped = []
        for p in sorted(res["dim0"], key=lambda q: q.persistence, reverse=True)[:15]:
            grouped.append((0, p))
        for p in sorted(res["dim1"], key=lambda q: q.persistence, reverse=True)[:15]:
            grouped.append((1, p))
        for i, (dim, p) in enumerate(grouped):
            b = p.birth
            d = min(p.death, 1.05) if not np.isinf(p.death) else 1.05
            color = "#1f77b4" if dim == 0 else "#d62728"
            ax.hlines(i, b, d, color=color, linewidth=2)
        ax.set_xlim(0, 1.1)
        ax.set_yticks([])
        ax.set_xlabel("Filtration value")
        ax.set_title("Persistence barcode")

    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="H₀ (components)"),
        plt.Line2D([0], [0], color="#d62728", linewidth=2, label="H₁ (loops)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "TopoMedVision — topological fingerprint of MRI slices",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    raw, arr, result = _load_and_preprocess(DEMO_SAMPLE)

    print(f"Generating screenshots from {DEMO_SAMPLE}…")
    _screenshot_hero(ASSETS_DIR / "hero.png")
    print(f"  → {ASSETS_DIR / 'hero.png'}")
    _screenshot_original(raw, arr, ASSETS_DIR / "screenshot_original.png")
    print(f"  → {ASSETS_DIR / 'screenshot_original.png'}")
    _screenshot_persistence(result, ASSETS_DIR / "screenshot_persistence.png")
    print(f"  → {ASSETS_DIR / 'screenshot_persistence.png'}")
    _screenshot_highlight(arr, result, ASSETS_DIR / "screenshot_highlight.png")
    print(f"  → {ASSETS_DIR / 'screenshot_highlight.png'}")
    _screenshot_prediction(result, arr, ASSETS_DIR / "screenshot_prediction.png")
    print(f"  → {ASSETS_DIR / 'screenshot_prediction.png'}")
    _screenshot_baseline(arr, result, ASSETS_DIR / "screenshot_baseline.png")
    print(f"  → {ASSETS_DIR / 'screenshot_baseline.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
