"""Visualizations for the Streamlit UI.

We use matplotlib for static plots (persistence diagram, barcode) and Plotly
for the interactive scatter so users can hover individual points.  Returning
figure objects (not images) lets Streamlit render them with its native
matplotlib / plotly support and keeps interactivity intact.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .persistence import PersistencePair


# Colours chosen to be colourblind-friendly and to read on a dark Streamlit
# background as well as a light notebook one.
_DIM_COLORS = {0: "#1f77b4", 1: "#d62728"}
_DIM_LABELS = {0: "H₀ (components)", 1: "H₁ (loops)"}


def persistence_diagram(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    *,
    max_value: float = 1.0,
):
    """Static matplotlib persistence diagram.

    Points above the diagonal are real features; the further from the
    diagonal, the more persistent (and thus "shape-meaningful") the feature.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Diagonal — features die instantly here.
    diag = np.linspace(0, max_value, 50)
    ax.plot(diag, diag, color="gray", linestyle="--", linewidth=1)

    for dim, pairs in ((0, pairs_dim0), (1, pairs_dim1)):
        if not pairs:
            continue
        xs = [p.birth for p in pairs]
        # Clip infinite deaths to the top of the plot.
        ys = [min(p.death, max_value * 1.05) if not np.isinf(p.death) else max_value * 1.05
              for p in pairs]
        ax.scatter(xs, ys, s=22, alpha=0.7, color=_DIM_COLORS[dim], label=_DIM_LABELS[dim])

    ax.set_xlim(0, max_value * 1.05)
    ax.set_ylim(0, max_value * 1.1)
    ax.set_xlabel("Birth (filtration value)")
    ax.set_ylabel("Death")
    ax.set_title("Persistence diagram")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def persistence_barcode(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    *,
    max_value: float = 1.0,
    max_bars: int = 40,
):
    """Static matplotlib barcode.

    Each horizontal segment is one feature.  Long bars = robust shape.
    We cap the number of bars displayed because filtration noise can produce
    hundreds of near-zero-length bars that visually drown out the signal.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Sort by persistence descending; show the most prominent bars first so
    # the eye lands on the meaningful ones.
    grouped: list[tuple[int, PersistencePair]] = []
    for p in sorted(pairs_dim0, key=lambda q: q.persistence, reverse=True):
        grouped.append((0, p))
    for p in sorted(pairs_dim1, key=lambda q: q.persistence, reverse=True):
        grouped.append((1, p))
    grouped = grouped[:max_bars]

    for i, (dim, p) in enumerate(grouped):
        b = p.birth
        d = min(p.death, max_value * 1.05) if not np.isinf(p.death) else max_value * 1.05
        ax.hlines(i, b, d, color=_DIM_COLORS[dim], linewidth=2)

    ax.set_xlim(0, max_value * 1.1)
    ax.set_yticks([])
    ax.set_xlabel("Filtration value")
    ax.set_title(f"Persistence barcode (top {len(grouped)})")
    # Custom legend (hlines doesn't auto-populate one).
    handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=_DIM_LABELS[d])
               for d, c in _DIM_COLORS.items()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


def interactive_diagram(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    *,
    max_value: float = 1.0,
) -> go.Figure:
    """Plotly interactive persistence diagram with per-point hover."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for dim, pairs in ((0, pairs_dim0), (1, pairs_dim1)):
        if not pairs:
            continue
        xs = [p.birth for p in pairs]
        ys = [min(p.death, max_value * 1.05) if not np.isinf(p.death) else max_value * 1.05
              for p in pairs]
        text = [
            f"dim={p.dim}<br>birth={p.birth:.3f}<br>death={p.death:.3f}<br>"
            f"persistence={p.persistence:.3f}<br>"
            + (f"birth_xy=({p.birth_xy[0]}, {p.birth_xy[1]})" if p.birth_xy else "")
            for p in pairs
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=8, color=_DIM_COLORS[dim], opacity=0.75),
                name=_DIM_LABELS[dim],
                text=text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Persistence diagram (interactive)",
        xaxis_title="Birth",
        yaxis_title="Death",
        xaxis=dict(range=[0, max_value * 1.05]),
        yaxis=dict(range=[0, max_value * 1.1]),
        height=460,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def lifetime_histogram(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    *,
    bins: int = 25,
):
    """Histogram of persistence lifetimes — quick "is there signal?" gut check."""
    fig, ax = plt.subplots(figsize=(6, 3))
    finite0 = [p.persistence for p in pairs_dim0 if not np.isinf(p.persistence)]
    finite1 = [p.persistence for p in pairs_dim1 if not np.isinf(p.persistence)]

    if finite0:
        ax.hist(finite0, bins=bins, alpha=0.6, color=_DIM_COLORS[0], label=_DIM_LABELS[0])
    if finite1:
        ax.hist(finite1, bins=bins, alpha=0.6, color=_DIM_COLORS[1], label=_DIM_LABELS[1])

    ax.set_xlabel("Persistence (lifetime)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of feature lifetimes")
    if finite0 or finite1:
        ax.legend()
    fig.tight_layout()
    return fig


def annotate_birth_points(
    image: np.ndarray,
    pairs: Iterable[PersistencePair],
    *,
    top_k: int = 5,
    max_value: float = 1.0,
):
    """Show the image with the top-k highest-persistence birth pixels marked.

    H₀ markers (bright peaks) are blue, H₁ markers (loops) are red — same
    colour convention as the persistence diagram, so a viewer can match a
    spatial annotation to the corresponding bar/point.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=max_value)
    ax.axis("off")

    finite = [p for p in pairs if p.birth_xy is not None and not np.isinf(p.persistence)]
    finite.sort(key=lambda p: p.persistence, reverse=True)

    for rank, p in enumerate(finite[:top_k]):
        x, y = p.birth_xy
        color = _DIM_COLORS.get(p.dim, "#ff3333")
        ax.scatter([x], [y], s=140, facecolors="none", edgecolors=color, linewidths=2.2)
        label = f"#{rank+1} H{p.dim}\nπ={p.persistence:.2f}"
        ax.text(x + 5, y + 5, label, color=color, fontsize=9, fontweight="bold")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markeredgecolor=_DIM_COLORS[0],
                   markerfacecolor="none", markeredgewidth=2, markersize=10, label=_DIM_LABELS[0]),
        plt.Line2D([0], [0], marker="o", color="w", markeredgecolor=_DIM_COLORS[1],
                   markerfacecolor="none", markeredgewidth=2, markersize=10, label=_DIM_LABELS[1]),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title(f"Top-{top_k} persistent feature birth pixels")
    fig.tight_layout()
    return fig
