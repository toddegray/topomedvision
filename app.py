"""TopoMedVision — Streamlit demo.

Run from the project root::

    streamlit run app.py

The app is structured as five tabs:

1. **Original**     — what the topology layer sees after preprocessing.
2. **Persistence**  — barcodes, diagrams, lifetime histogram (the "shape fingerprint").
3. **Highlight**    — original image + topological mask overlay.
4. **Prediction**   — tumor likelihood score with a feature-driven explanation.
5. **Baseline**     — side-by-side with a classical Otsu threshold mask.

Implementation notes
--------------------
- The expensive step is :func:`backend.persistence.compute_cubical_persistence`.
  We cache it with ``@st.cache_data`` keyed on the image bytes hash so flipping
  between tabs doesn't recompute.
- All UI strings live here; backend modules return raw data so they're
  reusable from notebooks or scripts without Streamlit installed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from backend import hybrid_model, persistence, utils, visualization

REPO_ROOT = Path(__file__).parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MODEL_PATH = REPO_ROOT / "models" / "topo_classifier.joblib"

DISCLAIMER = (
    "**Disclaimer.** TopoMedVision is an educational research prototype. "
    "It is **not** FDA-approved and **not** intended for clinical "
    "decision-making. Shipped sample images are public-domain MRI slices "
    "(AFIP / Wikimedia Commons) used for demonstration only — do not "
    "interpret any output as medical advice. See `data/samples/labels.json` "
    "for per-image attribution."
)


# ---------------------------------------------------------------------------
# Page config & global styles.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TopoMedVision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached compute paths.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _persistence_cached(image_bytes: bytes, denoise: bool, equalize: bool) -> dict:
    """Run preprocessing + persistence on raw image bytes.

    Caching key is the bytes themselves, which hashes cheaply and makes the
    cache safe across re-runs and reloads of the same upload.
    """
    img = Image.open(io.BytesIO(image_bytes))
    arr = utils.from_pil(img)
    arr = utils.preprocess(arr, denoise=denoise, equalize=equalize)
    result = persistence.compute_cubical_persistence(arr, persistence_threshold=0.0)
    result["image"] = arr
    return result


def _load_image_bytes(uploaded, sample_path: Path | None) -> bytes | None:
    if uploaded is not None:
        return uploaded.getvalue()
    if sample_path is not None and sample_path.exists():
        return sample_path.read_bytes()
    return None


# ---------------------------------------------------------------------------
# Sidebar — input controls.
# ---------------------------------------------------------------------------


def _sidebar() -> tuple[bytes | None, dict]:
    st.sidebar.title("🧠 TopoMedVision")
    st.sidebar.caption("Topological deep learning for MRI shape analysis")

    st.sidebar.markdown("### 1 · Choose an image")
    samples = utils.list_sample_images(SAMPLES_DIR)
    sample_options = ["— upload your own —"] + [s.name for s in samples]

    choice = st.sidebar.selectbox(
        "Sample slice",
        sample_options,
        index=min(3, len(sample_options) - 1) if len(sample_options) > 1 else 0,
        help="Public-domain MRI slices ship with the repo (AFIP via "
        "Wikimedia Commons). See `data/samples/labels.json` for sources.",
    )
    sample_path = None
    uploaded = None
    if choice == "— upload your own —":
        uploaded = st.sidebar.file_uploader(
            "Upload a 2D grayscale slice (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )
    else:
        sample_path = SAMPLES_DIR / choice

    image_bytes = _load_image_bytes(uploaded, sample_path)

    st.sidebar.markdown("### 2 · Preprocessing")
    denoise = st.sidebar.checkbox(
        "Gaussian denoise (σ=1)",
        value=True,
        help="Suppresses single-pixel speckle that would create many "
        "short-lived persistence pairs.",
    )
    equalize = st.sidebar.checkbox(
        "Local contrast equalization (CLAHE)",
        value=False,
        help="Off by default — CLAHE rescales local contrast, which can "
        "wash out the global 'tumor is brighter' signal. Turn on only if "
        "comparing slices acquired with very different scanner gain.",
    )

    st.sidebar.markdown("### 3 · Highlighting")
    persistence_threshold = st.sidebar.slider(
        "Min persistence to plot", 0.00, 0.50, 0.05, 0.01,
        help="Bars below this lifetime are dropped before plotting / scoring.",
    )
    top_k = st.sidebar.slider(
        "Top-k features to highlight", 1, 12, 5,
        help="Budget is split across H₀ (bright components) and H₁ (loops) so "
        "loop features stay visible on non-skull-stripped images where H₀ "
        "tends to be dominated by skull/scalp brightness.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(DISCLAIMER)
    st.sidebar.markdown(
        "[GitHub](https://github.com/) · "
        "[Math primer](#math-background)"
    )

    return image_bytes, dict(
        denoise=denoise,
        equalize=equalize,
        persistence_threshold=persistence_threshold,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Tab renderers.
# ---------------------------------------------------------------------------


def _tab_original(arr: np.ndarray, raw_bytes: bytes) -> None:
    st.subheader("Preprocessed input")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**As uploaded**")
        st.image(Image.open(io.BytesIO(raw_bytes)), use_container_width=True)
    with col2:
        st.markdown("**After preprocessing** (denoise + CLAHE, resized to 192×192)")
        st.image(arr, clamp=True, use_container_width=True)
    st.caption(
        "The topology layer operates on the right-hand image. Sublevel-set "
        "persistent homology treats pixel intensity as a height function and "
        "tracks how connected components and loops appear, merge, and "
        "disappear as we sweep a threshold."
    )


def _tab_persistence(result: dict, options: dict) -> None:
    st.subheader("Persistence diagrams & barcode")

    dim0 = [p for p in result["dim0"] if p.persistence >= options["persistence_threshold"]]
    dim1 = [p for p in result["dim1"] if p.persistence >= options["persistence_threshold"]]

    badge = "🟢 gudhi" if result["backend"] == "gudhi" else "🟡 numpy fallback"
    st.caption(
        f"Backend: {badge} · "
        f"H₀ pairs: **{len(dim0)}** · H₁ pairs: **{len(dim1)}** · "
        f"image: {result['image_shape'][0]}×{result['image_shape'][1]}"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Persistence diagram**")
        st.plotly_chart(
            visualization.interactive_diagram(dim0, dim1, max_value=1.0),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Barcode**")
        st.pyplot(visualization.persistence_barcode(dim0, dim1, max_value=1.0),
                  clear_figure=True)

    st.markdown("**Lifetime histogram**")
    st.pyplot(visualization.lifetime_histogram(dim0, dim1), clear_figure=True)

    with st.expander("How to read this", expanded=False):
        st.markdown(
            """
- Each point in the **diagram** is one feature: a connected component (H₀) or
  a loop (H₁). The further a point sits from the diagonal, the more
  *persistent* the feature — i.e., the more robust to small intensity
  perturbations.
- The **barcode** is the same data laid out as horizontal bars; long bars =
  meaningful shape, short bars = noise.
- Tumor masses tend to produce a few **long H₀ bars** (dense bright blobs).
  Ring-enhanced lesions produce **persistent H₁ bars** (the dark core
  surrounded by a bright rim is geometrically a loop).
            """
        )


def _tab_highlight(result: dict, options: dict) -> None:
    st.subheader("Topological highlighting")

    pairs = list(result["dim0"]) + list(result["dim1"])
    mask = persistence.topology_mask(
        result["image"], pairs,
        top_k=options["top_k"],
        min_persistence=max(0.05, options["persistence_threshold"]),
    )
    overlay = utils.overlay_mask(result["image"], mask)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top-k birth pixels**")
        st.pyplot(
            visualization.annotate_birth_points(
                result["image"], pairs, top_k=options["top_k"]
            ),
            clear_figure=True,
        )
    with col2:
        st.markdown("**Highlight mask overlay**")
        st.image(overlay, use_container_width=True)

    st.caption(
        "Red regions are flood-filled from the most persistent feature birth "
        "pixels. Tolerance scales with each feature's persistence — robust "
        "features get larger, more confident masks."
    )


def _tab_prediction(result: dict) -> None:
    st.subheader("Tumor likelihood score")

    classifier = hybrid_model.load_classifier(MODEL_PATH)
    score_result = hybrid_model.score_tumor_likelihood(
        result["dim0"], result["dim1"], image=result["image"], classifier=classifier
    )

    col_score, col_meter = st.columns([1, 2])
    with col_score:
        st.metric(
            "Likelihood",
            f"{score_result.score * 100:.0f}%",
            help=(
                "The shipped classifier is a 6-sample stub demonstrating "
                "the load path; it adds no generalizable signal. Score = "
                "½·rule + ½·stub.predict_proba when loaded, else rule only. "
                f"Mode: {'rule + stub blend' if classifier else 'rule-based only'}"
            ),
        )
        if score_result.score > 0.7:
            st.error("Elevated topological signature.")
        elif score_result.score > 0.4:
            st.warning("Inconclusive.")
        else:
            st.success("Low likelihood.")
    with col_meter:
        st.progress(score_result.score)
        st.markdown(score_result.explanation)

    with st.expander("Feature vector & contributions", expanded=False):
        feat_table = sorted(
            (
                {
                    "feature": name,
                    "value": round(score_result.features.get(name, 0.0), 4),
                    "weighted contribution": round(
                        score_result.contributions.get(name, 0.0), 4
                    ),
                }
                for name in hybrid_model.FEATURE_NAMES
            ),
            key=lambda r: abs(r["weighted contribution"]),
            reverse=True,
        )
        st.dataframe(feat_table, use_container_width=True)


def _tab_baseline(result: dict, options: dict) -> None:
    st.subheader("Baseline: Otsu threshold vs. topological highlight")

    img = result["image"]
    otsu = utils.otsu_mask(img)
    pairs = list(result["dim0"]) + list(result["dim1"])
    topo = persistence.topology_mask(
        img, pairs,
        top_k=options["top_k"],
        min_persistence=max(0.05, options["persistence_threshold"]),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original**")
        st.image(img, clamp=True, use_container_width=True)
    with col2:
        st.markdown("**Otsu threshold (intensity only)**")
        st.image(utils.overlay_mask(img, otsu, color=(40, 130, 220)),
                 use_container_width=True)
    with col3:
        st.markdown("**Topological highlight**")
        st.image(utils.overlay_mask(img, topo), use_container_width=True)

    st.caption(
        "Otsu picks every bright pixel — including skull / bone — because it "
        "has no notion of *spatial structure*. The topological mask uses "
        "persistent connected components, so it preferentially keeps "
        "compact, robust bright masses. Both views together let you see "
        "what each method is and isn't picking up."
    )


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    image_bytes, options = _sidebar()

    st.title("TopoMedVision")
    st.markdown(
        "Topological deep learning prototype for highlighting tumor-like "
        "regions in 2D brain MRI slices. Upload a slice or pick a sample "
        "to see its persistence barcode, topology-derived mask, a "
        "lightweight likelihood score, and a side-by-side baseline."
    )

    if image_bytes is None:
        st.info("👈 Pick a sample or upload an image to begin.")
        st.markdown("---")
        st.markdown("### Math background <a id='math-background'></a>",
                    unsafe_allow_html=True)
        st.markdown(
            r"""
A standard neural unit is a weighted sum followed by a nonlinearity:
$y = \sigma(\sum_i w_i x_i + b)$. It learns features from raw pixel
intensities. **Topological deep learning** augments that picture with
features derived from the *shape* of the data — invariants that survive
small deformations and noise.

For a grayscale image, treat pixel intensity as a height function $f$.
The **sublevel-set filtration** is the family of binarizations
$X_t = \{p : f(p) \le t\}$ for $t$ sweeping from $-\infty$ to $+\infty$.
As $t$ grows, $X_t$ gains connected components (H₀) and loops (H₁).
**Persistent homology** records the $(\text{birth}, \text{death})$ value
of each topological feature — the persistence of a feature is its
death minus birth, i.e. how robust that shape is to threshold
perturbations.

For 2D images we use a **cubical complex** (each pixel is a 2-cell), which
makes computation linear in the number of pixels using the
``cubicalripser`` / ``gudhi`` algorithms.
            """
        )
        return

    with st.spinner("Computing cubical persistent homology…"):
        result = _persistence_cached(image_bytes, options["denoise"], options["equalize"])

    tabs = st.tabs([
        "Original",
        "Persistence",
        "Highlight",
        "Prediction",
        "Baseline",
    ])
    with tabs[0]:
        _tab_original(result["image"], image_bytes)
    with tabs[1]:
        _tab_persistence(result, options)
    with tabs[2]:
        _tab_highlight(result, options)
    with tabs[3]:
        _tab_prediction(result)
    with tabs[4]:
        _tab_baseline(result, options)

    st.markdown("---")
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
