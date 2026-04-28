---
title: TopoMedVision
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.56.0"
app_file: app.py
pinned: false
license: mit
short_description: Cubical persistent homology for MRI tumor highlighting
---

# TopoMedVision 🧠

> **Topological deep learning for brain tumor highlighting in MRI.**
> A research prototype that pairs *cubical persistent homology* with a
> lightweight classifier to surface candidate tumor regions in 2D MRI
> slices — and explains *why* it flagged them.

> ⚠️ **Educational prototype only.** Not FDA-approved. Not for clinical use.
> Sample images shipped with the repo are real public-domain MRI slices
> (AFIP teaching cases via Wikimedia Commons) used for demonstration only —
> see [Image attribution](#image-attribution) below.

![Healthy vs tumor topological fingerprint](./assets/hero.png)

---

## Why topological deep learning?

A standard neural unit is a weighted sum followed by a nonlinearity —
[Karpathy's neuron](https://cs231n.github.io/neural-networks-1/). It learns
features from raw pixel intensities. **Topological deep learning (TDL)**
augments that picture with features derived from the *shape* of the data —
invariants that survive small deformations, noise, and acquisition
differences. That makes it especially attractive for medical imaging, where
robustness to scanner-to-scanner variation matters more than squeezing the
last 0.5 % of accuracy on a clean benchmark.

This project demonstrates the bridge:

1. Treat each MRI slice's pixel intensity as a height function.
2. Compute its **cubical persistent homology** — a multiscale fingerprint of
   connected components and loops.
3. Feed that fingerprint into a small classifier *and* use it directly to
   build a spatial highlight mask.
4. Show side-by-side how this differs from a classical Otsu intensity
   threshold.

## What the demo shows

The Streamlit app loads a 2D MRI slice and walks through the topology
pipeline across five tabs. The figures below are produced by
`scripts/generate_screenshots.py` from the same code paths the app runs,
using the ring-enhanced glioblastoma sample (`05_tumor_ring_axial.png`) as
the running example.

### 1 · Original — what the topology layer actually sees

![Original tab — raw vs preprocessed](./assets/screenshot_original.png)

The raw upload on the left, the preprocessed slice on the right. Light
Gaussian denoising suppresses single-pixel speckle that would otherwise
inflate the diagram with hundreds of short-lived bars. CLAHE is off by
default — it can wash out the global "tumor is brighter" signal that
makes the persistence story work.

### 2 · Persistence — the topological fingerprint

![Persistence tab — diagram, barcode, lifetime histogram](./assets/screenshot_persistence.png)

The **diagram** plots every (birth, death) pair; points far from the
diagonal are robust features. The **barcode** ranks them by lifetime —
each long bar is a real shape feature, each short one is noise. The
**histogram** at the bottom shows the lifetime distribution: healthy
slices concentrate near zero, tumor slices grow a heavy tail. H₀ (blue)
tracks connected components (bright blobs); H₁ (red) tracks loops
(bright rings around dark cores).

### 3 · Highlight — turning persistence into a spatial mask

![Highlight tab — top-k birth pixels and overlay mask](./assets/screenshot_highlight.png)

The top-`k` most persistent features have their **birth pixels** marked
on the slice (left), then flood-filled into an **overlay mask** (right).
This is what makes persistence interpretable as a *highlight*: every
red/blue circle traces back to a specific bar in the barcode you can
click in the app.

### 4 · Prediction — the score and the reason for it

![Prediction tab — tumor likelihood gauge and feature drivers](./assets/screenshot_prediction.png)

The 14-dimensional persistence feature vector is fed into a Random
Forest (or, if no model is loaded, a hand-tuned rule-based scorer). The
**gauge** shows the calibrated tumor likelihood (95 % on this slice);
the **bar chart** breaks out which features pushed the score up or down,
and by how much. The point of the persistence pipeline is that those
feature names map back to *concrete shape facts about the image* — so a
"95 %" verdict comes with a structural reason, not a black-box vibe.
For this ring-enhanced glioblastoma the top drivers read as:

- **`max_persistence_dim0` (+3.4)** — the longest-lived H₀ bar. Plain
  English: there's a bright connected component (the lesion) that
  survives across a wide threshold range. Healthy brains rarely have
  any bright blob with persistence this high.
- **`n_long_dim0_30` (+2.0)** — the count of H₀ bars with lifetime
  ≥ 0.30. Plain English: more than one bright component remains after
  aggressive threshold filtering — typical of a focal mass plus
  surrounding enhancement, atypical of normal parenchyma.
- **`n_long_dim1_15` (+1.2)** and **`max_persistence_dim1` (+1.0)** —
  the count and lifetime of long H₁ (loop) bars. Plain English: the
  ring lesion's bright rim around a darker core registers as a
  persistent loop in the diagram. This is the single feature that most
  cleanly separates ring-enhanced lesions from solid masses and from
  healthy slices.
- **`image_max_intensity` (+1.0)** — the brightest pixel value after
  preprocessing. Contrast-enhanced tumors saturate higher than
  parenchyma; this is the one non-topological hint the model gets.
- **`n_long_dim0_15` (–0.85)** — note the *negative* contribution.
  Plain English: at a looser threshold (0.15) almost any slice produces
  many medium-lived H₀ bars (gyri, sulci, ventricle edges). The model
  has learned that "lots of medium-persistence components" is *not*
  diagnostic on its own, so it actively discounts that signal in favor
  of the high-threshold counts above.

Two things are worth noticing. First, every driver is a number you
could have computed by hand from the persistence diagram in tab 2 — the
explanation isn't post-hoc storytelling, it's the same scalars the
model voted on. Second, the model's signs *agree with the math*: the
features the stability theorem says should be robust (long bars,
persistent loops) are the ones with positive weight; the features that
co-vary with image noise (medium bars at low thresholds) get penalized.

### 5 · Baseline — vs. classical intensity thresholding

![Baseline tab — Otsu vs. topology highlight](./assets/screenshot_baseline.png)

Side-by-side comparison with **Otsu thresholding**, the classical
intensity-only baseline. Otsu fires on every bright pixel — including
skull, scalp, and the contrast-enhanced lesion lumped together
(~19 % coverage on the example). The **topology highlight** uses the
*shape* of the intensity surface and concentrates on the few features
with high persistence (~4 % coverage), focusing on the lesion's bright
rim and a couple of structural landmarks instead of the whole bright
foreground.

## Run it locally

```bash
git clone <your-fork>
cd topomedvision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train_classifier.py         # trains the demo Random Forest
python scripts/generate_screenshots.py     # optional: refreshes assets/

streamlit run app.py                       # then open http://localhost:8501
```

Pick a sample from the sidebar (e.g. `05_tumor_ring_axial.png`) and
click through the five tabs above.

> The six sample MRIs in `data/samples/` are real public-domain images
> shipped with the repo — no generation step is needed. The setup scripts
> are one-shot: rerun them only when you want to retrain the classifier or
> regenerate the README screenshots. The Streamlit app picks up
> `models/topo_classifier.joblib` automatically when present, and falls
> back to the rule-based scorer if it's missing.

## Tests

```bash
pip install pytest
pytest tests/                              # 23 tests across both backends
```

The suite verifies the persistence math (single-peak, two-peak, four-peak,
threshold filtering, sort order, mask shape) on **both** the gudhi backend
and the pure-NumPy fallback, and includes a cross-backend agreement check
for clean inputs.

## Deploying to Hugging Face Spaces

The repo is HF-Spaces-ready: the YAML frontmatter at the top of this
README declares it as a Streamlit Space, and `models/topo_classifier.joblib`
+ `data/samples/*.png` are tracked so the app starts immediately on first
load — no build step.

```bash
# One-time: create a new Space at https://huggingface.co/new-space
# Choose SDK = "Streamlit", note its git URL, then:

git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

After the push, HF Spaces installs `requirements.txt` and runs `app.py`
automatically. The first start takes ~60s while gudhi compiles into the
container. The Space's public URL is shareable as your demo link.

## Architecture

```
                  ┌────────────────────┐
   image upload → │  utils.preprocess  │  → 192×192 float [0,1]
                  └─────────┬──────────┘
                            │
                  ┌─────────▼─────────────────────────┐
                  │  persistence.compute_cubical_…    │  gudhi ▸ numpy fallback
                  │  → dim0 / dim1 PersistencePairs   │
                  │     with spatial birth coords     │
                  └─────────┬─────────────────────────┘
                            │
              ┌─────────────┼─────────────────────────────┐
              │             │                             │
   ┌──────────▼─────┐ ┌─────▼─────────┐  ┌────────────────▼──────┐
   │ topology_mask  │ │ persistence_  │  │  hybrid_model.        │
   │ (flood-fill    │ │ features      │→ │  score_tumor_likelihood│
   │  overlay)      │ │ (14-D vector) │  │  (rule + optional RF) │
   └──────────┬─────┘ └───────────────┘  └────────────────┬──────┘
              │                                            │
              └─────────────  Streamlit UI  ───────────────┘
```

### Module layout

```
topomedvision/
├── app.py                   # Streamlit demo (5 tabs)
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── utils.py             # image I/O, preprocessing, overlay helpers
│   ├── persistence.py       # cubical PH (gudhi + union-find fallback)
│   ├── hybrid_model.py      # 14-D feature vector + rule/learned scorer
│   └── visualization.py     # matplotlib + plotly views
├── data/
│   ├── generate_samples.py  # synthesizes 6 MRI-like PNGs
│   ├── samples/             # populated by the script above
│   └── README_data.md       # how to plug in real BraTS slices
├── scripts/
│   └── train_classifier.py  # trains the demo Random Forest on samples
├── notebooks/
│   └── exploration.ipynb    # end-to-end walk-through
├── cpp_wrapper/
│   └── README.md            # CubicalRipser interop (stretch)
├── models/                  # pickled scikit-learn classifier (gitignored)
└── assets/                  # logos, screenshots, diagrams
```

## Math background

Treat a grayscale image $f \colon \Omega \to \mathbb{R}$ as a height function.
The **sublevel-set filtration** is the family of binarizations
$X_t = \{p \in \Omega : f(p) \le t\}$, indexed by $t \in \mathbb{R}$.
As $t$ grows from $-\infty$ to $+\infty$:

- **0-dimensional homology** ($H_0$) tracks connected components of $X_t$.
- **1-dimensional homology** ($H_1$) tracks loops (1-cycles that aren't
  boundaries of 2-cells in $X_t$).

Each topological feature has a *birth* value $b$ (the $t$ at which it
appears) and a *death* value $d$ (the $t$ at which it merges with an older
feature, or vanishes). The **persistence** of a feature is $|d - b|$ — its
robustness to threshold perturbations.

For 2D images we work on a **cubical complex** (each pixel is a 2-cell). The
algorithms in [CubicalRipser](https://github.com/shizuo-kaji/CubicalRipser_3dim)
and [GUDHI](https://gudhi.inria.fr/) compute the persistence diagram in
near-linear time on the number of pixels.

In a tumor-vs-healthy MRI slice you typically see:

- **Tumor masses** → a few **long $H_0$ bars** (compact bright blobs that
  stay connected over a wide threshold range).
- **Ring-enhanced lesions** (bright rim, dark necrotic core) → at least
  one **persistent $H_1$ bar**, because a dark interior surrounded by
  bright pixels is geometrically a loop.
- **Healthy parenchyma** → many short bars (texture noise) and very few
  long ones.

`backend/hybrid_model.py` turns these intuitions into a 14-dimensional
feature vector (Betti curve summaries + lifetime statistics + intensity
stats) and feeds it to either a hand-tuned weighted sum or a scikit-learn
Random Forest.

## How robustness works in practice

Pixel-level CNNs can get derailed by intensity scaling, scanner gain
differences, or small geometric warps. Persistence is **stable** in a strong
mathematical sense: if you perturb $f$ by at most $\varepsilon$ in the
sup-norm, the bottleneck distance between persistence diagrams is at most
$\varepsilon$ (Cohen-Steiner–Edelsbrunner–Harer 2007). The demo's
preprocessing tab lets you toggle denoising and contrast equalization to
see this in action — long bars stay long; only short, noisy bars shuffle.

## Backends

| Backend | Status | Notes |
|---|---|---|
| `gudhi` | ✅ default if installed | Used for both 0D and 1D persistence; recovers spatial coordinates via `cofaces_of_persistence_pairs`. |
| Pure NumPy union-find | ✅ always available | Implements 0D persistence via the elder rule on a 4-connected pixel filtration. Approximates 1D via the dual filtration. |
| `cripser` (CubicalRipser binding) | 🟡 stretch goal | See [`cpp_wrapper/README.md`](./cpp_wrapper/README.md) for the drop-in path. |
| `topomodelx` simplicial NN | 🟡 stretch goal | Add to `requirements.txt` and replace the Random Forest in `hybrid_model.py`. |

The active backend is reported under the **Persistence** tab so you always
know which code path produced the diagram.

## Limitations

- **2D only.** Real diagnostic radiology reads volumes; this is a single
  axial slice. Extending to 3D is a `gudhi.CubicalComplex` argument change
  but increases compute cost.
- **No clinical training data.** The shipped Random Forest is trained on
  six real MRI slices by `scripts/train_classifier.py` — far too few to
  generalize, so it trivially memorizes them. Drop a larger labelled
  corpus (e.g. BraTS slices) into `data/samples/`, update `labels.json`,
  and rerun the training script for a meaningful classifier. The
  rule-based scorer remains as a fallback when no model is loaded.
- **Tiny sample set.** Six images is enough to demonstrate the topology
  fingerprint qualitatively but nowhere near enough to make accuracy
  claims. Real BraTS-scale evaluation needs hundreds-to-thousands of
  slices and proper train / val / test splits.
- **Skull and scalp signal.** The bundled samples are *not*
  skull-stripped, so persistent H₀ components from bright skull/scalp
  pixels contribute to the topology mask. A real pipeline would prepend
  a skull-stripping step (e.g. HD-BET) before the persistence layer.
- **Compute.** A 192×192 slice takes ~0.5–1 s with `gudhi`, a few seconds
  with the pure-NumPy fallback. Larger slices grow super-linearly with
  pixel count.
- **No segmentation guarantee.** The flood-fill mask is a *highlight*, not
  a segmentation. It will under-cover diffuse lesions and miss any
  feature whose birth pixel sits outside the visible mass.

## Image attribution

All six bundled MRI slices in `data/samples/` are real, public-domain
images. Per-image source URLs and licenses live in
[`data/samples/labels.json`](./data/samples/labels.json); the summary:

| File | Subject | Source | License |
|---|---|---|---|
| `01_healthy_axial_t1.png` | Healthy brain, axial T1 | Wikimedia Commons — [`File:T1t2PD.jpg`](https://commons.wikimedia.org/wiki/File:T1t2PD.jpg) (left panel cropped), KieranMaher | Public domain |
| `02_healthy_sagittal_t1.png` | Healthy brain, sagittal T1 | Wikimedia Commons — [`File:MRI_brain.jpg`](https://commons.wikimedia.org/wiki/File:MRI_brain.jpg) | Public domain |
| `03_tumor_solid_axial.png` | Sub-ependymal giant cell astrocytoma, axial T1+C | Wikimedia Commons — [`File:MRI_of_brain_with_sub-ependymal_giant_cell_astrocytoma.jpg`](https://commons.wikimedia.org/wiki/File:MRI_of_brain_with_sub-ependymal_giant_cell_astrocytoma.jpg) (AFIP) | Public domain (17 U.S.C. § 105) |
| `04_tumor_solid_sagittal.png` | Pilocytic astrocytoma, hypothalamic, sagittal T1+C | Wikimedia Commons — [`File:405615R-PA-HYPOTHALAMIC.jpg`](https://commons.wikimedia.org/wiki/File:405615R-PA-HYPOTHALAMIC.jpg) (AFIP) | Public domain (17 U.S.C. § 105) |
| `05_tumor_ring_axial.png` | Glioblastoma multiforme (textbook ring enhancement), axial T1+C | Wikimedia Commons — [`File:AFIP-00405558-Glioblastoma-Radiology.jpg`](https://commons.wikimedia.org/wiki/File:AFIP-00405558-Glioblastoma-Radiology.jpg) | Public domain (17 U.S.C. § 105) |
| `06_tumor_ring_recurrent.png` | Recurrent multifocal glioblastoma, axial T1+C | Wikimedia Commons — [`File:AFIP-00405589-Glioblastoma-Radiology.jpg`](https://commons.wikimedia.org/wiki/File:AFIP-00405589-Glioblastoma-Radiology.jpg) | Public domain (17 U.S.C. § 105) |

All images were center-cropped to square and resampled to 192 × 192
grayscale. The originals (variable resolution and aspect ratio) are
linked above. The four AFIP images are works of the U.S. federal
government and are therefore in the public domain in the United States.

## Future work

- **Full 3D volumes** with proper sub-volume cropping around the brain.
- **TopoModelX integration**: build a simplicial complex from the image
  superlevel-set graph and run a learned simplicial attention layer on top.
- **Persistence images** as input to a small CNN for an end-to-end learned
  variant; compare to the rule-based scorer on a held-out set.
- **CubicalRipser C++ subprocess** for an order-of-magnitude speedup on
  larger slices; see [`cpp_wrapper/README.md`](./cpp_wrapper/README.md).
- **Regulatory note.** Any clinical adaptation would need IRB review,
  prospective validation on the target scanner population, and FDA SaMD
  classification.

## References

- Edelsbrunner, Letscher & Zomorodian, *Topological Persistence and
  Simplification*, 2002.
- Cohen-Steiner, Edelsbrunner & Harer, *Stability of Persistence
  Diagrams*, 2007. — the stability theorem cited above.
- Kaji, Sudo & Ahara, *CubicalRipser: software for computing persistent
  homology of image and volume data*, 2020.
  https://github.com/shizuo-kaji/CubicalRipser_3dim
- The GUDHI project. https://gudhi.inria.fr/
- Hajij et al., *Topological Deep Learning: Going Beyond Graph Data*,
  2023. (TopoModelX / TopoNetX).
- BraTS challenge. https://www.med.upenn.edu/cbica/brats2020/
- Karpathy, [CS231n notes on the neuron model](https://cs231n.github.io/neural-networks-1/).
- SIIM TDL track / educational resources.
