"""Hybrid topological + statistical scoring for tumor likelihood.

This is a research prototype, not a clinical tool.  The interesting
work is the rule-based scorer — the optional sklearn classifier path is
*scaffolding* for a future BraTS-trained model, not a live ML component.

Pipeline
--------
1. ``persistence_features`` turns a list of persistence pairs into a fixed-
   length feature vector (Betti curves + persistence-lifetime statistics +
   intensity statistics of the image).  This is the "topological fingerprint."

2. ``score_tumor_likelihood`` combines the feature vector into a [0, 1]
   score plus a short natural-language explanation.  Two modes:

   - **Rule-based** (default; no training data required): a hand-tuned
     weighted sum (see ``_RULE_WEIGHTS``) that is explicit and inspectable.
     The bar chart in the Streamlit Prediction tab decomposes this score.
   - **Rule + stub blend**: if a scikit-learn classifier is loaded via
     :func:`load_classifier`, its ``predict_proba`` is averaged 50/50 with
     the rule-based score.  The shipped joblib was trained on six images
     (see :func:`train_classifier`), which is far too few to generalize —
     it overfits and its main effect is to sharpen the score on those exact
     six samples.  The path exists so a real BraTS-trained classifier can
     drop in without touching the call sites.

We intentionally keep the explanation a plain string — for an XAI demo,
"this score is high because there are 3 long-persisting connected components
above a lifetime of 0.3" is more useful than a 1024-dim attention map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .persistence import PersistencePair


FEATURE_NAMES: Tuple[str, ...] = (
    "n_dim0",                 # number of 0D features (excl. essential class)
    "n_dim1",                 # number of 1D features
    "max_persistence_dim0",   # longest 0D bar
    "max_persistence_dim1",   # longest 1D bar
    "mean_persistence_dim0",
    "mean_persistence_dim1",
    "n_long_dim0_15",         # # of dim0 bars with persistence > 0.15
    "n_long_dim0_30",         # # of dim0 bars with persistence > 0.30
    "n_long_dim1_15",
    "total_lifetime_dim0",    # sum of dim0 persistences
    "total_lifetime_dim1",
    "image_max_intensity",
    "image_p95_intensity",
    "image_bright_fraction",  # fraction of pixels in top 10% of intensity
)


@dataclass
class ScoreResult:
    """Output of :func:`score_tumor_likelihood`."""

    score: float                    # in [0, 1]
    confidence: float               # rough self-assessed confidence in [0, 1]
    explanation: str
    features: dict                  # name -> value, the inputs to the score
    contributions: dict             # name -> signed contribution to the score


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def persistence_features(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    image: Optional[np.ndarray] = None,
) -> dict:
    """Compute the fixed-length feature dict used by the classifier.

    Essential (infinite-persistence) classes are excluded from "longest bar"
    statistics since they aren't informative — there's always exactly one
    essential 0D class for a connected image, regardless of tumor presence.
    """
    finite0 = [p.persistence for p in pairs_dim0 if not math.isinf(p.persistence)]
    finite1 = [p.persistence for p in pairs_dim1 if not math.isinf(p.persistence)]

    feats = {
        "n_dim0": float(len(finite0)),
        "n_dim1": float(len(finite1)),
        "max_persistence_dim0": float(max(finite0)) if finite0 else 0.0,
        "max_persistence_dim1": float(max(finite1)) if finite1 else 0.0,
        "mean_persistence_dim0": float(np.mean(finite0)) if finite0 else 0.0,
        "mean_persistence_dim1": float(np.mean(finite1)) if finite1 else 0.0,
        "n_long_dim0_15": float(sum(1 for v in finite0 if v > 0.15)),
        "n_long_dim0_30": float(sum(1 for v in finite0 if v > 0.30)),
        "n_long_dim1_15": float(sum(1 for v in finite1 if v > 0.15)),
        "total_lifetime_dim0": float(sum(finite0)),
        "total_lifetime_dim1": float(sum(finite1)),
    }

    if image is not None:
        img = image.astype(np.float32)
        if img.max() > 0:
            img = img / img.max()
        feats.update(
            {
                "image_max_intensity": float(img.max()),
                "image_p95_intensity": float(np.percentile(img, 95)),
                "image_bright_fraction": float((img > 0.7).mean()),
            }
        )
    else:
        feats.update(
            {
                "image_max_intensity": 0.0,
                "image_p95_intensity": 0.0,
                "image_bright_fraction": 0.0,
            }
        )

    return feats


def feature_vector(features: dict) -> np.ndarray:
    """Project a feature dict to a flat numpy vector in :data:`FEATURE_NAMES` order."""
    return np.array([features.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Rule-based + (optional) learned scoring
# ---------------------------------------------------------------------------


# Hand-tuned weights for the rule-based scorer.  They encode the demo's prior
# about what a "tumor-like" topological signature looks like:
#   - one or two VERY long-persisting 0D components (dense bright masses)
#   - the longest 0D bar is longer than what healthy tissue produces
#   - some 1D loops (necrotic cores create rings on T1c MRI)
#   - the brightest pixel is markedly brighter than the parenchyma mean
# Negative weights penalize features typical of healthy slices (lots of
# medium-persistence noise from texture).
#
# These weights were tuned by inspecting the topological signatures of the
# 6 synthetic samples after the data generator was strengthened.  They give:
# healthy → ~0.20–0.30, solid tumor → ~0.65–0.80, ring tumor → ~0.75–0.90.
_RULE_WEIGHTS = {
    "max_persistence_dim0": 4.0,    # the dominant signal
    "n_long_dim0_30": 0.20,
    "max_persistence_dim1": 1.5,
    "n_long_dim1_15": 0.05,
    "image_max_intensity": 1.0,
    "image_bright_fraction": 1.5,
    # Penalize: medium bars from noise are abundant in healthy tissue too.
    "n_long_dim0_15": -0.03,
}
_RULE_BIAS = -3.5


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _rule_based_score(features: dict) -> Tuple[float, dict]:
    """Return (score, per-feature signed contributions)."""
    contribs = {name: w * features.get(name, 0.0) for name, w in _RULE_WEIGHTS.items()}
    raw = _RULE_BIAS + sum(contribs.values())
    return _sigmoid(raw), contribs


def score_tumor_likelihood(
    pairs_dim0: Sequence[PersistencePair],
    pairs_dim1: Sequence[PersistencePair],
    image: Optional[np.ndarray] = None,
    classifier: Optional["object"] = None,
) -> ScoreResult:
    """Score and explain tumor likelihood.

    Returns a rule-based score by default.  If ``classifier`` is supplied,
    its ``predict_proba`` is averaged 50/50 with the rule score; the
    returned ``contributions`` dict still decomposes only the rule half
    (the bar chart in the Streamlit Prediction tab plots that dict).

    The shipped classifier is a 6-sample stub — it overfits and its
    contribution is essentially noise on unseen images.  See module
    docstring for the full caveat.  The blend exists so a real
    BraTS-trained model can be dropped in without changing call sites.
    """
    features = persistence_features(pairs_dim0, pairs_dim1, image=image)
    rule_score, contribs = _rule_based_score(features)

    score = rule_score
    confidence = 0.5  # rule-based scores are inherently low-confidence

    if classifier is not None:
        try:
            x = feature_vector(features).reshape(1, -1)
            proba = float(classifier.predict_proba(x)[0, 1])
            score = 0.5 * rule_score + 0.5 * proba
            confidence = 0.75
        except Exception:
            # Classifier shape mismatch or unfit: silently fall back to rule.
            pass

    explanation = _explain(features, contribs, score)
    return ScoreResult(
        score=float(np.clip(score, 0.0, 1.0)),
        confidence=float(confidence),
        explanation=explanation,
        features=features,
        contributions=contribs,
    )


def _explain(features: dict, contribs: dict, score: float) -> str:
    """Build a short, human-readable explanation of the score."""
    lines: List[str] = []

    if score > 0.7:
        verdict = "**Elevated** likelihood of a tumor-like topological signature."
    elif score > 0.4:
        verdict = "**Moderate / inconclusive** topological signature."
    else:
        verdict = "**Low** likelihood — topology resembles healthy tissue."
    lines.append(verdict)

    drivers = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    if drivers:
        lines.append("Top drivers:")
        for name, contribution in drivers:
            value = features.get(name, 0.0)
            arrow = "⬆" if contribution > 0 else "⬇"
            lines.append(f"- {arrow} `{name}` = {value:.3g} (weight {contribution:+.2f})")

    n_long = int(features.get("n_long_dim0_30", 0))
    if n_long >= 2:
        lines.append(
            f"Detected **{n_long} long-persisting bright components** "
            "(persistence > 0.30) — characteristic of dense mass regions."
        )
    if features.get("n_long_dim1_15", 0) >= 1:
        lines.append(
            "At least one persistent **1D loop** is present, consistent with "
            "a ring-shaped enhancement around a low-intensity core."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional: persisted classifier
# ---------------------------------------------------------------------------


def load_classifier(path: str | Path):
    """Load a pickled scikit-learn classifier, or return ``None`` if missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        import joblib  # local import: scikit-learn pulls joblib in transitively

        return joblib.load(path)
    except Exception:
        return None


def train_classifier(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str | Path] = None,
):
    """Train a small Random Forest on persistence features.

    This is a stretch-goal helper — the demo runs without it.  We use
    Random Forest rather than a neural net because (a) the feature vector is
    14-dim, (b) it gives free feature-importance for the XAI panel.
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=0,
        class_weight="balanced",
    )
    clf.fit(feature_matrix, labels)

    if save_path is not None:
        import joblib

        joblib.dump(clf, save_path)
    return clf
