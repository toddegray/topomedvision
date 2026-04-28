"""Train the demo classifier on the shipped MRI samples.

This script lives separate from the app because training is a one-off
operation: run it once, save the model to
``models/topo_classifier.joblib``, and the Streamlit app picks it up
automatically.

The classifier is a small Random Forest trained on the 14-D persistence
feature vector defined in :mod:`backend.hybrid_model`.  With only six
training images it will trivially memorize them — that's by design for
the demo, not a generalization claim.  Swap in a larger labelled corpus
(e.g. BraTS slices) to get a meaningful classifier.

Run::

    python scripts/train_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import hybrid_model, persistence, utils  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "topo_classifier.joblib"


def main() -> None:
    labels_path = SAMPLES_DIR / "labels.json"
    if not labels_path.exists():
        raise SystemExit(
            f"No labels.json found at {labels_path}. "
            "The repo ships one alongside the sample PNGs in data/samples/."
        )

    manifest = json.loads(labels_path.read_text())
    feats: list[np.ndarray] = []
    labels: list[int] = []

    print(f"Computing persistence features for {len(manifest)} samples…")
    for entry in manifest:
        path = SAMPLES_DIR / entry["file"]
        img = utils.preprocess(utils.load_image(path))
        result = persistence.compute_cubical_persistence(img, persistence_threshold=0.05)
        f = hybrid_model.persistence_features(result["dim0"], result["dim1"], image=img)
        feats.append(hybrid_model.feature_vector(f))
        labels.append(1 if entry["label"] == "tumor" else 0)
        print(f"  {entry['file']:32s} label={entry['label']:7s} "
              f"max_H0={f['max_persistence_dim0']:.3f} "
              f"max_H1={f['max_persistence_dim1']:.3f}")

    X = np.stack(feats)
    y = np.array(labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf = hybrid_model.train_classifier(X, y, save_path=MODEL_PATH)

    train_acc = clf.score(X, y)
    print()
    print(f"Wrote {MODEL_PATH}")
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Feature importances (top 5):")
    for name, imp in sorted(
        zip(hybrid_model.FEATURE_NAMES, clf.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )[:5]:
        print(f"  {name:30s} {imp:.3f}")


if __name__ == "__main__":
    main()
