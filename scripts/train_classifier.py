"""Train (or retrain) the stub Random Forest classifier.

The shipped ``models/topo_classifier.joblib`` is a 6-sample stub: with
that many training images, a Random Forest just memorizes them and
contributes nothing generalizable to unseen scores.  Its purpose in
the repo is to exercise the train -> save -> load -> blend code path
end-to-end, so a real BraTS-trained model can be dropped in later
without touching ``backend/hybrid_model.py`` or ``app.py``.

The actual diagnostic-style logic lives in ``_RULE_WEIGHTS`` in
``backend/hybrid_model.py`` (a hand-coded weighted sum), and is what
the Prediction tab's bar chart decomposes regardless of whether this
classifier is present.

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
