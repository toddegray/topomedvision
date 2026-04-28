"""TopoMedVision backend: topological feature extraction and hybrid scoring.

Modules
-------
utils          - image I/O and preprocessing
persistence    - cubical persistent homology (gudhi or pure-numpy fallback)
hybrid_model   - persistence-based feature vectors + lightweight classifier
visualization  - matplotlib / plotly views of diagrams, barcodes, overlays
"""

from . import utils, persistence, hybrid_model, visualization

__all__ = ["utils", "persistence", "hybrid_model", "visualization"]
