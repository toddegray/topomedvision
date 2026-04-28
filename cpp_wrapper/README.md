# C++ wrapper notes (stretch goal)

This directory documents how to wire the original C++ implementation of
**CubicalRipser** into the demo. The Streamlit app does *not* require this —
it ships with a `gudhi` backend and a pure-NumPy fallback. The notes here
are a stretch path for swapping in the C++ implementation if you want the
extra speed or a clean process boundary for profiling.

## Why CubicalRipser?

CubicalRipser ([Kaji et al.](https://github.com/shizuo-kaji/CubicalRipser_3dim))
is one of the fastest implementations of cubical persistent homology for
image and voxel data. It is written in C++ and exposed through a thin
Python binding called `cripser`.

## Option A — pip install the binding

If you have a working C++ toolchain (Xcode CLT on macOS, build-essentials on
Linux, or MSVC Build Tools on Windows):

```bash
pip install cripser
```

Then add the following branch to `backend/persistence.py::compute_cubical_persistence`:

```python
import cripser
pd = cripser.computePH(image, maxdim=1)  # (N, 5): dim, birth, death, x, y
```

`pd[:, 0]` gives the homological dimension and `pd[:, 1:3]` the birth/death
filtration values, with `pd[:, 3:5]` providing the spatial location of the
birth cell — a drop-in replacement for the demo's `birth_xy` field.

## Option B — call the binary via subprocess

For a fully decoupled "model server" style integration, build CubicalRipser
from source and shell out to it. The wrapper below is a sketch:

```python
import json, subprocess, tempfile, numpy as np

def cubical_ripser_subprocess(image: np.ndarray) -> dict:
    """Call a prebuilt cubicalripser binary, returning persistence as JSON."""
    with tempfile.NamedTemporaryFile(suffix=".npy") as fin, \
         tempfile.NamedTemporaryFile(suffix=".json") as fout:
        np.save(fin.name, image.astype("float64"))
        subprocess.check_call([
            "./cubicalripser_bin",
            "--input", fin.name,
            "--output", fout.name,
            "--format", "json",
            "--maxdim", "1",
        ])
        with open(fout.name) as f:
            return json.load(f)
```

You'd compile a small CLI on top of CubicalRipser that reads a `.npy` file
and writes a JSON list of `{dim, birth, death, x, y}` records. This keeps
the C++ code unmodified and gives you a clean process boundary for
profiling.

## Option C — thin C++ header

If you want to ship a wrapper header that other C++ projects can link:

```cpp
// topomedvision_wrapper.hpp
#pragma once
#include <vector>
#include <cubicalripser/dense_cubical_grids.h>

namespace topomedvision {

struct PersistencePair {
    int dim;
    double birth;
    double death;
    int x;
    int y;
};

std::vector<PersistencePair>
compute_persistence(const float* image, int height, int width, int maxdim = 1);

}  // namespace topomedvision
```

The implementation file would construct CubicalRipser's internal grid
representation, run the algorithm, and copy results into the `PersistencePair`
struct. This is left as an exercise; see CubicalRipser's `main.cpp` for the
canonical call sequence.
