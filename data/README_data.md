# Data

This directory holds the sample MRI slices used by the demo.

## Bundled samples (default)

The six PNGs in `samples/` are **real public-domain MRI slices** drawn
from the AFIP teaching collection and other public-domain sources on
Wikimedia Commons. They are committed to the repo so the Streamlit demo
loads instantly on first launch — no generation step required.

Per-image source URLs, authors, and licenses are recorded in
[`samples/labels.json`](./samples/labels.json), and a summary table lives
in the [Image attribution](../README.md#image-attribution) section of the
top-level README.

| File | Label | Notes |
|---|---|---|
| `01_healthy_axial_t1.png` | healthy | Normal brain, axial T1 |
| `02_healthy_sagittal_t1.png` | healthy | Normal brain, sagittal T1 |
| `03_tumor_solid_axial.png` | tumor | Sub-ependymal giant cell astrocytoma — solid central mass |
| `04_tumor_solid_sagittal.png` | tumor | Pilocytic astrocytoma, hypothalamic — solid mass |
| `05_tumor_ring_axial.png` | tumor | Glioblastoma multiforme — textbook ring enhancement (H₁ loop) |
| `06_tumor_ring_recurrent.png` | tumor | Recurrent multifocal glioblastoma |

> ⚠️ **Disclaimer:** These are real diagnostic images released for
> teaching, but TopoMedVision is a research prototype — not FDA-approved
> and not for clinical decision-making. Do not use any output from this
> tool to inform patient care.

## Using your own data

Drop additional 2D grayscale slices (PNG/JPG) into `samples/` and add an
entry to `labels.json` to include them in classifier training. Any image
in the directory is also pickable from the Streamlit sidebar, regardless
of whether it appears in the manifest.

To extract a 2D slice from a NIfTI volume (e.g. BraTS):

```python
import nibabel as nib, numpy as np
from PIL import Image

vol = nib.load("BraTS20_Training_001_t1ce.nii.gz").get_fdata()
slc = vol[:, :, vol.shape[2] // 2]                            # mid-axial
slc = (slc - slc.min()) / (slc.ptp() + 1e-8) * 255
Image.fromarray(slc.astype("uint8")).save("data/samples/brats_t1ce_mid.png")
```

## Synthetic generator (legacy, not used by the demo)

The script [`generate_samples.py`](./generate_samples.py) procedurally
synthesizes cartoon brain slices that *qualitatively* exhibit the
expected topology (multi-peak hotspots for solid tumors, bright rim with
dark core for ring-enhanced lesions). It was the original sample source
before the repo was migrated to real public-domain images, and is kept
around purely as a reference / fallback for environments where you can't
ship real medical images. **The demo no longer calls it.** Running it
will overwrite the bundled real samples and the `labels.json` manifest
— think of it as "if all else fails, you can still get *something* on
screen," not as part of the normal setup path.
