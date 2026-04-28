# Assets

Static images referenced from the project README. Regenerate them with:

```bash
python scripts/generate_screenshots.py
```

The script renders composite matplotlib figures from the same backend
code paths the Streamlit app uses, so the assets stay in sync with the
demo whenever the persistence layer or scorer changes.

| File | Description |
|---|---|
| `hero.png` | Healthy vs. ring-enhanced tumor side-by-side, with persistence barcodes — top of the README. |
| `screenshot_original.png` | Raw upload vs. preprocessed slice. |
| `screenshot_persistence.png` | Persistence diagram + barcode + lifetime histogram. |
| `screenshot_highlight.png` | Top-`k` birth pixels and the flood-filled overlay mask. |
| `screenshot_prediction.png` | Tumor-likelihood gauge + top feature drivers. |
| `screenshot_baseline.png` | Otsu intensity threshold vs. topology highlight side-by-side. |

The example slice used for tabs 1–5 is `data/samples/05_tumor_ring_axial.png`
(public-domain glioblastoma MRI from AFIP). The hero image pairs that
with `01_healthy_axial_t1.png` (public-domain normal axial T1).
