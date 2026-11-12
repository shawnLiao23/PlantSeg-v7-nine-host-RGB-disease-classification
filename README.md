# AIP304-Machine-learning-coursework

AIP304 Coursework 2, Option 1: PlantSeg v7 nine-host RGB disease classification.

The project keeps the original train/validation/test partitions for all 4,349 selected images and 43 host–disease classes. Source data, local Python environments, and model weights are intentionally kept outside Git. Obtain PlantSeg v7 from the [versioned Zenodo record](https://zenodo.org/records/17719108) and extract it as `plantseg/plantseg/`.

The notebook is [`code/AIP304_CW2_Option1.ipynb`](code/AIP304_CW2_Option1.ipynb). The installable source package is [`code/plantseg_cw2/`](code/plantseg_cw2/), with supplementary analyses in its `analysis/` subpackage. Setup and commands are in [`code/README.md`](code/README.md). Saved metrics, audit records, plots, and the current report are under `result/`; the video guide and requirement checklist are under `docs/`.

The original validation-selected class-weighted model has test macro-F1 **0.6317**. A separately frozen, post-final fine-tuning extension has test macro-F1 **0.6629** on the same 867 original test images. This extension was designed after the original test results were visible and is labelled supplementary in the saved records. The current PDF still describes the original frozen plan; its final revision is pending.
