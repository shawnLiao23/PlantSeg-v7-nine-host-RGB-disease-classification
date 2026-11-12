# AIP304 Coursework 2 - Option 1

This project classifies RGB photographs into host-disease classes for the nine specified PlantSeg hosts. The notebook is `AIP304_CW2_Option1.ipynb`; the installable `plantseg_cw2` package holds the audit, training, evaluation, plotting, and inference code used by the notebook. The `analysis` subpackage contains supplementary experiments and diagnostics.

## Dataset and paths

Obtain the versioned **PlantSeg v7** archive from [Zenodo record 17719108](https://zenodo.org/records/17719108). Keep its supplied `train`, `val`, and `test` assignments. The expected local layout is:

```text
coursework2/
  plantseg/plantseg/Metadata.csv
  plantseg/plantseg/images/{train,val,test}/
  code/AIP304_CW2_Option1.ipynb
  code/plantseg_cw2/{dataset,train,evaluate,run_experiments,...}.py
  code/plantseg_cw2/analysis/
  docs/
  result/
```

The notebook can also use `PLANTSEG_ROOT` to locate the extracted directory containing `Metadata.csv`, and `CW2_RESULT_DIR` to choose a result directory. Both are optional. The code uses relative paths in the manifest so that it does not depend on this computer's private Windows path. It never modifies the original dataset.

The local source is verified **PlantSeg v7**: the project-root `plantseg.zip` matches the official v7 MD5 (`9358a66dff88cdd15c4fe009763c40a3`), and all 15,552 extracted files match its entries by size and CRC. The check is recorded in `result/source_archive_verification.json` and the dataset audit. The notebook also checks the archive checksum when available. Do not mix releases.

The fixed local manifest includes all 4,349 nine-host images in 43 classes under their supplied splits: 3,007 train, 475 validation, and 867 test. No duplicate image is excluded or reassigned. The audit found 73 exact-duplicate groups spanning splits and visually reviewed 142 near-duplicate candidate pairs (140 same-photo, two distinct). Because the original splits contain some of the same photographs, validation and test scores may be optimistic for genuinely new images. Seventeen same-split exact-duplicate groups carry conflicting class labels and remain a documented label-noise limitation.

## Dependencies

Use Python with PyTorch, torchvision, Pillow, NumPy, Matplotlib, and Jupyter. From the project root, run `python -m pip install -r code/requirements.txt` and `python -m pip install . --no-deps` to install the package and command-line entry points. A regular install has been checked with this project's Chinese Windows path. For source edits without reinstalling, set `PYTHONPATH` to the absolute `code` directory before using `python -m plantseg_cw2...`. If running from outside the repository, set `CW2_PROJECT_ROOT` to its absolute path. The notebook embeds its required package source and adds `code` to `sys.path` automatically. The full run is practical on a CUDA GPU; CPU execution is supported but much slower.

## Run the project

In the notebook, set `MODE = "full"` to audit, train, validate, freeze the model choice, run the final test, and regenerate figures. Run the cells sequentially. `MODE = "demo"` loads existing outputs and a checkpoint for a short inference demonstration without training again.

The command-line stages provide the same workflow when run from the project root after the package is installed or `PYTHONPATH` is set:

```powershell
python -m plantseg_cw2.run_experiments --audit
python -m plantseg_cw2.run_experiments --train
python -m plantseg_cw2.run_experiments --final-test
python -m plantseg_cw2.analysis.sensitivity_analysis
python -m plantseg_cw2.run_experiments --plots
python -m plantseg_cw2.build_report --student "[Name]" --student-id "[ID]" --contributions "[Actual contributions]"
```

`--final-test` is a separate stage. Training writes `result/frozen_plan.json` after model selection by validation macro-F1. Once final test results exist, the runner refuses to train or tune again against them. Running `--plots` and `build_report.py` only regenerates figures and the PDF from saved results.

## Experiments and evaluation

The required majority-class predictor uses only training class frequency. The CNN baseline trains a new classifier head on ImageNet-pretrained ResNet-18 features. Two controlled changes are evaluated separately against this baseline: training augmentation and a class-weighted cross-entropy loss calculated from training counts. The validation macro-F1 decides checkpoints and the final model. The frozen systems are then evaluated once on the test split.

The outputs include macro-F1, top-1 accuracy, balanced accuracy, a per-class precision/recall/F1/support table, and a confusion matrix in a fixed class order. Where a metric denominator is zero, the code reports zero for that class and includes all mapped classes in macro averages. Validation and test use deterministic transforms, evaluation mode, and disabled gradients. The complete training settings are stored in `result/experiment_config.json`; actual environment and elapsed training times are recorded alongside the results.

### Supplementary fine-tuning experiment

After the original final test had already been run, a separate experiment fine-tuned ResNet-18's last residual block and classifier head from the original CNN baseline checkpoint. It used the same original 3,007 training and 475 validation images, deterministic transforms, plain cross-entropy, and validation macro-F1 for checkpoint selection. A 10-epoch pilot improved through epoch 10; a continuation from that checkpoint, with a fresh Adam optimizer, stopped after five further epochs without improvement. The saved epoch-10 checkpoint achieved validation macro-F1 **0.6638**, versus **0.6276** for the original class-weighted model. On the 414 validation images without a *detected* training-photo match, the scores were **0.6401** and **0.6092**, respectively. The original dataset and splits were never changed.

The checkpoint was frozen in `result/exploratory_finetune/frozen_extension_plan.json` before one supplementary test evaluation on all 867 original test images. It achieved macro-F1 **0.6629**, accuracy **0.7093**, and balanced accuracy **0.6568**; the original selected model achieved **0.6317**, **0.6690**, and **0.6444**. Per-class differences and paired predictions are in `result/exploratory_finetune/`. This extension is labelled **post-final** because the original test results were already visible when it was designed; it must not be described as a fully blind final test or used to justify further tuning.

The reproducible supplementary commands, run after the original experiment, are:

```powershell
python -m plantseg_cw2.analysis.explore_finetune --epochs 10 --patience 4
python -m plantseg_cw2.analysis.explore_finetune --continue-from-best-pilot --epochs 15 --patience 5
python -m plantseg_cw2.analysis.validation_overlap_analysis
python -m plantseg_cw2.analysis.evaluate_extension --freeze
python -m plantseg_cw2.analysis.evaluate_extension --test
python -m plantseg_cw2.analysis.compare_extension
```

The freeze and test commands intentionally refuse a second run in the same result directory. To reproduce from scratch, use a separate result directory and perform the original audit and training there first.

After the frozen test run, `sensitivity_analysis.py` checks the saved predictions against the audit's exact matches and visually confirmed same-photo pairs. It reports scores for the 760 test images without a detected match in training, while leaving all 867 original test images and all model choices untouched. This is a post-test limitation check, not a new validation or test-selection step. It cannot rule out undetected overlap or overlap between validation and training. Results are in `result/overlap_sensitivity.json`.

## Main outputs

```text
result/manifest.csv                 fixed original split labels and image audit fields
result/class_mapping.json           stable class IDs and names
result/dataset_audit.json           image checks, duplicate audit, split counts
result/experiment_summary.csv       validation results and actual training times
result/frozen_plan.json              validation-selected model, before final test
result/final_test_metrics.json       frozen test metrics for every system
result/overlap_sensitivity.json      post-test check using saved predictions
result/validation_overlap_analysis.json  validation-only overlap sensitivity
result/exploratory_finetune/         post-final fine-tuning, frozen extension and test
result/tables/                      class counts, per-class scores, predictions
result/figures/                     data and model visualisations
result/checkpoints/                 best-validation CNN checkpoints
result/AIP304_CW2_Report.pdf        concise report, at most five pages
```

The notebook regenerates the report's data-derived tables and figures from these files. Large inline image outputs are removed from the saved notebook; the figures remain in `result/figures/` and are displayed when its cells run. The PDF is built by `plantseg_cw2.build_report`, which refuses to create a report without the recorded manifest, audit, validation summary, and final test metrics. Fill in student names, identifiers, and actual team contributions before submission. The recording guide is `docs/video_script.md`; a real video with the presenter visible is still required.

The repository contains saved result tables but omits raw data and model checkpoints. A fresh clone can inspect the saved outputs. To execute inference, use locally regenerated checkpoints. To rerun the full experiment without overwriting the archived results, set `CW2_RESULT_DIR` to a new directory such as `result_local` and place a copy of `result/tables/near_duplicate_review.csv` at `result_local/tables/near_duplicate_review.csv` before auditing. This preserves the documented visual review decisions.

## Sources

- Wei et al., *A large-scale in-the-wild dataset for plant disease segmentation*, Zenodo v7, 2025. [doi:10.5281/zenodo.17719108](https://doi.org/10.5281/zenodo.17719108)
- Wei et al., *Scientific Data* 13, 205, 2026. [doi:10.1038/s41597-025-06513-4](https://doi.org/10.1038/s41597-025-06513-4)
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR, 770-778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
