# Coursework requirement checklist (2026-11-13)

| Requirement | Status | Evidence / limitation |
|---|---|---|
| PlantSeg v7 and nine required hosts | Completed | Official ZIP MD5 and all 15,552 extracted files verified; 4,349 images, 43 classes. |
| Original train / validation / test splits | Completed | 3,007 / 475 / 867 images; no reassignment. |
| File integrity, class distribution, duplicate audit | Completed | `dataset_audit.json`, fixed manifest, review CSV, class counts and figures. |
| Prevent identical photographs crossing evaluation splits | Partially Completed | Exact and near duplicates were identified, but all images were retained in their original splits at the user's direction. Scores may be optimistic for new photographs. |
| Majority-class and basic CNN baselines | Completed | Validation and final test metrics saved. |
| Two separate controlled changes | Completed | Training augmentation and class-weighted loss were each compared directly with the basic CNN. A later, clearly labelled post-final fine-tuning extension was also run. |
| Validation macro-F1 model selection | Completed | Frozen plan selected class-weighted ResNet-18 before final test. |
| Final test: macro-F1, accuracy, balanced accuracy, per-class metrics and confusion matrix | Completed | `final_test_metrics.json` and `tables/`; selected model macro-F1 0.6317, accuracy 0.6690, balanced accuracy 0.6444. |
| Quantify detected same-photo leakage limitation | Completed | Post-test sensitivity analysis excludes 107 test photos with a detected training-photo match from analysis only; 760 remaining photos yield macro-F1 0.6122, accuracy 0.6526, balanced accuracy 0.6238. Original splits and frozen model remain unchanged. |
| Supplementary fine-tuning analysis | Completed | Layer4 plus head fine-tuning reached validation macro-F1 0.6638; a further five epochs without improvement triggered early stopping. Separate frozen supplementary test reached macro-F1 0.6629, accuracy 0.7093, balanced accuracy 0.6568 on all 867 original test images. Prior test results were already visible, so this is not a blind replacement for the original plan. |
| Training logs, checkpoints, environment and reproducibility settings | Completed | Best checkpoints, per-epoch CSV logs, config, fingerprints, and environment JSON saved. |
| Notebook full and demo modes | Partially Completed | Self-contained notebook executed end to end in local demo mode; full retraining was performed through the same embedded source modules locally, but a fresh Colab/Kaggle notebook run was not performed. |
| Data figures and representative error analysis | Completed | Saved distribution, curves, comparison, confusion matrix, prediction gallery and inspected error note. |
| PDF report, at most five pages | Partially Completed | Five-page PDF generated and visually checked; student identity and contribution fields remain placeholders until supplied. |
| AI-use disclosure | Completed | Brief two-sentence disclosure appears within the five-page report. |
| Three-minute video | Not Completed | English script and screen plan are prepared; a real presenter-visible recording is still required. |

The task sheet says a small set of well-explained experiments is sufficient; the two required controlled improvements were completed. The later fine-tuning extension is documented separately from the originally frozen test plan.
