# Saved experiment artifacts

The fixed manifest, class mapping, audit, configuration, validation summary, frozen plan, and original final test result live at this level. `tables/`, `logs/`, and `figures/` contain the corresponding per-class data, histories, and plots. `exploratory_finetune/` and `exploratory_finetune_continued/` hold the separately labelled post-final experiment.

The repository keeps small result records and report figures. PlantSeg images, the source ZIP, pretrained weights, trained checkpoints, and large audit contact sheets stay on the local machine and are ignored by Git. Paths inside the manifest are relative to the extracted PlantSeg root. The archived `source_archive_verification.json` records the ZIP label and checksum; use `python -m plantseg_cw2.analysis.verify_archive` to recheck another local copy.
