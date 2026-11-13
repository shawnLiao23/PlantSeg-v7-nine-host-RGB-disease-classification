"""Run the overlap analysis against copies of local frozen inputs.

Usage: PYTHONPATH=code python code/tests/manual_overlap_smoke.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from plantseg_cw2.analysis.validation_overlap_analysis import main


source = Path(__file__).resolve().parents[2] / "result"
required = [
    "frozen_plan.json", "experiment_summary.csv", "manifest.csv",
    "class_mapping.json", "dataset_audit.json",
    "tables/near_duplicate_review.csv",
    "exploratory_finetune/summary.json",
    "exploratory_finetune_continued/summary.json",
]
with tempfile.TemporaryDirectory(prefix="cw2_overlap_smoke_") as temp:
    output = Path(temp) / "result"
    for relative in required:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    for folder in ("checkpoints", "exploratory_finetune", "exploratory_finetune_continued"):
        for path in (source / folder).glob("*.pth"):
            destination = output / folder / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    report = main(output)
    assert report["all_validation_count"] == 475
    assert len(report["experiments"]) == 4
    print("Overlap analysis passed on copied frozen inputs.")
