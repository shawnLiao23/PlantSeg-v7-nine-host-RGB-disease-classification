"""Check saved best checkpoints against their recorded validation scores."""

from __future__ import annotations

import csv
import json

from ..dataset import dataset_root, load_manifest, result_root
from ..evaluate import predict_model
from ..run_experiments import _load_model
from ..train import device


def main() -> None:
    output = result_root()
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    manifest = load_manifest(output / "manifest.csv")
    with (output / "experiment_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    target_device = device()
    for row in summary:
        if not row["checkpoint"]:
            continue
        model = _load_model(output / row["checkpoint"], mapping, row, target_device)
        observed, _ = predict_model(model, "val", manifest, dataset_root(),
                                    len(mapping["classes"]), target_device)
        for key, column in (("macro_f1", "val_macro_f1"),
                            ("accuracy", "val_accuracy"),
                            ("balanced_accuracy", "val_balanced_accuracy")):
            expected = float(row[column])
            if abs(observed[key] - expected) > 1e-6:
                raise AssertionError(f"{row['experiment']} {key}: saved {expected}, reloaded {observed[key]}")
        print(f"{row['experiment']}: checkpoint and validation metrics match")


if __name__ == "__main__":
    main()
