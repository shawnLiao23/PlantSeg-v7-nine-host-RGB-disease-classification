"""Run the coursework stages. Test evaluation is an explicit final stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torchvision.models import resnet18

from .audit_dataset import audit_dataset
from .dataset import dataset_root, is_included, load_manifest, result_root
from .evaluate import majority_prediction, make_final_plots, predict_model, write_rows
from .train import WEIGHTS_ID, train_experiments, device


def _load_model(checkpoint_path: Path, mapping: dict, summary_row: dict,
                target_device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (payload.get("architecture") != "resnet18"
            or payload.get("class_mapping") != mapping
            or payload.get("weights_identifier") != WEIGHTS_ID
            or int(payload.get("best_epoch", -1)) != int(summary_row["best_epoch"])
            or abs(float(payload.get("validation_macro_f1", -1))
                   - float(summary_row["val_macro_f1"])) > 1e-9):
        raise ValueError(f"Checkpoint does not match frozen class mapping or validation summary: {checkpoint_path}")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(mapping["classes"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model.to(target_device)


def final_test(output_dir: str | Path | None = None) -> dict:
    output = result_root(output_dir)
    summary_path = output / "experiment_summary.csv"
    plan_path = output / "frozen_plan.json"
    test_path = output / "final_test_metrics.json"
    if not plan_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Train and freeze experiments before final test evaluation")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if summary_sha != plan["summary_sha256"]:
        raise ValueError("Experiment summary changed after plan freeze; refusing test evaluation")
    for filename, key in (("manifest.csv", "manifest_sha256"),
                          ("class_mapping.json", "class_mapping_sha256")):
        actual = hashlib.sha256((output / filename).read_bytes()).hexdigest()
        if actual != plan.get(key):
            raise ValueError(f"{filename} changed after plan freeze; refusing test evaluation")
    if test_path.exists():
        existing = json.loads(test_path.read_text(encoding="utf-8"))
        if existing.get("frozen_plan_summary_sha256") != summary_sha:
            raise ValueError("Test results exist for a different frozen plan")
        print("Final test already exists for this frozen plan; reusing saved results.", flush=True)
        return existing
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    if set(plan["experiments"]) != {row["experiment"] for row in summary}:
        raise ValueError("Frozen experiment list does not match summary")
    manifest = load_manifest(output / "manifest.csv")
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    class_count = len(mapping["classes"])
    target_device = device()
    test_results = {}
    selected_predictions = None
    for row in summary:
        name = row["experiment"]
        if name == "majority_class":
            metrics = majority_prediction(manifest, "test", class_count)
        else:
            checkpoint = output / row["checkpoint"]
            model = _load_model(checkpoint, mapping, row, target_device)
            metrics, predictions = predict_model(model, "test", manifest, dataset_root(),
                                                 class_count, target_device)
            if name == plan["selected_model"]:
                selected_predictions = predictions
            del model
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
        test_results[name] = metrics
        print(f"Test {name}: macro-F1={metrics['macro_f1']:.4f}, "
              f"accuracy={metrics['accuracy']:.4f}, "
              f"balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
    selected = plan["selected_model"]
    if selected == "majority_class":
        majority_id = test_results[selected]["majority_class_id"]
        selected_predictions = [
            {"image_id": row["image_id"], "image_path": row["image_path"],
             "true_class": int(row["class_id"]), "pred_class": majority_id, "confidence": 1.0}
            for row in manifest if row["split"] == "test" and is_included(row)
        ]
    chosen_metrics = test_results[selected]
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    report = {"selected_model": selected, "selection_metric": "validation macro-F1",
              "frozen_plan_summary_sha256": summary_sha, "experiments": test_results,
              "test_set_size": len(selected_predictions),
              "release_provenance": audit.get("dataset_release", "unverified"),
              "release_version_note": audit.get("version_note", "")}
    # Test outputs are written only after all models have been evaluated under the frozen plan.
    write_rows(output / "tables" / "test_predictions.csv", selected_predictions,
               ["image_id", "image_path", "true_class", "pred_class", "confidence"])
    per_class = []
    for entry in chosen_metrics["per_class"]:
        item = dict(entry)
        item["class_name"] = mapping["classes"][item["class_id"]]["class_name"]
        per_class.append(item)
    write_rows(output / "tables" / "per_class_test.csv", per_class,
               ["class_id", "class_name", "precision", "recall", "f1", "support"])
    matrix = np.asarray(chosen_metrics["confusion_matrix"], dtype=int)
    matrix_rows = [{"true_class_id": i, **{str(j): int(matrix[i, j]) for j in range(class_count)}}
                   for i in range(class_count)]
    write_rows(output / "tables" / "confusion_matrix.csv", matrix_rows,
               ["true_class_id"] + [str(i) for i in range(class_count)])
    confusions = []
    for i in range(class_count):
        for j in range(class_count):
            if i != j and matrix[i, j] > 0:
                confusions.append({"true_class": mapping["classes"][i]["class_name"],
                                   "predicted_class": mapping["classes"][j]["class_name"],
                                   "count": int(matrix[i, j])})
    confusions.sort(key=lambda r: -r["count"])
    (output / "error_analysis.json").write_text(json.dumps({
        "top_confusions": confusions[:15],
        "error_count": int(len(selected_predictions) - np.trace(matrix)),
        "note": "Confusions are observations; symptom-level causes require visual inspection."
    }, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_final_plots(output)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--audit", action="store_true", help="Build manifest and audit dataset")
    stages.add_argument("--train", action="store_true", help="Train/validate and freeze experiment plan")
    stages.add_argument("--full", action="store_true", help="Audit then train; do not evaluate test")
    stages.add_argument("--final-test", action="store_true", help="Evaluate the frozen plan on test once")
    stages.add_argument("--plots", action="store_true", help="Regenerate figures from saved results")
    args = parser.parse_args(argv)
    if args.audit or args.full:
        audit_dataset(dataset_root(), result_root())
    if args.train or args.full:
        if (result_root() / "final_test_metrics.json").exists():
            raise RuntimeError("Final test has been evaluated; do not retrain/tune against its results")
        train_experiments()
    if args.final_test:
        final_test()
    if args.plots:
        make_final_plots()


if __name__ == "__main__":
    main()
