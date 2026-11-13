"""Compare saved validation scores under the historical and evaluation protocols.

Reads frozen inputs and writes only to result/diagnostics. No test images are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torchvision.models import resnet18

from ..dataset import dataset_root, load_manifest, result_root
from ..evaluate import predict_model
from ..train import BATCH_SIZE, WEIGHTS_ID, device


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entries_for(output: Path) -> list[dict]:
    with (output / "experiment_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        entries = [row for row in csv.DictReader(handle) if row["checkpoint"]]
    pilot_path = output / "exploratory_finetune" / "summary.json"
    continued_path = output / "exploratory_finetune_continued" / "summary.json"
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        latest = (json.loads(continued_path.read_text(encoding="utf-8"))
                  if continued_path.is_file() else pilot)
        checkpoint = latest.get("best_available_checkpoint") or pilot.get("checkpoint")
        if checkpoint:
            entries.append({
                "experiment": "finetune_layer4_postfinal",
                "checkpoint": checkpoint,
                "best_epoch": pilot["best_epoch"] if checkpoint == pilot["checkpoint"]
                              else latest["best_epoch"],
                "val_macro_f1": latest["best_validation"]["macro_f1"]
                                if checkpoint == latest.get("best_available_checkpoint")
                                else pilot["best_validation"]["macro_f1"],
            })
    return entries


def main(output_dir: str | Path | None = None, experiment: str | None = None) -> dict:
    output = result_root(output_dir)
    needed = [output / name for name in
              ("frozen_plan.json", "experiment_summary.csv", "manifest.csv", "class_mapping.json")]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: " + ", ".join(missing))
    plan = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    hashes = {name: sha(output / name) for name in
              ("frozen_plan.json", "experiment_summary.csv", "manifest.csv", "class_mapping.json")}
    for field, name in (("summary_sha256", "experiment_summary.csv"),
                        ("manifest_sha256", "manifest.csv"),
                        ("class_mapping_sha256", "class_mapping.json")):
        if hashes[name] != plan[field]:
            raise ValueError(f"Frozen {name} hash differs: expected={plan[field]} actual={hashes[name]}")
    entries = entries_for(output)
    if experiment:
        entries = [row for row in entries if row["experiment"] == experiment]
        if not entries:
            raise ValueError(f"Unknown experiment: {experiment}")
    source = dataset_root()
    manifest = load_manifest(output / "manifest.csv")
    val_rows = [row for row in manifest if row["split"] == "val" and
                str(row.get("excluded", "0")).lower() not in {"1", "true", "yes"}]
    if len({row["image_id"] for row in val_rows}) != len(val_rows):
        raise ValueError("Validation manifest contains duplicate image_id values")
    missing_images = [str(source / row["image_path"]) for row in val_rows
                      if not (source / row["image_path"]).is_file()]
    missing_checkpoints = [str(output / row["checkpoint"]) for row in entries
                           if not (output / row["checkpoint"]).is_file()]
    if missing_images or missing_checkpoints:
        raise FileNotFoundError(
            f"Missing {len(missing_images)} validation images and "
            f"{len(missing_checkpoints)} checkpoints: "
            + ", ".join((missing_images + missing_checkpoints)[:20])
        )
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    target = device()
    report = {
        "frozen_input_sha256": hashes,
        "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                        "cuda": torch.version.cuda, "device": str(target),
                        "gpu": torch.cuda.get_device_name(0) if target.type == "cuda" else None},
        "validation_count": len(val_rows), "experiments": {},
    }
    differences = []
    for entry in entries:
        name = entry["experiment"]
        checkpoint = output / entry["checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["class_mapping"] != mapping or payload["weights_identifier"] != WEIGHTS_ID
                or int(payload["best_epoch"]) != int(entry["best_epoch"])
                or abs(float(payload["validation_macro_f1"]) - float(entry["val_macro_f1"])) > 1e-10):
            raise ValueError(f"Checkpoint metadata disagrees with summary: {checkpoint}; sha256={sha(checkpoint)}")
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, mapping["class_count"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(target).eval()
        historical = "full_amp" if name == "finetune_layer4_postfinal" else "cached_features"
        modes = [(historical, BATCH_SIZE), ("feature_fp32", 32), ("feature_fp32", 64),
                 ("full_amp", 32), ("full_amp", 64)]
        if historical == "cached_features":
            modes.append(("cached_features", 64))
        unique = list(dict.fromkeys(modes))
        observed = {}
        for protocol, batch_size in unique:
            metrics, predictions = predict_model(
                model, "val", manifest, source, mapping["class_count"], target,
                batch_size=batch_size, protocol=protocol, include_margin=True)
            if [row["image_id"] for row in predictions] != [row["image_id"] for row in val_rows]:
                raise ValueError(f"Prediction image_id order differs from manifest: {name}")
            if any(row["true_class"] != int(item["class_id"])
                   for row, item in zip(predictions, val_rows)):
                raise ValueError(f"Prediction labels differ from manifest: {name}")
            key = f"{protocol}_bs{batch_size}"
            observed[key] = (metrics, predictions)
        reference_key = f"{historical}_bs{BATCH_SIZE}"
        reference = observed[reference_key][1]
        protocol_metrics = {}
        for key, (metrics, predictions) in observed.items():
            protocol_metrics[key] = {metric: metrics[metric] for metric in
                                     ("macro_f1", "accuracy", "balanced_accuracy")}
            protocol_metrics[key]["different_predictions_from_historical"] = sum(
                a["pred_class"] != b["pred_class"] for a, b in zip(reference, predictions))
            if key == reference_key:
                continue
            for a, b in zip(reference, predictions):
                if a["pred_class"] != b["pred_class"]:
                    differences.append({
                        "experiment": name, "comparison_protocol": key,
                        "image_id": a["image_id"], "true_class": a["true_class"],
                        "historical_pred_class": a["pred_class"],
                        "comparison_pred_class": b["pred_class"],
                        "historical_top2_logit_margin": a["top2_logit_margin"],
                        "comparison_top2_logit_margin": b["top2_logit_margin"],
                    })
        report["experiments"][name] = {
            "checkpoint": entry["checkpoint"], "checkpoint_sha256": sha(checkpoint),
            "checkpoint_epoch": payload["best_epoch"],
            "saved_validation_macro_f1": float(entry["val_macro_f1"]),
            "historical_protocol": reference_key,
            "historical_matches_saved": abs(
                protocol_metrics[reference_key]["macro_f1"] - float(entry["val_macro_f1"])) <= 1e-6,
            "protocols": protocol_metrics,
        }
        print(f"{name}: saved={entry['val_macro_f1']} historical="
              f"{protocol_metrics[reference_key]['macro_f1']}", flush=True)
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    folder = output / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    diff_path = folder / "validation_prediction_differences.csv"
    with diff_path.open("w", encoding="utf-8", newline="") as handle:
        columns = ["experiment", "comparison_protocol", "image_id", "true_class",
                   "historical_pred_class", "comparison_pred_class",
                   "historical_top2_logit_margin", "comparison_top2_logit_margin"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(differences)
    report["prediction_differences_sha256"] = sha(diff_path)
    (folder / "validation_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment")
    options = parser.parse_args()
    report = main(options.output_dir, options.experiment)
    failures = [name for name, item in report["experiments"].items()
                if not item["historical_matches_saved"]]
    if failures:
        parser.exit(1, "Historical validation does not match saved scores: "
                    + ", ".join(failures) + "\nSee diagnostics/validation_consistency.json.\n")
