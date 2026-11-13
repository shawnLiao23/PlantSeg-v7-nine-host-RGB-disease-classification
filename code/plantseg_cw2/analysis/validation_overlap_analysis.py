"""Check frozen validation predictions with detected train-photo overlaps omitted.

The original selection and full-split scores remain unchanged. This script
evaluates saved frozen checkpoints on validation only and records a sensitivity
analysis for interpretation; it never reads the test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models import resnet18

from ..dataset import dataset_root, load_manifest, result_root
from ..evaluate import predict_model, scores
from ..train import BATCH_SIZE, WEIGHTS_ID, device


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output_dir: Path | None = None) -> dict:
    output = result_root(output_dir)
    plan = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    for field, filename in (("manifest_sha256", "manifest.csv"),
                            ("class_mapping_sha256", "class_mapping.json"),
                            ("summary_sha256", "experiment_summary.csv")):
        if sha(output / filename) != plan[field]:
            raise ValueError(f"Frozen input changed: {filename}")
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    review_path = output / "tables" / "near_duplicate_review.csv"
    if sha(review_path) != audit["near_duplicate_review_sha256"]:
        raise ValueError("Audited visual review changed")
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    manifest = load_manifest(output / "manifest.csv")
    val_rows = [row for row in manifest if row["split"] == "val"]
    if len({row["image_id"] for row in val_rows}) != len(val_rows):
        raise ValueError("Validation manifest contains duplicate image_id values")
    exact = set()
    for group in audit["exact_duplicate_groups"]:
        members = group["members"]
        if any(member["split"] == "train" for member in members):
            exact.update(member["image_id"] for member in members
                         if member["split"] == "val")
    near = set()
    for pair in read_rows(review_path):
        if pair["decision"] != "same_photo":
            continue
        if pair["split_a"] == "train" and pair["split_b"] == "val":
            near.add(pair["image_b"])
        if pair["split_b"] == "train" and pair["split_a"] == "val":
            near.add(pair["image_a"])
    overlap = exact | near
    if not overlap.issubset({row["image_id"] for row in val_rows}):
        raise ValueError("Audit references unknown validation images")
    summary_rows = read_rows(output / "experiment_summary.csv")
    pilot_path = output / "exploratory_finetune" / "summary.json"
    continued_path = output / "exploratory_finetune_continued" / "summary.json"
    if pilot_path.is_file():
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        latest = (json.loads(continued_path.read_text(encoding="utf-8"))
                  if continued_path.is_file() else pilot)
        if latest.get("best_available_checkpoint"):
            checkpoint = latest["best_available_checkpoint"]
            expected = latest["best_validation"]["macro_f1"]
            best_epoch = (pilot["best_epoch"] if checkpoint == pilot["checkpoint"]
                          else latest["best_epoch"])
        else:
            checkpoint = pilot["checkpoint"]
            expected = pilot["best_validation"]["macro_f1"]
            best_epoch = pilot["best_epoch"]
        summary_rows.append({"experiment": "finetune_layer4_postfinal",
                             "checkpoint": checkpoint,
                             "best_epoch": str(best_epoch),
                             "val_macro_f1": expected})
    target = device()
    prediction_rows = {row["image_id"]: {
        "image_id": row["image_id"], "true_class": int(row["class_id"]),
        "train_photo_overlap": int(row["image_id"] in overlap),
    } for row in val_rows}
    results = {}
    for entry in summary_rows:
        name = entry["experiment"]
        if name == "majority_class":
            continue
        payload = torch.load(output / entry["checkpoint"], map_location="cpu", weights_only=True)
        if (payload["class_mapping"] != mapping or payload["weights_identifier"] != WEIGHTS_ID
                or payload["best_epoch"] != int(entry["best_epoch"])
                or abs(float(entry["val_macro_f1"]) - payload["validation_macro_f1"]) > 1e-10):
            raise ValueError(f"Frozen checkpoint does not match: {name}")
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, mapping["class_count"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(target).eval()
        protocol = "full_amp" if name == "finetune_layer4_postfinal" else "cached_features"
        full, predictions = predict_model(model, "val", manifest, dataset_root(),
                                          mapping["class_count"], target,
                                          batch_size=BATCH_SIZE, protocol=protocol)
        if abs(full["macro_f1"] - float(entry["val_macro_f1"])) > 1e-6:
            raise ValueError(
                f"Recomputed validation result differs from frozen summary: {name}; "
                f"saved={entry['val_macro_f1']}, observed={full['macro_f1']}, "
                f"checkpoint_sha256={sha(output / entry['checkpoint'])}, "
                f"protocol={protocol}, batch_size={BATCH_SIZE}. "
                "Run python -m plantseg_cw2.analysis.diagnose_validation_consistency "
                "before continuing."
            )
        for row in predictions:
            prediction_rows[row["image_id"]][name] = row["pred_class"]
        clean_predictions = [row for row in predictions if row["image_id"] not in overlap]
        clean = scores([row["true_class"] for row in clean_predictions],
                       [row["pred_class"] for row in clean_predictions], mapping["class_count"])
        results[name] = {
            "all_validation": {key: full[key] for key in ("macro_f1", "accuracy", "balanced_accuracy")},
            "without_detected_train_overlap": {key: clean[key] for key in
                                                 ("macro_f1", "accuracy", "balanced_accuracy")},
        }
        print(f"{name}: full {full['macro_f1']:.4f}, without overlap {clean['macro_f1']:.4f}", flush=True)
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    table_path = output / "tables" / "validation_overlap_predictions.csv"
    columns = ["image_id", "true_class", "train_photo_overlap"] + list(results)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(prediction_rows[row["image_id"]] for row in val_rows)
    report = {
        "status": "interpretive sensitivity analysis; original selection unchanged",
        "exact_validation_train_overlap": len(exact),
        "near_validation_train_overlap": len(near),
        "overlap_union": len(overlap),
        "all_validation_count": len(val_rows),
        "without_overlap_count": len(val_rows) - len(overlap),
        "experiments": results,
        "predictions_sha256": sha(table_path),
        "test_evaluated": False,
    }
    (output / "validation_overlap_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
