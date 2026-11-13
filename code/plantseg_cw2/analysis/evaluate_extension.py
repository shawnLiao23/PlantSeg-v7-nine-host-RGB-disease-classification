"""Freeze and evaluate the post-final fine-tuning extension once.

This is explicitly separate from the original frozen test plan. Prior test
results already existed when this extension was designed, so this evaluation
must be labelled supplementary and cannot be presented as a blind final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models import resnet18

from ..dataset import dataset_root, load_manifest, result_root
from ..evaluate import predict_model, scores, write_rows
from .sensitivity_analysis import read_csv
from ..train import BATCH_SIZE, WEIGHTS_ID, device


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension_folder(output: Path) -> Path:
    return output / "exploratory_finetune"


def freeze(output: Path) -> dict:
    folder = extension_folder(output)
    plan_path = folder / "frozen_extension_plan.json"
    if plan_path.exists():
        raise FileExistsError("Extension plan already frozen")
    original = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    continued_path = output / "exploratory_finetune_continued" / "summary.json"
    continued = json.loads(continued_path.read_text(encoding="utf-8"))
    pilot_checkpoint = summary.get("checkpoint")
    if not pilot_checkpoint:
        raise ValueError("Pilot has no improved checkpoint to freeze")
    selected_checkpoint = continued["best_available_checkpoint"]
    if selected_checkpoint not in (pilot_checkpoint, continued.get("checkpoint")):
        raise ValueError("Continuation selected an unknown checkpoint")
    selected_scores = continued["best_validation"]
    if selected_scores["macro_f1"] < summary["best_validation"]["macro_f1"]:
        raise ValueError("Continuation best is worse than the pilot best")
    selected_epoch = (summary["best_epoch"] if selected_checkpoint == pilot_checkpoint
                      else continued["best_epoch"])
    checkpoint = output / selected_checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    if (payload["class_mapping"] != mapping or payload["weights_identifier"] != WEIGHTS_ID
            or payload["best_epoch"] != selected_epoch
            or abs(payload["validation_macro_f1"] - selected_scores["macro_f1"]) > 1e-10):
        raise ValueError("Checkpoint does not match validation-selected extension")
    for field, filename in (("manifest_sha256", "manifest.csv"),
                            ("class_mapping_sha256", "class_mapping.json"),
                            ("summary_sha256", "experiment_summary.csv")):
        if original[field] != sha(output / filename):
            raise ValueError(f"Original frozen input changed: {filename}")
    validation = json.loads((output / "validation_overlap_analysis.json").read_text(encoding="utf-8"))
    if abs(validation["experiments"]["finetune_layer4_postfinal"]["all_validation"]["macro_f1"]
           - selected_scores["macro_f1"]) > 1e-10:
        raise ValueError("Independent validation prediction check differs")
    plan = {
        "label": "supplementary post-final extension; prior test results were already visible",
        "selection_rule": "highest validation macro-F1 across fine-tuning pilot and continuation",
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_sha256": sha(checkpoint),
        "validation_macro_f1": selected_scores["macro_f1"],
        "pilot_best_epoch": summary["best_epoch"],
        "selected_stage": "pilot" if selected_checkpoint == pilot_checkpoint else "continuation",
        "selected_epoch": selected_epoch,
        "continued_epochs_without_improvement": continued["epochs_run"],
        "original_frozen_plan_sha256": sha(output / "frozen_plan.json"),
        "manifest_sha256": original["manifest_sha256"],
        "class_mapping_sha256": original["class_mapping_sha256"],
        "validation_analysis_sha256": sha(output / "validation_overlap_analysis.json"),
        "prediction_protocol": "full_amp",
        "prediction_batch_size": BATCH_SIZE,
        "prior_test_results_available": (output / "final_test_metrics.json").exists(),
        "test_used_to_select_this_checkpoint": False,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen supplementary extension: {selected_checkpoint} at validation macro-F1 "
          f"{plan['validation_macro_f1']:.4f}")
    return plan


def evaluate(output: Path) -> dict:
    folder = extension_folder(output)
    plan_path = folder / "frozen_extension_plan.json"
    test_path = folder / "supplementary_test_metrics.json"
    if test_path.exists():
        raise FileExistsError("Supplementary test has already been evaluated")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for field, path in (("checkpoint_sha256", output / plan["selected_checkpoint"]),
                        ("manifest_sha256", output / "manifest.csv"),
                        ("class_mapping_sha256", output / "class_mapping.json"),
                        ("validation_analysis_sha256", output / "validation_overlap_analysis.json"),
                        ("original_frozen_plan_sha256", output / "frozen_plan.json")):
        if sha(path) != plan[field]:
            raise ValueError(f"Frozen extension input changed: {path.name}")
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    payload = torch.load(output / plan["selected_checkpoint"], map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, mapping["class_count"])
    model.load_state_dict(payload["model_state_dict"])
    target = device()
    model.to(target).eval()
    full, predictions = predict_model(model, "test", load_manifest(output / "manifest.csv"),
                                      dataset_root(), mapping["class_count"], target,
                                      protocol=plan.get("prediction_protocol", "feature_fp32"),
                                      batch_size=plan.get("prediction_batch_size", 64))
    if len(predictions) != 867:
        raise ValueError("Expected all 867 official test images")
    prediction_path = folder / "supplementary_test_predictions.csv"
    write_rows(prediction_path, predictions,
               ["image_id", "image_path", "true_class", "pred_class", "confidence"])
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    exact = set()
    for group in audit["exact_duplicate_groups"]:
        members = group["members"]
        if any(member["split"] == "train" for member in members):
            exact.update(member["image_id"] for member in members if member["split"] == "test")
    near = set()
    review_path = output / "tables" / "near_duplicate_review.csv"
    if sha(review_path) != audit["near_duplicate_review_sha256"]:
        raise ValueError("Audited duplicate review changed")
    for pair in read_csv(review_path):
        if pair["decision"] != "same_photo":
            continue
        if pair["split_a"] == "train" and pair["split_b"] == "test":
            near.add(pair["image_b"])
        if pair["split_b"] == "train" and pair["split_a"] == "test":
            near.add(pair["image_a"])
    overlap = exact | near
    clean_rows = [row for row in predictions if row["image_id"] not in overlap]
    clean = scores([row["true_class"] for row in clean_rows],
                   [row["pred_class"] for row in clean_rows], mapping["class_count"])
    output_metrics = {
        "label": plan["label"],
        "frozen_extension_plan_sha256": sha(plan_path),
        "prior_test_results_available": plan["prior_test_results_available"],
        "test_used_to_select_this_checkpoint": False,
        "all_official_test": {key: full[key] for key in
                              ("macro_f1", "accuracy", "balanced_accuracy", "per_class", "confusion_matrix")},
        "without_detected_train_photo_overlap": {key: clean[key] for key in
                                                  ("macro_f1", "accuracy", "balanced_accuracy")},
        "overlap_union": len(overlap),
        "remaining_count": len(clean_rows),
        "predictions_sha256": sha(prediction_path),
    }
    test_path.write_text(json.dumps(output_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Supplementary test macro-F1 {full['macro_f1']:.4f}; "
          f"{len(clean_rows)} without detected train overlap: {clean['macro_f1']:.4f}")
    return output_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--test", action="store_true")
    args = parser.parse_args()
    output = result_root()
    freeze(output) if args.freeze else evaluate(output)
