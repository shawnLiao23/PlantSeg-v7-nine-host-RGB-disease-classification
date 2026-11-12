"""Post-test sensitivity check using the frozen model's saved predictions.

This does not select a model or alter the supplied PlantSeg split. It reports
how the already-frozen selected model scores after omitting test photographs
that the audit linked to a training photograph.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
RESULT = ROOT / "result"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(rows: list[dict], class_count: int) -> dict:
    if not rows:
        raise ValueError("Sensitivity subset is empty")
    tp = [0] * class_count
    fp = [0] * class_count
    fn = [0] * class_count
    support = [0] * class_count
    for row in rows:
        true = int(row["true_class"])
        predicted = int(row["pred_class"])
        if not (0 <= true < class_count and 0 <= predicted < class_count):
            raise ValueError("Prediction has an invalid class ID")
        support[true] += 1
        if true == predicted:
            tp[true] += 1
        else:
            fn[true] += 1
            fp[predicted] += 1
    f1 = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])
          if 2 * tp[i] + fp[i] + fn[i] else 0.0
          for i in range(class_count)]
    recall = [tp[i] / support[i] if support[i] else 0.0
              for i in range(class_count)]
    return {
        "count": len(rows),
        "classes_with_support": sum(bool(n) for n in support),
        "macro_f1": sum(f1) / class_count,
        "accuracy": sum(tp) / len(rows),
        "balanced_accuracy": sum(recall) / class_count,
    }


def main(result_dir: Path = RESULT) -> dict:
    plan = json.loads((result_dir / "frozen_plan.json").read_text(encoding="utf-8"))
    for field, filename in (("manifest_sha256", "manifest.csv"),
                            ("class_mapping_sha256", "class_mapping.json"),
                            ("summary_sha256", "experiment_summary.csv")):
        if sha256_file(result_dir / filename) != plan[field]:
            raise ValueError(f"Frozen input changed: {filename}")
    final = json.loads((result_dir / "final_test_metrics.json").read_text(encoding="utf-8"))
    if final["selected_model"] != plan["selected_model"]:
        raise ValueError("Selected model differs from the frozen plan")
    audit = json.loads((result_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    mapping = json.loads((result_dir / "class_mapping.json").read_text(encoding="utf-8"))
    predictions_path = result_dir / "tables" / "test_predictions.csv"
    review_path = result_dir / "tables" / "near_duplicate_review.csv"
    if sha256_file(review_path) != audit["near_duplicate_review_sha256"]:
        raise ValueError("Visual duplicate review differs from audited review")
    predictions = read_csv(predictions_path)
    if len(predictions) != final["test_set_size"]:
        raise ValueError("Saved predictions do not cover the final test set")
    exact_test_ids = set()
    for group in audit["exact_duplicate_groups"]:
        members = group["members"]
        if any(member["split"] == "train" for member in members):
            exact_test_ids.update(member["image_id"] for member in members
                                  if member["split"] == "test")
    near_test_ids = set()
    for pair in read_csv(review_path):
        if pair["decision"] != "same_photo":
            continue
        for a, b in (("a", "b"), ("b", "a")):
            if pair[f"split_{a}"] == "train" and pair[f"split_{b}"] == "test":
                near_test_ids.add(pair[f"image_{b}"])
    overlap_ids = exact_test_ids | near_test_ids
    observed_ids = [row["image_id"] for row in predictions]
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("Saved predictions repeat an image ID")
    if not overlap_ids.issubset(set(observed_ids)):
        raise ValueError("Audited test overlap is missing from saved predictions")
    class_count = mapping["class_count"]
    full = scores(predictions, class_count)
    recorded = final["experiments"][plan["selected_model"]]
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        if abs(full[metric] - recorded[metric]) > 1e-10:
            raise ValueError(f"Saved predictions disagree with final {metric}")
    clean = scores([row for row in predictions if row["image_id"] not in overlap_ids],
                   class_count)
    output = {
        "purpose": "Post-hoc sensitivity analysis only; no model selection or retraining",
        "selected_model": plan["selected_model"],
        "definition": "Omit test images with an exact or visually confirmed same-photo match in training",
        "official_split_unchanged": True,
        "exact_test_train_overlap": len(exact_test_ids),
        "near_test_train_overlap": len(near_test_ids),
        "overlap_union": len(overlap_ids),
        "all_test": full,
        "without_detected_train_overlap": clean,
        "remaining_limitation": "Undetected overlaps may remain; validation also has audited overlaps",
        "predictions_sha256": sha256_file(predictions_path),
        "review_sha256": sha256_file(review_path),
    }
    destination = result_dir / "overlap_sensitivity.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {destination}: {full['count']} full, {clean['count']} without detected train overlap")
    return output


if __name__ == "__main__":
    main()
