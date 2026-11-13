"""Compare the frozen original model with the supplementary fine-tuned model.

Uses only saved, paired predictions. The bootstrap interval is descriptive:
the extension was designed after original test results were already visible,
and original split overlaps can reduce effective sample independence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..dataset import result_root


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_f1(true: np.ndarray, pred: np.ndarray, count: int) -> float:
    matrix = np.bincount(true * count + pred, minlength=count**2).reshape(count, count)
    correct = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * correct, denominator, out=np.zeros(count, dtype=float),
                           where=denominator > 0).mean())


def plot_extension(output: Path, summary: dict) -> None:
    folder = output / "exploratory_finetune"
    pilot_summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with (output / "experiment_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        weighted_validation = float(next(row for row in csv.DictReader(handle)
                                         if row["experiment"] == "weighted_resnet18")["val_macro_f1"])
    pilot = read_csv(folder / "validation_history.csv")
    continued = read_csv(output / "exploratory_finetune_continued" / "validation_history.csv")
    history = pilot + continued
    plan = json.loads((folder / "frozen_extension_plan.json").read_text(encoding="utf-8"))
    selected_x = (pilot_summary["best_epoch"] if plan.get("selected_stage", "pilot") == "pilot"
                  else len(pilot) + plan["selected_epoch"])
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    axes[0].plot(epochs, [float(r["train_loss"]) for r in history], marker="o", ms=3,
                 label="Training loss")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in history], marker="o", ms=3,
                 label="Validation loss")
    axes[0].set(ylabel="Cross-entropy", xlabel="Epoch", title="Fine-tuning loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epochs, [float(r["val_macro_f1"]) for r in history], marker="o", ms=4,
                 color="#157a75", label="Fine-tuning")
    axes[1].axhline(weighted_validation, ls="--", color="#536475", lw=1,
                    label="Original class weighting")
    axes[1].scatter([selected_x], [plan["validation_macro_f1"]], color="#bd4c35", zorder=5,
                    label="Selected checkpoint")
    axes[1].set(ylabel="Macro-F1", xlabel="Epoch", title="Validation selection")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axvline(len(pilot) + .5, color="#b5bec5", ls=":", lw=1)
        ax.grid(alpha=.2)
    fig.suptitle("The continuation restarted Adam from the pilot's best checkpoint", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "figures" / "finetune_curves.png", dpi=170)
    plt.close(fig)

    names = ["Macro-F1", "Accuracy", "Balanced acc."]
    keys = ["macro_f1", "accuracy", "balanced_accuracy"]
    original = [summary["metrics"][k]["original"] for k in keys]
    fine = [summary["metrics"][k]["finetune"] for k in keys]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(x - .18, original, width=.36, label="Original class weighting", color="#536475")
    ax.bar(x + .18, fine, width=.36, label="Supplementary fine-tuning", color="#157a75")
    ax.set(xticks=x, xticklabels=names, ylabel="Score", ylim=(0, .80),
           title="Official test split: all 867 images")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.18)
    for i, value in enumerate(original):
        ax.text(i - .18, value + .012, f"{value:.3f}", ha="center", fontsize=8)
    for i, value in enumerate(fine):
        ax.text(i + .18, value + .012, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "extension_comparison.png", dpi=170)
    plt.close(fig)


def main(output_dir: Path | None = None, bootstrap_replicates: int = 3000) -> dict:
    output = result_root(output_dir)
    folder = output / "exploratory_finetune"
    original = json.loads((output / "final_test_metrics.json").read_text(encoding="utf-8"))
    extension = json.loads((folder / "supplementary_test_metrics.json").read_text(encoding="utf-8"))
    plan_path = folder / "frozen_extension_plan.json"
    if extension["frozen_extension_plan_sha256"] != sha(plan_path):
        raise ValueError("Supplementary test is detached from its frozen plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original_path = output / "tables" / "test_predictions.csv"
    extension_path = folder / "supplementary_test_predictions.csv"
    if extension["predictions_sha256"] != sha(extension_path):
        raise ValueError("Supplementary predictions changed")
    first = {row["image_id"]: row for row in read_csv(original_path)}
    second = read_csv(extension_path)
    if len(first) != 867 or len(second) != 867 or set(first) != {r["image_id"] for r in second}:
        raise ValueError("The paired test predictions do not cover the same 867 images")
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    count = mapping["class_count"]
    true = np.array([int(row["true_class"]) for row in second])
    earlier = np.array([int(first[row["image_id"]]["pred_class"]) for row in second])
    later = np.array([int(row["pred_class"]) for row in second])
    if any(int(first[row["image_id"]]["true_class"]) != int(row["true_class"])
           for row in second):
        raise ValueError("The two test prediction tables use different labels")
    original_metrics = original["experiments"][original["selected_model"]]
    extension_metrics = extension["all_official_test"]
    if (abs(macro_f1(true, earlier, count) - original_metrics["macro_f1"]) > 1e-10
            or abs(macro_f1(true, later, count) - extension_metrics["macro_f1"]) > 1e-10):
        raise ValueError("Paired prediction scores differ from saved test metrics")
    by_class = [np.where(true == class_id)[0] for class_id in range(count)]
    generator = np.random.default_rng(2026)
    differences = np.empty(bootstrap_replicates)
    for index in range(bootstrap_replicates):
        sample = np.concatenate([generator.choice(ids, size=len(ids), replace=True)
                                 for ids in by_class])
        differences[index] = (macro_f1(true[sample], later[sample], count)
                              - macro_f1(true[sample], earlier[sample], count))
    original_per_class = original_metrics["per_class"]
    new_per_class = extension_metrics["per_class"]
    comparison_rows = []
    for class_id in range(count):
        old = original_per_class[class_id]
        new = new_per_class[class_id]
        if old["support"] != new["support"]:
            raise ValueError("Per-class support differs between models")
        comparison_rows.append({
            "class_id": class_id,
            "class_name": mapping["classes"][class_id]["class_name"],
            "test_support": old["support"],
            "original_precision": old["precision"], "finetune_precision": new["precision"],
            "original_recall": old["recall"], "finetune_recall": new["recall"],
            "original_f1": old["f1"], "finetune_f1": new["f1"],
            "f1_change": new["f1"] - old["f1"],
        })
    with (folder / "per_class_test_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    matrix = extension_metrics["confusion_matrix"]
    with (folder / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class_id"] + list(range(count)))
        for class_id, row in enumerate(matrix):
            writer.writerow([class_id] + row)
    summary = {
        "label": "supplementary post-final comparison; extension was not blind to earlier test results",
        "test_split_unchanged": True,
        "sample_count": len(true),
        "original_model": original["selected_model"],
        "finetune_model": (f"layer4 plus head, {plan.get('selected_stage', 'pilot')} "
                           f"epoch {plan.get('selected_epoch', plan['pilot_best_epoch'])} checkpoint"),
        "metrics": {
            name: {
                "original": original_metrics[name],
                "finetune": extension_metrics[name],
                "change": extension_metrics[name] - original_metrics[name],
            } for name in ("macro_f1", "accuracy", "balanced_accuracy")
        },
        "finetune_only_correct": int(np.sum((later == true) & (earlier != true))),
        "original_only_correct": int(np.sum((earlier == true) & (later != true))),
        "paired_stratified_bootstrap": {
            "seed": 2026,
            "replicates": bootstrap_replicates,
            "macro_f1_change_95_percent_interval": np.quantile(differences, [.025, .975]).tolist(),
            "interpretation": "Descriptive only: prior test visibility and duplicate overlap limit inference",
        },
        "largest_class_gains": [row["class_name"] for row in
                                sorted(comparison_rows, key=lambda r: r["f1_change"], reverse=True)[:5]],
        "largest_class_losses": [row["class_name"] for row in
                                 sorted(comparison_rows, key=lambda r: r["f1_change"])[:5]],
        "original_predictions_sha256": sha(original_path),
        "finetune_predictions_sha256": sha(extension_path),
    }
    (folder / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_extension(output, summary)
    print(f"Paired test macro-F1 change {summary['metrics']['macro_f1']['change']:.4f}; "
          f"95% descriptive bootstrap interval "
          f"{summary['paired_stratified_bootstrap']['macro_f1_change_95_percent_interval']}")
    return summary


if __name__ == "__main__":
    main()
