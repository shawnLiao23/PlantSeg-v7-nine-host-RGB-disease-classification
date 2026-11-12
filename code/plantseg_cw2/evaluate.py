"""Metrics and figures for the fixed PlantSeg classification protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PlantDiseaseDataset, dataset_root, load_manifest, make_transforms, result_root


def scores(y_true, y_pred, class_count: int) -> dict:
    """Compute per-class scores over the same fixed class list in every run."""
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if len(true) != len(pred) or len(true) == 0:
        raise ValueError("Evaluation needs matching nonempty labels and predictions")
    if (true < 0).any() or (true >= class_count).any() or (pred < 0).any() or (pred >= class_count).any():
        raise ValueError("Class id outside the fixed class mapping")
    matrix = np.bincount(true * class_count + pred, minlength=class_count**2).reshape(class_count, class_count)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(class_count, dtype=float), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(class_count, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(class_count, dtype=float), where=(precision + recall) != 0)
    return {
        "accuracy": float(correct.sum() / len(true)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {"class_id": i, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i in range(class_count)
        ],
        "confusion_matrix": matrix.tolist(),
        "class_count": class_count,
        "undefined_metric_rule": "Precision, recall and F1 are 0 when their denominator is 0; all mapped classes are included in macro averages.",
    }


def majority_prediction(manifest: list[dict], split: str, class_count: int):
    from .dataset import is_included
    training = [r for r in manifest if r["split"] == "train" and is_included(r)]
    target = [r for r in manifest if r["split"] == split and is_included(r)]
    counts = np.bincount([r["class_id"] for r in training], minlength=class_count)
    majority_id = int(np.argmax(counts))
    result = scores([r["class_id"] for r in target], [majority_id] * len(target), class_count)
    result["majority_class_id"] = majority_id
    return result


def predict_model(model, split: str, manifest: list[dict], data_dir: Path, class_count: int,
                  device: torch.device, batch_size: int = 64) -> tuple[dict, list[dict]]:
    ds = PlantDiseaseDataset(manifest, data_dir, split, make_transforms(False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    y_true, y_pred, confidence = [], [], []
    model.eval()
    feature_model = torch.nn.Sequential(*list(model.children())[:-1]).eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                features = feature_model(images)
            logits = model.fc(features.flatten(1).float())
            probabilities = torch.softmax(logits.float(), dim=1)
            conf, prediction = probabilities.max(dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(prediction.cpu().numpy().tolist())
            confidence.extend(conf.cpu().numpy().tolist())
    result = scores(y_true, y_pred, class_count)
    predictions = [
        {"image_id": row["image_id"], "image_path": row["image_path"],
         "true_class": int(row["class_id"]), "pred_class": int(pred),
         "confidence": float(conf)}
        for row, pred, conf in zip(ds.rows, y_pred, confidence)
    ]
    return result, predictions


def write_rows(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not columns:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(output: Path, experiments: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for entry in experiments:
        path = output / "logs" / f"{entry['experiment']}_history.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epoch = [int(r["epoch"]) for r in rows]
        axes[0].plot(epoch, [float(r["train_loss"]) for r in rows], label=entry["experiment"])
        axes[1].plot(epoch, [float(r["val_loss"]) for r in rows], label=entry["experiment"])
        axes[2].plot(epoch, [float(r["val_macro_f1"]) for r in rows], label=entry["experiment"])
        best = int(entry["best_epoch"])
        axes[2].scatter([best], [float(rows[best - 1]["val_macro_f1"])], s=24)
    for ax, title, ylabel in zip(axes, ["Training loss", "Validation loss", "Validation macro-F1"],
                                 ["Cross-entropy", "Cross-entropy", "Macro-F1"]):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "figures" / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_model_comparison(output: Path, summary: list[dict], test: dict | None = None) -> None:
    rows = summary
    names = [r["experiment"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.1))
    for offset, key, label in [(-width, "val_macro_f1", "Macro-F1"), (0, "val_accuracy", "Accuracy"),
                               (width, "val_balanced_accuracy", "Balanced accuracy")]:
        ax.bar(x + offset, [float(r[key]) for r in rows], width, label=label)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Validation score",
           title="Controlled experiment comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "figures" / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_confusion(output: Path, metrics: dict) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.035)
    ax.set(title="Selected model: test confusion matrix", xlabel="Predicted class ID", ylabel="True class ID")
    ax.set_xticks(np.arange(len(matrix)))
    ax.set_yticks(np.arange(len(matrix)))
    ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(output / "figures" / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_examples(output: Path, predictions: list[dict], mapping: dict, data_dir: Path) -> None:
    from PIL import Image
    correct = [r for r in predictions if r["true_class"] == r["pred_class"]]
    incorrect = [r for r in predictions if r["true_class"] != r["pred_class"]]
    chosen = correct[:3] + sorted(incorrect, key=lambda r: -r["confidence"])[:3]
    if not chosen:
        return
    names = [x["class_name"] for x in mapping["classes"]]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.3))
    for ax, row in zip(axes.flat, chosen):
        with Image.open(data_dir / row["image_path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.axis("off")
        ax.set_title(f"True: {names[row['true_class']]}\nPred: {names[row['pred_class']]} ({row['confidence']:.2f})", fontsize=8)
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Test examples (softmax score is not calibrated probability)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "figures" / "prediction_examples.png", dpi=180)
    plt.close(fig)


def make_final_plots(output_dir: str | Path | None = None) -> None:
    output = result_root(output_dir)
    with (output / "experiment_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    plot_training_curves(output, summary)
    plot_model_comparison(output, summary)
    test_path = output / "final_test_metrics.json"
    if test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        chosen = test["selected_model"]
        plot_confusion(output, test["experiments"][chosen])
        mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
        pred_path = output / "tables" / "test_predictions.csv"
        if pred_path.exists():
            with pred_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["true_class"] = int(row["true_class"])
                row["pred_class"] = int(row["pred_class"])
                row["confidence"] = float(row["confidence"])
            plot_examples(output, rows, mapping, dataset_root())
