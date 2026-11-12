"""Validation-only, post-final pilot: fine-tune ResNet-18 layer4 and head.

The original frozen plan and test outputs are deliberately untouched. This
script reads only training and validation images and writes under a separate
exploratory directory. Its results cannot replace the original final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from ..dataset import PlantDiseaseDataset, dataset_root, load_manifest, make_transforms, result_root
from ..evaluate import scores
from ..train import BATCH_SIZE, SEED, WEIGHTS_ID, device, seed_everything


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(epochs: int = 10, patience: int = 4, output_dir: Path | None = None,
         continue_from_best_pilot: bool = False) -> dict:
    output = result_root(output_dir)
    plan = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
    for key, name in (("manifest_sha256", "manifest.csv"),
                      ("class_mapping_sha256", "class_mapping.json"),
                      ("summary_sha256", "experiment_summary.csv")):
        if plan[key] != digest(output / name):
            raise ValueError(f"Frozen input changed: {name}")
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    manifest = load_manifest(output / "manifest.csv")
    baseline_path = output / "checkpoints" / "baseline_resnet18_best.pth"
    original_baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    if original_baseline["class_mapping"] != mapping or original_baseline["weights_identifier"] != WEIGHTS_ID:
        raise ValueError("Baseline checkpoint does not match frozen data or pretrained weights")
    with (output / "experiment_summary.csv").open(encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["experiment"] == "baseline_resnet18")
    if (int(row["best_epoch"]) != original_baseline["best_epoch"] or
            abs(float(row["val_macro_f1"]) - original_baseline["validation_macro_f1"]) > 1e-10):
        raise ValueError("Baseline checkpoint differs from frozen validation summary")
    initial_path = baseline_path
    initial_scores = {"macro_f1": float(row["val_macro_f1"]),
                      "accuracy": float(row["val_accuracy"]),
                      "balanced_accuracy": float(row["val_balanced_accuracy"])}
    if continue_from_best_pilot:
        pilot_dir = output / "exploratory_finetune"
        pilot = json.loads((pilot_dir / "summary.json").read_text(encoding="utf-8"))
        if pilot["best_epoch"] < 1 or not pilot["checkpoint"]:
            raise ValueError("Pilot has no improved checkpoint to continue")
        initial_path = output / pilot["checkpoint"]
        initial_scores = pilot["best_validation"]
    baseline = torch.load(initial_path, map_location="cpu", weights_only=True)
    if baseline["class_mapping"] != mapping or baseline["weights_identifier"] != WEIGHTS_ID:
        raise ValueError("Continuation checkpoint does not match frozen data")

    seed_everything(SEED)
    target = device()
    source = dataset_root()
    train_set = PlantDiseaseDataset(manifest, source, "train", make_transforms(False))
    val_set = PlantDiseaseDataset(manifest, source, "val", make_transforms(False))
    if len(train_set) != 3007 or len(val_set) != 475:
        raise ValueError("Unexpected original training/validation split size")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=target.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=target.type == "cuda")
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, mapping["class_count"])
    model.load_state_dict(baseline["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    model.to(target)
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ])
    criterion = nn.CrossEntropyLoss()
    amp = target.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    folder = output / ("exploratory_finetune_continued" if continue_from_best_pilot
                       else "exploratory_finetune")
    folder.mkdir(parents=True, exist_ok=True)
    history_path = folder / "validation_history.csv"
    checkpoint_path = folder / "best_validation.pth"
    config = {
        "status": "post-final validation-only exploration; do not use for frozen final test",
        "stage": "continued from best pilot with freshly initialized Adam optimizer"
                 if continue_from_best_pilot else "initial fine-tuning pilot",
        "initial_checkpoint": str(initial_path.relative_to(output)).replace("\\", "/"),
        "initial_checkpoint_sha256": digest(initial_path),
        "initial_validation": initial_scores,
        "manifest_sha256": plan["manifest_sha256"],
        "class_mapping_sha256": plan["class_mapping_sha256"],
        "seed": SEED, "max_epochs": epochs, "patience": patience,
        "train_size": len(train_set), "validation_size": len(val_set),
        "batch_size": BATCH_SIZE, "optimizer": "Adam",
        "layer4_lr": 1e-5, "head_lr": 1e-4, "loss": "plain cross entropy",
        "augmentation": "none; same deterministic train/validation transform as original baseline",
        "trainable": "layer4 and fc", "device": str(target), "amp": amp,
        "checkpoint_rule": "highest validation macro-F1; first epoch wins exact ties",
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_f1 = float(initial_scores["macro_f1"])
    best_epoch = 0
    best_scores = initial_scores
    history = []
    no_gain = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # Frozen early blocks keep their ImageNet batch-normalization statistics.
        for module in (model.bn1, *[m for block in (model.layer1, model.layer2, model.layer3)
                                      for m in block.modules() if isinstance(m, nn.BatchNorm2d)]):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images = images.to(target, non_blocking=True)
            labels = labels.to(target, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * len(labels)
            train_correct += int((logits.argmax(1) == labels).sum().item())
        model.eval()
        val_true, val_pred = [], []
        val_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(target, non_blocking=True)
                labels = labels.to(target, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                val_loss += float(loss.item()) * len(labels)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
        measured = scores(val_true, val_pred, mapping["class_count"])
        record = {
            "epoch": epoch, "train_loss": train_loss / len(train_set),
            "train_accuracy": train_correct / len(train_set),
            "val_loss": val_loss / len(val_set),
            "val_macro_f1": measured["macro_f1"],
            "val_accuracy": measured["accuracy"],
            "val_balanced_accuracy": measured["balanced_accuracy"],
            "epoch_time_sec": time.perf_counter() - epoch_start,
        }
        history.append(record)
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        if measured["macro_f1"] > best_f1 + 1e-12:
            best_f1 = measured["macro_f1"]
            best_epoch = epoch
            best_scores = {key: measured[key] for key in
                           ("macro_f1", "accuracy", "balanced_accuracy")}
            no_gain = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "class_mapping": mapping, "architecture": "resnet18",
                        "best_epoch": epoch, "validation_macro_f1": best_f1,
                        "pretrained_weights_sha256": baseline["pretrained_weights_sha256"],
                        "weights_identifier": WEIGHTS_ID,
                        "preprocessing": baseline["preprocessing"],
                        "training_config": config}, checkpoint_path)
        else:
            no_gain += 1
        print(f"epoch {epoch:02d}/{epochs} train_loss={record['train_loss']:.4f} "
              f"val_macro_f1={measured['macro_f1']:.4f} best={best_f1:.4f} "
              f"time={record['epoch_time_sec']:.1f}s", flush=True)
        if no_gain >= patience:
            print(f"Stopped after {patience} epochs without validation macro-F1 gain", flush=True)
            break
    summary = {"status": config["status"], "stage": config["stage"],
               "baseline_validation_macro_f1": float(row["val_macro_f1"]),
               "initial_validation_macro_f1": float(initial_scores["macro_f1"]),
               "best_epoch": best_epoch, "best_validation": best_scores,
               "epochs_run": len(history), "training_time_sec": time.perf_counter() - start,
               "checkpoint": str(checkpoint_path.relative_to(output)).replace("\\", "/")
               if best_epoch else None,
               "best_available_checkpoint": str((checkpoint_path if best_epoch else initial_path)
                                                 .relative_to(output)).replace("\\", "/"),
               "test_evaluated": False}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--continue-from-best-pilot", action="store_true")
    args = parser.parse_args()
    main(args.epochs, args.patience, continue_from_best_pilot=args.continue_from_best_pilot)
