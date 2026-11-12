"""Train the fixed ResNet-18 head experiments using validation macro-F1."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

# Must be set before the first CUDA BLAS operation for repeatable head training.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet18

from .dataset import PlantDiseaseDataset, dataset_root, is_included, load_manifest, make_transforms, result_root
from .evaluate import majority_prediction, scores, write_rows


SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
EXPERIMENTS = (
    ("baseline_resnet18", False, False, "frozen ResNet-18, plain cross-entropy"),
    ("augmentation_resnet18", True, False, "training augmentation only"),
    ("weighted_resnet18", False, True, "inverse-square-root class-weighted loss only"),
)
SUMMARY_COLUMNS = ["experiment", "best_epoch", "val_macro_f1", "val_accuracy",
                   "val_balanced_accuracy", "train_time_sec", "checkpoint", "change"]
HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
                   "val_macro_f1", "epoch_time_sec"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        memory = MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            info["ram_bytes"] = memory.ullTotalPhys
    return info


def _weights_path(output: Path) -> Path:
    path = output / "checkpoints" / "imagenet_resnet18_f37072fd.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"Official pretrained weights missing: {path}. Download from "
            "https://download.pytorch.org/models/resnet18-f37072fd.pth and place at that path."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"Official pretrained weight SHA-256 mismatch: {digest}")
    return path


def make_feature_extractor(output: Path, target_device: torch.device) -> nn.Module:
    model = resnet18(weights=None)
    state = torch.load(_weights_path(output), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model.to(target_device)


def extract_features(extractor: nn.Module, dataset: PlantDiseaseDataset,
                     target_device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=target_device.type == "cuda")
    features, labels = [], []
    extractor.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(target_device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=target_device.type == "cuda"):
                values = extractor(images)
            features.append(values.float())
            labels.append(targets.to(target_device, non_blocking=True))
    return torch.cat(features), torch.cat(labels)


def _class_weights(manifest: list[dict], class_count: int, target_device: torch.device) -> tuple[torch.Tensor, list[int]]:
    counts = np.bincount([r["class_id"] for r in manifest if r["split"] == "train" and is_included(r)],
                         minlength=class_count)
    if (counts == 0).any():
        raise ValueError("A mapped class has no training examples")
    weights = 1 / np.sqrt(counts.astype(float))
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32, device=target_device), counts.tolist()


def _train_head_epoch(head: nn.Linear, features: torch.Tensor, labels: torch.Tensor,
                      criterion: nn.Module, optimizer: torch.optim.Optimizer,
                      batch_size: int, epoch: int) -> tuple[float, float]:
    head.train()
    generator = torch.Generator(device=features.device).manual_seed(SEED + epoch)
    order = torch.randperm(len(labels), generator=generator, device=features.device)
    total_loss, correct = 0.0, 0
    for ids in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features[ids])
        loss = criterion(logits, labels[ids])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(ids)
        correct += int((logits.argmax(dim=1) == labels[ids]).sum().item())
    return total_loss / len(labels), correct / len(labels)


def _validate_head(head: nn.Linear, features: torch.Tensor, labels: torch.Tensor,
                   class_count: int) -> tuple[float, dict]:
    head.eval()
    with torch.inference_mode():
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, labels).item()
        prediction = logits.argmax(dim=1).cpu().numpy()
    result = scores(labels.cpu().numpy(), prediction, class_count)
    return float(loss), result


def _checkpoint(output: Path, name: str, head: nn.Linear, extractor: nn.Module,
                class_mapping: dict, best_epoch: int, val_scores: dict, config: dict) -> Path:
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(class_mapping["classes"]))
    model.load_state_dict({**extractor.state_dict(),
                           "fc.weight": head.weight.detach().cpu(),
                           "fc.bias": head.bias.detach().cpu()})
    path = output / "checkpoints" / f"{name}_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(), "architecture": "resnet18",
        "class_mapping": class_mapping, "best_epoch": best_epoch,
        "validation_macro_f1": val_scores["macro_f1"], "seed": SEED,
        "weights_identifier": WEIGHTS_ID, "pretrained_weights_sha256": WEIGHTS_SHA256,
        "preprocessing": {"resize": 256, "center_crop": 224,
                          "normalization": "ImageNet IMAGENET1K_V1 mean/std"},
        "training_config": config,
    }, path)
    return path


def train_experiments(output_dir: str | Path | None = None) -> list[dict]:
    output = result_root(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "logs", "tables", "figures"):
        (output / folder).mkdir(exist_ok=True)
    manifest = load_manifest(output / "manifest.csv")
    class_mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    class_count = len(class_mapping["classes"])
    fingerprint = {
        "manifest_sha256": hashlib.sha256((output / "manifest.csv").read_bytes()).hexdigest(),
        "class_mapping_sha256": hashlib.sha256((output / "class_mapping.json").read_bytes()).hexdigest(),
    }
    fingerprint_path = output / "training_data_fingerprint.json"
    if (output / "experiment_summary.csv").exists():
        if not fingerprint_path.exists() or json.loads(fingerprint_path.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("Data manifest or class mapping changed since training began; remove stale generated training artifacts before starting a new run")
    else:
        fingerprint_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    source = dataset_root()
    seed_everything()
    target_device = device()
    config = {
        "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "baseline_lr": LEARNING_RATE, "optimizer": "Adam", "weights_identifier": WEIGHTS_ID,
        "weights_sha256": WEIGHTS_SHA256, "image_size": 224,
        "validation_transform": "Resize(256), CenterCrop(224), ToTensor, ImageNet V1 Normalize",
        "augmentation": "RandomHorizontalFlip, RandomRotation(10), ColorJitter(0.1,0.1,0.1) on training images only",
        "class_weight_formula": "clip((1/sqrt(training class count))/mean(1/sqrt(training class count)), 0.25, 4.0)",
        "backbone": "frozen ResNet-18; cached deterministic features for basic and weighted experiments",
        "device": str(target_device), "num_workers": 0,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "split_policy": "Original supplied train/val/test partitions; all images retained",
        "duplicate_exclusions": "None (duplicates audited and reported)",
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "environment_info.json").write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
    summary_path = output / "experiment_summary.csv"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            summary = list(csv.DictReader(handle))
    else:
        summary = []
    complete = {r["experiment"] for r in summary}
    if "majority_class" not in complete:
        majority = majority_prediction(manifest, "val", class_count)
        summary.append({"experiment": "majority_class", "best_epoch": 0,
                        "val_macro_f1": majority["macro_f1"], "val_accuracy": majority["accuracy"],
                        "val_balanced_accuracy": majority["balanced_accuracy"],
                        "train_time_sec": 0.0, "checkpoint": "", "change": "training-set majority class"})
        write_rows(summary_path, summary, SUMMARY_COLUMNS)
    extractor = make_feature_extractor(output, target_device)
    val_dataset = PlantDiseaseDataset(manifest, source, "val", make_transforms(False))
    val_features, val_labels = extract_features(extractor, val_dataset, target_device, BATCH_SIZE)
    train_dataset = PlantDiseaseDataset(manifest, source, "train", make_transforms(False))
    class_weights, counts = _class_weights(manifest, class_count, target_device)
    write_rows(output / "tables" / "training_class_weights.csv", [
        {"class_id": i, "class_name": class_mapping["classes"][i]["class_name"],
         "training_count": counts[i], "weight": float(class_weights[i].item())}
        for i in range(class_count)
    ])

    for name, augmented, weighted, change in EXPERIMENTS:
        if name in complete and (output / "checkpoints" / f"{name}_best.pth").exists():
            print(f"Skipping completed experiment: {name}", flush=True)
            continue
        print(f"Starting {name} on {target_device}", flush=True)
        seed_everything()
        head = nn.Linear(512, class_count).to(target_device)
        optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(weight=class_weights if weighted else None)
        deterministic_features = None
        history = []
        best_f1, best_epoch, best_metrics = -1.0, 0, None
        start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            if augmented:
                augmented_dataset = PlantDiseaseDataset(manifest, source, "train", make_transforms(True))
                train_features, train_labels = extract_features(extractor, augmented_dataset, target_device, BATCH_SIZE)
            else:
                if deterministic_features is None:
                    deterministic_features = extract_features(extractor, train_dataset, target_device, BATCH_SIZE)
                train_features, train_labels = deterministic_features
            train_loss, train_acc = _train_head_epoch(head, train_features, train_labels, criterion,
                                                      optimizer, BATCH_SIZE, epoch)
            val_loss, val_scores = _validate_head(head, val_features, val_labels, class_count)
            epoch_time = time.perf_counter() - epoch_start
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "train_accuracy": train_acc, "val_accuracy": val_scores["accuracy"],
                            "val_macro_f1": val_scores["macro_f1"], "epoch_time_sec": epoch_time})
            write_rows(output / "logs" / f"{name}_history.csv", history, HISTORY_COLUMNS)
            if val_scores["macro_f1"] > best_f1 + 1e-12:
                best_f1, best_epoch, best_metrics = val_scores["macro_f1"], epoch, val_scores
                _checkpoint(output, name, head, extractor, class_mapping, epoch, val_scores, config)
            print(f"{name} epoch {epoch:02d}/{EPOCHS} train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_macro_f1={val_scores['macro_f1']:.4f} "
                  f"time={epoch_time:.1f}s", flush=True)
        elapsed = time.perf_counter() - start
        summary = [r for r in summary if r["experiment"] != name]
        summary.append({"experiment": name, "best_epoch": best_epoch,
                        "val_macro_f1": best_metrics["macro_f1"],
                        "val_accuracy": best_metrics["accuracy"],
                        "val_balanced_accuracy": best_metrics["balanced_accuracy"],
                        "train_time_sec": elapsed,
                        "checkpoint": f"checkpoints/{name}_best.pth", "change": change})
        write_rows(summary_path, summary, SUMMARY_COLUMNS)
    selected = max(summary, key=lambda row: float(row["val_macro_f1"]))["experiment"]
    summary_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    plan = {"selected_model": selected, "selection_metric": "validation macro-F1",
            "summary_sha256": summary_digest, "experiments": [r["experiment"] for r in summary],
            **fingerprint,
            "test_used_for_selection": False}
    (output / "frozen_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Frozen plan: selected {selected} by validation macro-F1", flush=True)
    return summary
