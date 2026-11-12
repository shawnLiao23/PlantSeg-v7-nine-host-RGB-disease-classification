"""Dataset utilities for the PlantSeg classification coursework."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights


HOSTS = (
    "apple", "banana", "citrus", "corn", "potato",
    "soybean", "tomato", "wheat", "zucchini",
)
SPLIT_NAMES = {"Training": "train", "Validation": "val", "Test": "test"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = (
    "image_id", "image_path", "host", "disease", "class_name", "class_id",
    "split", "source_index", "annotation_path", "reported_resolution", "metadata_row", "sha256",
    "dhash", "excluded", "exclusion_reason",
)


def dataset_root(path: str | Path | None = None) -> Path:
    """Find the extracted directory that contains Metadata.csv."""
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get("PLANTSEG_ROOT"):
        candidates = [Path(os.environ["PLANTSEG_ROOT"])]
    else:
        candidates = [
            PROJECT_ROOT / "plantseg" / "plantseg",
            PROJECT_ROOT / "plantseg",
            Path.cwd() / "plantseg" / "plantseg",
            Path.cwd() / "plantseg",
        ]
    for candidate in candidates:
        if (candidate / "Metadata.csv").is_file():
            return candidate.resolve()
        if (candidate / "plantseg" / "Metadata.csv").is_file():
            return (candidate / "plantseg").resolve()
    raise FileNotFoundError(
        "Metadata.csv was not found. Set PLANTSEG_ROOT to the extracted PlantSeg folder."
    )


def result_root(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("CW2_RESULT_DIR") or PROJECT_ROOT / "result").resolve()


def write_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: str | Path | None = None) -> list[dict]:
    manifest_path = Path(path) if path else result_root() / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["class_id"] = int(row["class_id"])
        row["source_index"] = int(row["source_index"])
        row["metadata_row"] = int(row["metadata_row"])
    return rows


def build_manifest(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Use only official metadata labels and splits; keep paths relative."""
    source = dataset_root(data_dir)
    destination = result_root(output_dir)
    with (source / "Metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = [
            (line, row) for line, row in enumerate(csv.DictReader(handle), start=2)
            if row["Plant"].strip().lower() in HOSTS
        ]
    if not metadata:
        raise ValueError("No rows for the nine requested hosts were found in Metadata.csv")

    pairs = sorted({
        (row["Plant"].strip().lower(), row["Disease"].strip().lower())
        for _, row in metadata
    })
    pair_to_id = {pair: index for index, pair in enumerate(pairs)}
    source_indices: dict[tuple[str, str], set[int]] = {pair: set() for pair in pairs}
    rows = []
    seen_names = set()
    for line, item in metadata:
        name = item["Name"].strip()
        if Path(name).name != name:
            raise ValueError(f"Unsafe image name on metadata row {line}: {name}")
        if name in seen_names:
            raise ValueError(f"Duplicate image name in metadata: {name}")
        seen_names.add(name)
        host = item["Plant"].strip().lower()
        disease = item["Disease"].strip().lower()
        pair = (host, disease)
        source_index = int(item["Index"])
        source_indices[pair].add(source_index)
        split = SPLIT_NAMES.get(item["Split"].strip())
        if split is None:
            raise ValueError(f"Unknown official split on metadata row {line}")
        label_name = item["Label file"].strip()
        if Path(label_name).name != label_name:
            raise ValueError(f"Unsafe annotation name on metadata row {line}")
        rows.append({
            "image_id": name,
            "image_path": (Path("images") / split / name).as_posix(),
            "host": host,
            "disease": disease,
            "class_name": f"{host} | {disease}",
            "class_id": pair_to_id[pair],
            "split": split,
            "source_index": source_index,
            "annotation_path": (Path("annotations") / split / label_name).as_posix(),
            "reported_resolution": item["Resolution"].strip(),
            "metadata_row": line,
            "sha256": "",
            "dhash": "",
            "excluded": "0",
            "exclusion_reason": "",
        })
    inconsistent = [pair for pair, indices in source_indices.items() if len(indices) != 1]
    if inconsistent:
        raise ValueError(f"Metadata index varies within classes: {inconsistent}")
    if len({next(iter(v)) for v in source_indices.values()}) != len(pairs):
        raise ValueError("A metadata Index was shared by different host-disease classes")

    mapping = {
        "class_definition": "(host, disease)",
        "class_count": len(pairs),
        "classes": [
            {
                "class_id": index,
                "class_name": f"{host} | {disease}",
                "host": host,
                "disease": disease,
                "source_index": next(iter(source_indices[(host, disease)])),
            }
            for index, (host, disease) in enumerate(pairs)
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(rows, destination / "manifest.csv")
    return rows


def is_included(row: dict) -> bool:
    return str(row.get("excluded", "0")).strip().lower() not in {"1", "true", "yes"}


def make_transforms(augment: bool = False):
    """Use the ResNet-18 ImageNet normalization for every split."""
    norm = transforms.Normalize(
        mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
        std=ResNet18_Weights.IMAGENET1K_V1.transforms().std,
    )
    # Match the deterministic preprocessing specified by the pretrained weights.
    steps = [transforms.Resize(256), transforms.CenterCrop(224)]
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])
    steps.extend([transforms.ToTensor(), norm])
    return transforms.Compose(steps)


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | list[dict],
        data_dir: str | Path | None,
        split: str,
        transform=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        all_rows = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.rows = [row for row in all_rows if row["split"] == split and is_included(row)]
        self.data_dir = dataset_root(data_dir)
        self.transform = transform or make_transforms(False)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.data_dir / row["image_path"]) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), int(row["class_id"])
