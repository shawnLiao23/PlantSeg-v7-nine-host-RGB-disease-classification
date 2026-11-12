"""Extra near-duplicate candidates from frozen ImageNet ResNet-18 features."""

from __future__ import annotations

import csv

import torch

from ..dataset import PlantDiseaseDataset, dataset_root, is_included, load_manifest, make_transforms, result_root
from ..train import device, extract_features, make_feature_extractor


def scan(threshold: float = 0.985) -> list[dict]:
    output = result_root()
    manifest = load_manifest()
    active = [r for r in manifest if is_included(r)]
    target_device = device()
    extractor = make_feature_extractor(output, target_device)
    vectors = {}
    rows_by_split = {}
    for split in ("train", "val", "test"):
        dataset = PlantDiseaseDataset(manifest, dataset_root(), split, make_transforms(False))
        features, _ = extract_features(extractor, dataset, target_device, 32)
        vectors[split] = torch.nn.functional.normalize(features.float(), dim=1)
        rows_by_split[split] = dataset.rows
    pairs = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        similarity = vectors[first] @ vectors[second].T
        matches = (similarity >= threshold).nonzero(as_tuple=False)
        for i, j in matches.cpu().tolist():
            a, b = rows_by_split[first][i], rows_by_split[second][j]
            dhash_distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if dhash_distance <= 4:
                continue
            pairs.append({"image_a": a["image_id"], "split_a": first,
                          "class_a": a["class_name"], "image_b": b["image_id"],
                          "split_b": second, "class_b": b["class_name"],
                          "cosine_similarity": float(similarity[i, j].item()),
                          "dhash_distance": dhash_distance})
    pairs.sort(key=lambda r: (-r["cosine_similarity"], r["image_a"], r["image_b"]))
    path = output / "tables" / "embedding_extra_candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]) if pairs else [
            "image_a", "split_a", "class_a", "image_b", "split_b", "class_b",
            "cosine_similarity", "dhash_distance"])
        writer.writeheader()
        writer.writerows(pairs)
    print(f"ResNet-18 cosine >= {threshold}: {len(pairs)} extra cross-split candidates; wrote {path}")
    return pairs


if __name__ == "__main__":
    scan()
