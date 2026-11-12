"""Find extra cross-split near-duplicate candidates using a 64-bit pHash."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import dataset_root, is_included, load_manifest, result_root


def phash(path: Path, cosine: np.ndarray) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = (cosine @ pixels @ cosine.T)[:8, :8]
    bits = low > np.median(low.ravel()[1:])
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def scan(threshold: int = 8) -> list[dict]:
    output = result_root()
    rows = [r for r in load_manifest() if is_included(r)]
    source = dataset_root()
    n = 32
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    cosine = np.sqrt(2 / n) * np.cos(np.pi * (x + 0.5) * k / n)
    cosine[0] = np.sqrt(1 / n)
    hashes = np.array([phash(source / r["image_path"], cosine) for r in rows], dtype=np.uint64)
    by_split = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s], dtype=int)
                for s in ("train", "val", "test")}
    found = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        left, right = by_split[first], by_split[second]
        for start in range(0, len(left), 64):
            chunk = left[start:start + 64]
            xor = np.bitwise_xor(hashes[chunk, None], hashes[right][None, :])
            distances = np.unpackbits(xor.view(np.uint8).reshape(len(chunk), len(right), 8, 1),
                                      axis=3).sum(axis=(2, 3))
            for a, b in zip(*np.where(distances <= threshold)):
                row_a, row_b = rows[chunk[a]], rows[right[b]]
                dhash_distance = (int(row_a["dhash"], 16) ^ int(row_b["dhash"], 16)).bit_count()
                if dhash_distance <= 4:
                    continue
                found.append({
                    "image_a": row_a["image_id"], "split_a": first,
                    "class_a": row_a["class_name"], "image_b": row_b["image_id"],
                    "split_b": second, "class_b": row_b["class_name"],
                    "phash_distance": int(distances[a, b]), "dhash_distance": dhash_distance,
                })
    found.sort(key=lambda r: (r["phash_distance"], r["dhash_distance"], r["image_a"], r["image_b"]))
    path = output / "tables" / "phash_extra_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(found[0]) if found else [
            "image_a", "split_a", "class_a", "image_b", "split_b", "class_b",
            "phash_distance", "dhash_distance"])
        writer.writeheader()
        writer.writerows(found)
    print(f"pHash <= {threshold}: {len(found)} extra cross-split candidate pairs; wrote {path}")
    return found


if __name__ == "__main__":
    scan()
