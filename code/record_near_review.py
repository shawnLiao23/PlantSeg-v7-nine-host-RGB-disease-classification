"""Record the visual review of the fixed cross-split candidate sheets.

The decisions below were made by comparing the paired images in
result/figures/near_duplicate_review_*.png and phash_review_*.png.
Run this only if those sheets have been checked against the same image files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from dataset import load_manifest, result_root


FALSE_PAIRS = {
    ("apple_rust_google_0013.jpg", "wheat_leaf_rust_Google_0339.jpg"),
    ("tomato_bacterial_leaf_spot_google_0315.jpg", "wheat_leaf_rust_Google_0339.jpg"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_review(output: Path | None = None) -> Path:
    output = result_root(output)
    dhash = read_csv(output / "tables" / "near_duplicate_candidates.csv")
    phash = read_csv(output / "tables" / "phash_extra_candidates.csv")
    embeddings = read_csv(output / "tables" / "embedding_extra_candidates.csv")
    if len(dhash) != 122 or len(phash) != 20:
        raise ValueError("Candidate set changed; the visual decisions must be reviewed again")
    all_pairs = {(row["image_a"], row["image_b"]) for row in dhash + phash}
    if len(all_pairs) != 142 or not FALSE_PAIRS.issubset(all_pairs):
        raise ValueError("Candidate pair IDs changed; visual review is stale")
    embedding_pairs = {(row["image_a"], row["image_b"]) for row in embeddings}
    rows = {row["image_id"]: row for row in load_manifest(output / "manifest.csv")}

    decisions = []
    for index, item in enumerate(dhash + phash):
        pair = item["image_a"], item["image_b"]
        is_dhash = index < len(dhash)
        if pair in FALSE_PAIRS:
            decision = "different_photo"
            evidence = "Different host, plant structure, and scene; hash collision."
        else:
            decision = "same_photo"
            evidence = (
                "Matching lesion pattern and scene landmarks; only crop, colour, "
                "compression, or text overlay differs."
            )
        decisions.append({
            "review_index": index,
            "image_a": item["image_a"],
            "split_a": item["split_a"],
            "class_a": item["class_a"],
            "sha256_a": rows[item["image_a"]]["sha256"],
            "image_b": item["image_b"],
            "split_b": item["split_b"],
            "class_b": item["class_b"],
            "sha256_b": rows[item["image_b"]]["sha256"],
            "dhash_distance": item["hamming_distance"] if is_dhash else item["dhash_distance"],
            "phash_distance": "" if is_dhash else item["phash_distance"],
            "candidate_source": "dhash" if is_dhash else "phash",
            "embedding_support": int(pair in embedding_pairs),
            "review_sheet": (
                f"figures/near_duplicate_review_{index // 6:02d}.png"
                if is_dhash else f"figures/phash_review_{(index - len(dhash)) // 5:02d}.png"
            ),
            "decision": decision,
            "visual_evidence": evidence,
        })
    path = output / "tables" / "near_duplicate_review.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    return path


if __name__ == "__main__":
    path = write_review()
    print(f"Wrote 142 visual decisions to {path}")
