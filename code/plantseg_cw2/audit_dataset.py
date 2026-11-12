"""Build the fixed classification manifest and audit its source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from .dataset import HOSTS, build_manifest, dataset_root, is_included, result_root, write_manifest
from .paths import PROJECT_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    """Return a 64-bit dHash; low Hamming distance suggests visual similarity."""
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def inspect_images(rows: list[dict], source: Path) -> tuple[list[dict], list[dict]]:
    image_errors = []
    resolution_mismatches = []
    for row in rows:
        path = source / row["image_path"]
        if not path.is_file():
            row["excluded"] = "1"
            row["exclusion_reason"] = "missing_image"
            image_errors.append({"image_id": row["image_id"], "problem": "missing"})
            continue
        try:
            row["sha256"] = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                row["dhash"] = f"{difference_hash(image):016x}"
                if width < 1 or height < 1:
                    raise ValueError("invalid image dimensions")
                if row["reported_resolution"] != f"{width}x{height}":
                    resolution_mismatches.append({
                        "image_id": row["image_id"],
                        "metadata": row["reported_resolution"],
                        "actual": f"{width}x{height}",
                    })
        except (OSError, ValueError) as error:
            row["excluded"] = "1"
            row["exclusion_reason"] = "corrupt_image"
            image_errors.append({"image_id": row["image_id"], "problem": str(error)})
    return image_errors, resolution_mismatches


def exclude_exact_duplicates(rows: list[dict]) -> list[dict]:
    """Filter a temporary copy for near-duplicate candidate discovery only."""
    groups = defaultdict(list)
    for row in rows:
        if is_included(row) and row["sha256"]:
            groups[row["sha256"]].append(row)
    split_priority = {"train": 0, "val": 1, "test": 2}
    duplicates = []
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda row: (
            split_priority[row["split"]], row["image_id"]
        ))
        best_split = ordered[0]["split"]
        retained = [row for row in ordered if row["split"] == best_split]
        for row in ordered:
            if row["split"] != best_split:
                row["excluded"] = "1"
                row["exclusion_reason"] = f"cross_split_exact_duplicate_of:{retained[0]['image_id']}"
        duplicates.append({
            "sha256": digest,
            "retained": [row["image_id"] for row in retained],
            "members": [
                {"image_id": row["image_id"], "split": row["split"],
                 "class_name": row["class_name"]}
                for row in ordered
            ],
            "cross_split": len({row["split"] for row in ordered}) > 1,
            "conflicting_labels": len({row["class_id"] for row in ordered}) > 1,
        })
    return duplicates


def find_near_duplicates(rows: list[dict], output: Path, threshold: int = 4) -> list[dict]:
    """Record cross-split candidates only; do not delete by hash threshold."""
    by_split = {
        split: [(row, int(row["dhash"], 16)) for row in rows
                if row["split"] == split and is_included(row) and row["dhash"]]
        for split in ("train", "val", "test")
    }
    candidates = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        for row_a, hash_a in by_split[first]:
            for row_b, hash_b in by_split[second]:
                distance = (hash_a ^ hash_b).bit_count()
                if distance <= threshold:
                    candidates.append({
                        "image_a": row_a["image_id"], "split_a": first,
                        "class_a": row_a["class_name"],
                        "image_b": row_b["image_id"], "split_b": second,
                        "class_b": row_b["class_name"],
                        "hamming_distance": distance,
                    })
    candidates.sort(key=lambda item: (
        item["hamming_distance"], item["image_a"], item["image_b"]
    ))
    with (output / "tables" / "near_duplicate_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        columns = (
            "image_a", "split_a", "class_a", "image_b", "split_b", "class_b",
            "hamming_distance",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)
    return candidates


def apply_reviewed_near_duplicates(
    rows: list[dict], candidates: list[dict], output: Path
) -> dict:
    """Replay visual decisions to document overlaps, without altering model rows."""
    review_path = output / "tables" / "near_duplicate_review.csv"
    if not review_path.is_file():
        for filename in ("near_duplicate_overlap_risk.csv", "near_duplicate_exclusions.csv"):
            with (output / "tables" / filename).open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(("image_id", "split", "matching_image_id", "matching_split"))
        return {"reviewed_pairs": 0, "confirmed_pairs": 0,
                "uncertain_pairs": 0, "excluded_images": [],
                "potential_overlap_images": [],
                "review_sha256": None, "dhash_reviewed_pairs": 0,
                "dhash_confirmed_pairs": 0, "phash_extra_pairs": 0,
                "phash_extra_confirmed_pairs": 0}
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    expected = {(item["image_a"], item["image_b"]) for item in candidates}
    observed = {(item["image_a"], item["image_b"]) for item in decisions}
    if not expected.issubset(observed) or len(observed) != len(decisions):
        raise ValueError(
            "Near-duplicate review must cover every current dHash candidate exactly once."
        )
    allowed = {"same_photo", "different_photo", "uncertain"}
    if any(item["decision"] not in allowed for item in decisions):
        raise ValueError("Review decisions must be same_photo, different_photo, or uncertain")

    lookup = {row["image_id"]: row for row in rows if is_included(row)}
    for item in decisions:
        for suffix in ("a", "b"):
            image_id = item[f"image_{suffix}"]
            if image_id not in lookup:
                raise ValueError(f"Reviewed image is absent from audited images: {image_id}")
            source_row = lookup[image_id]
            if (item[f"split_{suffix}"] != source_row["split"]
                    or item[f"class_{suffix}"] != source_row["class_name"]
                    or item[f"sha256_{suffix}"] != source_row["sha256"]):
                raise ValueError(f"Reviewed image changed since visual review: {image_id}")
        if item["split_a"] == item["split_b"]:
            raise ValueError("Near-duplicate review must contain cross-split pairs only")

    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first, second):
        parent[find(first)] = find(second)

    confirmed = [item for item in decisions if item["decision"] == "same_photo"]
    for item in confirmed:
        union(item["image_a"], item["image_b"])
    groups = defaultdict(list)
    for image_id in parent:
        groups[find(image_id)].append(lookup[image_id])
    split_priority = {"train": 0, "val": 1, "test": 2}
    potential_overlap = []
    for group in groups.values():
        best_split = min((row["split"] for row in group), key=split_priority.get)
        reference_id = min(row["image_id"] for row in group if row["split"] == best_split)
        for row in group:
            if row["split"] != best_split:
                potential_overlap.append({
                    "image_id": row["image_id"], "split": row["split"],
                    "matching_image_id": reference_id, "matching_split": best_split,
                })
    with (output / "tables" / "near_duplicate_overlap_risk.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image_id", "split", "matching_image_id", "matching_split")
        )
        writer.writeheader()
        writer.writerows(potential_overlap)
    # Overwrite the previous run's exclusion table to avoid stale claims.
    with (output / "tables" / "near_duplicate_exclusions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("image_id", "split", "retained_image_id", "retained_split")
        )
        writer.writeheader()
    return {
        "reviewed_pairs": len(decisions),
        "confirmed_pairs": len(confirmed),
        "uncertain_pairs": sum(item["decision"] == "uncertain" for item in decisions),
        "review_sha256": sha256_file(review_path),
        "dhash_reviewed_pairs": sum(item["candidate_source"] == "dhash" for item in decisions),
        "dhash_confirmed_pairs": sum(item["candidate_source"] == "dhash"
                                      and item["decision"] == "same_photo" for item in decisions),
        "phash_extra_pairs": sum(item["candidate_source"] == "phash" for item in decisions),
        "phash_extra_confirmed_pairs": sum(item["candidate_source"] == "phash"
                                           and item["decision"] == "same_photo" for item in decisions),
        "excluded_images": [],
        "potential_overlap_images": potential_overlap,
    }


def count_by_class(rows: list[dict], mapping: dict, output: Path) -> list[dict]:
    raw = Counter((int(row["class_id"]), row["split"]) for row in rows)
    used = Counter((int(row["class_id"]), row["split"]) for row in rows if is_included(row))
    counts = []
    for item in mapping["classes"]:
        class_id = item["class_id"]
        record = {key: item[key] for key in ("class_id", "class_name", "host", "disease")}
        for split in ("train", "val", "test"):
            record[f"raw_{split}"] = raw[class_id, split]
            record[f"used_{split}"] = used[class_id, split]
        record["raw_total"] = sum(record[f"raw_{split}"] for split in ("train", "val", "test"))
        record["used_total"] = sum(record[f"used_{split}"] for split in ("train", "val", "test"))
        counts.append(record)
    with (output / "tables" / "class_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    return counts


def plot_distribution(counts: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [item["class_name"] for item in counts]
    indices = list(range(len(counts)))
    fig, ax = plt.subplots(figsize=(11, 15))
    left = [0] * len(counts)
    for split, color in (("train", "#4C78A8"), ("val", "#F2CF5B"), ("test", "#E07B54")):
        values = [item[f"used_{split}"] for item in counts]
        ax.barh(indices, values, left=left, color=color, label=split, height=0.8)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(indices, names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Images in the supplied official splits")
    ax.set_title("PlantSeg nine-host class distribution")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.subplots_adjust(left=0.35, right=0.98)
    fig.savefig(output / "figures" / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_examples(rows: list[dict], source: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = []
    for host in HOSTS:
        candidate = next((row for row in rows if row["host"] == host
                          and row["split"] == "train" and is_included(row)), None)
        if candidate:
            examples.append(candidate)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for axis, row in zip(axes.flat, examples):
        with Image.open(source / row["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(row["class_name"], fontsize=8)
        axis.axis("off")
    for axis in list(axes.flat)[len(examples):]:
        axis.axis("off")
    fig.suptitle("Representative training images (one per host)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "figures" / "training_examples.png", dpi=150)
    plt.close(fig)


def plot_near_duplicate_preview(
    candidates: list[dict], rows: list[dict], source: Path, output: Path,
    limit: int = 10, filename: str = "near_duplicate_preview.png",
) -> None:
    """Make a small review sheet, keeping the candidate status explicit."""
    if not candidates:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {row["image_id"]: row for row in rows}
    selected = candidates[:limit]
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3 * len(selected)))
    if len(selected) == 1:
        axes = [axes]
    for pair, (left, right) in zip(selected, axes):
        for axis, key, split_key, class_key in (
            (left, "image_a", "split_a", "class_a"),
            (right, "image_b", "split_b", "class_b"),
        ):
            with Image.open(source / lookup[pair[key]]["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(
                f"{pair[split_key]}: {pair[key]}\n{pair[class_key]}", fontsize=7
            )
            axis.axis("off")
        left.text(
            0, -0.05,
            f"candidate {pair.get('_review_index', '?')}; "
            f"dHash distance = {pair['hamming_distance']}; visual review required",
            transform=left.transAxes, fontsize=8,
        )
    fig.suptitle("Cross-split near-duplicate candidates (not automatically excluded)")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output / "figures" / filename, dpi=130)
    plt.close(fig)


def source_summary(source: Path) -> dict:
    counts = {}
    total_bytes = 0
    total_files = 0
    for kind in ("images", "annotations"):
        counts[kind] = {}
        for split in ("train", "val", "test"):
            files = [path for path in (source / kind / split).iterdir() if path.is_file()]
            counts[kind][split] = len(files)
            total_files += len(files)
            total_bytes += sum(path.stat().st_size for path in files)
    for path in source.iterdir():
        if path.is_file():
            total_files += 1
            total_bytes += path.stat().st_size
    counts["all_files"] = total_files
    counts["all_bytes"] = total_bytes
    with (source / "Metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    counts["metadata_rows"] = len(metadata)
    counts["metadata_missing_images"] = sum(
        not (source / "images" / {
            "Training": "train", "Validation": "val", "Test": "test"
        }[row["Split"]] / row["Name"]).is_file()
        for row in metadata
    )
    return counts


def verify_archive_extraction(archive: Path, source: Path) -> dict:
    """Check every extracted file against the versioned ZIP's size and CRC."""
    missing = []
    size_mismatches = []
    crc_mismatches = []
    archive_paths = set()
    with zipfile.ZipFile(archive) as bundle:
        members = [item for item in bundle.infolist() if not item.is_dir()]
        for item in members:
            parts = PurePosixPath(item.filename).parts
            if not parts or parts[0] != "plantseg" or ".." in parts:
                raise ValueError(f"Unexpected or unsafe ZIP member: {item.filename}")
            relative = Path(*parts[1:])
            archive_paths.add(relative.as_posix())
            extracted = source / relative
            if not extracted.is_file():
                missing.append(item.filename)
                continue
            if extracted.stat().st_size != item.file_size:
                size_mismatches.append(item.filename)
                continue
            checksum = 0
            with extracted.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum = zlib.crc32(block, checksum)
            if checksum != item.CRC:
                crc_mismatches.append(item.filename)
    extracted_paths = {path.relative_to(source).as_posix()
                       for path in source.rglob("*") if path.is_file()}
    extra = sorted(extracted_paths - archive_paths)
    return {
        "archive_file_count": len(members),
        "archive_uncompressed_bytes": sum(item.file_size for item in members),
        "missing_count": len(missing), "size_mismatch_count": len(size_mismatches),
        "crc_mismatch_count": len(crc_mismatches), "extra_extracted_count": len(extra),
        "missing_examples": missing[:10], "size_mismatch_examples": size_mismatches[:10],
        "crc_mismatch_examples": crc_mismatches[:10], "extra_extracted_examples": extra[:10],
        "matches_extraction": not (missing or size_mismatches or crc_mismatches or extra),
    }


def archive_extraction_check(archive: Path, source: Path, output: Path,
                             archive_md5: str) -> dict:
    """Reuse a fresh full-file verification; otherwise verify every file now."""
    cached_path = output / "source_archive_verification.json"
    if cached_path.is_file() and cached_path.stat().st_mtime >= archive.stat().st_mtime:
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        if (cached.get("archive") == str(archive.resolve())
                and cached.get("archive_md5") == archive_md5
                and cached.get("extraction_matches_archive") is True):
            latest_extracted_change = max(
                (path.stat().st_mtime for path in source.rglob("*") if path.is_file()),
                default=0,
            )
            if cached_path.stat().st_mtime >= latest_extracted_change:
                return {
                    "archive_file_count": cached["zip_file_count"],
                    "archive_uncompressed_bytes": cached["zip_uncompressed_bytes"],
                    "missing_count": len(cached["missing_files"]),
                    "size_mismatch_count": len(cached["size_mismatch"]),
                    "crc_mismatch_count": len(cached["crc_mismatch"]),
                    "extra_extracted_count": len(cached["extra_files"]),
                    "matches_extraction": True,
                    "verification_source": "full size/CRC scan in source_archive_verification.json",
                }
    result = verify_archive_extraction(archive, source)
    result["verification_source"] = "full size/CRC scan by audit_dataset.py"
    return result


def write_split_stats(summary: dict, output: Path) -> None:
    path = output / "tables" / "split_stats.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "split", "raw_images", "used_images", "excluded_images", "class_count"
        ))
        writer.writeheader()
        for split, counts in summary["split_counts"].items():
            writer.writerow({
                "split": split,
                "raw_images": counts["raw"],
                "used_images": counts["used"],
                "excluded_images": counts["raw"] - counts["used"],
                "class_count": summary["classes"],
            })
        writer.writerow({
            "split": "total",
            "raw_images": summary["raw_images"],
            "used_images": summary["used_images"],
            "excluded_images": summary["raw_images"] - summary["used_images"],
            "class_count": summary["classes"],
        })


def audit_dataset(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    near_threshold: int = 4,
) -> dict:
    source = dataset_root(data_dir)
    output = result_root(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    rows = build_manifest(source, output)
    mapping = json.loads((output / "class_mapping.json").read_text(encoding="utf-8"))
    image_errors, resolution_mismatches = inspect_images(rows, source)
    missing_masks = [row["image_id"] for row in rows
                     if not (source / row["annotation_path"]).is_file()]
    # The model uses every readable source row in its supplied split. Make a
    # separate copy solely to reproduce the earlier near-duplicate review set;
    # exact copies are omitted from candidate discovery, not model training.
    review_rows = [dict(row) for row in rows]
    duplicates = exclude_exact_duplicates(review_rows)
    for group in duplicates:
        group["retained_in_model"] = [member["image_id"] for member in group["members"]]
        group.pop("retained", None)
    candidates = find_near_duplicates(review_rows, output, near_threshold)
    review = apply_reviewed_near_duplicates(review_rows, candidates, output)
    write_manifest(rows, output / "manifest.csv")
    counts = count_by_class(rows, mapping, output)
    plot_distribution(counts, output)
    plot_examples(rows, source, output)
    plot_near_duplicate_preview(candidates, rows, source, output)
    used = [row for row in rows if is_included(row)]
    split_counts = {
        split: {"raw": sum(row["split"] == split for row in rows),
                "used": sum(row["split"] == split for row in used)}
        for split in ("train", "val", "test")
    }
    train_counts = [item["used_train"] for item in counts]
    info = json.loads((source / "annotation_train.json").read_text(encoding="utf-8"))["info"]
    archive = next((path for path in (PROJECT_ROOT / "plantseg.zip",
               source.parent / "plantseg.zip") if path.is_file()), None)
    archive_md5 = md5_file(archive) if archive else None
    official_md5 = "9358a66dff88cdd15c4fe009763c40a3"
    extraction_check = archive_extraction_check(archive, source, output, archive_md5) if archive else None
    release_verified = archive_md5 == official_md5 and bool(extraction_check and extraction_check["matches_extraction"])
    summary = {
        "dataset_release": "PlantSeg v7 verified" if release_verified else "unverified",
        "version_note": (
            "Local plantseg.zip matches the official v7 MD5; every extracted file matches the archive's size and CRC."
            if release_verified else
            "The supplied extraction could not be verified against the PlantSeg v7 archive. "
            "COCO annotation info.version is %s and does not establish the PlantSeg release."
            % info.get("version", "missing")
        ),
        "official_v7_record": "https://zenodo.org/records/17719108",
        "official_v7_archive_md5": official_md5,
        "local_archive_present": archive is not None,
        "local_archive_bytes": archive.stat().st_size if archive else None,
        "local_archive_md5": archive_md5,
        "archive_extraction_check": extraction_check,
        "source": source_summary(source),
        "nine_hosts": list(HOSTS),
        "raw_images": len(rows),
        "used_images": len(used),
        "classes": len(counts),
        "split_counts": split_counts,
        "missing_or_corrupt_images": image_errors,
        "missing_masks": missing_masks,
        "resolution_mismatches": resolution_mismatches,
        "exact_duplicate_groups": duplicates,
        "exact_duplicate_group_count": len(duplicates),
        "cross_split_exact_duplicate_group_count": sum(item["cross_split"] for item in duplicates),
        "cross_split_conflicting_label_group_count": sum(
            item["cross_split"] and item["conflicting_labels"] for item in duplicates
        ),
        "within_split_conflicting_label_group_count": sum(
            not item["cross_split"] and item["conflicting_labels"] for item in duplicates
        ),
        "near_duplicate_threshold": near_threshold,
        "cross_split_near_duplicate_candidate_count": len(candidates),
        "near_duplicate_reviewed_pair_count": review["reviewed_pairs"],
        "near_duplicate_confirmed_pair_count": review["confirmed_pairs"],
        "near_duplicate_uncertain_pair_count": review["uncertain_pairs"],
        "near_duplicate_review_sha256": review["review_sha256"],
        "dhash_reviewed_pair_count": review["dhash_reviewed_pairs"],
        "dhash_confirmed_pair_count": review["dhash_confirmed_pairs"],
        "phash_extra_reviewed_pair_count": review["phash_extra_pairs"],
        "phash_extra_confirmed_pair_count": review["phash_extra_confirmed_pairs"],
        "near_duplicate_exclusions": review["excluded_images"],
        "near_duplicate_overlap_risk_images": review.get("potential_overlap_images", []),
        "near_duplicate_note": (
            "64-bit dHash and pHash generated candidate pairs; visual decisions were recorded. "
            "All readable images remain in their original splits, including confirmed cross-split copies. "
            "Validation and test scores may therefore be optimistic."
        ),
        "train_class_min": min(train_counts),
        "train_class_max": max(train_counts),
        "train_imbalance_ratio": max(train_counts) / min(train_counts) if min(train_counts) else None,
        "exclusion_rule": (
            "Use every readable image in its supplied train, validation, or test split. "
            "No duplicate-based exclusions or split reassignments are applied. "
            "This preserves the supplied split assignments but retains same-photo overlap across partitions."
        ),
        "duplicate_policy": "original_splits_no_duplicate_exclusions",
    }
    (output / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_split_stats(summary, output)
    lines = [
        "# Dataset audit summary", "",
        f"Selected nine-host images: {len(rows):,} raw; {len(used):,} used in original splits.",
        f"Host-disease classes: {len(counts)}.",
        "Official split counts (raw → used): " + ", ".join(
            f"{split} {data['raw']:,} → {data['used']:,}"
            for split, data in split_counts.items()
        ) + ".",
        f"Missing/corrupt selected images: {len(image_errors)}; missing masks: {len(missing_masks)}.",
        f"Exact duplicate groups: {len(duplicates)}; cross-split groups: "
        f"{summary['cross_split_exact_duplicate_group_count']}.",
        f"Groups with conflicting class labels: "
        f"{summary['cross_split_conflicting_label_group_count']} cross-split and "
        f"{summary['within_split_conflicting_label_group_count']} within one split; "
        "all copies remain in their supplied splits.",
        f"Cross-split dHash candidates (distance ≤ {near_threshold}): {len(candidates)}; "
        f"additional pHash candidates: {review['phash_extra_pairs']}; "
        f"visually confirmed pairs: {review['confirmed_pairs']}; "
        "all copies retained at the user's direction.",
        f"Train class size range: {min(train_counts)}–{max(train_counts)} "
        f"(ratio {summary['train_imbalance_ratio']:.2f}:1).",
        "", "## Version", "", summary["version_note"], "",
        "## Leakage rule", "", summary["exclusion_rule"], "",
    ]
    (output / "tables" / "dataset_audit_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--near-threshold", type=int, default=4)
    options = parser.parse_args()
    audit = audit_dataset(options.data_dir, options.output_dir, options.near_threshold)
    print(json.dumps({
        "raw_images": audit["raw_images"],
        "used_images": audit["used_images"],
        "classes": audit["classes"],
        "split_counts": audit["split_counts"],
        "exact_duplicate_groups": audit["exact_duplicate_group_count"],
        "cross_split_near_duplicate_candidates": audit["cross_split_near_duplicate_candidate_count"],
    }, indent=2))
