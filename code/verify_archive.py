"""Verify the supplied PlantSeg ZIP and its extracted files without changing either."""

from __future__ import annotations

import hashlib
import json
import zlib
import zipfile
from pathlib import Path, PurePosixPath

from dataset import PROJECT_ROOT, dataset_root, result_root


OFFICIAL_V7_MD5 = "9358a66dff88cdd15c4fe009763c40a3"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xFFFFFFFF


def verify(archive: Path | None = None, extracted: Path | None = None) -> dict:
    archive = archive or PROJECT_ROOT / "plantseg.zip"
    extracted = extracted or dataset_root()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    md5 = file_md5(archive)
    expected = {}
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            if not parts or parts[0] != "plantseg" or ".." in parts:
                raise ValueError(f"Unexpected ZIP member path: {item.filename}")
            relative = Path(*parts[1:])
            if not relative.parts:
                raise ValueError(f"Unexpected root file in ZIP: {item.filename}")
            expected[relative.as_posix()] = (item.file_size, item.CRC)
    actual = {p.relative_to(extracted).as_posix(): p for p in extracted.rglob("*") if p.is_file()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    size_mismatch = []
    crc_mismatch = []
    for name in sorted(set(expected) & set(actual)):
        path = actual[name]
        size, crc = expected[name]
        if path.stat().st_size != size:
            size_mismatch.append(name)
        elif file_crc32(path) != crc:
            crc_mismatch.append(name)
    report = {
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": md5,
        "official_v7_md5": OFFICIAL_V7_MD5,
        "archive_is_official_v7": md5 == OFFICIAL_V7_MD5,
        "zip_file_count": len(expected),
        "zip_uncompressed_bytes": sum(size for size, _ in expected.values()),
        "extracted_file_count": len(actual),
        "matching_file_count": len(set(expected) & set(actual)) - len(size_mismatch) - len(crc_mismatch),
        "missing_files": missing,
        "extra_files": extra,
        "size_mismatch": size_mismatch,
        "crc_mismatch": crc_mismatch,
    }
    report["extraction_matches_archive"] = not any((missing, extra, size_mismatch, crc_mismatch))
    target = result_root() / "source_archive_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))
    print(f"Mismatch counts: missing={len(missing)}, extra={len(extra)}, "
          f"size={len(size_mismatch)}, crc={len(crc_mismatch)}")
    return report


if __name__ == "__main__":
    verify()
