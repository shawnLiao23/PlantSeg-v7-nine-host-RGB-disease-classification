# Dataset audit summary

Selected nine-host images: 4,349 raw; 4,349 used in original splits.
Host-disease classes: 43.
Official split counts (raw → used): train 3,007 → 3,007, val 475 → 475, test 867 → 867.
Missing/corrupt selected images: 0; missing masks: 0.
Exact duplicate groups: 161; cross-split groups: 73.
Groups with conflicting class labels: 16 cross-split and 17 within one split; all copies remain in their supplied splits.
Cross-split dHash candidates (distance ≤ 4): 122; additional pHash candidates: 20; visually confirmed pairs: 140; all copies retained at the user's direction.
Train class size range: 19–225 (ratio 11.84:1).

## Version

Local plantseg.zip matches the official v7 MD5; every extracted file matches the archive's size and CRC.

## Leakage rule

Use every readable image in its supplied train, validation, or test split. No duplicate-based exclusions or split reassignments are applied. This preserves the supplied split assignments but retains same-photo overlap across partitions.
