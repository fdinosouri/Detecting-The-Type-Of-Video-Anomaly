"""Sanity-check an annotation pair before spending hours training on it.

Checks, in order of how expensive the mistake would be:

1. train/test leakage — the same video appearing in both files
2. paths that do not exist under --root
3. labels outside 0..NUM_CLASSES-1, or classes with no entries
4. non-positive frame counts
5. duplicate rows inside a single file

    python verify_split.py --root D:/dataset \\
        --train labels/UCF_std_train.txt --test labels/UCF_std_test.txt
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_multiclass import CLASS_NAMES, parse_annotation_label


def parse_total_frames(parts):
    """Mirror FrameDataset: 4 fields -> end-start, otherwise -> end."""
    if len(parts) == 4:
        return int(parts[2]) - int(parts[1])

    return int(parts[1])


def read_split(path):
    rows = []

    with path.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, start=1):
            parts = line.split()

            if not parts:
                continue

            rows.append({
                "lineno": lineno,
                "path": parts[0],
                "name": parts[0].split("/")[-1].split("\\")[-1],
                "label": parse_annotation_label(parts),
                "frames": parse_total_frames(parts),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=len(CLASS_NAMES))
    args = parser.parse_args()

    train = read_split(args.train)
    test = read_split(args.test)
    problems = 0

    print(f"train: {len(train)} rows   test: {len(test)} rows")

    # 1. leakage
    train_names = {r["name"] for r in train}
    test_names = {r["name"] for r in test}
    overlap = sorted(train_names & test_names)

    if overlap:
        problems += 1
        print(f"\nLEAKAGE: {len(overlap)} videos appear in both files")
        for name in overlap[:10]:
            print(f"   {name}")
    else:
        print("\nno train/test leakage")

    # 2. missing files
    for tag, rows in (("train", train), ("test", test)):
        missing = [r for r in rows if not (args.root / r["path"]).exists()]

        if missing:
            problems += 1
            print(f"\n{tag}: {len(missing)} paths do not exist under "
                  f"{args.root}")
            for r in missing[:5]:
                print(f"   line {r['lineno']}: {r['path']}")
        else:
            print(f"{tag}: all {len(rows)} paths exist")

    # 3. labels
    for tag, rows in (("train", train), ("test", test)):
        counts = Counter(r["label"] for r in rows)
        bad = [r for r in rows if not 0 <= r["label"] < args.num_classes]

        if bad:
            problems += 1
            print(f"\n{tag}: {len(bad)} rows have a label outside "
                  f"0..{args.num_classes - 1}")

        absent = [
            CLASS_NAMES[c] for c in range(args.num_classes)
            if c not in counts
        ]

        if absent:
            problems += 1
            print(f"\n{tag}: no entries for {absent}")

        singles = [
            CLASS_NAMES[c] for c in range(args.num_classes)
            if 0 < counts.get(c, 0) < 5
        ]

        if singles:
            print(f"{tag}: fewer than 5 videos for {singles} "
                  f"-- per-class metrics there are close to meaningless")

    # 4. frame counts
    for tag, rows in (("train", train), ("test", test)):
        bad = [r for r in rows if r["frames"] <= 0]

        if bad:
            problems += 1
            print(f"\n{tag}: {len(bad)} rows have a non-positive frame count")
            for r in bad[:5]:
                print(f"   line {r['lineno']}: {r['path']} -> {r['frames']}")

    # 5. duplicate rows within a file
    for tag, rows in (("train", train), ("test", test)):
        seen = defaultdict(list)
        for r in rows:
            seen[r["name"]].append(r["lineno"])

        dupes = {k: v for k, v in seen.items() if len(v) > 1}

        if dupes:
            problems += 1
            print(f"\n{tag}: {len(dupes)} video names appear more than once")
            for name, lines in list(dupes.items())[:5]:
                print(f"   {name} on lines {lines}")

    print()
    if problems:
        print(f"{problems} problem(s) found -- fix before training")
        sys.exit(1)

    print("split looks usable")


if __name__ == "__main__":
    main()
