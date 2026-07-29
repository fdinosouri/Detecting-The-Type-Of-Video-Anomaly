"""Build train/test annotation lists from a UCF-Crime folder tree.

Walks the dataset root, infers each video's class from its path
(``.../Abuse/Abuse036_x264.mp4`` -> Abuse, normal folders -> Normal) and
writes ``train.txt``/``test.txt`` with ``<relative_path> <label>`` lines.

The official split file (``Anomaly_Detection_splits/Anomaly_Test.txt``
from the UCF-Crime release) decides which videos go to the test list; it
is matched by file name, so folder layout differences don't matter. If no
split file is given, a stratified random split is used instead.

Usage:
    python annotations/make_annotations.py --root /path/to/UCF_Crime \
        --test-split /path/to/Anomaly_Test.txt
    python annotations/make_annotations.py --root /path/to/UCF_Crime \
        --test-ratio 0.2
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from videomae_multiclass.labels import label_from_path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="UCF-Crime dataset root")
    parser.add_argument("--test-split", default=None,
                        help="official Anomaly_Test.txt (matched by file name)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="fallback stratified split ratio when no split file")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    videos = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise SystemExit(f"No video files found under {root}")

    labeled = [(path, label_from_path(path)) for path in videos]

    if args.test_split:
        test_names = set()
        with open(args.test_split, "r", encoding="utf-8") as f:
            for line in f:
                # first token only, so files with extra columns (e.g. the
                # temporal annotation file) work too; match by basename
                tokens = line.split()
                if not tokens:
                    continue
                test_names.add(tokens[0].replace("\\", "/").rsplit("/", 1)[-1])
        train = [(p, l) for p, l in labeled if Path(p).name not in test_names]
        test = [(p, l) for p, l in labeled if Path(p).name in test_names]
        if not test:
            raise SystemExit(
                f"No video from {root} matched the names in {args.test_split}. "
                "Check that the split file lists the same file names as the "
                "dataset, or use --test-ratio instead."
            )
    else:
        rng = random.Random(args.seed)
        by_class = defaultdict(list)
        for item in labeled:
            by_class[item[1]].append(item)
        train, test = [], []
        for items in by_class.values():
            rng.shuffle(items)
            n_test = max(1, int(round(len(items) * args.test_ratio)))
            test.extend(items[:n_test])
            train.extend(items[n_test:])
        train.sort()
        test.sort()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.txt", train), ("test.txt", test)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for path, label in split:
                f.write(f"{path} {label}\n")
        print(f"{name}: {len(split)} videos -> {out_dir / name}")
        for label, count in sorted(Counter(l for _, l in split).items()):
            print(f"    {label:<15} {count}")


if __name__ == "__main__":
    main()
