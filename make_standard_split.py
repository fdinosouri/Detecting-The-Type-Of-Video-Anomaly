"""Rewrite the official UCF-Crime split against your local frame layout.

The split files shipped with UMIL (`labels/UCF_train.txt`,
`labels/UCF_test.txt`) are already 14-class labelled and use the same
class ids as this project (Normal 0, Abuse 1 ... Vandalism 13). What
they are not is compatible with an arbitrary folder layout: every path
is written as `frames/<Class>/<video>.mp4`, and the frame counts come
from the authors' extraction.

This script matches each entry by video name against the frame
directories actually present under `--root`, and rewrites the split with
your real relative paths and your real frame counts. Videos it cannot
find are reported and skipped, so a partial dataset still produces a
usable (smaller) split.

    python make_standard_split.py --root D:/dataset --source labels

Output (formats chosen to match datasets/build.py FrameDataset):
    labels/UCF_std_train.txt   path 0 frames label            (4 fields)
    labels/UCF_std_test.txt    path frames label s e s2 e2    (7 fields)

The test file keeps the temporal annotation columns, so frame-level AUC
stays computable later.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism",
]


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def index_frame_dirs(root, pattern):
    """Map video name -> [(relative path, frame count)] for frame folders."""
    index = defaultdict(list)

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        count = sum(1 for _ in path.glob(pattern))

        if count:
            index[path.name].append(
                (path.relative_to(root).as_posix(), count)
            )

    return index


def index_video_files(root):
    """Map video name -> [(relative path, None)] for raw video files.

    Frame counts are filled in later, either from an existing annotation
    file or by decoding, because opening ~1900 videos is slow.
    """
    index = defaultdict(list)

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            index[path.name].append(
                (path.relative_to(root).as_posix(), None)
            )

    return index


def load_known_counts(paths):
    """Read `name -> total_frames` from existing annotation files.

    Uses the same column convention as FrameDataset, so the effective
    `total_frames` is preserved rather than recomputed.
    """
    counts = {}

    for path in paths:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.split()

                if len(parts) < 3:
                    continue

                name = parts[0].split("/")[-1].split("\\")[-1]

                try:
                    if len(parts) == 4:
                        total = int(parts[2]) - int(parts[1])
                    else:
                        total = int(parts[1])
                except ValueError:
                    continue

                if total > 0:
                    counts.setdefault(name, total)

    return counts


def count_frames_by_decoding(root, rel):
    import cv2

    cap = cv2.VideoCapture(str(root / rel))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total if total > 0 else 0


def pick(candidates, expected_class):
    """Prefer a candidate whose path mentions the expected class folder."""
    if len(candidates) == 1:
        return candidates[0]

    hinted = [
        c for c in candidates
        if expected_class.lower() in c[0].lower()
    ]

    return hinted[0] if hinted else candidates[0]


def convert(src, dst, index, is_test, resolve_frames):
    written = 0
    missing = []
    unreadable = []
    per_class = Counter()

    with src.open("r", encoding="utf-8") as fin, \
            dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.split()

            if not parts:
                continue

            name = parts[0].split("/")[-1]
            expected = parts[0].split("/")[1] if "/" in parts[0] else ""

            # Same column convention as FrameDataset: 4 fields put the
            # label last, everything else puts it at index 2.
            label = int(parts[3]) if len(parts) == 4 else int(parts[2])

            if name not in index:
                missing.append(parts[0])
                continue

            rel, frames = pick(index[name], expected)

            if frames is None:
                frames = resolve_frames(name, rel)

            if not frames:
                unreadable.append(parts[0])
                continue

            if is_test:
                temporal = parts[3:7] if len(parts) >= 7 else ["-1"] * 4
                fout.write(
                    f"{rel} {frames} {label} {' '.join(temporal)}\n"
                )
            else:
                fout.write(f"{rel} 0 {frames} {label}\n")

            per_class[label] += 1
            written += 1

    return written, missing, unreadable, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="DATA.ROOT — the folder holding the frame dirs")
    parser.add_argument("--source", type=Path, default=Path("labels"),
                        help="folder containing UCF_train.txt / UCF_test.txt")
    parser.add_argument("--out", type=Path, default=Path("labels"))
    parser.add_argument("--pattern", default="img_*.jpg",
                        help="frame filename glob, matching DATA.FILENAME_TMPL")
    parser.add_argument("--counts-from", type=Path, nargs="*", default=None,
                        help="existing annotation files to take frame counts "
                             "from; defaults to labels/UCF_full_*.txt")
    args = parser.parse_args()

    if not args.root.is_dir():
        sys.exit(f"--root not found: {args.root}")

    src_train = args.source / "UCF_train.txt"
    src_test = args.source / "UCF_test.txt"

    for path in (src_train, src_test):
        if not path.exists():
            sys.exit(
                f"{path} not found. Copy UCF_train.txt and UCF_test.txt "
                f"from the UMIL repo's labels/ folder into {args.source}."
            )

    print(f"Scanning {args.root} ...", flush=True)
    index = index_frame_dirs(args.root, args.pattern)
    print(f"  frame directories containing {args.pattern}: {len(index)}")

    if not index:
        index = index_video_files(args.root)
        print(f"  video files: {len(index)}")

        if not index:
            sys.exit(
                f"Nothing usable under {args.root}.\n"
                f"Expected either frame folders holding {args.pattern} files, "
                f"or video files ({', '.join(sorted(VIDEO_EXTS))})."
            )

    duplicates = {k: v for k, v in index.items() if len(v) > 1}
    if duplicates:
        print(f"  warning: {len(duplicates)} names appear in more than one "
              f"folder; picking the one matching the class name")

    known = load_known_counts(
        args.counts_from
        or sorted(args.source.glob("UCF_full_*.txt"))
    )
    if known:
        print(f"  reusing frame counts for {len(known)} videos from "
              f"existing annotation files")

    decoded = {"n": 0}

    def resolve_frames(name, rel):
        if name in known:
            return known[name]

        total = count_frames_by_decoding(args.root, rel)
        decoded["n"] += 1

        if decoded["n"] % 100 == 0:
            print(f"    decoded {decoded['n']} videos ...", flush=True)

        return total

    args.out.mkdir(parents=True, exist_ok=True)
    total_missing = []
    total_unreadable = []

    for src, name, is_test in (
        (src_train, "UCF_std_train.txt", False),
        (src_test, "UCF_std_test.txt", True),
    ):
        dst = args.out / name
        written, missing, unreadable, per_class = convert(
            src, dst, index, is_test, resolve_frames
        )
        total_missing += missing
        total_unreadable += unreadable

        print()
        print(f"{name}: wrote {written} entries, "
              f"not found {len(missing)}, unreadable {len(unreadable)}")
        for cid in sorted(per_class):
            print(f"   {cid:2d} {CLASS_NAMES[cid]:15s} {per_class[cid]:4d}")

        if len(per_class) < len(CLASS_NAMES):
            absent = [
                CLASS_NAMES[c] for c in range(len(CLASS_NAMES))
                if c not in per_class
            ]
            print(f"   WARNING: no entries for {absent}")

    if decoded["n"]:
        print(f"\ndecoded {decoded['n']} videos to count frames")

    if total_missing:
        report = args.out / "UCF_std_missing.txt"
        report.write_text("\n".join(total_missing), encoding="utf-8")
        print(f"\n{len(total_missing)} videos are not present under "
              f"{args.root}; list written to {report}")
        print("Download them, or accept the reduced split.")

    if total_unreadable:
        report = args.out / "UCF_std_unreadable.txt"
        report.write_text("\n".join(total_unreadable), encoding="utf-8")
        print(f"\n{len(total_unreadable)} videos were found but reported zero "
              f"frames; list written to {report}")


if __name__ == "__main__":
    main()
