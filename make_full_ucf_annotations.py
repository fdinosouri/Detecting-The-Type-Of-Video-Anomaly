import os
from pathlib import Path
import cv2
from collections import Counter

DATA_ROOT = Path("D:/dataset")
OUT_FILE = Path("labels/UCF_full_all.txt")

CLASS_TO_ID = {
    "Normal": 0,
    "Abuse": 1,
    "Arrest": 2,
    "Arson": 3,
    "Assault": 4,
    "Burglary": 5,
    "Explosion": 6,
    "Fighting": 7,
    "RoadAccidents": 8,
    "Robbery": 9,
    "Shooting": 10,
    "Shoplifting": 11,
    "Stealing": 12,
    "Vandalism": 13,
}

def get_label(video_path: Path):
    parts = [p.lower() for p in video_path.parts]
    name = video_path.name.lower()

    if "normal" in name or any("normal" in p for p in parts):
        return 0

    for class_name, class_id in CLASS_TO_ID.items():
        if class_name.lower() in parts or class_name.lower() in name:
            return class_id

    return None

def get_frame_count(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if count <= 0:
        count = 1

    return count

def main():
    OUT_FILE.parent.mkdir(exist_ok=True)

    print("Scanning videos...", flush=True)

    videos = []
    for ext in ["*.mp4", "*.avi", "*.mkv"]:
        found = list(DATA_ROOT.rglob(ext))
        print(f"Found {len(found)} files with {ext}", flush=True)
        videos.extend(found)

    print("Total video files:", len(videos), flush=True)

    lines = []
    skipped = []
    counter = Counter()

    for idx, video_path in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] {video_path.name}", flush=True)

        label = get_label(video_path)

        if label is None:
            skipped.append(str(video_path))
            print("  skipped: unknown class", flush=True)
            continue

        frame_count = get_frame_count(video_path)
        rel_path = video_path.relative_to(DATA_ROOT).as_posix()

        lines.append(f"{rel_path} 1 {frame_count} {label}")
        counter[label] += 1

        print(f"  label={label}, frames={frame_count}", flush=True)

    lines = sorted(lines)

    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("DONE", flush=True)
    print("Total videos found:", len(videos), flush=True)
    print("Total annotated videos:", len(lines), flush=True)
    print("Skipped videos:", len(skipped), flush=True)
    print("Output:", OUT_FILE, flush=True)

    print()
    print("Class counts:", flush=True)
    for class_id in sorted(counter.keys()):
        class_name = [k for k, v in CLASS_TO_ID.items() if v == class_id][0]
        print(class_id, class_name, counter[class_id], flush=True)

    if skipped:
        Path("labels/UCF_full_skipped.txt").write_text("\n".join(skipped), encoding="utf-8")
        print()
        print("Skipped list saved to labels/UCF_full_skipped.txt", flush=True)

if __name__ == "__main__":
    main()