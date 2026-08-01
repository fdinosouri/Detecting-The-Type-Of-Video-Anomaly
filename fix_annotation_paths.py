from pathlib import Path
import os

DATA_ROOT = Path("G:/dataset")

FILES_TO_FIX = [
    Path("labels/UCF_train_split.txt"),
    Path("labels/UCF_val_split.txt"),
    Path("labels/UCF_test_split.txt"),
]

print("Scanning videos in:", DATA_ROOT)

video_map = {}

for video_path in DATA_ROOT.rglob("*.mp4"):
    video_map[video_path.name] = video_path

print("Found videos:", len(video_map))

for annotation_file in FILES_TO_FIX:
    print("\nFixing:", annotation_file)

    lines = annotation_file.read_text(encoding="utf-8").splitlines()
    new_lines = []

    fixed_count = 0
    missing_count = 0

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()

        old_path = parts[0]
        video_name = Path(old_path).name

        if video_name not in video_map:
            print("MISSING:", video_name)
            missing_count += 1
            new_lines.append(line)
            continue

        real_full_path = video_map[video_name]
        relative_path = real_full_path.relative_to(DATA_ROOT)

        parts[0] = str(relative_path).replace("\\", "/")

        new_line = " ".join(parts)
        new_lines.append(new_line)

        fixed_count += 1

    backup_file = annotation_file.with_suffix(annotation_file.suffix + ".backup")
    backup_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    annotation_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    print("Fixed:", fixed_count)
    print("Missing:", missing_count)
    print("Backup saved:", backup_file)

print("\nDONE")
