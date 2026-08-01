from pathlib import Path

print("SCRIPT STARTED", flush=True)

FRAMES_ROOT = Path(r"G:\dataset\frames")
OUT_FILE = Path(r"labels\UCF_train_small.txt")

LABELS = {
    "Training_Normal_Videos_Anomaly": 0,
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

lines = []

print("FRAMES_ROOT =", FRAMES_ROOT, flush=True)
print("Exists =", FRAMES_ROOT.exists(), flush=True)

for class_name, label in LABELS.items():
    class_dir = FRAMES_ROOT / class_name
    print("Checking:", class_name, flush=True)

    if not class_dir.exists():
        continue

    for video_dir in class_dir.iterdir():
        if not video_dir.is_dir():
            continue

        # سریع‌تر: فقط فایل‌های img را پیدا می‌کند و شماره آخرین را می‌گیرد
        max_frame = 0
        for img in video_dir.glob("img_*.jpg"):
            try:
                num = int(img.stem.replace("img_", ""))
                if num > max_frame:
                    max_frame = num
            except:
                pass

        if max_frame == 0:
            continue

        line = f"frames/{class_name}/{video_dir.name} 1 {max_frame} {label}"
        lines.append(line)
        print("Added:", line, flush=True)

OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("DONE", flush=True)
print("Total videos:", len(lines), flush=True)
print("Saved to:", OUT_FILE, flush=True)
input("Press Enter to exit...")