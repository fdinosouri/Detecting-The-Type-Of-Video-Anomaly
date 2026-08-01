import cv2
from pathlib import Path

VIDEO_ROOT = Path(r"D:\UCF-Crime")      # مسیر دیتاست ویدیویی خودت
OUT_ROOT = VIDEO_ROOT / "frames"

VIDEO_EXTS = [".mp4", ".avi", ".mkv", ".mov"]

for video_path in VIDEO_ROOT.rglob("*"):
    if video_path.suffix.lower() not in VIDEO_EXTS:
        continue

    # از خود پوشه frames دوباره فریم نساز
    if "frames" in video_path.parts:
        continue

    rel_path = video_path.relative_to(VIDEO_ROOT)
    out_dir = OUT_ROOT / rel_path.as_posix()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open:", video_path)
        continue

    frame_id = 1
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_file = out_dir / f"img_{frame_id:08d}.jpg"
        cv2.imwrite(str(out_file), frame)

        frame_id += 1
        saved += 1

    cap.release()
    print(f"Done: {video_path} -> {saved} frames")