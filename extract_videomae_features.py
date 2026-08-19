"""Cache VideoMAE clip features so the MIL head can be trained in minutes.

Fine-tuning VideoMAE end-to-end inside the UMIL loop is not practical
here: its ViT-B/16 encoder over 16 frames is roughly 8x the cost of the
CLIP ViT-B/32 backbone this project used, which turns a 16-hour run into
several days. Running the encoder once and caching what it produces
costs a few hours total, after which each training epoch reads floats
off disk and finishes in seconds.

For every video the script samples NUM_CLIPS clips, runs the encoder,
mean-pools the tokens of each clip and writes one
`[NUM_CLIPS, hidden]` array per video. The whole UCF-Crime dataset fits
in well under a gigabyte.

    python extract_videomae_features.py --root D:/dataset \
        --annotations labels/UCF_std_train.txt labels/UCF_std_test.txt \
        --out features/videomae

Re-running skips videos already cached, so an interrupted job resumes.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# VideoMAE was pretrained on ImageNet-normalised RGB.
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_annotations(paths):
    """Collect unique (relative path, video name) pairs across split files."""
    seen = {}

    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.split()

                if not parts:
                    continue

                rel = parts[0]
                name = rel.split("/")[-1].split("\\")[-1]
                seen.setdefault(name, rel)

    return sorted(seen.items())


def sample_indices(total_frames, num_clips, num_frames):
    """Split the video into num_clips segments, sample uniformly in each.

    Deterministic, so cached features match between runs and between
    train and test.
    """
    if total_frames < 1:
        return None

    bounds = np.linspace(0, total_frames, num_clips + 1)
    indices = []

    for i in range(num_clips):
        start, end = bounds[i], bounds[i + 1]

        if end - start < 1:
            picks = np.full(num_frames, int(start), dtype=np.int64)
        else:
            picks = np.linspace(
                start, end - 1, num_frames
            ).round().astype(np.int64)

        indices.append(np.clip(picks, 0, total_frames - 1))

    return np.concatenate(indices)


def decode(video_path, wanted):
    """Decode only the requested frame indices.

    Uses grab() to skip past unwanted frames without paying for their
    decode, which matters when 256 frames are needed out of several
    thousand.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    order = np.argsort(wanted)
    frames = [None] * len(wanted)
    pos = 0
    last = None
    cursor = 0

    for slot in order:
        target = int(wanted[slot])

        while pos <= target:
            ok = cap.grab()

            if not ok:
                break

            if pos == target:
                ok, image = cap.retrieve()
                last = image if ok else last

            pos += 1

        frames[slot] = last
        cursor += 1

    cap.release()

    if last is None:
        return None

    # Any leading slots before the first successful decode reuse the
    # first frame that did decode.
    first = next((f for f in frames if f is not None), None)

    if first is None:
        return None

    return [f if f is not None else first for f in frames]


def preprocess(frames, size):
    import cv2

    out = np.empty((len(frames), size, size, 3), dtype=np.float32)

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        scale = size / min(h, w)
        resized = cv2.resize(
            frame, (max(size, int(round(w * scale))),
                    max(size, int(round(h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

        rh, rw = resized.shape[:2]
        top, left = (rh - size) // 2, (rw - size) // 2
        crop = resized[top:top + size, left:left + size]

        # cv2 gives BGR
        out[i] = crop[:, :, ::-1].astype(np.float32) / 255.0

    out = (out - IMAGE_MEAN) / IMAGE_STD

    return torch.from_numpy(out).permute(0, 3, 1, 2)  # T, C, H, W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("features/videomae"))
    parser.add_argument("--model", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16,
                        help="frames per clip; VideoMAE expects 16")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-clips", type=int, default=4,
                        help="clips per forward pass; lower it if you OOM")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N videos (smoke test)")
    args = parser.parse_args()

    from transformers import VideoMAEModel

    videos = read_annotations(args.annotations)

    if args.limit:
        videos = videos[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)

    todo = [
        (name, rel) for name, rel in videos
        if not (args.out / f"{Path(name).stem}.npy").exists()
    ]

    print(f"{len(videos)} videos referenced, {len(todo)} still to extract")

    if not todo:
        print("nothing to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...", flush=True)

    model = VideoMAEModel.from_pretrained(args.model).to(device).eval()
    hidden = model.config.hidden_size
    print(f"  hidden size {hidden}, expects {model.config.num_frames} frames")

    if model.config.num_frames != args.num_frames:
        print(f"  note: --num-frames {args.num_frames} differs from the "
              f"checkpoint's {model.config.num_frames}")

    failed = []
    start = time.time()

    for done, (name, rel) in enumerate(todo, start=1):
        path = args.root / rel

        if not path.exists():
            failed.append(rel)
            continue

        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        wanted = sample_indices(total, args.num_clips, args.num_frames)

        if wanted is None:
            failed.append(rel)
            continue

        frames = decode(path, wanted)

        if frames is None:
            failed.append(rel)
            continue

        tensor = preprocess(frames, args.size)
        tensor = tensor.view(
            args.num_clips, args.num_frames, 3, args.size, args.size
        )

        feats = []

        with torch.no_grad():
            for i in range(0, args.num_clips, args.batch_clips):
                batch = tensor[i:i + args.batch_clips].to(device)
                tokens = model(batch).last_hidden_state
                feats.append(tokens.mean(dim=1).float().cpu())

        np.save(
            args.out / f"{Path(name).stem}.npy",
            torch.cat(feats).numpy().astype(np.float32),
        )

        if done % 20 == 0 or done == len(todo):
            rate = done / (time.time() - start)
            eta = (len(todo) - done) / rate if rate else 0
            print(f"  {done}/{len(todo)}  {rate * 60:.1f} videos/min  "
                  f"eta {eta / 60:.0f} min", flush=True)

    print(f"\ndone in {(time.time() - start) / 60:.1f} min")

    if failed:
        report = args.out / "failed.txt"
        report.write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} videos could not be read; listed in {report}")


if __name__ == "__main__":
    main()
