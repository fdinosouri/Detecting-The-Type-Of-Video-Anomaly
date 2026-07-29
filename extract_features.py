"""Extract VideoMAE clip features to use as a drop-in backbone for UMIL.

For each video in the list this saves ``features/<video>.npy`` of shape
(num_clips, hidden_size): clips are sampled with a sliding window across
the whole video and each clip is mean-pooled over VideoMAE's last hidden
states. These can replace UMIL's original backbone features so the MIL
head trains on much stronger representations.

Usage:
    python extract_features.py --config configs/videomae_ucf_crime.yaml \
        --list annotations/train.txt --out features/train
    # add --checkpoint checkpoints/best.pth to use fine-tuned weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from videomae_multiclass.dataset import (
    VideoClipDataset,
    frames_to_tensor,
    read_video_frames,
    video_num_frames,
    _center_crop,
    _resize_short_side,
)
from videomae_multiclass.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/videomae_ucf_crime.yaml")
    parser.add_argument("--list", required=True, help="annotation list file")
    parser.add_argument("--out", default="features", help="output directory")
    parser.add_argument("--checkpoint", default=None, help="optional fine-tuned .pth")
    parser.add_argument("--stride", type=int, default=None,
                        help="frames between clip starts (default: one clip span)")
    parser.add_argument("--max-clips", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))
    backbone = model.videomae.to(device).eval()

    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    num_frames = data_cfg.get("num_frames", 16)
    sampling_rate = data_cfg.get("sampling_rate", 4)
    crop_size = data_cfg.get("crop_size", 224)
    short_side = data_cfg.get("short_side", 256)
    span = num_frames * sampling_rate
    stride = args.stride or span

    dataset = VideoClipDataset(
        list_file=args.list, root=str(root), mode="test", num_clips=1
    )
    out_dir = Path(args.out)

    for rel_path, _ in tqdm(dataset.records, desc="videos"):
        out_path = out_dir / Path(rel_path).with_suffix(".npy")
        if out_path.exists():
            continue
        path = str(root / rel_path)
        total = video_num_frames(path)

        starts = list(range(0, max(total - span, 0) + 1, stride)) or [0]
        if len(starts) > args.max_clips:
            starts = np.linspace(0, starts[-1], args.max_clips).astype(int).tolist()

        features = []
        for start in starts:
            if total >= span:
                indices = start + np.arange(num_frames) * sampling_rate
            else:
                indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
            frames = read_video_frames(path, indices)
            frames = _resize_short_side(frames, short_side)
            frames = _center_crop(frames, crop_size)
            clip = frames_to_tensor(frames).unsqueeze(0).to(device)
            hidden = backbone(pixel_values=clip).last_hidden_state  # (1, tokens, dim)
            features.append(hidden.mean(dim=1).squeeze(0).float().cpu().numpy())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.stack(features))

    print(f"Features written under {out_dir}")


if __name__ == "__main__":
    main()
