"""Video dataset for VideoMAE multiclass anomaly-type classification.

Annotation list format (one video per line)::

    <relative/path/to/video.mp4> <label>

where ``<label>`` is either a class name (e.g. ``Robbery``) or an integer
class index. Paths are resolved against ``root``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import LABEL2ID, label_from_path

try:
    from decord import VideoReader, cpu

    HAS_DECORD = True
except ImportError:  # pragma: no cover - depends on local install
    HAS_DECORD = False

import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_annotation_line(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.rsplit(maxsplit=1)
    if len(parts) == 2 and (parts[1] in LABEL2ID or parts[1].lstrip("-").isdigit()):
        path, label = parts
        label_id = LABEL2ID[label] if label in LABEL2ID else int(label)
    else:
        path = line
        label_id = LABEL2ID[label_from_path(line)]
    return path, label_id


def load_annotations(list_file: str):
    records = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is not None:
                records.append(parsed)
    if not records:
        raise ValueError(f"No annotations found in {list_file}")
    return records


def _read_frames_decord(path: str, indices: np.ndarray) -> np.ndarray:
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    indices = np.clip(indices, 0, len(vr) - 1)
    batch = vr.get_batch(indices)
    return batch.asnumpy()


def _read_frames_cv2(path: str, indices: np.ndarray) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.clip(indices, 0, total - 1)
    wanted = {}
    for idx in np.unique(indices):
        wanted[int(idx)] = None
    pos = 0
    remaining = set(wanted)
    while remaining:
        target = min(remaining)
        if target != pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            pos = target
        ok, frame = cap.read()
        if not ok:
            break
        wanted[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        remaining.discard(pos)
        pos += 1
    cap.release()
    fallback = next((f for f in wanted.values() if f is not None), None)
    if fallback is None:
        raise IOError(f"Failed to decode any frame from: {path}")
    frames = [wanted[int(i)] if wanted[int(i)] is not None else fallback for i in indices]
    return np.stack(frames)


def read_video_frames(path: str, indices: np.ndarray) -> np.ndarray:
    """Return frames as uint8 RGB array of shape (T, H, W, C)."""
    if HAS_DECORD:
        return _read_frames_decord(path, indices)
    return _read_frames_cv2(path, indices)


def video_num_frames(path: str) -> int:
    if HAS_DECORD:
        return len(VideoReader(path, ctx=cpu(0), num_threads=1))
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 1)


def sample_clip_indices(
    total: int,
    num_frames: int,
    sampling_rate: int,
    num_clips: int,
    random_shift: bool,
) -> list[np.ndarray]:
    """Sample ``num_clips`` clips of ``num_frames`` frames each.

    For training (``random_shift=True``) the clip start is random; for
    inference clips are spread uniformly over the whole video so that a
    short anomaly somewhere in a long video is still covered.
    """
    span = num_frames * sampling_rate
    clips = []
    if total >= span:
        if random_shift:
            starts = [random.randint(0, total - span) for _ in range(num_clips)]
        else:
            starts = np.linspace(0, total - span, num_clips).astype(int).tolist()
        for start in starts:
            clips.append(start + np.arange(num_frames) * sampling_rate)
    else:
        base = np.linspace(0, max(total - 1, 0), num_frames).astype(int)
        for _ in range(num_clips):
            clips.append(base.copy())
    return clips


def _resize_short_side(frames: np.ndarray, short_side: int) -> np.ndarray:
    _, h, w, _ = frames.shape
    scale = short_side / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return np.stack([cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR) for f in frames])


def _center_crop(frames: np.ndarray, size: int) -> np.ndarray:
    _, h, w, _ = frames.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return frames[:, top : top + size, left : left + size]


def _random_resized_crop(frames: np.ndarray, size: int, scale=(0.56, 1.0)) -> np.ndarray:
    _, h, w, _ = frames.shape
    for _ in range(10):
        area = h * w * random.uniform(*scale)
        aspect = random.uniform(3 / 4, 4 / 3)
        crop_w = int(round((area * aspect) ** 0.5))
        crop_h = int(round((area / aspect) ** 0.5))
        if crop_w <= w and crop_h <= h:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            crop = frames[:, top : top + crop_h, left : left + crop_w]
            return np.stack(
                [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR) for f in crop]
            )
    frames = _resize_short_side(frames, size)
    return _center_crop(frames, size)


def _color_jitter(frames: np.ndarray, brightness=0.2, contrast=0.2) -> np.ndarray:
    out = frames.astype(np.float32)
    if brightness > 0:
        out *= 1.0 + random.uniform(-brightness, brightness)
    if contrast > 0:
        mean = out.mean(axis=(1, 2, 3), keepdims=True)
        out = (out - mean) * (1.0 + random.uniform(-contrast, contrast)) + mean
    return np.clip(out, 0, 255)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) uint8/float -> normalized float tensor (T, C, H, W)."""
    arr = frames.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class VideoClipDataset(Dataset):
    """Yields VideoMAE-ready clips.

    Train mode returns a single augmented clip ``(T, C, H, W)``; test mode
    returns ``num_clips`` uniformly spread clips ``(N, T, C, H, W)`` so the
    caller can aggregate clip-level predictions into a video-level one.
    """

    def __init__(
        self,
        list_file: str,
        root: str,
        num_frames: int = 16,
        sampling_rate: int = 4,
        crop_size: int = 224,
        short_side: int = 256,
        mode: str = "train",
        num_clips: int = 1,
    ):
        assert mode in ("train", "test")
        self.records = load_annotations(list_file)
        self.root = Path(root)
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.crop_size = crop_size
        self.short_side = short_side
        self.mode = mode
        self.num_clips = num_clips if mode == "test" else 1

    def __len__(self):
        return len(self.records)

    @property
    def labels(self):
        return [label for _, label in self.records]

    def _load_clip(self, path: str, indices: np.ndarray) -> torch.Tensor:
        frames = read_video_frames(path, indices)
        if self.mode == "train":
            frames = _random_resized_crop(frames, self.crop_size)
            if random.random() < 0.5:
                frames = frames[:, :, ::-1].copy()
            frames = _color_jitter(frames)
        else:
            frames = _resize_short_side(frames, self.short_side)
            frames = _center_crop(frames, self.crop_size)
        return frames_to_tensor(frames)

    def __getitem__(self, index: int):
        rel_path, label = self.records[index]
        path = str(self.root / rel_path)
        total = video_num_frames(path)
        clip_indices = sample_clip_indices(
            total,
            self.num_frames,
            self.sampling_rate,
            self.num_clips,
            random_shift=self.mode == "train",
        )
        clips = torch.stack([self._load_clip(path, idx) for idx in clip_indices])
        if self.mode == "train":
            return clips[0], label
        return clips, label, rel_path
