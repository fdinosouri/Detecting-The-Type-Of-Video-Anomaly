"""Video-level inference: clip scoring + top-k gated aggregation.

Anomalies in UCF-Crime are short events inside long videos. Averaging
clip predictions over the whole video drowns the anomaly under Normal
clips (rare classes collapse to Normal), while taking a plain max is
noisy (Normal videos flip to random anomaly classes — the
"Normal -> Vandalism/Stealing" mistakes). The middle ground used here:

1. score every clip, take the k clips with the highest anomaly
   probability (1 - p(Normal)),
2. if even those top-k clips are not anomalous enough on average, the
   video is Normal (the "gate"),
3. otherwise the anomaly type is decided only from those top-k clips,
   with the Normal class excluded from the argmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import NORMAL_INDEX


@torch.no_grad()
def predict_clip_logits(model, clips: torch.Tensor, device, batch_size: int = 4) -> torch.Tensor:
    """clips: (N, T, C, H, W) for one video -> logits (N, num_classes)."""
    model.eval()
    outputs = []
    for start in range(0, clips.size(0), batch_size):
        batch = clips[start : start + batch_size].to(device, non_blocking=True)
        logits = model(pixel_values=batch).logits
        outputs.append(logits.float().cpu())
    return torch.cat(outputs, dim=0)


def aggregate_video_prediction(
    clip_logits: torch.Tensor,
    topk: int = 3,
    gate: float = 0.5,
    mode: str = "topk_gated",
    normal_index: int = NORMAL_INDEX,
):
    """Aggregate clip logits (N, C) into one video-level prediction.

    Returns ``(pred_index, video_probs, anomaly_score)``.
    """
    probs = F.softmax(clip_logits, dim=-1)

    if mode == "mean":
        video_probs = probs.mean(dim=0)
        return int(video_probs.argmax()), video_probs, float(1.0 - video_probs[normal_index])

    anomaly_per_clip = 1.0 - probs[:, normal_index]
    k = min(topk, probs.size(0))
    top_scores, top_idx = anomaly_per_clip.topk(k)
    anomaly_score = float(top_scores.mean())
    video_probs = probs[top_idx].mean(dim=0)

    if anomaly_score < gate:
        return normal_index, video_probs, anomaly_score

    anomaly_probs = video_probs.clone()
    anomaly_probs[normal_index] = float("-inf")
    return int(anomaly_probs.argmax()), video_probs, anomaly_score


@torch.no_grad()
def predict_dataset(model, dataset, device, cfg: dict, progress: bool = True):
    """Run video-level inference over a test-mode ``VideoClipDataset``.

    Returns ``(predictions, labels, paths, anomaly_scores)`` where
    ``predictions``/``labels`` are lists of class indices.
    """
    infer_cfg = cfg.get("inference", {})
    topk = int(infer_cfg.get("topk", 3))
    gate = float(infer_cfg.get("gate", 0.5))
    mode = infer_cfg.get("aggregation", "topk_gated")
    batch_size = int(infer_cfg.get("clip_batch_size", 4))

    iterator = range(len(dataset))
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="videos")

    predictions, labels, paths, scores = [], [], [], []
    for i in iterator:
        clips, label, rel_path = dataset[i]
        clip_logits = predict_clip_logits(model, clips, device, batch_size)
        pred, _, anomaly_score = aggregate_video_prediction(
            clip_logits, topk=topk, gate=gate, mode=mode
        )
        predictions.append(pred)
        labels.append(int(label))
        paths.append(rel_path)
        scores.append(anomaly_score)
    return predictions, labels, paths, scores
