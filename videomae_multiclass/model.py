"""VideoMAE backbone + 14-class head, replacing the UMIL backbone.

Uses the HuggingFace port of MCG-NJU/VideoMAE. Start from the checkpoint
that was already fine-tuned on Kinetics-400 — its action-recognition
features (fights, crashes, explosions...) transfer far better to anomaly
*types* than the self-supervised-only weights.
"""

from __future__ import annotations

import torch
from transformers import VideoMAEForVideoClassification

from .labels import CLASSES, ID2LABEL, LABEL2ID


def build_model(cfg: dict) -> VideoMAEForVideoClassification:
    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained", "MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained,
        num_labels=len(CLASSES),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    freeze_blocks = int(model_cfg.get("freeze_blocks", 0))
    if freeze_blocks > 0:
        for param in model.videomae.embeddings.parameters():
            param.requires_grad = False
        for layer in model.videomae.encoder.layer[:freeze_blocks]:
            for param in layer.parameters():
                param.requires_grad = False
    return model


def _param_depth(name: str, num_layers: int) -> int:
    """Map a parameter name to a depth id: embeddings=0, block i=i+1, head=last."""
    if "embeddings" in name:
        return 0
    if "encoder.layer." in name:
        layer_id = int(name.split("encoder.layer.")[1].split(".")[0])
        return layer_id + 1
    return num_layers + 1


def build_optimizer(model, cfg: dict) -> torch.optim.AdamW:
    """AdamW with layer-wise LR decay.

    On a small dataset like UCF-Crime, updating early transformer blocks
    at the full LR destroys the pretrained features; LLRD keeps early
    blocks nearly frozen while the head adapts quickly.
    """
    train_cfg = cfg.get("train", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    llrd = float(train_cfg.get("layer_decay", 0.75))
    num_layers = model.config.num_hidden_layers

    groups: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        depth = _param_depth(name, num_layers)
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = (depth, no_decay)
        if key not in groups:
            scale = llrd ** (num_layers + 1 - depth)
            groups[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)

    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
