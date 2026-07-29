"""Fine-tune VideoMAE for 14-class UCF-Crime anomaly-type classification.

Usage:
    python train.py --config configs/videomae_ucf_crime.yaml

The best checkpoint is selected by macro-F1 (not accuracy), because with
148/296 Normal videos a model can reach ~62% accuracy while ignoring half
of the anomaly classes entirely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from videomae_multiclass.dataset import VideoClipDataset
from videomae_multiclass.inference import predict_dataset
from videomae_multiclass.labels import CLASSES, NUM_CLASSES
from videomae_multiclass.losses import build_criterion
from videomae_multiclass.model import build_model, build_optimizer, cosine_warmup_scheduler
from videomae_multiclass.samplers import ClassBalancedSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/videomae_ucf_crime.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    return parser.parse_args()


def evaluate(model, test_dataset, device, cfg):
    predictions, labels, _, _ = predict_dataset(model, test_dataset, device, cfg)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    accuracy = sum(p == t for p, t in zip(predictions, labels)) / len(labels)
    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        zero_division=0,
    )
    return macro_f1, accuracy, report


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VideoClipDataset(
        list_file=data_cfg["train_list"],
        root=data_cfg["root"],
        num_frames=data_cfg.get("num_frames", 16),
        sampling_rate=data_cfg.get("sampling_rate", 4),
        crop_size=data_cfg.get("crop_size", 224),
        short_side=data_cfg.get("short_side", 256),
        mode="train",
    )
    test_dataset = VideoClipDataset(
        list_file=data_cfg["test_list"],
        root=data_cfg["root"],
        num_frames=data_cfg.get("num_frames", 16),
        sampling_rate=data_cfg.get("sampling_rate", 4),
        crop_size=data_cfg.get("crop_size", 224),
        short_side=data_cfg.get("short_side", 256),
        mode="test",
        num_clips=cfg.get("inference", {}).get("eval_num_clips", 5),
    )

    sampler = ClassBalancedSampler(
        train_dataset.labels,
        samples_per_class=train_cfg.get("samples_per_class"),
        seed=train_cfg.get("seed", 0),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg, train_dataset.labels, NUM_CLASSES).to(device)
    optimizer = build_optimizer(model, cfg)

    epochs = train_cfg.get("epochs", 30)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=train_cfg.get("warmup_epochs", 2) * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_macro_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_macro_f1 = ckpt.get("best_macro_f1", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Device: {device} | train videos: {len(train_dataset)} | "
          f"test videos: {len(test_dataset)} | steps/epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        sampler.set_epoch(epoch)
        running_loss, running_correct, running_total = 0.0, 0, 0

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for clips, labels in progress:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(pixel_values=clips).logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=-1) == labels).sum().item()
            running_total += labels.size(0)
            progress.set_postfix(
                loss=f"{running_loss / running_total:.4f}",
                acc=f"{running_correct / running_total:.3f}",
            )

        eval_every = train_cfg.get("eval_every", 1)
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            macro_f1, accuracy, report = evaluate(model, test_dataset, device, cfg)
            print(f"\n[epoch {epoch + 1}] test accuracy={accuracy:.4f} macro-F1={macro_f1:.4f}")
            print(report)

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_macro_f1": max(best_macro_f1, macro_f1),
                "classes": CLASSES,
                "config": cfg,
            }
            torch.save(state, output_dir / "last.pth")
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(state, output_dir / "best.pth")
                with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"epoch": epoch + 1, "macro_f1": macro_f1, "accuracy": accuracy},
                        f,
                        indent=2,
                    )
                print(f"New best macro-F1 {macro_f1:.4f} -> saved {output_dir / 'best.pth'}")

    print(f"\nDone. Best macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
