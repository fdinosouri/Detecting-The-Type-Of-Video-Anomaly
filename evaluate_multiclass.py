"""Evaluate a fine-tuned VideoMAE checkpoint on the 14-class test list.

Prints the same style of report as the original UMIL evaluation script
(accuracy, per-class precision/recall/F1, confusion matrix, wrong
predictions) and saves predictions to JSON.

Usage:
    python evaluate_multiclass.py --config configs/videomae_ucf_crime.yaml \
        --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from videomae_multiclass.dataset import VideoClipDataset
from videomae_multiclass.inference import predict_dataset
from videomae_multiclass.labels import CLASSES, NUM_CLASSES
from videomae_multiclass.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/videomae_ucf_crime.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output", default="predictions.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    data_cfg = cfg["data"]
    test_dataset = VideoClipDataset(
        list_file=data_cfg["test_list"],
        root=data_cfg["root"],
        num_frames=data_cfg.get("num_frames", 16),
        sampling_rate=data_cfg.get("sampling_rate", 4),
        crop_size=data_cfg.get("crop_size", 224),
        short_side=data_cfg.get("short_side", 256),
        mode="test",
        num_clips=cfg.get("inference", {}).get("num_clips", 10),
    )

    predictions, labels, paths, anomaly_scores = predict_dataset(
        model, test_dataset, device, cfg
    )

    correct = sum(p == t for p, t in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total

    print("=" * 70)
    print(f"Matched Videos: {total}")
    print(f"Multiclass Accuracy: {accuracy:.4f}")
    print(f"Multiclass Accuracy Percent: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {correct}")
    print(f"Total Predictions: {total}")
    print("=" * 70)
    print("\nClassification Report:\n")
    print(
        classification_report(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0,
        )
    )
    print("Confusion Matrix:\n")
    print(confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))))
    print("\nRows: True labels")
    print("Columns: Predicted labels")

    wrong = [
        (path, CLASSES[t], CLASSES[p])
        for path, t, p in zip(paths, labels, predictions)
        if p != t
    ]
    print(f"\nWrong Predictions: {len(wrong)}\n")
    for path, true_name, pred_name in wrong:
        print(f"{path} | true={true_name} | predicted={pred_name}")

    result = {
        "prd": {
            path: {
                "pred": CLASSES[p],
                "true": CLASSES[t],
                "anomaly_score": round(score, 4),
            }
            for path, p, t, score in zip(paths, predictions, labels, anomaly_scores)
        },
        "accuracy": accuracy,
    }
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved predictions to {output_path}")


if __name__ == "__main__":
    main()
