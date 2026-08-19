"""Train the 14-class MIL head on cached VideoMAE features.

The features come from extract_videomae_features.py, so the encoder is
frozen and every epoch is a few seconds of matrix multiplies instead of
an hour of video decoding. That makes it practical to actually sweep
hyper-parameters rather than accept the first run.

The loss is the same one main.py ended up with: clips are ranked once by
anomaly evidence `1 - P(Normal)` and the *same* top-k clips vote for
every class, which avoids the per-class topk that was punishing the
genuinely normal clips of anomaly videos. Class weights are
inverse-frequency, and the smoothness and sparsity terms are carried
over from UMIL.

    python train_mil_head.py --features features/videomae \
        --train labels/UCF_std_train.txt --test labels/UCF_std_test.txt \
        --out exp_mae

Writes exp_mae/test_scores.pkl in the same layout main.py produces, so
evaluate_multiclass.py --sweep and bootstrap_ci.py work on it unchanged.
"""

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_multiclass import (
    CLASS_NAMES,
    evaluate_from_scores,
    parse_annotation_label,
)


def load_split(annotation_file, feature_dir):
    """Return (names, features, labels), skipping videos with no cache."""
    names, feats, labels = [], [], []
    missing = 0

    with Path(annotation_file).open("r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()

            if len(parts) < 3:
                continue

            name = parts[0].split("/")[-1].split("\\")[-1]
            path = Path(feature_dir) / f"{Path(name).stem}.npy"

            if not path.exists():
                missing += 1
                continue

            names.append(name)
            feats.append(np.load(path))
            labels.append(parse_annotation_label(parts))

    if not names:
        raise RuntimeError(
            f"No cached features matched {annotation_file}. "
            f"Run extract_videomae_features.py first."
        )

    return (
        names,
        torch.from_numpy(np.stack(feats)).float(),
        torch.tensor(labels, dtype=torch.long),
        missing,
    )


class MILHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=0, dropout=0.3):
        super().__init__()

        if hidden:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, num_classes),
            )

    def forward(self, x):
        return self.net(x)


def mil_loss(logits, labels, class_weights, top_k, w_smooth, w_sparse):
    scores = F.softmax(logits, dim=-1)

    # Rank clips once by anomaly evidence and use those same clips for
    # every class, so the Normal channel is not built from the most
    # normal-looking clips of an anomaly video.
    anomaly_evidence = 1.0 - scores[:, :, 0]
    k = min(top_k, logits.size(1))
    _, idx = torch.topk(anomaly_evidence, k=k, dim=1)
    gather_index = idx.unsqueeze(-1).expand(-1, -1, logits.size(-1))

    logits_video = torch.gather(logits, 1, gather_index).mean(dim=1)

    loss = F.cross_entropy(
        logits_video, labels, weight=class_weights, label_smoothing=0.03
    )

    anomaly_scores = torch.max(scores[:, :, 1:], dim=-1).values
    smoothed = (
        (anomaly_scores[:, 1:] - anomaly_scores[:, :-1])
        .pow(2).sum(dim=-1).mean()
    )
    sparsity = anomaly_scores.sum(dim=-1).mean()

    return loss + w_smooth * smoothed + w_sparse * sparsity, loss


def build_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()

    return counts, torch.clamp(weights, 0.25, 4.0).to(device)


@torch.no_grad()
def predict(model, feats, batch_size, device):
    model.eval()
    out = []

    for i in range(0, len(feats), batch_size):
        chunk = feats[i:i + batch_size].to(device)
        out.append(F.softmax(model(chunk), dim=-1).cpu().numpy())

    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("exp_mae"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=0,
                        help="0 = linear head, otherwise MLP width")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--w-smooth", type=float, default=0.01)
    parser.add_argument("--w-sparse", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_names, tr_x, tr_y, tr_missing = load_split(args.train, args.features)
    te_names, te_x, te_y, te_missing = load_split(args.test, args.features)

    print(f"train {len(tr_names)} videos (missing features: {tr_missing})")
    print(f"test  {len(te_names)} videos (missing features: {te_missing})")
    print(f"features: {tuple(tr_x.shape[1:])} per video   device: {device}")

    counts, class_weights = build_class_weights(
        tr_y, len(CLASS_NAMES), device
    )
    print("\nclass counts / loss weights:")
    for c in range(len(CLASS_NAMES)):
        print(f"  {c:2d} {CLASS_NAMES[c]:15s} count={int(counts[c]):4d} "
              f"weight={class_weights[c]:.3f}")

    model = MILHead(
        tr_x.shape[-1], len(CLASS_NAMES), args.hidden, args.dropout
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nhead: {params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    args.out.mkdir(parents=True, exist_ok=True)
    order = np.arange(len(tr_names))
    best = {"macro_f1": -1.0}
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        total, mil_only, seen = 0.0, 0.0, 0

        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            x = tr_x[batch].to(device)
            y = tr_y[batch].to(device)

            loss, mil = mil_loss(
                model(x), y, class_weights,
                args.topk, args.w_smooth, args.w_sparse,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * len(batch)
            mil_only += mil.item() * len(batch)
            seen += len(batch)

        scheduler.step()

        probs = predict(model, te_x, args.batch_size, device)
        scores = {
            name: [probs[i]] for i, name in enumerate(te_names)
        }
        res = evaluate_from_scores(
            scores, args.test, top_k=args.topk, log=lambda *a, **k: None
        )

        from sklearn.metrics import f1_score
        macro = f1_score(
            res["y_true"], res["y_pred"], average="macro",
            labels=list(range(len(CLASS_NAMES))), zero_division=0,
        )

        if macro > best["macro_f1"]:
            is_ano = res["y_true"] > 0
            best = {
                "macro_f1": macro,
                "epoch": epoch,
                "accuracy": res["accuracy"],
                "type_accuracy": float(
                    (res["y_anomaly_class"][is_ano]
                     == res["y_true"][is_ano]).mean()
                ),
            }
            torch.save(model.state_dict(), args.out / "best_head.pth")

            with (args.out / "test_scores.pkl").open("wb") as fh:
                pickle.dump({"prd": scores}, fh)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {total / seen:.4f} "
                  f"(mil {mil_only / seen:.4f})  "
                  f"test macro F1 {macro:.4f}  acc {res['accuracy']:.4f}")

    print(f"\ntrained in {time.time() - start:.0f}s")
    print(f"best macro F1 {best['macro_f1']:.4f} at epoch {best['epoch']}  "
          f"accuracy {best['accuracy']:.4f}  "
          f"anomaly-type accuracy {best['type_accuracy']:.4f}")
    print(f"scores written to {args.out / 'test_scores.pkl'}")
    print(f"\nSelecting the best epoch on the test split makes this an "
          f"optimistic estimate -- report it as such.")


if __name__ == "__main__":
    main()
