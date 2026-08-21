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

from sklearn.metrics import f1_score

from evaluate_multiclass import (
    ANOMALY_THRESHOLD,
    CLASS_NAMES,
    evaluate_from_scores,
    parse_annotation_label,
    scores_to_prediction,
)

LABELS = list(range(len(CLASS_NAMES)))


def predict_labels(probs, threshold, top_k):
    """Two-stage video decision for a batch of clip-probability arrays."""
    preds = [
        scores_to_prediction(p, anomaly_threshold=threshold, top_k=top_k)[0]
        for p in probs
    ]

    return np.asarray(preds, dtype=np.int64)


def stratified_split(labels, val_frac, seed):
    """Hold out val_frac of each class, so rare classes appear in both."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)

        # Never take the last example of a class away from training.
        n_val = min(int(round(len(idx) * val_frac)), len(idx) - 1)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def load_split(annotation_file, feature_dirs):
    """Return (names, features, labels), skipping videos with no cache.

    Several feature directories are concatenated along the channel axis,
    so features from different encoders can be combined without
    re-extracting anything.
    """
    if isinstance(feature_dirs, (str, Path)):
        feature_dirs = [feature_dirs]

    names, feats, labels = [], [], []
    missing = 0

    with Path(annotation_file).open("r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.split()

            if len(parts) < 3:
                continue

            name = parts[0].split("/")[-1].split("\\")[-1]
            paths = [
                Path(d) / f"{Path(name).stem}.npy" for d in feature_dirs
            ]

            if not all(q.exists() for q in paths):
                missing += 1
                continue

            names.append(name)
            feats.append(np.concatenate([np.load(q) for q in paths], axis=-1))
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


class TemporalHead(nn.Module):
    """Let each clip see the other clips before it is scored.

    The per-clip MLP scores every clip in isolation, so a clip can only
    say "this looks like a fight" from its own 16 frames. Attention over
    the 16 clips lets a clip be read in the context of the rest of the
    video, which is the distinction the theft classes need: the same shop
    interior means shoplifting or robbery depending on what the
    surrounding clips show.
    """

    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.3,
                 layers=2, heads=4):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.project = nn.Linear(in_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, 64, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)

        encoder = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.project(self.norm(x))
        x = x + self.position[:, :x.size(1)]
        x = self.encoder(x)

        return self.classifier(self.dropout(x))


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
def predict(model, feats, batch_size, device, logit_offset=None):
    """Clip probabilities, optionally with a long-tail logit adjustment.

    Subtracting tau * log(prior) from the logits at inference is the
    standard correction for long-tailed classification (Menon et al.,
    2021). Normal is half the training data while Shooting is under 2%,
    and macro F1 weights them equally, so shifting the decision boundary
    toward the rare classes is exactly what this metric rewards.
    """
    model.eval()
    out = []

    for i in range(0, len(feats), batch_size):
        chunk = feats[i:i + batch_size].to(device)
        logits = model(chunk)

        if logit_offset is not None:
            logits = logits - logit_offset

        out.append(F.softmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(out)


def build_head(arch, in_dim, num_classes, hidden, dropout):
    if arch == "temporal":
        return TemporalHead(in_dim, num_classes, hidden or 256, dropout)

    return MILHead(in_dim, num_classes, hidden if arch == "mlp" else 0, dropout)


def train_one(args, tr_x, tr_y, class_weights, eval_sets, seed, device,
              verbose, logit_offset=None):
    """Train one head from scratch and return probabilities for eval_sets."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_head(
        args.arch, tr_x.shape[-1], len(CLASS_NAMES), args.hidden, args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    order = np.arange(len(tr_y))

    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        total, seen = 0.0, 0

        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            loss, _ = mil_loss(
                model(tr_x[batch].to(device)), tr_y[batch].to(device),
                class_weights, args.topk, args.w_smooth, args.w_sparse,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * len(batch)
            seen += len(batch)

        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == args.epochs - 1):
            print(f"    epoch {epoch:3d}  loss {total / seen:.4f}", flush=True)

    return (
        model,
        [predict(model, x, args.batch_size, device, logit_offset)
         for x in eval_sets],
    )


def macro_f1_of(probs, labels, top_k):
    return f1_score(
        labels, predict_labels(probs, ANOMALY_THRESHOLD, top_k),
        average="macro", labels=LABELS, zero_division=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, nargs="+", required=True,
                        help="one or more cached-feature directories; "
                             "several are concatenated per clip")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("exp_mae"))
    parser.add_argument("--arch", choices=["linear", "mlp", "temporal"],
                        default="mlp",
                        help="temporal = attention across the clips of a video")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=1,
                        help="train this many heads and average their "
                             "probabilities; averaging cuts the run-to-run "
                             "variance that dominates a 290-video test set")
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="hold out this much of training and report a "
                             "validation score; it is never used to select "
                             "anything, the epoch budget is fixed in advance")
    parser.add_argument("--w-smooth", type=float, default=0.01)
    parser.add_argument("--w-sparse", type=float, default=0.001)
    parser.add_argument("--logit-adjust", type=float, default=0.0,
                        help="tau for the long-tail logit adjustment at "
                             "inference; 1.0 is the standard setting, "
                             "0 disables it")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_names, tr_x, tr_y, tr_missing = load_split(args.train, args.features)
    te_names, te_x, te_y, te_missing = load_split(args.test, args.features)

    print(f"train {len(tr_names)} videos (missing features: {tr_missing})")
    print(f"test  {len(te_names)} videos (missing features: {te_missing})")
    print(f"features: {tuple(tr_x.shape[1:])} per video   device: {device}")

    va_x = va_y = None

    if args.val_frac > 0:
        fit_idx, val_idx = stratified_split(
            tr_y.numpy(), args.val_frac, args.seed
        )
        va_x, va_y = tr_x[val_idx], tr_y[val_idx].numpy()
        tr_x, tr_y = tr_x[fit_idx], tr_y[fit_idx]
        print(f"holding out {len(val_idx)} videos to report a validation "
              f"score, fitting on {len(fit_idx)}")

    counts, class_weights = build_class_weights(
        tr_y, len(CLASS_NAMES), device
    )
    print("\nclass counts / loss weights:")
    for c in range(len(CLASS_NAMES)):
        print(f"  {c:2d} {CLASS_NAMES[c]:15s} count={int(counts[c]):4d} "
              f"weight={class_weights[c]:.3f}")

    logit_offset = None

    if args.logit_adjust:
        prior = (counts / counts.sum()).to(device)
        logit_offset = args.logit_adjust * torch.log(prior)
        print(f"\nlogit adjustment tau={args.logit_adjust} "
              f"(Normal {prior[0]:.3f} vs Shooting {prior[10]:.3f})")

    eval_sets = [te_x] + ([va_x] if va_x is not None else [])
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    te_probs, va_probs, members = [], [], []

    for run in range(args.seeds):
        seed = args.seed + run
        print(f"\nseed {seed} ({run + 1}/{args.seeds})", flush=True)

        model, outs = train_one(
            args, tr_x, tr_y, class_weights, eval_sets, seed, device,
            verbose=(args.seeds == 1), logit_offset=logit_offset,
        )

        te_probs.append(outs[0])
        if va_x is not None:
            va_probs.append(outs[1])

        members.append(macro_f1_of(outs[0], te_y.numpy(), args.topk))
        print(f"  test macro F1 {members[-1]:.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nhead: {args.arch}, {params:,} trainable parameters")
    print(f"trained {args.seeds} head(s) in {time.time() - start:.0f}s")

    probs = np.mean(te_probs, axis=0)
    scores = {name: [probs[i]] for i, name in enumerate(te_names)}

    with (args.out / "test_scores.pkl").open("wb") as fh:
        pickle.dump({"prd": scores}, fh)

    res = evaluate_from_scores(
        scores, args.test, top_k=args.topk, log=lambda *a, **k: None
    )
    is_ano = res["y_true"] > 0

    print(f"\n=== test, fixed {args.epochs}-epoch budget, no selection ===")
    if args.seeds > 1:
        print(f"  individual seeds  {', '.join(f'{m:.4f}' for m in members)}")
        print(f"  seed mean         {np.mean(members):.4f} "
              f"(std {np.std(members):.4f})")
    print(f"  macro F1                   "
          f"{macro_f1_of(probs, te_y.numpy(), args.topk):.4f}")
    print(f"  accuracy                   {res['accuracy']:.4f}")
    print(f"  anomaly-type accuracy      "
          f"{float((res['y_anomaly_class'][is_ano] == res['y_true'][is_ano]).mean()):.4f}")

    if va_x is not None:
        print(f"  (validation macro F1       "
              f"{macro_f1_of(np.mean(va_probs, axis=0), va_y, args.topk):.4f})")

    print(f"\nscores written to {args.out / 'test_scores.pkl'}")


if __name__ == "__main__":
    main()
