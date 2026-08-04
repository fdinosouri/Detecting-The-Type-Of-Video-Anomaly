"""Bootstrap confidence intervals for the 14-class test metrics.

Nine of the fourteen classes have only 8 test videos, so a single video
moving between classes shifts macro F1 noticeably. This resamples the
matched test videos with replacement to estimate how much of the
reported number is sampling noise.

    python bootstrap_ci.py --scores exp_v3/test_scores.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from evaluate_multiclass import (
    ANNOTATION_FILE,
    ANOMALY_THRESHOLD,
    CLASS_NAMES,
    SCORES_FILE,
    TOP_K,
    evaluate_from_scores,
)

LABELS = list(range(len(CLASS_NAMES)))


def macro_f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=LABELS,
        zero_division=0,
    )


def summarise(name, point, samples, log=print):
    samples = np.asarray(samples)
    log(
        f"{name:24s} {point:.4f}   "
        f"std={samples.std():.4f}   "
        f"95% CI=[{np.percentile(samples, 2.5):.4f}, "
        f"{np.percentile(samples, 97.5):.4f}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES_FILE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.scores.exists():
        raise FileNotFoundError(f"File not found: {args.scores}")

    with args.scores.open("rb") as file:
        result = pickle.load(file)

    if "prd" not in result:
        raise KeyError(
            f"'prd' key not found. Available keys: {list(result.keys())}"
        )

    res = evaluate_from_scores(
        result["prd"],
        args.annotations,
        anomaly_threshold=args.threshold,
        top_k=args.topk,
        log=lambda *a, **k: None,
    )

    y_true = res["y_true"]
    y_pred = res["y_pred"]
    y_cls = res["y_anomaly_class"]
    y_score = res["y_anomaly_score"]
    n = len(y_true)

    point_f1 = macro_f1(y_true, y_pred)
    point_acc = float((y_true == y_pred).mean())
    is_ano = y_true > 0
    point_type = float((y_cls[is_ano] == y_true[is_ano]).mean())
    point_auc = roc_auc_score(is_ano, y_score)

    rng = np.random.default_rng(args.seed)
    boot_f1, boot_acc, boot_type, boot_auc = [], [], [], []

    for _ in range(args.iterations):
        idx = rng.integers(0, n, n)
        bt, bp = y_true[idx], y_pred[idx]

        boot_f1.append(macro_f1(bt, bp))
        boot_acc.append(float((bt == bp).mean()))

        ano = bt > 0
        if ano.any():
            boot_type.append(float((y_cls[idx][ano] == bt[ano]).mean()))

        # AUC is undefined unless both classes are present in the resample
        if 0 < ano.sum() < len(ano):
            boot_auc.append(roc_auc_score(ano, y_score[idx]))

    print(f"Videos: {n}   bootstrap iterations: {args.iterations}")
    print(f"Threshold: {args.threshold}   top-k: {args.topk}")
    print()
    print(f"{'metric':24s} {'point':>6s}")
    summarise("macro F1", point_f1, boot_f1)
    summarise("multiclass accuracy", point_acc, boot_acc)
    summarise("anomaly-type accuracy", point_type, boot_type)
    summarise("binary video-level AUC", point_auc, boot_auc)
    print()
    print(
        f"A macro-F1 difference below ~{2 * np.std(boot_f1):.3f} is not "
        f"distinguishable from sampling noise on this test split."
    )


if __name__ == "__main__":
    main()
