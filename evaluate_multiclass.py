import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

SCORES_FILE = Path("exp/test_scores.pkl")
ANNOTATION_FILE = Path("labels/UCF_full_test_split.txt")

# How many of the most anomalous clips vote for the anomaly type.
TOP_K = 4

# A video whose max clip-level anomaly evidence stays below this
# threshold is predicted Normal.
ANOMALY_THRESHOLD = 0.5

CLASS_NAMES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


def normalize_video_name(value):
    value = str(value).replace("\\", "/")
    filename = value.split("/")[-1]
    return Path(filename).stem.lower()


def parse_annotation_label(parts):
    """Read the class label exactly as datasets/build.py FrameDataset does.

        4 fields  -> path start end label   (label at index 3)
        otherwise -> path end label ...     (label at index 2)

    The custom `UCF_full_*` splits have 4 fields, but the standard
    `UCF_test.txt` has 7 (`path frames class start end start2 end2`).
    Reading `parts[-1]` therefore picks the trailing temporal marker
    `-1` on the standard split, which silently labels every video as an
    anomaly. Evaluation must read the same column the dataset reads.
    """
    if len(parts) == 4:
        return int(parts[3])

    return int(parts[2])


def read_ground_truth(annotation_file):
    ground_truth = {}
    original_paths = {}

    with Path(annotation_file).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            video_path = parts[0]
            label = parse_annotation_label(parts)
            video_name = normalize_video_name(video_path)

            ground_truth[video_name] = label
            original_paths[video_name] = video_path

    return ground_truth, original_paths


def scores_to_prediction(
    scores,
    anomaly_threshold=ANOMALY_THRESHOLD,
    top_k=TOP_K,
):
    """Two-stage video-level decision.

    Stage 1 (is it anomalous?): the video's anomaly score is the MAX
    clip-level anomaly evidence, 1 - P(Normal). Mean pooling over all
    clips is wrong here: an anomaly usually covers only a few clips,
    so the average is dominated by the normal clips and nearly every
    video ends up predicted as Normal.

    Stage 2 (which anomaly?): only the top-k most anomalous clips
    vote, by averaging their probabilities over classes 1..13.

    The stage-2 vote vector (classes 1..13, in that order) is returned
    as well, because a class with zero F1 cannot be diagnosed from the
    argmax alone: it matters whether the true class came second in that
    vote or ninth.
    """
    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim == 1:
        clip_probs = scores.reshape(1, -1)
    else:
        clip_probs = scores.reshape(-1, scores.shape[-1])

    if clip_probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, "
            f"but got shape {clip_probs.shape}"
        )

    anomaly_evidence = 1.0 - clip_probs[:, 0]
    video_anomaly_score = float(anomaly_evidence.max())

    k = min(top_k, clip_probs.shape[0])
    top_clips = np.argsort(anomaly_evidence)[-k:]
    class_scores = clip_probs[top_clips, 1:].mean(axis=0)
    anomaly_class = 1 + int(np.argmax(class_scores))

    if video_anomaly_score < anomaly_threshold:
        predicted_class = 0
    else:
        predicted_class = anomaly_class

    return predicted_class, anomaly_class, video_anomaly_score, class_scores


EPSILON = 1e-12


def class_diagnostics(
    y_true,
    y_pred,
    y_anomaly_class,
    y_class_scores,
    log=print,
):
    """Explain a per-class F1 of 0.0000 instead of just reporting it.

    A dead class has three possible causes, and they need different
    fixes, so the report has to tell them apart:

    1. *Stage 1 ate it.* The video never passed the anomaly threshold,
       so it was called Normal and the type vote was discarded. Fix the
       threshold, not the classifier.
    2. *The vote is close.* The true class is ranked second or third in
       the stage-2 vote, losing by a small margin. A per-class bias --
       the logit adjustment, or a different class-weight scheme -- can
       recover it, and the `boost` column says exactly how much is
       needed and how many other videos it would cost.
    3. *The vote is not close.* The true class ranks eighth of thirteen.
       No re-weighting recovers that; the features do not carry the
       distinction.

    `y_class_scores` is the stage-2 vote matrix, [videos, 13], columns
    ordered as classes 1..13.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_anomaly_class = np.asarray(y_anomaly_class)
    scores = np.asarray(y_class_scores, dtype=np.float64)

    if scores.ndim != 2 or scores.shape[1] != len(CLASS_NAMES) - 1:
        log("Per-class diagnosis needs the stage-2 vote matrix; skipping.")
        return {}

    log("=" * 78)
    log("Per-class diagnosis (anomaly classes only)")
    log("")
    log(f"{'class':15s} {'sup':>4s} {'pred':>5s} {'TP':>3s} "
        f"{'norm':>5s} {'r=1':>4s} {'r<=3':>5s} {'med r':>6s} {'margin':>7s}")

    log_scores = np.log(np.clip(scores, EPSILON, None))
    report = {}

    for class_id in range(1, len(CLASS_NAMES)):
        column = class_id - 1
        is_true = y_true == class_id
        support = int(is_true.sum())

        if support == 0:
            continue

        # How often the true class wins, and by how much it loses when
        # it does not. The margin is in log space, so it is directly
        # comparable to the logit adjustment's per-class shift.
        rows = log_scores[is_true]
        own = rows[:, column]
        other = np.delete(rows, column, axis=1).max(axis=1)
        margins = other - own
        ranks = 1 + (rows > own[:, None]).sum(axis=1)

        blocked = int(((y_pred == 0) & is_true).sum())
        positive = margins[margins > 0]
        needed = float(np.median(positive)) if positive.size else 0.0

        report[CLASS_NAMES[class_id]] = {
            "support": support,
            "predicted": int((y_pred == class_id).sum()),
            "true_positives": int(((y_pred == class_id) & is_true).sum()),
            "blocked_by_threshold": blocked,
            "median_rank": float(np.median(ranks)),
            "median_margin": needed,
        }

        log(f"{CLASS_NAMES[class_id]:15s} {support:4d} "
            f"{int((y_pred == class_id).sum()):5d} "
            f"{int(((y_pred == class_id) & is_true).sum()):3d} "
            f"{blocked:5d} {int((ranks == 1).sum()):4d} "
            f"{int((ranks <= 3).sum()):5d} {np.median(ranks):6.1f} "
            f"{needed:7.2f}")

    log("")
    log("sup = true videos, pred = times predicted, norm = called Normal by")
    log("stage 1, r = rank of the true class in the stage-2 vote, margin =")
    log("log-space boost that class needs to win the videos it currently")
    log("loses. A class with median rank 1-2 is a calibration problem; one")
    log("with median rank above ~4 is a feature problem.")
    log("")
    log("What a per-class boost would buy (applied to that class alone):")
    log("")
    log(f"{'class':15s} {'boost':>6s} {'gained':>7s} {'lost':>6s} {'net':>5s}")

    for class_id in range(1, len(CLASS_NAMES)):
        name = CLASS_NAMES[class_id]

        if name not in report or report[name]["median_margin"] <= 0:
            continue

        boost = report[name]["median_margin"] + 1e-6
        shifted = log_scores.copy()
        shifted[:, class_id - 1] += boost
        new_class = 1 + shifted.argmax(axis=1)

        # Stage 1 is untouched: a video below the anomaly threshold
        # stays Normal however the type vote is shifted.
        new_pred = np.where(y_pred == 0, 0, new_class)

        gained = int(((new_pred == y_true) & (y_pred != y_true)).sum())
        lost = int(((new_pred != y_true) & (y_pred == y_true)).sum())

        report[name]["boost_gained"] = gained
        report[name]["boost_lost"] = lost

        log(f"{name:15s} {boost:6.2f} {gained:7d} {lost:6d} "
            f"{gained - lost:5d}")

    log("")
    log("Each row is that class's boost applied on its own, at the median")
    log("margin above. Fit any boost on validation data, never on this")
    log("table -- these are test videos, and with 3-8 videos per class the")
    log("gain is one or two videos wide.")
    log("=" * 78)

    return report


def evaluate_from_scores(
    predictions,
    annotation_file,
    anomaly_threshold=ANOMALY_THRESHOLD,
    top_k=TOP_K,
    log=print,
    diagnose=False,
):
    if not isinstance(predictions, dict):
        raise TypeError(
            f"Expected predictions dict, got {type(predictions)}"
        )

    ground_truth, original_paths = read_ground_truth(annotation_file)

    y_true = []
    y_pred = []
    y_anomaly_class = []
    y_anomaly_score = []
    y_class_scores = []
    matched_names = []
    missing_predictions = []
    unknown_prediction_keys = []

    for prediction_key, scores in predictions.items():
        video_name = normalize_video_name(prediction_key)

        if video_name not in ground_truth:
            unknown_prediction_keys.append(prediction_key)
            continue

        (
            predicted_class,
            anomaly_class,
            anomaly_score,
            class_scores,
        ) = scores_to_prediction(
            scores,
            anomaly_threshold=anomaly_threshold,
            top_k=top_k,
        )

        y_true.append(ground_truth[video_name])
        y_pred.append(predicted_class)
        y_anomaly_class.append(anomaly_class)
        y_anomaly_score.append(anomaly_score)
        y_class_scores.append(class_scores)
        matched_names.append(video_name)

    prediction_names = {
        normalize_video_name(key)
        for key in predictions.keys()
    }

    for video_name in ground_truth:
        if video_name not in prediction_names:
            missing_predictions.append(video_name)

    if not y_true:
        raise RuntimeError(
            "No prediction keys matched the annotation file."
        )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_anomaly_class = np.asarray(y_anomaly_class, dtype=np.int64)
    y_anomaly_score = np.asarray(y_anomaly_score, dtype=np.float64)
    y_class_scores = np.asarray(y_class_scores, dtype=np.float64)

    accuracy = accuracy_score(y_true, y_pred)
    correct = int((y_true == y_pred).sum())

    log("=" * 70)
    log(f"Matched Videos: {len(y_true)}")
    log(f"Anomaly Threshold: {anomaly_threshold}  Top-K clips: {top_k}")
    log(f"Multiclass Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    log(f"Correct Predictions: {correct} / {len(y_true)}")

    is_anomaly = y_true > 0

    if len(set(is_anomaly.tolist())) == 2:
        binary_auc = roc_auc_score(is_anomaly, y_anomaly_score)
        log(f"Binary Video-Level AUC (1 - P(Normal), max over clips): "
            f"{binary_auc:.4f}")

    # Threshold-free measure of the 14-class conversion quality:
    # among the truly anomalous videos, how often is the TYPE right?
    if is_anomaly.any():
        type_correct = (
            y_anomaly_class[is_anomaly] == y_true[is_anomaly]
        )
        type_accuracy = float(type_correct.mean())
        log(f"Anomaly-Type Accuracy (on true anomaly videos only, "
            f"threshold-free): {type_accuracy:.4f} "
            f"({int(type_correct.sum())}/{int(is_anomaly.sum())})")

    log("=" * 70)

    log("Classification Report:")
    log("\n" + classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    log("Confusion Matrix (rows: true, columns: predicted):")
    log("\n" + np.array2string(matrix))

    if diagnose:
        class_diagnostics(
            y_true, y_pred, y_anomaly_class, y_class_scores, log=log
        )

    wrong_indices = np.where(y_true != y_pred)[0]

    log(f"Wrong Predictions: {len(wrong_indices)}")

    for index in wrong_indices[:20]:
        video_name = matched_names[index]
        true_class = CLASS_NAMES[int(y_true[index])]
        predicted_class = CLASS_NAMES[int(y_pred[index])]

        log(
            f"{original_paths[video_name]} | "
            f"true={true_class} | "
            f"predicted={predicted_class} | "
            f"anomaly_score={y_anomaly_score[index]:.3f}"
        )

    log(f"Missing predictions: {len(missing_predictions)}")

    for video_name in missing_predictions[:20]:
        log(original_paths[video_name])

    log(
        f"Prediction keys not found in annotation: "
        f"{len(unknown_prediction_keys)}"
    )

    for key in unknown_prediction_keys[:20]:
        log(key)

    return {
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_anomaly_class": y_anomaly_class,
        "y_anomaly_score": y_anomaly_score,
        "y_class_scores": y_class_scores,
        "confusion_matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES_FILE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="explain each class's score: how often it was blocked by the "
             "anomaly threshold, where the true class ranked in the type "
             "vote, and what a per-class boost would gain or cost",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="try a range of thresholds and top-k values on the saved scores "
             "instead of printing one full report",
    )
    args = parser.parse_args()

    if not args.scores.exists():
        raise FileNotFoundError(f"File not found: {args.scores}")

    if not args.annotations.exists():
        raise FileNotFoundError(f"File not found: {args.annotations}")

    with args.scores.open("rb") as file:
        result = pickle.load(file)

    if not isinstance(result, dict):
        raise TypeError(f"Expected result dict, got {type(result)}")

    if "prd" not in result:
        raise KeyError(
            f"'prd' key not found. Available keys: {list(result.keys())}"
        )

    print("Result type:", type(result))
    print("Result keys:", list(result.keys()))
    print("Prediction count:", len(result["prd"]))

    if args.sweep:
        quiet = lambda *a, **k: None
        print()
        print(f"{'top-k':>6s} {'thresh':>7s} {'accuracy':>9s} {'macro F1':>9s} "
              f"{'type acc':>9s}")

        best = None

        for top_k in (1, 2, 4, 8):
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                res = evaluate_from_scores(
                    result["prd"],
                    args.annotations,
                    anomaly_threshold=threshold,
                    top_k=top_k,
                    log=quiet,
                )

                macro_f1 = f1_score(
                    res["y_true"],
                    res["y_pred"],
                    average="macro",
                    labels=list(range(len(CLASS_NAMES))),
                    zero_division=0,
                )

                is_anomaly = res["y_true"] > 0
                type_acc = float(
                    (res["y_anomaly_class"][is_anomaly]
                     == res["y_true"][is_anomaly]).mean()
                )

                print(f"{top_k:6d} {threshold:7.2f} {res['accuracy']:9.4f} "
                      f"{macro_f1:9.4f} {type_acc:9.4f}")

                if best is None or macro_f1 > best[0]:
                    best = (macro_f1, top_k, threshold)

        print()
        print(f"best macro F1 = {best[0]:.4f} at top-k={best[1]} "
              f"threshold={best[2]}")
        print("Note: this picks hyper-parameters on the test split, so treat "
              "the swept numbers as an upper bound, not a clean test score.")
        return

    evaluate_from_scores(
        result["prd"],
        args.annotations,
        anomaly_threshold=args.threshold,
        top_k=args.topk,
        diagnose=args.diagnose,
    )


if __name__ == "__main__":
    main()
