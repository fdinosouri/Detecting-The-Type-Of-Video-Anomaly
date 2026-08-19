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

    return predicted_class, anomaly_class, video_anomaly_score


def evaluate_from_scores(
    predictions,
    annotation_file,
    anomaly_threshold=ANOMALY_THRESHOLD,
    top_k=TOP_K,
    log=print,
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
    matched_names = []
    missing_predictions = []
    unknown_prediction_keys = []

    for prediction_key, scores in predictions.items():
        video_name = normalize_video_name(prediction_key)

        if video_name not in ground_truth:
            unknown_prediction_keys.append(prediction_key)
            continue

        predicted_class, anomaly_class, anomaly_score = scores_to_prediction(
            scores,
            anomaly_threshold=anomaly_threshold,
            top_k=top_k,
        )

        y_true.append(ground_truth[video_name])
        y_pred.append(predicted_class)
        y_anomaly_class.append(anomaly_class)
        y_anomaly_score.append(anomaly_score)
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
        "confusion_matrix": matrix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES_FILE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD)
    parser.add_argument("--topk", type=int, default=TOP_K)
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
    )


if __name__ == "__main__":
    main()
