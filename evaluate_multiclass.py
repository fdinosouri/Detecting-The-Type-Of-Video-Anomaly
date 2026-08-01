import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
SCORES_FILE = Path("exp/test_scores.pkl")
ANNOTATION_FILE = Path("labels/UCF_full_test_split.txt")

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


def read_ground_truth(annotation_file):
    ground_truth = {}
    original_paths = {}

    with annotation_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            video_path = parts[0]
            label = int(parts[-1])
            video_name = normalize_video_name(video_path)

            ground_truth[video_name] = label
            original_paths[video_name] = video_path

    return ground_truth, original_paths


def scores_to_prediction(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim == 3:
        video_scores = scores.mean(axis=(0, 1))
    elif scores.ndim == 2:
        video_scores = scores.mean(axis=0)
    elif scores.ndim == 1:
        video_scores = scores
    else:
        raise ValueError(
            f"Unexpected prediction shape: {scores.shape}"
        )

    if video_scores.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class scores, "
            f"but got shape {video_scores.shape}"
        )

    predicted_class = int(np.argmax(video_scores))

    return predicted_class, video_scores


def main():
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {SCORES_FILE}"
        )

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {ANNOTATION_FILE}"
        )

    ground_truth, original_paths = read_ground_truth(
        ANNOTATION_FILE
    )

    with SCORES_FILE.open("rb") as file:
        result = pickle.load(file)

    if not isinstance(result, dict):
        raise TypeError(
            f"Expected result dict, got {type(result)}"
        )

    if "prd" not in result:
        raise KeyError(
            f"'prd' key not found. Available keys: {list(result.keys())}"
        )

    predictions = result["prd"]

    if not isinstance(predictions, dict):
        raise TypeError(
            f"Expected predictions dict, got {type(predictions)}"
        )

    print("Result type:", type(result))
    print("Result keys:", list(result.keys()))
    print("Prediction container type:", type(predictions))
    print("Prediction count:", len(predictions))
    print("Ground-truth count:", len(ground_truth))

    y_true = []
    y_pred = []
    matched_names = []
    missing_predictions = []
    unknown_prediction_keys = []

    for prediction_key, scores in predictions.items():
        video_name = normalize_video_name(prediction_key)

        if video_name not in ground_truth:
            unknown_prediction_keys.append(prediction_key)
            continue

        predicted_class, _ = scores_to_prediction(scores)
        true_class = ground_truth[video_name]

        y_true.append(true_class)
        y_pred.append(predicted_class)
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

    accuracy = accuracy_score(y_true, y_pred)
    correct = int((y_true == y_pred).sum())

    print()
    print("=" * 70)
    print(f"Matched Videos: {len(y_true)}")
    print(f"Multiclass Accuracy: {accuracy:.4f}")
    print(f"Multiclass Accuracy Percent: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {correct}")
    print(f"Total Predictions: {len(y_true)}")
    print("=" * 70)

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Rows: True labels")
    print("Columns: Predicted labels")

    wrong_indices = np.where(y_true != y_pred)[0]

    print()
    print("Wrong Predictions:", len(wrong_indices))

    for index in wrong_indices[:20]:
        video_name = matched_names[index]
        true_class = CLASS_NAMES[int(y_true[index])]
        predicted_class = CLASS_NAMES[int(y_pred[index])]

        print(
            f"{original_paths[video_name]} | "
            f"true={true_class} | "
            f"predicted={predicted_class}"
        )

    print()
    print("Missing predictions:", len(missing_predictions))

    for video_name in missing_predictions[:20]:
        print(original_paths[video_name])

    print()
    print(
        "Prediction keys not found in annotation:",
        len(unknown_prediction_keys),
    )

    for key in unknown_prediction_keys[:20]:
        print(key)


if __name__ == "__main__":
    main()