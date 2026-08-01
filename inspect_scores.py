from pathlib import Path

import mmcv
import numpy as np


SCORES_FILE = Path("exp/test_scores.pkl")


def main():
    if not SCORES_FILE.exists():
        raise FileNotFoundError(f"Not found: {SCORES_FILE}")

    result = mmcv.load(str(SCORES_FILE))
    predictions = result["prd"]

    all_scores = []
    predicted_classes = []
    nan_videos = []
    inf_videos = []

    for video_name, value in predictions.items():
        score = np.asarray(value, dtype=np.float64)

        if score.ndim == 3:
            score = score.reshape(-1, score.shape[-1])
        elif score.ndim == 1:
            score = score.reshape(1, -1)

        if np.isnan(score).any():
            nan_videos.append(video_name)

        if np.isinf(score).any():
            inf_videos.append(video_name)

        all_scores.append(score)

        video_score = np.nanmean(score, axis=0)
        predicted_classes.append(int(np.argmax(video_score)))

    merged = np.concatenate(all_scores, axis=0)
    predicted_classes = np.asarray(predicted_classes)

    unique, counts = np.unique(
        predicted_classes,
        return_counts=True,
    )

    print("Videos:", len(predictions))
    print("Merged score shape:", merged.shape)
    print("NaN count:", int(np.isnan(merged).sum()))
    print("Inf count:", int(np.isinf(merged).sum()))
    print("NaN videos:", len(nan_videos))
    print("Inf videos:", len(inf_videos))

    print("\nPredicted class distribution:")
    for class_id, count in zip(unique, counts):
        print(f"class {class_id}: {count}")

    print("\nPer-class score statistics:")
    for class_id in range(merged.shape[1]):
        values = merged[:, class_id]

        print(
            f"class {class_id:2d} | "
            f"min={np.nanmin(values):.6f} | "
            f"max={np.nanmax(values):.6f} | "
            f"mean={np.nanmean(values):.6f} | "
            f"std={np.nanstd(values):.6f}"
        )

    print("\nFirst 5 video predictions:")

    for video_name, value in list(predictions.items())[:5]:
        score = np.asarray(value, dtype=np.float64)

        if score.ndim == 3:
            score = score.reshape(-1, score.shape[-1])
        elif score.ndim == 1:
            score = score.reshape(1, -1)

        video_score = np.nanmean(score, axis=0)

        print("\n", video_name)
        print("shape:", score.shape)
        print("mean score:", video_score)
        print("predicted:", int(np.argmax(video_score)))


if __name__ == "__main__":
    main()