from collections import Counter
from pathlib import Path

import torch


TRAIN_FILE = Path("labels/UCF_full_train_split.txt")

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

NUM_CLASSES = len(CLASS_NAMES)


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train split not found: {TRAIN_FILE}")

    class_counts = Counter()

    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line {line_number}: {line}"
                )

            try:
                label_id = int(parts[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at line {line_number}: {line}"
                ) from exc

            if not 0 <= label_id < NUM_CLASSES:
                raise ValueError(
                    f"Label {label_id} out of range at line {line_number}"
                )

            class_counts[label_id] += 1

    counts = torch.tensor(
        [class_counts[i] for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )

    if torch.any(counts == 0):
        missing = [
            CLASS_NAMES[i]
            for i, count in enumerate(counts)
            if count == 0
        ]
        raise ValueError(f"Classes without training samples: {missing}")

    total_samples = counts.sum()

    # Inverse-frequency balanced weights
    weights = total_samples / (NUM_CLASSES * counts)

    # Normalize weights around 1
    weights = weights / weights.mean()

    # Prevent extremely large or small weights
    weights = torch.clamp(weights, min=0.25, max=4.0)

    print("\nTraining class distribution")
    print("=" * 65)

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_id:2d} | "
            f"{class_name:15s} | "
            f"count={int(counts[class_id]):4d} | "
            f"weight={weights[class_id].item():.4f}"
        )

    print("=" * 65)
    print("Total:", int(total_samples.item()))

    print("\nPython class weights:")
    print([
        round(value, 6)
        for value in weights.tolist()
    ])


if __name__ == "__main__":
    main()