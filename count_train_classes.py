from collections import Counter
from pathlib import Path


ANNOTATION_FILE = Path("lists/ucf_CLIP_rgb_train.csv")


def infer_class(path_text):
    class_names = [
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

    normalized = path_text.lower()

    for class_name in class_names:
        if class_name.lower() in normalized:
            return class_name

    return "Unknown"


def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    counter = Counter()

    with ANNOTATION_FILE.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            class_name = infer_class(line)
            counter[class_name] += 1

    for class_name, count in counter.most_common():
        print(f"{class_name:15s}: {count}")


if __name__ == "__main__":
    main()