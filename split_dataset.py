from pathlib import Path
import random
from collections import defaultdict

INPUT_FILE = Path("labels/UCF_train_small.txt")

TRAIN_OUT = Path("labels/UCF_train_split.txt")
VAL_OUT = Path("labels/UCF_val_split.txt")
TEST_OUT = Path("labels/UCF_test_split.txt")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SEED = 1024
random.seed(SEED)

lines = [line.strip() for line in INPUT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

by_class = defaultdict(list)

for line in lines:
    parts = line.split()

    if len(parts) < 4:
        print("Skipping bad line:", line)
        continue

    label = parts[-1]
    by_class[label].append(line)

train_lines = []
val_lines = []
test_lines = []

print("Total videos:", len(lines))
print("Classes found:", sorted(by_class.keys(), key=int))

for label in sorted(by_class.keys(), key=int):
    class_lines = by_class[label]
    random.shuffle(class_lines)

    total = len(class_lines)

    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    if total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
    elif total == 2:
        n_train = 1
        n_val = 0
    elif total == 1:
        n_train = 1
        n_val = 0

    train_part = class_lines[:n_train]
    val_part = class_lines[n_train:n_train + n_val]
    test_part = class_lines[n_train + n_val:]

    train_lines.extend(train_part)
    val_lines.extend(val_part)
    test_lines.extend(test_part)

    print(
        f"Class {label}: total={total}, "
        f"train={len(train_part)}, "
        f"val={len(val_part)}, "
        f"test={len(test_part)}"
    )

random.shuffle(train_lines)
random.shuffle(val_lines)
random.shuffle(test_lines)

TRAIN_OUT.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
VAL_OUT.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
TEST_OUT.write_text("\n".join(test_lines) + "\n", encoding="utf-8")

print("\nDONE")
print("Train:", len(train_lines), "->", TRAIN_OUT)
print("Validation:", len(val_lines), "->", VAL_OUT)
print("Test:", len(test_lines), "->", TEST_OUT)