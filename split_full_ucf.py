from pathlib import Path
from collections import defaultdict
import random

INPUT_FILE = Path("labels/UCF_full_all.txt")

TRAIN_FILE = Path("labels/UCF_full_train_split.txt")
VAL_FILE = Path("labels/UCF_full_val_split.txt")
TEST_FILE = Path("labels/UCF_full_test_split.txt")

random.seed(1024)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

items_by_class = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        label = int(parts[-1])
        items_by_class[label].append(line)

train_lines = []
val_lines = []
test_lines = []

for label, items in sorted(items_by_class.items()):
    random.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    train_lines.extend(train)
    val_lines.extend(val)
    test_lines.extend(test)

    print(f"class {label}: total={n}, train={len(train)}, val={len(val)}, test={len(test)}")

TRAIN_FILE.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
VAL_FILE.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
TEST_FILE.write_text("\n".join(test_lines) + "\n", encoding="utf-8")

print()
print("DONE")
print("train:", len(train_lines), TRAIN_FILE)
print("val:", len(val_lines), VAL_FILE)
print("test:", len(test_lines), TEST_FILE)