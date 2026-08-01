from pathlib import Path
from collections import defaultdict, Counter
import random

random.seed(1024)

src = Path("labels/UCF_full_train_split.txt")
dst = Path("labels/UCF_full_train_split_balanced.txt")

rows = [
    line.strip()
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

by_class = defaultdict(list)

for row in rows:
    label = int(row.split()[-1])
    by_class[label].append(row)

# ????? ??? ?? ????
targets = {
    0: 420,   # Normal
    1: 105,
    2: 105,
    3: 105,
    4: 105,
    5: 140,
    6: 105,
    7: 105,
    8: 140,
    9: 140,
    10: 140,  # Shooting
    11: 105,
    12: 140,
    13: 140,  # Vandalism
}

balanced = []

for label in range(14):
    items = by_class[label]
    target = targets[label]

    if target <= len(items):
        selected = random.sample(items, target)
    else:
        selected = items.copy()

        while len(selected) < target:
            selected.append(random.choice(items))

    balanced.extend(selected)

random.shuffle(balanced)

dst.write_text("\n".join(balanced) + "\n", encoding="utf-8")

counts = Counter(int(row.split()[-1]) for row in balanced)

print("Created:", dst)
print("Total samples:", len(balanced))
print("Balanced counts:")

for label in range(14):
    print(label, counts[label])
