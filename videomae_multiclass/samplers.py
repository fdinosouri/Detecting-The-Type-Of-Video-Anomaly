"""Class-balanced sampling so every batch sees rare anomaly classes.

With natural sampling, a UCF-Crime epoch is mostly Normal videos and the
rare classes (Arrest, Assault, Explosion, Fighting, ...) appear so seldom
that the classifier never learns them. This sampler draws the same number
of videos per class each epoch (oversampling rare classes with
replacement), which is one of the most effective single changes for
raising macro-F1 on long-tailed data.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    def __init__(self, labels, samples_per_class: int | None = None, seed: int = 0):
        self.labels = np.asarray(labels)
        self.classes = np.unique(self.labels)
        self.indices_by_class = {
            c: np.flatnonzero(self.labels == c) for c in self.classes
        }
        if samples_per_class is None:
            counts = [len(v) for v in self.indices_by_class.values()]
            samples_per_class = int(np.median(counts))
        self.samples_per_class = max(samples_per_class, 1)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.samples_per_class * len(self.classes)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        chosen = []
        for c in self.classes:
            pool = self.indices_by_class[c]
            replace = len(pool) < self.samples_per_class
            chosen.append(rng.choice(pool, size=self.samples_per_class, replace=replace))
        chosen = np.concatenate(chosen)
        rng.shuffle(chosen)
        return iter(chosen.tolist())
