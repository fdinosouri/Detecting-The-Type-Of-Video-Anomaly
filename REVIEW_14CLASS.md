# Review: 14-class UMIL conversion — why the results looked bad, and what was fixed

## What was already working

The binary side of the model is fine. `test_epoch14_log.txt` shows a video-level
AUC of **0.944** (normal vs. anomaly, max over clips of `1 - P(Normal)`), so the
backbone and the MIL fine-tuning are learning real anomaly evidence. The
problems were in (1) how the 14-class prediction was aggregated at test time,
(2) a stale-scores trap in `--only_test`, and (3) how top-k clips were selected
during training.

## Problem 1 — mean pooling over all clips at evaluation (critical)

`evaluate_multiclass.py` averaged the softmax scores over all 16 clips and took
the argmax. In an anomaly video the anomaly usually covers only a few clips;
the other clips are genuinely normal, so the average is dominated by the
`Normal` class and almost every anomaly video gets predicted `Normal`. This
alone makes the multiclass accuracy look terrible even when the model is good.

**Fix:** two-stage decision in `scores_to_prediction`:

1. *Is it anomalous?* — video anomaly score = **max** over clips of
   `1 - P(Normal)` (same aggregation the binary AUC already uses). Below the
   threshold (`--threshold`, default 0.5) → predict `Normal`.
2. *Which anomaly?* — only the top-k most anomalous clips (`--topk`, default 4)
   vote, by averaging their probabilities over classes 1–13.

The script now also reports a threshold-free **Anomaly-Type Accuracy** computed
only on truly anomalous videos — this is the cleanest measure of the 14-class
conversion quality, independent of the normal/anomaly threshold.

## Problem 2 — stale `test_scores.pkl` in `--only_test` (critical)

`main.py` reused `exp/test_scores.pkl` whenever it existed, even if it was
produced by a different checkpoint or a different test split. In
`validation_debug.txt` this actually happened: **288 of 298 videos** were
"excluded on the result!" and the reported AUC (0.875) was computed on ~10
videos — a meaningless number. This also explains why the same
`checkpoint_epoch_4` produced 0.875 at 09:21 and 0.924 at 09:34 the same day.

**Fix:** scores are recomputed by default; pass `--reuse_scores` to reuse an
existing pkl deliberately. `--only_test` now also runs the full multiclass
evaluation (report + confusion matrix) automatically.

## Problem 3 — per-class top-k in the training MIL loss

`torch.topk(scores, k, dim=1)` on a `[B, clips, classes]` tensor takes the
top-k **independently for every class**: the `Normal` channel of the video
logits was built from the most normal-looking clips, the `Shooting` channel
from the most shooting-like clips, and so on. For an anomaly video the
cross-entropy then pushes down the `Normal` score of clips that really are
normal — systematic label noise that corrupts the 14-class training.

**Fix:** clips are ranked once by anomaly evidence `1 - P(Normal)`, and the
**same** top-k clips are used for every class (this is the multiclass
generalization of the original UMIL max-clip selection). For normal videos this
picks the hardest negatives, exactly like the original binary code.

## Recommendations (not changed in code)

- **Don't double-correct class imbalance.** The balanced train file already
  oversamples Shooting/Vandalism, and the loss additionally weights them 2.0.
  Pick one mechanism, or the model may over-predict those classes.
- **Retrain before judging.** Problem 3 affected training, so re-run training
  with the fixed `main.py`, then evaluate with
  `python main.py --config configs/ucf/32_5.yaml --only_test --resume <ckpt>`
  or `python evaluate_multiclass.py --scores exp/test_scores.pkl`.
- **Keep experiments comparable.** The logs mix runs with
  `UCF_full_train_split.txt` (1365 videos) and the balanced file (1995), and 15
  vs. 20 epochs. Fix one protocol when comparing checkpoints.
- The 0.944 AUC is a *video-level* AUC on a custom 70/15/15 split — it is not
  comparable to the frame-level AUC (~86.75) reported in the UMIL paper on the
  standard UCF-Crime split.
