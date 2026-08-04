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

## Round 2 — why macro avg stays low after retraining

Retraining with the fixed loss moved overall accuracy 61.8% → 63.9% and
Normal recall 0.87 → 0.91, but Anomaly-Type Accuracy stayed at ~37%.
Counting how often each class is *predicted* over the 296 test videos
explains where the remaining error lives:

| class | true videos | times predicted | ratio |
|---|---|---|---|
| Fighting | 8 | **2** | 0.25 |
| Vandalism | 8 | 5 | 0.62 |
| Assault | 8 | 3 | 0.38 |
| RoadAccidents | 23 | **39** | 1.70 |
| Stealing | 15 | 20 | 1.33 |

Fighting is emitted twice in the whole test set. The model has not
"failed" on Fighting — it has learned never to say Fighting, and the
probability mass moved to the frequent neighbours. Three causes:

**1. Class imbalance was never actually corrected.** Training used
`UCF_full_train_split.txt`, where Normal is ~50% of videos and the rare
types ~3% each, while the loss weights were 1.0 everywhere except a
hand-picked 2.0 for Shooting and Vandalism. Normal therefore carried
~19x more gradient mass than Fighting. `build_class_weights` (a table of
hand-tuned constants) existed but was never called. It is now replaced by
inverse-frequency weights computed from the actual training file, which
brings that ratio down to ~3.6x, and the counts and weights are logged at
epoch 0 so the balance is visible.

**2. The class prototypes are CLIP text embeddings of bare class names.**
`generate_text` uses the template `"{}"`, so the classifier weight vector
for each class is the CLIP embedding of a single word, and
`MODEL.FIX_TEXT: True` freezes them. "Stealing", "Robbery", "Shoplifting"
and "Burglary" are near-synonyms in language space, so their prototypes
are nearly collinear and cannot separate. The confusion matrix shows
exactly this shape: of the 61 theft-cluster videos, 66% stay inside the
cluster but only 27 land on the right member.

Measured on the real CLIP ViT-B/32 text encoder, the bare-name
prototypes have a mean pairwise cosine of 0.83 (Abuse/Stealing 0.91,
Assault/Arrest 0.91, Fighting/Shooting 0.90). Feeding a *perfect* video
feature — one exactly equal to its own class prototype — through
`softmax(logit_scale * cos)` then yields only P(true class) = 0.44 on
average and 0.32 for the worst class. That is a ceiling imposed by the
text geometry alone, before the visual encoder makes a single mistake.

`labels/ucf_14_labels_descriptive.csv` replaces the bare words with
concrete visual scene descriptions ("a burglar climbing through a broken
window of an empty dark house" rather than "Burglary"). The same
measurement on those prompts gives a mean ceiling of 0.86 and a worst
class of 0.67. Abstract legal terms cluster in CLIP's language space;
descriptions of what the camera actually sees do not.

**3. Eight test videos per class.** Nine of the fourteen classes have
support 8, so one video is 12.5% recall and a 0.0000 score is one unlucky
draw away from 0.125. macro avg weights those nine classes the same as
the 148-video Normal class, which is why it sits near 0.33 while weighted
avg is 0.63. Report both, and treat per-class numbers on support-8
classes as indicative only.

For context, on the anomaly videos the model gets the type right 37% of
the time against 15.5% for always predicting the most common anomaly type
and 7.7% for uniform random — the 14-class conversion is learning real
signal, it is the rare-class tail that collapses.

## Round 3 — results after the fixes, and where the bottleneck moved

`exp_v3` = descriptive prompts + inverse-frequency weights + balanced
train file, 15 epochs. Compared with `exp_v2` (bare prompts, ad-hoc
weights, unbalanced split):

| metric | exp_v2 | exp_v3 |
|---|---|---|
| Binary video-level AUC | 0.9473 | **0.9637** |
| Anomaly-Type Accuracy | 0.3716 | **0.4054** |
| macro F1 | 0.3327 | **0.3713** |
| macro precision | 0.3745 | **0.4093** |
| multiclass accuracy | 0.6385 | 0.6351 |

Per-class F1 for the classes that were dead: Arrest 0.000 → 0.556,
Vandalism 0.000 → 0.167, Abuse 0.462 → 0.714, Stealing 0.200 → 0.350.
Fighting is now *predicted* 9 times instead of 2, so the collapse is
fixed, but it is still 0/8 correct — three of those nine predictions are
Assault videos.

Overall accuracy is flat because the gains on rare classes are paid for
by Normal (13 → 18 normal videos called anomalous). That is the expected
trade when the loss stops being dominated by the majority class, and it
is the right trade for a 14-class task: macro F1 and type accuracy both
went up.

**The bottleneck is no longer the prompts.** The two worst remaining
confusions are Shoplifting → Robbery (6 of 8) and Burglary → Robbery
(5 of 15), yet those prompt pairs have cosine 0.71 and 0.67 — well
separated in text space. A rewrite aimed at pushing them further apart
was measured and made things *worse* (mean ceiling 0.86 → 0.74), so the
prompts in `ucf_14_labels_descriptive.csv` are kept. What remains is a
visual problem: 5 frames at 224px through ViT-B/32 do not carry enough
evidence to tell a shoplifter from an armed robber in the same shop, or
a one-sided assault from a group brawl.

## Warning: `TEST.NUM_CROP: 3` is broken upstream

`datasets/build.py` builds the validation pipeline as

```
[0] SampleFrames  [1] RawFrameDecode  [2] Resize  [3] CenterCrop
[4] Normalize     [5] FormatShape     [6] Collect [7] ToTensor
```

and then, for three-crop testing, does

```python
val_pipeline[3] = dict(type='Resize', ...)
val_pipeline[4] = dict(type='ThreeCrop', ...)
```

Index 4 is `Normalize`, so enabling `NUM_CROP: 3` silently **deletes
normalisation** and feeds raw 0-255 pixels to a CLIP backbone that
expects mean/std-normalised input. Results under that setting are
meaningless. Insert instead of overwrite if three-crop testing is wanted:

```python
if config.TEST.NUM_CROP == 3:
    val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
    val_pipeline.insert(4, dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE))
```

`TEST.NUM_CLIP > 1` (temporal multi-view) has no such problem — it only
replaces index 0 with a `multiview` `SampleFrames`, which the pipeline
supports.

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
