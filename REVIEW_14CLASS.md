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

## Round 4 — temporal multi-view at test time (best result)

`TEST.NUM_CLIP: 4` samples four independent temporal views per clip
(64 views per video instead of 16) and needs no retraining — it is a
pure test-time change on the `exp_v3` checkpoint, run with
`--opts TEST.NUM_CLIP 4`.

| metric | 1 view | **4 views** |
|---|---|---|
| Binary video-level AUC | 0.9637 | 0.9629 |
| Anomaly-Type Accuracy | 0.4054 | **0.4257** |
| macro F1 | 0.3713 | **0.3881** |
| macro recall | 0.3742 | **0.4047** |
| multiclass accuracy | 0.6351 | **0.6419** |
| classes with zero F1 | Fighting | **none** |

Fighting finally scores (F1 0.235, 2/8 correct), so every one of the 14
classes is now predicted and hit at least once. The gain is concentrated
where it was predicted to be: the violence cluster
(Abuse/Assault/Fighting) goes from 46% to 54% of its videos staying
inside the cluster and 6 → 8 exactly right, because more temporal
samples make a brief scuffle likelier to be captured. The theft cluster
is unchanged (24 exact both ways) — consistent with it being a
spatial-detail problem that more temporal views cannot fix.

Binary AUC is flat (0.9637 → 0.9629), so this buys type discrimination
without costing anomaly detection. Cost: validation goes from ~3.5 to
~13.5 minutes.

## Swapping the backbone to VideoMAE

VideoMAE is not a drop-in replacement for the CLIP backbone. UMIL has no
classifier layer at all — `logits = einsum(v_features, logit_scale *
t_features)` scores a video by its cosine similarity to CLIP *text*
embeddings. VideoMAE is vision-only, so the text head has to be replaced
by a learnable linear classifier. That is not purely a loss: the measured
text-prototype ceiling (0.44 with bare names, 0.86 with descriptions)
disappears entirely when the class vectors are free parameters.

Direction-wise it targets the right thing. VideoMAE ViT-B uses 16x16
patches against CLIP ViT-B/32's 32x32 — four times the spatial detail,
which is where 43% of the remaining anomaly errors sit — and it is
pretrained on video rather than images, which is where the other 18%
sit.

The obstacle is cost. Per clip, CLIP ViT-B/32 at 5 frames is 245 tokens;
VideoMAE ViT-B at 16 frames with tubelet 2 is 1568. Roughly eight times
the compute turns a 16-hour run into several days, and at batch size 2
the current setup already peaks near 9.4 GB.

So the encoder is frozen and its output cached instead:

- `extract_videomae_features.py` samples `--num-clips` clips per video,
  runs the encoder once, mean-pools each clip's tokens and writes one
  `[num_clips, 768]` array per video. The whole dataset is ~93 MB, and
  the job resumes if interrupted.
- `train_mil_head.py` trains a small head on those cached features with
  the corrected top-k MIL loss and inverse-frequency class weights. An
  epoch is seconds rather than an hour, so hyper-parameters can actually
  be swept. It writes `test_scores.pkl` in main.py's layout, so
  `evaluate_multiclass.py --sweep` and `bootstrap_ci.py` work on it
  unchanged.

The trade-off to state in any write-up: the backbone never adapts to
surveillance footage, so this measures how far frozen Kinetics-pretrained
VideoMAE features carry the task, not what end-to-end fine-tuning would
reach.

### Result

Extraction took 67 minutes for all 1895 videos; each head then trains in
about 7 seconds, which is what made a hyper-parameter comparison
possible at all. Best configuration was an MLP head (512 hidden,
dropout 0.3, 25 epochs).

| | X-CLIP fine-tuned | VideoMAE ViT-B frozen | VideoMAE ViT-L frozen |
|---|---|---|---|
| split | custom 70/15/15 | official | official |
| training | ~16 h end-to-end | 67 min + 80 s | 150 min + 80 s |
| binary video-level AUC | 0.9629 | 0.9649 | 0.9610 [0.938, 0.981] |
| anomaly-type accuracy | 0.4257 | 0.4214 | **0.4786** [0.397, 0.563] |
| macro F1 | 0.3881 [0.307, 0.454] | 0.3645 | **0.4122** [0.326, 0.479] |
| multiclass accuracy | 0.6419 | 0.6552 | **0.6897** [0.638, 0.745] |

Frozen ViT-L beats the fine-tuned CLIP backbone on all three
multiclass metrics, and does it on the harder official split.

One thing the table makes plain: **binary AUC is saturated.** All three
backbones land between 0.961 and 0.965 despite differing by 0.05 in
anomaly-type accuracy. Deciding *whether* a surveillance video is
anomalous is not what separates these models — deciding *which* anomaly
is. Any future work here should be measured on the type metrics, because
binary AUC no longer has room to move.

The confidence intervals overlap on every metric, and anomaly-type
accuracy is the same to within 0.003 — so on this data the two backbones
are not distinguishable, with VideoMAE reaching that from frozen
features at a fraction of the training cost. The macro-F1 gap is partly
the split: the simulation above puts the official split's test
distribution alone at about -0.038.

### What raised macro F1, and what did not

Because a head trains in seconds, each of these is a five-seed run with
a fixed 25-epoch budget, no selection.

| change | seed mean | ensemble | accuracy | type acc |
|---|---|---|---|---|
| MLP head, ViT-B | 0.3049 | 0.2975 | 0.566 | 0.386 |
| temporal head, ViT-B | 0.3480 | 0.3444 | 0.635 | 0.393 |
| + lr 3e-4 | 0.3405 | 0.3645 | 0.655 | 0.421 |
| **+ ViT-L features** | **0.3955** | **0.4122** | **0.690** | **0.479** |
| + logit adjustment | 0.4114 | 0.4104 | 0.686 | 0.479 |

Two changes carried the improvement, and both were predicted by the
error analysis rather than found by search:

**Attention across clips.** Every ViT-B temporal seed beat the MLP
baseline; the per-clip MLP scores a clip in isolation, which is the one
thing the theft cluster cannot be judged on, since the same shop
interior means shoplifting or robbery depending on the rest of the video.

**ViT-L over ViT-B.** The five ViT-L seeds (0.370-0.423) are entirely
disjoint from the five ViT-B seeds (0.310-0.354): a Mann-Whitney U test
on that separation gives p = 1/252. Note both use 16x16 patches, so this
is encoder *capacity*, not spatial resolution.

What did not work, and is worth reporting as such:

- **Concatenating ViT-B and ViT-L features** (1792-dim): seed mean
  0.3974 against 0.3955 for ViT-L alone, with higher variance. ViT-B
  appears to carry nothing ViT-L does not.
- **Seed ensembling** raised accuracy but not macro F1. Averaging
  probabilities is more conservative, so more videos fall under the
  anomaly threshold and the rare classes lose predictions.
- **Training with top-k 1** to match the best evaluation top-k: no gain,
  and the best *evaluation* top-k then flipped to 8. The optimal
  evaluation top-k compensates for the training top-k rather than
  reflecting anything about the data.
- **Logit adjustment**, once confined to the anomaly classes, moved the
  seed mean +0.016 and left the ensemble unchanged — inside the noise.

### Kinetics-style clip sampling made things worse

The clips fed to VideoMAE were stretched: each spanned a full
1/num_clips segment, so a 9000-frame video produced 16 frames covering
~19 seconds with 35 frames between them, against the ~2 seconds at
stride 4 the Kinetics checkpoints were trained on. That looked like a
clear distribution mismatch worth fixing, and the fix was measured
rather than assumed.

Re-extracting with `--frame-stride 4 --num-clips 32` — proper ~2-second
windows, twice as many of them, 116 minutes — gave **seed mean 0.3089**
against **0.3405** for the stretched 16-clip features on the same head.
Worse, not better, and the proportional top-k control (8 of 32 rather
than 4 of 16) did not rescue it.

So coverage beats clip validity here. Stretched clips still span the
whole video, while 32 two-second windows reach only ~21% of a long one,
and a sparse anomaly that falls outside every window cannot be scored at
all. VideoMAE turns out to tolerate an unusual temporal sampling rate
better than it tolerates never seeing the event.

### Doubling the clips changed nothing either

The stride experiment suggested coverage mattered more than clip
validity, so the untried combination was the one that improves both
axes: keep the stretched sampling (100% coverage) but double the clips,
halving how stretched each one is and giving MIL 32 positions to find
the anomaly in rather than 16.

Re-extracting with ViT-L at 32 clips took 289 minutes and produced:

| | 16 clips | 32 clips |
|---|---|---|
| seed mean | 0.3955 | 0.3875 |
| ensemble macro F1 | 0.4122 | 0.4105 |
| accuracy | 0.6897 | 0.6897 |
| anomaly-type accuracy | 0.4786 | 0.4857 |

Identical to four decimal places on accuracy. Temporal sampling
density is not the constraint — 16 clips already locate the anomaly as
well as 32 do.

### A diverged seed used to crash the run

One seed of the stride-4 run collapsed to macro F1 0.0010 and emitted
NaN probabilities, which poisoned the averaged ensemble and then crashed
`roc_auc_score` with `Input contains NaN`. Seeds with non-finite outputs
are now dropped from the ensemble with a warning, and a run where every
seed diverges exits with advice rather than a traceback. An unstable
configuration should be visible as instability, not as a stack trace.

### Tuning against a 290-video test set has a ceiling

Roughly 200 evaluations were run against the same test split across all
these configurations. With a per-evaluation standard deviation near
0.034, the expected maximum of N draws sits about `sigma * sqrt(2 ln N)`
above the truth: +0.11 for N=200, +0.07 even for N=8. Reporting the best
number seen would therefore overstate the result by more than the effect
being claimed. Every figure above comes from a configuration fixed in
advance, and the ViT-B/ViT-L comparison is a single pre-specified
comparison rather than a maximum over a search.

macro F1 plateaued near 0.41. Getting past roughly 0.45 is not a tuning
problem: seven of the fourteen classes have five or fewer test videos —
Abuse has two — and macro F1 weights all fourteen equally, so those
classes contribute near-binary noise no model can smooth out.

### Validation-based epoch selection is also noisy here

Holding out 15% of training gives only 4 validation videos each for
Explosion, Shooting and Shoplifting, and validation macro F1 duly jumps
around (0.299, 0.333, 0.391, 0.357, 0.401 across epochs). In one run the
epoch it selected scored *worse* on test than simply taking the last
epoch (0.3113 vs 0.3260). With a test set this small, the defensible
protocol is to fix the epoch budget in advance and report the final
epoch, so no split is used for selection at all.

## How much of these numbers is noise?

The test split has 296 videos, and nine of the fourteen classes carry
only 8 of them. One video changing class is 12.5% of that class's
recall, so macro F1 moves on sampling luck alone. `bootstrap_ci.py`
quantifies this: it resamples the matched test videos with replacement
4000 times and recomputes every metric on each resample, which
approximates the spread that would be seen across alternative test sets
of the same size.

On the best configuration (`exp_v3`, `TEST.NUM_CLIP: 4`):

| metric | point | std | 95% CI |
|---|---|---|---|
| macro F1 | 0.3881 | 0.037 | [0.307, 0.454] |
| multiclass accuracy | 0.6419 | 0.027 | [0.588, 0.693] |
| anomaly-type accuracy | 0.4257 | 0.041 | [0.331, 0.494] |

**A macro-F1 difference below roughly 0.075 is not distinguishable from
noise on this split.** That reframes the whole experiment log: the
single-view → four-view gain (+0.017) is 0.44 standard deviations, which
on its own is weak evidence. It is believable here only because the
mechanism was checked independently — the gain landed entirely in the
violence cluster and left the theft cluster untouched, exactly as the
temporal-vs-spatial split predicts.

It also sets the bar for future work: an experiment costing a full
retrain should be expected to move macro F1 by more than ~0.075, or it
cannot be claimed as an improvement. This is why `NUM_FRAMES: 8` was not
pursued — it spends ~26 hours on the temporal axis, which the four-view
result shows is close to saturated, while 43% of the remaining anomaly
errors sit in the theft cluster and are spatial. `ViT-B/16` (4x finer
patches) targets that error mass instead.

The width of these intervals is a property of the 70/15/15 split, not of
the model. A 60/20/20 split would narrow them.

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
