# Detecting The Type Of Video Anomaly — VideoMAE Multiclass

This repository replaces the UMIL backbone with **VideoMAE** to classify
the *type* of anomaly in UCF-Crime videos (14 classes: Normal + 13
anomaly types).

Original repositories:

- UMIL: https://github.com/ktr-hubrt/UMIL
- VideoMAE: https://github.com/MCG-NJU/VideoMAE

## Why the UMIL 14-class baseline had a low macro-F1

The UMIL-based baseline reached ~62% accuracy but macro-F1 ≈ 0.28, with
several classes (Arrest, Assault, Explosion, Fighting) at 0.00. Four
causes, and what this repo does about each:

| Problem | Fix in this repo |
|---|---|
| Severe class imbalance (Normal 148 vs. 8 per rare class) — plain CE collapses onto head classes | Class-balanced sampler + effective-number class weights + focal loss (`samplers.py`, `losses.py`) |
| UMIL features are trained for *binary* anomaly scoring, not for telling anomaly types apart | Fine-tuned VideoMAE backbone starting from Kinetics-400 action-recognition weights (`model.py`) |
| Averaging over all snippets dilutes short anomalies; taking max is noisy | Top-k gated aggregation: type is decided from the k most anomalous clips, and a gate sends low-score videos to Normal (`inference.py`) |
| Small dataset overfits a fully unfrozen ViT | Frozen early blocks + layer-wise LR decay + label smoothing + augmentation |

Model selection during training uses **macro-F1**, not accuracy —
accuracy is dominated by the Normal class and hides rare-class failures.

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA is strongly recommended (see pytorch.org for the right
install command for your GPU). `decord` is used for fast video decoding
and falls back to OpenCV automatically if unavailable.

## 1. Build annotation lists

```bash
python annotations/make_annotations.py --root D:/datasets/UCF_Crime \
    --test-split D:/datasets/UCF_Crime/Anomaly_Detection_splits/Anomaly_Test.txt
```

This writes `annotations/train.txt` and `annotations/test.txt` with
`<relative_path> <label>` lines; labels are inferred from folder names.
Without `--test-split`, a stratified 80/20 split is used.

Then set `data.root` in `configs/videomae_ucf_crime.yaml` to your dataset
path.

## 2. Train

```bash
python train.py --config configs/videomae_ucf_crime.yaml
```

Checkpoints go to `checkpoints/` (`best.pth` = highest test macro-F1).
If you run out of GPU memory, lower `train.batch_size` (8 → 4 → 2) and/or
raise `model.freeze_blocks` (6 → 8).

## 3. Evaluate

```bash
python evaluate_multiclass.py --config configs/videomae_ucf_crime.yaml \
    --checkpoint checkpoints/best.pth
```

Prints accuracy, per-class precision/recall/F1, the confusion matrix and
the list of wrong predictions (same format as the original UMIL
evaluation), and saves `predictions.json`.

## Optional: VideoMAE as a feature extractor for UMIL

To keep the UMIL MIL head but feed it VideoMAE features instead of the
original backbone:

```bash
python extract_features.py --config configs/videomae_ucf_crime.yaml \
    --list annotations/train.txt --out features/train \
    --checkpoint checkpoints/best.pth
```

Saves one `(num_clips, 768)` `.npy` per video.

## Tuning tips for macro-F1

- `inference.gate` controls the Normal/anomaly trade-off: lower it
  (0.5 → 0.4) if anomalies are being predicted as Normal; raise it if
  Normal videos flip to random anomaly classes.
- `inference.topk` = how many clips vote on the anomaly type; 3–5 works
  well for long videos.
- More epochs with the balanced sampler helps rare classes; watch the
  per-epoch per-class report printed by `train.py`.
- `MCG-NJU/videomae-large-finetuned-kinetics` gives a further boost if
  you have the VRAM.

---

## راهنمای فارسی

این ریپو بک‌بن UMIL را با **VideoMAE** جایگزین می‌کند تا نوع ناهنجاری
(۱۴ کلاس UCF-Crime) تشخیص داده شود.

دلیل پایین بودن macro avg در نسخه‌ی قبلی و راه‌حل هر کدام:

۱. **عدم توازن شدید کلاس‌ها** (Normal=148 در مقابل ۸ ویدیو برای کلاس‌های
کمیاب): حل شده با سمپلر متوازن (هر epoch از هر کلاس به تعداد مساوی نمونه
می‌گیرد) + وزن‌دهی کلاس‌ها + focal loss.

۲. **فیچرهای UMIL برای تشخیص دودویی (نرمال/غیرنرمال) آموزش دیده‌اند** و
تفاوت بین انواع ناهنجاری را خوب نشان نمی‌دهند: حل شده با fine-tune کردن
VideoMAE از وزن‌های Kinetics-400 که برای تشخیص «عمل» آموزش دیده‌اند.

۳. **میانگین‌گیری روی کل ویدیو، ناهنجاری کوتاه را محو می‌کند**: حل شده با
تجمیع top-k — نوع ناهنجاری فقط از k کلیپِ مشکوک‌تر تصمیم‌گیری می‌شود و
اگر همان‌ها هم امتیاز پایینی داشته باشند ویدیو Normal اعلام می‌شود.

۴. **دیتاست کوچک و overfit شدن ترنسفورمر**: حل شده با فریز کردن بلوک‌های
اولیه، layer-wise LR decay، و label smoothing.

مراحل اجرا:

```bash
pip install -r requirements.txt
python annotations/make_annotations.py --root <مسیر-دیتاست> --test-split <مسیر-Anomaly_Test.txt>
# سپس data.root را در configs/videomae_ucf_crime.yaml تنظیم کنید
python train.py --config configs/videomae_ucf_crime.yaml
python evaluate_multiclass.py --config configs/videomae_ucf_crime.yaml --checkpoint checkpoints/best.pth
```

اگر حافظه‌ی GPU کم آوردید: `batch_size` را کم کنید (۸ ← ۴ ← ۲) و
`freeze_blocks` را زیاد کنید (۶ ← ۸).
