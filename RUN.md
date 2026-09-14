# دستورهای اجرا

هر دو مسیر پروژه، از صفر تا عدد نهایی. مسیرها را با مسیر واقعی
مجموعه‌ی داده روی دستگاه خودتان جایگزین کنید.

---

## مسیر دوم — VideoMAE (نتیجه‌ی نهایی)

این مسیر نتیجه‌ی گزارش‌شده را تولید می‌کند و کاملاً در همین مخزن هست.

### گام ۰ — نصب

```bash
pip install torch torchvision transformers opencv-python numpy scikit-learn
```

### گام ۱ — ساخت تقسیم استاندارد

```bash
python make_standard_split.py --root D:/dataset --source labels
```

خروجی: `labels/UCF_std_train.txt` (۱۶۱۰ ویدیو) و
`labels/UCF_std_test.txt` (۲۹۰ ویدیو).

### گام ۲ — بررسی تقسیم

```bash
python verify_split.py --root D:/dataset \
    --train labels/UCF_std_train.txt \
    --test  labels/UCF_std_test.txt
```

اگر ویدیویی پیدا نشود یا برچسبی خارج از بازه باشد، همین‌جا معلوم می‌شود.

### گام ۳ — استخراج ویژگی (یک بار، سنگین)

نسخه‌ی بزرگ، همان که در نتیجه‌ی نهایی استفاده شده:

```bash
python extract_videomae_features.py --root D:/dataset \
    --annotations labels/UCF_std_train.txt labels/UCF_std_test.txt \
    --model MCG-NJU/videomae-large-finetuned-kinetics \
    --out features/videomae_large
```

حدود ۱۵۰ دقیقه روی کارت گرافیک. اجرای دوباره از ویدیوهای کش‌شده رد
می‌شود، پس اگر وسط کار قطع شد فقط دوباره همین دستور را بزنید.

نسخه‌ی پایه برای مقایسه (اختیاری، حدود ۶۷ دقیقه):

```bash
python extract_videomae_features.py --root D:/dataset \
    --annotations labels/UCF_std_train.txt labels/UCF_std_test.txt \
    --out features/videomae
```

### گام ۴ — آموزش سر طبقه‌بند (پیکربندی نهایی)

```bash
python train_mil_head.py --features features/videomae_large \
    --train labels/UCF_std_train.txt \
    --test  labels/UCF_std_test.txt \
    --arch temporal --lr 3e-4 --mixup 0.2 --logit-adjust 1.0 \
    --seeds 5 --out exp_final
```

حدود ۸۰ ثانیه روی کارت گرافیک، حدود ۸۵ دقیقه روی پردازنده.
خروجی: `exp_final/test_scores.pkl`.

پیش از شروع، تعداد ویدیوها، شکل آرایه‌ها، وزن هر کلاس و دامنه‌ی
جابه‌جایی لاجیت چاپ می‌شود. اگر «missing features» صفر نبود، بخشی از
داده کنار گذاشته شده و عدد نهایی معنای دیگری دارد.

### گام ۵ — ارزیابی

```bash
python evaluate_multiclass.py --scores exp_final/test_scores.pkl \
    --annotations labels/UCF_std_test.txt
```

چند ثانیه. ماکرو F1، صحت چندکلاسه و صحت نوع ناهنجاری را می‌دهد.

### گام ۶ — بازه‌های اطمینان

```bash
python bootstrap_ci.py --scores exp_final/test_scores.pkl \
    --annotations labels/UCF_std_test.txt --iterations 4000
```

### گام ۷ — تشخیص کلاس‌های با امتیاز صفر

```bash
python evaluate_multiclass.py --scores exp_final/test_scores.pkl \
    --annotations labels/UCF_std_test.txt --diagnose
```

برای هر کلاس می‌گوید چند ویدیو داشت، چند بار پیش‌بینی شد، چند تا را
مرحله‌ی اول رد کرد، و کلاس درست در رأی‌گیری مرحله‌ی دوم چندم شد.

---

## مسیر اول — X-CLIP

این مسیر به مخزن اصلی UMIL نیاز دارد. در مخزن حاضر فقط فایل‌هایی
هست که تغییر داده شده‌اند؛ پوشه‌های `utils/`، `datasets/`، `models/`،
`configs/` و `tools/` اینجا نیستند.

### گام ۱ — آماده‌سازی

```bash
git clone https://github.com/ktr-hubrt/UMIL.git
cd UMIL
```

سپس این فایل‌ها را از مخزن حاضر روی آن کپی کنید:

```
main.py
evaluate_multiclass.py
create_balanced_split.py
calculate_class_weights.py
labels/ucf_14_labels_descriptive.csv
```

### گام ۲ — ساخت تقسیم و متوازن‌سازی

```bash
python make_full_ucf_annotations.py
python split_full_ucf.py
python create_balanced_split.py
python calculate_class_weights.py
```

### گام ۳ — تنظیم فایل کانفیگ

در `configs/` فایل YAML مربوط به UCF:

```yaml
DATA:
  ROOT: D:/path/to/videos
  TRAIN_FILE: labels/UCF_full_train_split_balanced.txt
  VAL_FILE:   labels/UCF_full_test_split.txt
  LABEL_LIST: labels/ucf_14_labels_descriptive.csv
  NUM_CLASSES: 14
```

`LABEL_LIST` مهم‌ترین سطر است: همان جمله‌های توصیفی را به رمزگذار متن
می‌دهد.

### گام ۴ — آموزش

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train_recognizer.sh 2
```

حدود شانزده ساعت.

### گام ۵ — آزمون

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test_recognizer.sh 1
```

یا مستقیم، با چهار نمای زمانی که بهترین نتیجه‌ی این مسیر را داد:

```bash
python main.py --config configs/ucf/[نام فایل].yaml \
    --output exp_v3 --only_test \
    --pretrained path/to/checkpoint.pth \
    --opts TEST.NUM_CLIP 4
```

نمرات به‌طور پیش‌فرض بازمحاسبه می‌شوند. برای بازاستفاده از فایل موجود
باید صریحاً `--reuse_scores` بدهید.

---

## برای جلسه‌ی دفاع

هیچ‌کدام از دستورهای سنگین بالا لازم نیست. اگر `test_scores.pkl` را
دارید، این دو دستور در چند ثانیه همه‌ی اعداد را می‌دهند و نه به ویدیو
نیاز دارند نه به کارت گرافیک:

```bash
python evaluate_multiclass.py --scores exp_final/test_scores.pkl \
    --annotations labels/UCF_std_test.txt

python bootstrap_ci.py --scores exp_final/test_scores.pkl \
    --annotations labels/UCF_std_test.txt --iterations 4000
```

---

## گزینه‌های پرکاربرد `train_mil_head.py`

| گزینه | پیش‌فرض | کار |
|---|---|---|
| `--arch` | `mlp` | `temporal` سر ترنسفورمر، `mlp` سر نقطه‌ای، `linear` خطی |
| `--epochs` | ۲۵ | بودجه‌ی ثابت، بدون توقف زودهنگام |
| `--batch-size` | ۳۲ | تعداد ویدیو در هر دسته |
| `--lr` | ۱e-۳ | برای سر ترنسفورمر ۳e-۴ بدهید |
| `--hidden` | ۵۱۲ | بعد پنهان |
| `--dropout` | ۰٫۳ | نرخ حذف تصادفی |
| `--topk` | ۴ | تعداد قطعه‌های رأی‌دهنده |
| `--mixup` | ۰ | ۰٫۲ تا ۰٫۴ معمول است |
| `--logit-adjust` | ۰ | ۱٫۰ تنظیم استاندارد |
| `--seeds` | ۱ | تعداد بذر؛ ۵ در پیکربندی نهایی |
| `--seed` | ۱۰۲۴ | بذر آغازین |
| `--balanced-sampler` | خاموش | نمونه‌برداری با احتمال معکوس فراوانی |
| `--class-weight-power` | ۱٫۰ | توان وزن‌ها؛ با سمپلر متوازن ۰ یا ۰٫۵ بدهید |

دو گزینه‌ی آخر را با هم و هر دو با قدرت کامل روشن نکنید: یک نامتوازنی
را دو بار اصلاح می‌کنند و مدل کلاس‌های کمیاب را بیش از اندازه پیش‌بینی
می‌کند.
