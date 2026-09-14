# نقشه‌ی اسلاید به کد

هر ادعای اسلایدهای دفاع، با فایل و شماره‌ی خط آن. شماره‌ی خط‌ها مربوط
به وضعیت فعلی مخزن است.

---

# مسیر اول — X-CLIP

## اسلاید ۴ — پروتکل تقسیم داده

| موضوع | فایل | خط |
|---|---|---|
| ساخت فایل حاشیه‌نویسی از پوشه‌ها | `make_full_ucf_annotations.py` | کل فایل |
| تقسیم ۷۰/۱۵/۱۵ | `split_full_ucf.py` | کل فایل |
| متوازن‌سازی با تکرار نمونه | `create_balanced_split.py` | کل فایل |

هدف‌های متوازن‌سازی داخل `create_balanced_split.py` نوشته شده‌اند:
عادی ۴۲۰ سطر، هفت کلاس با ۱۰۵ سطر و شش کلاس با ۱۴۰ سطر، جمعاً ۱۹۹۵.

---

## اسلاید ۸ — X-CLIP چطور کار می‌کند

| موضوع | فایل | خط |
|---|---|---|
| ساخت بردارهای متن از چهارده جمله | `main.py` | ۲۰۵ |
| چهارده جمله‌ی توصیفی | `labels/ucf_14_labels_descriptive.csv` | ۱۵ سطر |
| عبور تصویر و متن از مدل | `main.py` | ۷۶۸ |

```python
text_labels = generate_text(train_data)          # main.py:205
output = model(_image, text_inputs)              # main.py:768
```

فایل CSV از طریق کلید `DATA.LABEL_LIST` در فایل YAML خوانده می‌شود.
خودِ `generate_text` در `utils/tools.py` مخزن UMIL است و اینجا کامیت
نشده.

---

## اسلاید ۹ — سه اشکال بنیادی

### اشکال ۱ — میانگین‌گیری در ارزیابی

| | |
|---|---|
| فایل | `evaluate_multiclass.py` |
| تابع | `scores_to_prediction()` خط ۹۲ |
| خط اصلاح | ۱۲۷ |

```python
video_anomaly_score = float(anomaly_evidence.max())   # خط ۱۲۷
```

بیشینه به جای میانگین. دلیلش در رشته‌ی مستندسازی همان تابع نوشته شده.

### اشکال ۲ — فایل نمرات کهنه

| | |
|---|---|
| فایل | `main.py` |
| تعریف پرچم | خط ۸۸ |
| محل استفاده | خط ۲۱۸ |

```python
parser.add_argument('--reuse_scores', action='store_true', ...)   # ۸۸
if args.reuse_scores and os.path.exists(out_path):                # ۲۱۸
```

کامنت خطوط ۲۱۶ و ۲۱۷ همان موردی را ثبت می‌کند که عدد روی ده ویدیو از
۲۹۸ حساب شده بود.

### اشکال ۳ — انتخاب قطعه‌ی برتر به تفکیک کلاس

| | |
|---|---|
| فایل | `main.py` |
| تابع | `train_one_epoch()` خط ۳۵۷ |
| خطوط اصلاح | ۴۵۷ تا ۴۸۰ |

```python
anomaly_evidence = 1.0 - scores[:, :, 0]                    # ۴۶۲
_, topk_indices = torch.topk(anomaly_evidence, ...)         # ۴۶۴
gather_index = topk_indices.unsqueeze(-1).expand(...)       # ۴۷۰
topk_logits = torch.gather(logits, dim=1, index=...)        # ۴۷۴
logits_video = topk_logits.mean(dim=1)                      # ۴۸۰
```

خط ۴۷۰ همان جایی است که یک رتبه‌بندی واحد برای هر چهارده کلاس تکرار
می‌شود. کامنت خطوط ۴۵۷ تا ۴۶۱ دلیلش را می‌گوید.

---

## اسلاید ۱۰ — تصمیم دومرحله‌ای

هر دو مرحله در یک تابع: `evaluate_multiclass.py` خط ۹۲.

| مرحله | خط | کد |
|---|---|---|
| شواهد ناهنجاری هر قطعه | ۱۲۶ | `anomaly_evidence = 1.0 - clip_probs[:, 0]` |
| مرحله‌ی اول: بیشینه | ۱۲۷ | `video_anomaly_score = float(anomaly_evidence.max())` |
| انتخاب قطعه‌های برتر | ۱۳۰ | `top_clips = np.argsort(anomaly_evidence)[-k:]` |
| مرحله‌ی دوم: رأی‌گیری | ۱۳۱ | `class_scores = clip_probs[top_clips, 1:].mean(axis=0)` |
| برنده | ۱۳۲ | `anomaly_class = 1 + int(np.argmax(class_scores))` |

قطعه‌ی `[:, 1:]` در خط ۱۳۱ همان «کلاس عادی وارد رقابت نمی‌شود» است، و
`1 +` در خط ۱۳۲ اندیس را به شماره‌ی کلاس واقعی برمی‌گرداند.

---

## اسلاید ۱۱ — کلاسی که مدل یاد گرفت هرگز نگوید

| موضوع | فایل | خط |
|---|---|---|
| وزن معکوس فراوانی | `main.py` | `build_class_weights()` خط ۲۸۷ |
| ابزار جداگانه‌ی محاسبه | `calculate_class_weights.py` | کل فایل |
| تشخیص کلاس با امتیاز صفر | `evaluate_multiclass.py` | `class_diagnostics()` خط ۱۴۵ |

```python
weights = counts.sum() / (num_classes * counts)      # عکس فراوانی
weights = weights / weights.mean()                   # میانگین برابر یک
weights = torch.clamp(weights, min=0.25, max=4.0)    # برش
```

خط `raise ValueError` بالای این سه، جلوی تقسیم بر صفر را می‌گیرد وقتی
کلاسی هیچ ویدیوی آموزشی ندارد.

---

## اسلاید ۱۲ — محدودیت بنیادی: هندسه فضای متنی

این اسلاید اندازه‌گیری است، نه کد. عددهایش در `REVIEW_14CLASS.md`
خطوط ۹۵ تا ۱۱۰ ثبت شده‌اند: میانگین کسینوس دوبه‌دوی اسم‌های خام ۰٫۸۳،
سقف احتمال کلاس درست ۰٫۴۴ و بدترین کلاس ۰٫۳۲؛ با جمله‌های توصیفی
به ترتیب ۰٫۸۶ و ۰٫۶۷.

جمله‌ها در `labels/ucf_14_labels_descriptive.csv` هستند.

---

## اسلاید ۱۳ — نتایج مسیر اول

| سطر جدول | کجا محاسبه می‌شود |
|---|---|
| AUC دودویی | `main.py` تابع `evaluate_video_level_auc()` خط ۱۴ |
| ماکرو F1، صحت، صحت نوع | `evaluate_multiclass.py` تابع `evaluate_from_scores()` خط ۲۷۲ |
| میانگین ماکرو | `evaluate_multiclass.py` خط ۴۹۷ `average="macro"` |
| چهار نمای زمانی | تنظیمات، نه کد: `--opts TEST.NUM_CLIP 4` |

جدول «کلاس‌هایی که از مرگ برگشتند» از مقایسه‌ی دو اجرا می‌آید و عددهایش
در `REVIEW_14CLASS.md` خط ۱۳۹ ثبت شده‌اند.

---

# مسیر دوم — VideoMAE

| اسلاید | موضوع | فایل | خط |
|---|---|---|---|
| ۱۶، ۱۷ | یخ‌زدن و خط لوله | `extract_videomae_features.py` | کل فایل، ۳۱۰ خط |
| ۱۷ | انتخاب فریم‌های هر قطعه | همان | `sample_indices()` خط ۵۴ |
| ۱۸ | سر ترنسفورمر | `train_mil_head.py` | `class TemporalHead` خط ۱۲۲ |
| ۱۸ | سر ساده برای مقایسه | همان | `class MILHead` خط ۱۶۳ |
| ۱۹ | ارتقای رمزگذار | — | تنها یک گزینه‌ی خط فرمان |
| ۲۰ | آمیزش | `train_mil_head.py` | خطوط ۳۱۳ تا ۳۳۱ |
| ۲۰ | تنظیم لاجیت | همان | خطوط ۴۵۴ تا ۴۷۸ |
| ۲۰ | وزن کلاس‌ها | همان | `build_class_weights()` خط ۲۱۴ |
| ۲۲ | نتایج منفی | همان | گزینه‌های `--stride`، `--num-clips` |
| ۲۳ | کف نوفه | `bootstrap_ci.py` | کل فایل |
| ۲۳ | پنج بذر | `train_mil_head.py` | خط ۴۸۶ |
| ۲۳ | بودجه‌ی ثابت دوره | همان | خط ۵۳۳، پیام `no selection` |

ارتقای رمزگذار هیچ خط کدی لازم نداشت چون پهنای ورودی از خود داده
استنتاج می‌شود: `build_head(args.arch, tr_x.shape[-1], ...)` خط ۲۷۰.

---

# اگر داور پرسید «کجای گیت این‌ها هست؟»

```
git show 3973746
```

پیام آن کامیت دقیقاً همان سه اشکال اسلاید ۹ را فهرست می‌کند، با این
آمار:

| فایل | خط اضافه | خط حذف |
|---|---|---|
| `evaluate_multiclass.py` | ۳۰۳ | ۲۳۷ |
| `main.py` | ۳۲ | ۶ |
| `REVIEW_14CLASS.md` | ۷۳ | ۰ |

فایل‌های مسیر دوم در کامیت‌های بعدی با وضعیت `A` (افزوده) ثبت شده‌اند،
نه `M` (تغییریافته) — یعنی از صفر نوشته شده‌اند.
