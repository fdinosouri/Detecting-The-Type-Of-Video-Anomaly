# -*- coding: utf-8 -*-
"""Persian terminology with the Latin original moved to a footnote.

The writing guide asks for the body to be Persian and the foreign term to
appear once, as a footnote, where the Persian equivalent is first used.
Doing that by hand in the content modules is error-prone: a term usually
reaches the page through several separate string literals, so "first
occurrence" is not something the author can see while writing.

This module applies the rule at build time instead. `annotate` walks the
content in reading order and rewrites only the first occurrence of each
entry, wrapping the Latin form in the delimiters `footnotes.split_footnotes`
understands.

Two kinds of entry:

SWAPS  Latin still sitting in Persian prose. The Latin is replaced by the
       Persian form everywhere, and footnoted at the first replacement.
TERMS  Persian already in the prose with no Latin anywhere. The footnote
       is attached the first time the Persian appears.
"""

import re

# Latin left in the Persian body -> (Persian replacement, footnote)
SWAPS = [
    ("UCF-Crime", "یوسی‌اف‑کرایم", "UCF-Crime"),
    ("Kinetics", "کینتیکس", "Kinetics"),
    ("F1", "اف‑یک", "F1 score"),
]

# Persian already in the body -> footnote carrying the Latin original
TERMS = [
    ("تشخیص ناهنجاری در ویدیو", "Video Anomaly Detection (VAD)"),
    ("یادگیری چندنمونه‌ای بی‌طرف", "Unbiased Multiple Instance Learning (UMIL)"),
    ("یادگیری چندنمونه‌ای", "Multiple Instance Learning (MIL)"),
    ("نظارت ضعیف", "Weak Supervision"),
    ("خودرمزگذار نقاب‌دار ویدیویی", "Video Masked Autoencoder (VideoMAE)"),
    ("مبدل بینایی", "Vision Transformer (ViT)"),
    ("سر مبدل", "Transformer Head"),
    ("رمزگذار", "Encoder"),
    ("رمزگشایی", "Decoding"),
    ("ماکرو", "Macro-averaging"),
    ("خودگردان‌سازی", "Bootstrapping"),
    ("بازه‌ی اطمینان", "Confidence Interval"),
    ("کف نوفه", "Noise Floor"),
    ("تنظیم لاجیت", "Logit Adjustment"),
    ("دم بلند", "Long-tailed Distribution"),
    ("آمیزش", "Mixup"),
    ("حذف تصادفی", "Dropout"),
    ("افت وزن", "Weight Decay"),
    ("نرخ یادگیری", "Learning Rate"),
    ("بیش‌برازش", "Overfitting"),
    ("خودتوجهی", "Self-Attention"),
    ("تعبیه‌ی جایگاهی", "Positional Embedding"),
    ("ماتریس درهم‌ریختگی", "Confusion Matrix"),
    ("سطح زیر منحنی", "Area Under the ROC Curve (AUC)"),
    ("آنتروپی متقاطع", "Cross Entropy"),
    ("بذر تصادفی", "Random Seed"),
    ("آزمون من-ویتنی", "Mann-Whitney U test"),
    ("وصله", "Patch"),
    ("تورم انتخاب", "Selection Inflation"),
    ("نمونه‌برداری", "Sampling"),
    ("تابع زیان", "Loss Function"),
    ("گرادیان", "Gradient"),
    ("دقت", "Precision"),
    ("یادآوری", "Recall"),
    ("صحت", "Accuracy"),
    ("کیف", "Bag"),
    ("قطعه", "Clip"),
    ("دوره", "Epoch"),
    ("دسته", "Batch"),
    ("ابرپارامتر", "Hyperparameter"),
    ("تقطیر دانش", "Knowledge Distillation"),
    ("یادگیری انتقالی", "Transfer Learning"),
]

MARK, END = "⟨", "⟩"

# A Persian term must not be matched inside a longer word, so the match has
# to end on something that is not a Persian letter or a zero-width joiner.
BOUNDARY = r"(?![؀-ۿ‌])"


class Annotator:
    """Rewrites content in reading order, footnoting each term once."""

    def __init__(self):
        self.done = set()

    def text(self, value):
        for latin, persian, note in SWAPS:
            if latin not in value:
                continue

            # Swap every occurrence first, then footnote the first result.
            # Footnoting first would leave the Latin original sitting inside
            # the note where the second pass would rewrite it too.
            value = value.replace(latin, persian)

            if latin not in self.done:
                self.done.add(latin)
                value = value.replace(
                    persian, f"{persian}{MARK}{note}{END}", 1
                )

        for persian, note in TERMS:
            if persian in self.done or persian not in value:
                continue

            pattern = re.compile(re.escape(persian) + BOUNDARY)
            replaced, count = pattern.subn(
                f"{persian}{MARK}{note}{END}", value, count=1
            )

            if count:
                self.done.add(persian)
                value = replaced

        return value

    def walk(self, node):
        """Rewrite every string in a nested list/tuple/dict, in order."""
        if isinstance(node, str):
            return self.text(node)

        if isinstance(node, tuple):
            return tuple(self.walk(item) for item in node)

        if isinstance(node, list):
            return [self.walk(item) for item in node]

        if isinstance(node, dict):
            return {key: self.walk(value) for key, value in node.items()}

        return node


def annotate(content):
    """Annotate a content module in place.

    Only the Persian side is touched: the English abstract, the English
    keywords and the English title metadata are left exactly as written.
    Figure filenames and equations are skipped too -- a footnote inside an
    equation would be nonsense, and a filename is not prose.
    """
    annotator = Annotator()

    content.ABSTRACT_FA = [annotator.text(p) for p in content.ABSTRACT_FA]

    for chapter in content.CHAPTERS:
        blocks = []

        for block in chapter["blocks"]:
            if block[0] == "fig":
                # ("fig", filename, caption): the caption is prose, the
                # filename is a path and must survive untouched.
                blocks.append((block[0], block[1], annotator.text(block[2])))
            elif block[0] in ("eq", "code"):
                blocks.append(block)
            else:
                blocks.append(annotator.walk(block))

        chapter["blocks"] = blocks

    return content
