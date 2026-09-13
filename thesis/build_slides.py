# -*- coding: utf-8 -*-
"""Build the Persian defence deck.

    python3 thesis/build_slides.py            # navy, Defense_Slides.pptx
    python3 thesis/build_slides.py --rose     # plum, Defense_Slides_Rose.pptx

Two styles, one set of slides. THEMES holds the palette and a DARK flag,
and the layout primitives branch on that flag, so the two differ in
structure and not only in colour:

    navy   light ground, kicker over the title, badged step numbers,
           bordered white cards, a solid header band on every table
    rose   plum ground throughout, kicker under the title, bare numerals
           for steps, flat panels, tables ruled by row with the header
           set in the accent rather than filled

The rose style exists because the navy one reads as the common template
for this kind of talk. Where a deck has to look unlike its neighbours,
it is the structure that has to change, not the hue.

Written with python-pptx rather than pptxgenjs. The JavaScript library
emits packages PowerPoint refuses to open — one slideMaster Override per
slide against a single master, shape ids that collide between tables and
text boxes, an undeclared chart axis id — and none of that is visible to
LibreOffice, python-pptx or the XSD. python-pptx starts from a template
PowerPoint itself authored, so the package is well formed by
construction.

Every figure here comes from REVIEW_14CLASS.md. The per-class test counts
on the split slide are the standard UCF-Crime distribution; regenerate
them from the real split files before presenting:

    python -c "import collections,sys; from evaluate_multiclass import \
parse_annotation_label, CLASS_NAMES; \
c=collections.Counter(parse_annotation_label(l.split()) \
for l in open(sys.argv[1]) if len(l.split())>=3); \
[print(f'{CLASS_NAMES[i]:16s} {c[i]}') for i in range(len(CLASS_NAMES))]" \
labels/UCF_std_test.txt
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent


def _rgb(value):
    return RGBColor(value >> 16, (value >> 8) & 0xFF, value & 0xFF)


# NAVY carries the dark slides and headings, DEEP the shapes drawn on
# them, ACCENT every highlight, GOOD the positive readings and WARN the
# negative ones. LIGHT is the slide ground, CARD the tinted card fill,
# BAND the table stripe and HILITE the marked table row.
THEMES = {
    "night": dict(
        NAVY=0x16264F, DEEP=0x1E3566, ICE=0xBFD4F2, ACCENT=0xE8A33D,
        GOOD=0x3FA7A0, WARN=0xC9483B, LIGHT=0xF6F8FC, INK=0x1B2236,
        MUTED=0x6A748C, CARD=0xFFF7EA, COOL=0xEAF6F5, ALERT=0xFDEEEC,
        BAND=0xEDF1F8, RULE=0xD9E0EC, HILITE=0xFFF2DC, ONACCENT=0x1B1300,
        DEEP2=0x24427D, GOOD_TINT=0x7FD3CC, ACCENT_TINT=0xF5C27A,
        DARK=False, TITLE=0x16264F, PANEL=0xFFFFFF, BAR=0x16264F,
        GRID=0xE4E9F2, AXIS=0xC7D0E0, FAINT=0x8FA6D4,
    ),
    # A deck lit from the dark side: plum ground throughout, rose for
    # every highlight, panels instead of cards, numerals instead of
    # badges, and tables ruled rather than banded.
    "rose": dict(
        DARK=True,
        NAVY=0x220B18, DEEP=0x4A1B38, DEEP2=0x6B2750, ICE=0xF6CFDE,
        ACCENT=0xFF6FA5, GOOD=0x7FD8C0, WARN=0xFF8A80,
        LIGHT=0x2E1020, TITLE=0xFBEEF4, INK=0xF0DCE6, MUTED=0xB58FA2,
        PANEL=0x3D1730, CARD=0x4A1B38, COOL=0x1F3A38, ALERT=0x4A1520,
        BAND=0x361428, RULE=0x4A1B38, HILITE=0x5A1F3E, ONACCENT=0x2E1020,
        GOOD_TINT=0x9FE6D2, ACCENT_TINT=0xFFA0C4,
        GRID=0x45203A, AXIS=0x6B3352, FAINT=0x9C7186, BAR=0x7C3C63,
    ),
}

STYLE = "rose" if "--rose" in sys.argv else "night"
_P = THEMES[STYLE]

NAVY = _rgb(_P["NAVY"])
DEEP = _rgb(_P["DEEP"])
DEEP2 = _rgb(_P["DEEP2"])
ICE = _rgb(_P["ICE"])
AMBER = _rgb(_P["ACCENT"])
TEAL = _rgb(_P["GOOD"])
RED = _rgb(_P["WARN"])
LIGHT = _rgb(_P["LIGHT"])
INK = _rgb(_P["INK"])
MUTED = _rgb(_P["MUTED"])
CREAM = _rgb(_P["CARD"])
MINT = _rgb(_P["COOL"])
BLUSH = _rgb(_P["ALERT"])
BAND = _rgb(_P["BAND"])
RULE = _rgb(_P["RULE"])
HILITE = _rgb(_P["HILITE"])
ONACCENT = _rgb(_P["ONACCENT"])
GRID = _rgb(_P["GRID"])
AXIS = _rgb(_P["AXIS"])
FAINT = _rgb(_P["FAINT"])
TITLE = _rgb(_P["TITLE"])
PANEL = _rgb(_P["PANEL"])
DARK = _P["DARK"]
ACCENT = AMBER          # the accent reads better by name in layout code
BAR = _rgb(_P["BAR"])
GOOD_TINT = _rgb(_P["GOOD_TINT"])
ACCENT_TINT = _rgb(_P["ACCENT_TINT"])
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

OUTPUT = HERE / ("Defense_Slides_Rose.pptx" if STYLE == "rose"
                 else "Defense_Slides.pptx")

FA = "Arial"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---- primitives ------------------------------------------------------
def _style_run(run, size, *, bold=False, color=INK, italic=False):
    font = run.font
    font.name = FA
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.color.rgb = color

    # Persian needs the complex-script typeface set as well as the latin
    # one, or PowerPoint substitutes a default for the Arabic glyphs.
    rPr = run._r.get_or_add_rPr()
    if rPr.find(qn("a:cs")) is None:
        cs = rPr.makeelement(qn("a:cs"), {"typeface": FA})
        rPr.append(cs)


def _style_para(para, *, rtl=True, align=None, space_after=0, spacing=None):
    pPr = para._p.get_or_add_pPr()
    pPr.set("rtl", "1" if rtl else "0")

    if align is not None:
        para.alignment = align
    else:
        para.alignment = PP_ALIGN.RIGHT if rtl else PP_ALIGN.LEFT

    para.space_after = Pt(space_after)
    para.space_before = Pt(0)

    if spacing is not None:
        para.line_spacing = Pt(spacing)


def textbox(slide, x, y, w, h, lines, *, size=14, bold=False, color=INK,
            rtl=True, align=None, spacing=None, space_after=0, italic=False):
    """One text box; `lines` is a string or a list of paragraphs."""
    if isinstance(lines, str):
        lines = lines.split("\n")

    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    for index, line in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        _style_para(para, rtl=rtl, align=align, spacing=spacing,
                    space_after=space_after)
        run = para.add_run()
        run.text = line
        _style_run(run, size, bold=bold, color=color, italic=italic)

    return box


def shape(slide, kind, x, y, w, h, fill, *, line=None):
    item = slide.shapes.add_shape(kind, Inches(x), Inches(y),
                                  Inches(w), Inches(h))
    item.fill.solid()
    item.fill.fore_color.rgb = fill

    if line is None:
        item.line.fill.background()
    else:
        item.line.color.rgb = line
        item.line.width = Pt(0.75)

    item.shadow.inherit = False
    item.text_frame.text = ""

    return item


def card(slide, x, y, w, h, fill=None, *, line=RULE):
    """A bordered card on the light design, a flat panel on the dark one."""
    if fill is None:
        fill = PANEL

    return shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill,
                 line=None if DARK else line)


def dot(slide, x, y, number, fill=NAVY):
    if DARK:
        # no badge: the numeral carries the step on its own
        tint = {NAVY: ICE, AMBER: ACCENT_TINT, RED: RED}.get(fill, fill)

        return textbox(slide, x - 0.2, y - 0.1, 0.9, 0.62, str(number),
                       size=27, bold=True, color=tint, rtl=False,
                       align=PP_ALIGN.CENTER)

    circle = shape(slide, MSO_SHAPE.OVAL, x, y, 0.52, 0.52, fill)
    frame = circle.text_frame
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    _style_para(para, rtl=False, align=PP_ALIGN.CENTER)
    run = para.add_run()
    run.text = str(number)
    _style_run(run, 16, bold=True, color=WHITE)

    return circle


def stat(slide, x, y, w, value, label, color=None):
    color = TITLE if color is None else color
    textbox(slide, x, y, w, 0.8, value, size=38, bold=True, color=color,
            rtl=False, align=PP_ALIGN.CENTER)
    textbox(slide, x, y + 0.78, w, 0.5, label, size=12, color=MUTED,
            align=PP_ALIGN.CENTER)


def _cell_lines(cell, *, horizontal, vertical):
    """Set a cell's four borders; None means no line at all.

    The default table style draws pale hairlines on every edge, which
    turn into a harsh grid over a dark ground. Ruling only the rows
    reads as a table without boxing every number in.
    """
    tcPr = cell._tc.get_or_add_tcPr()

    for tag, colour in (("a:lnB", horizontal), ("a:lnT", horizontal),
                        ("a:lnR", vertical), ("a:lnL", vertical)):
        existing = tcPr.find(qn(tag))

        if existing is not None:
            tcPr.remove(existing)

        line = tcPr.makeelement(qn(tag), {"w": "6350", "cap": "flat",
                                          "cmpd": "sng", "algn": "ctr"})

        if colour is None:
            line.append(line.makeelement(qn("a:noFill"), {}))
        else:
            fill = line.makeelement(qn("a:solidFill"), {})
            fill.append(fill.makeelement(qn("a:srgbClr"),
                                         {"val": str(colour)}))
            line.append(fill)

        # lines precede the fill inside a:tcPr
        tcPr.insert(0, line)


def table(slide, x, y, w, head, rows, widths, *, row_h=0.3, size=11,
          head_size=12, highlight=()):
    graphic = slide.shapes.add_table(len(rows) + 1, len(head), Inches(x),
                                     Inches(y), Inches(w),
                                     Inches(row_h * (len(rows) + 1)))
    tbl = graphic.table

    # banding and the header emphasis are painted per cell below
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")

    for index, width in enumerate(widths):
        tbl.columns[index].width = Inches(width)

    for index in range(len(rows) + 1):
        tbl.rows[index].height = Inches(row_h)

    def fill_cell(cell, text, *, bold, color, background, align):
        cell.fill.solid()
        cell.fill.fore_color.rgb = background

        if DARK:
            _cell_lines(cell, horizontal=RULE, vertical=None)
        cell.margin_left = cell.margin_right = Inches(0.04)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

        frame = cell.text_frame
        frame.word_wrap = True
        para = frame.paragraphs[0]
        _style_para(para, rtl=True, align=align)
        run = para.add_run()
        run.text = str(text)
        _style_run(run, size if not bold or color != WHITE else head_size,
                   bold=bold, color=color)

    head_colour = ACCENT if DARK else WHITE
    head_fill = LIGHT if DARK else NAVY

    for index, title in enumerate(head):
        fill_cell(tbl.cell(0, index), title, bold=True, color=head_colour,
                  background=head_fill, align=PP_ALIGN.CENTER)

    for r, row in enumerate(rows):
        marked = r in highlight
        plain = PANEL if r % 2 else BAND
        background = HILITE if marked else plain

        for c, text in enumerate(row):
            fill_cell(tbl.cell(r + 1, c), text, bold=marked,
                      color=(ACCENT_TINT if DARK else NAVY) if marked else INK,
                      background=background,
                      align=PP_ALIGN.RIGHT if c == 0 else PP_ALIGN.CENTER)

    return tbl


def bullets(slide, x, y, w, items, *, size=13, color=INK, spacing=19):
    return textbox(slide, x, y, w, 0.42 * len(items) + 0.3,
                   ["●  " + item for item in items],
                   size=size, color=color, spacing=spacing, space_after=7)


def dark_slide():
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY

    return slide


def light_slide(title, kicker=None):
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = LIGHT

    textbox(slide, 0.6, 0.45, 12.1, 0.8, title, size=29, bold=True,
            color=TITLE)

    if kicker:
        textbox(slide, 0.6, 1.24, 12.1, 0.32, kicker, size=12.5, bold=True,
                color=AMBER)

    return slide


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# =====================================================================
# 1 — title
# =====================================================================
s = dark_slide()
shape(s, MSO_SHAPE.OVAL, 9.3, -1.6, 6.2, 6.2, DEEP)
shape(s, MSO_SHAPE.OVAL, 10.9, 3.6, 3.6, 3.6, DEEP2)
textbox(s, 0.8, 1.5, 8.4, 0.4, "پایان‌نامه کارشناسی مهندسی کامپیوتر",
        size=14, bold=True, color=AMBER)
textbox(s, 0.8, 2.05, 8.4, 1.9,
        ["تشخیص نوع ناهنجاری", "در ویدیوهای نظارتی"],
        size=38, bold=True, color=WHITE, spacing=46)
textbox(s, 0.8, 4.15, 8.4, 0.5,
        "یادگیری چندنمونه‌ای ضعیف‌نظارت با ستون فقرات ترنسفورمر ویدیویی",
        size=16, color=ICE)
textbox(s, 0.8, 5.15, 3.0, 0.3, "ماکرو F1 نهایی", size=12, color=ICE)
textbox(s, 0.8, 5.45, 3.0, 0.8, "0.4298", size=40, bold=True, color=AMBER,
        rtl=False, align=PP_ALIGN.LEFT)
textbox(s, 4.0, 5.72, 3.2, 0.4, "از خط پایه 0.2975", size=13, color=ICE)
textbox(s, 0.8, 6.6, 8.4, 0.35,
        "دانشگاه صنعتی همدان  ·  گروه مهندسی کامپیوتر", size=12,
        color=FAINT)
notes(s, "پروژه تبدیل یک سامانه تشخیص ناهنجاری دودویی به سامانه "
         "چهارده‌کلاسه است. عدد نهایی ماکرو F1 برابر 0.4298 در برابر خط "
         "پایه 0.2975.")

# =====================================================================
# 2 — the problem
# =====================================================================
s = light_slide("صورت مسئله", "فصل اول")
card(s, 0.6, 1.75, 5.9, 2.1)
textbox(s, 0.95, 1.95, 5.2, 0.4, "مسئله متداول: تشخیص ناهنجاری", size=17,
        bold=True, color=MUTED)
textbox(s, 0.95, 2.45, 5.2, 1.0,
        ["آیا این ویدیو ناهنجار است؟", "جواب: بله یا خیر"], size=15,
        spacing=24)
card(s, 6.8, 1.75, 5.9, 2.1, CREAM)
textbox(s, 7.15, 1.95, 5.2, 0.4, "مسئله ما: تشخیص نوع ناهنجاری", size=17,
        bold=True, color=AMBER)
textbox(s, 7.15, 2.45, 5.2, 1.0,
        ["کدام نوع ناهنجاری رخ داده؟", "جواب: یکی از ۱۴ کلاس"], size=15,
        spacing=24)
textbox(s, 0.6, 4.15, 12.1, 0.4, "چرا این تفاوت مهم است", size=18,
        bold=True, color=TITLE)
bullets(s, 0.6, 4.65, 12.1, [
    "پاسخ عملیاتی به تصادف رانندگی، آمبولانس است؛ به سرقت مسلحانه، نیروی مسلح.",
    "یک هشدار خام «ناهنجاری رخ داد» برای اپراتور قابل اقدام نیست.",
    "از نظر دشواری، مسئله چهارده‌کلاسه با نسخه دودویی قابل مقایسه نیست.",
], size=14)
notes(s, "بیشتر پژوهش‌ها مسئله را دودویی تعریف می‌کنند. ما یک گام جلوتر "
         "رفتیم.")

# =====================================================================
# 3 — dataset
# =====================================================================
s = light_slide("مجموعه داده UCF-Crime", "داده")
stat(s, 0.6, 1.8, 2.9, "1900", "کل ویدیوها")
stat(s, 3.7, 1.8, 2.9, "14", "کلاس")
stat(s, 6.8, 1.8, 2.9, "1895", "رمزگشایی موفق", TEAL)
stat(s, 9.9, 1.8, 2.9, "5", "فایل خراب", RED)
card(s, 0.6, 3.5, 5.9, 3.1)
textbox(s, 0.95, 3.7, 5.2, 0.4, "ویژگی‌های داده", size=17, bold=True,
        color=TITLE)
bullets(s, 0.95, 4.2, 5.2, [
    "ویدیوی واقعی دوربین مداربسته، نه صحنه‌پردازی",
    "وضوح پایین، نور ضعیف، زاویه نامناسب",
    "مدت‌زمان از چند ثانیه تا چند دقیقه",
    "برچسب فقط در سطح کل ویدیو",
])
card(s, 6.8, 3.5, 5.9, 3.1, CREAM)
textbox(s, 7.15, 3.7, 5.2, 0.4, "چالش اصلی: نامتوازنی شدید", size=17,
        bold=True, color=AMBER)
bullets(s, 7.15, 4.2, 5.2, [
    "کلاس عادی حدود نیمی از داده آموزش است",
    "کلاس تیراندازی کمتر از دو درصد",
    "در آزمون، هفت کلاس پنج ویدیو یا کمتر دارند",
    "کلاس Assault تنها سه ویدیوی آزمون دارد",
])
notes(s, "نامتوازنی هم آموزش را دشوار می‌کند و هم ارزیابی را نوفه‌آلود.")

# =====================================================================
# 4 — train split
# =====================================================================
s = light_slide("تعداد ویدیو در هر کلاس — مجموعه آموزش", "تقسیم داده")
table(s, 0.6, 1.75, 5.9, ["کلاس", "تعداد ویدیو"], [
    ["Normal — عادی", "800"],
    ["Robbery — سرقت", "145"],
    ["RoadAccidents — تصادف", "127"],
    ["Stealing — دزدی", "95"],
    ["Burglary — سرقت از منزل", "87"],
    ["Abuse — سوءاستفاده", "48"],
    ["Assault — تعرض", "47"],
], [4.1, 1.8], row_h=0.34, size=12)
table(s, 6.8, 1.75, 5.9, ["کلاس", "تعداد ویدیو"], [
    ["Arrest — بازداشت", "45"],
    ["Fighting — درگیری", "45"],
    ["Vandalism — تخریب", "45"],
    ["Arson — آتش‌افروزی", "41"],
    ["Explosion — انفجار", "29"],
    ["Shoplifting — دزدی از مغازه", "29"],
    ["Shooting — تیراندازی", "27"],
], [4.1, 1.8], row_h=0.34, size=12)
card(s, 0.6, 5.15, 12.1, 1.5, CREAM)
textbox(s, 0.95, 5.35, 11.4, 0.4,
        "جمع: ۱۶۱۰ ویدیوی آموزش  ·  ۸۰۰ عادی و ۸۱۰ ناهنجار", size=16,
        bold=True, color=TITLE)
textbox(s, 0.95, 5.8, 11.4, 0.7,
        "نسبت کلاس عادی به کمیاب‌ترین کلاس حدود ۳۰ به ۱ است. همین نسبت "
        "است که وزن‌دهی معکوس فراوانی را ضروری می‌کند.", size=12.5,
        spacing=19)
notes(s, "این توزیع استاندارد UCF-Crime است و جمعش دقیقا 1610 می‌شود.")

# =====================================================================
# 5 — test split
# =====================================================================
s = light_slide("تعداد ویدیو در هر کلاس — مجموعه آزمون", "تقسیم داده")
table(s, 0.6, 1.72, 7.4,
      ["کلاس", "تقسیم استاندارد", "تقسیم مسیر اول"], [
          ["Normal — عادی", "150", "148"],
          ["RoadAccidents — تصادف رانندگی", "23", "23"],
          ["Shooting — تیراندازی", "23", "8"],
          ["Explosion — انفجار", "21", "8"],
          ["Shoplifting — دزدی از مغازه", "21", "8"],
          ["Burglary — سرقت از منزل", "13", "15"],
          ["Arson — آتش‌افروزی", "9", "8"],
          ["Arrest — بازداشت", "5", "8"],
          ["Fighting — درگیری", "5", "8"],
          ["Robbery — سرقت", "5", "23"],
          ["Stealing — دزدی", "5", "15"],
          ["Vandalism — تخریب اموال", "4", "8"],
          ["Abuse — سوءاستفاده", "3", "8"],
          ["Assault — تعرض", "3", "8"],
      ], [3.6, 1.9, 1.9], row_h=0.27, size=11, head_size=11.5,
      highlight=(7, 8, 9, 10, 11, 12, 13))
textbox(s, 0.6, 5.86, 7.4, 0.4,
        "جمع ستون دوم ۲۹۰ و جمع ستون سوم ۲۹۶ است. ردیف‌های پررنگ، هفت "
        "کلاسی هستند که پنج ویدیوی آزمون یا کمتر دارند.", size=11,
        color=MUTED, spacing=16)
card(s, 8.3, 1.72, 4.4, 1.95, CREAM)
textbox(s, 8.6, 1.9, 3.8, 0.35, "دو تقسیم متفاوت", size=15, bold=True,
        color=AMBER)
textbox(s, 8.6, 2.3, 3.8, 1.25,
        "مسیر X-CLIP روی تقسیمی با ۲۹۶ ویدیوی آزمون اجرا شد و مسیر "
        "VideoMAE روی تقسیم استاندارد با ۲۹۰ ویدیو.", size=11.5,
        spacing=18)
card(s, 8.3, 3.85, 4.4, 1.85)
textbox(s, 8.6, 4.03, 3.8, 0.35, "چرا ارزیابی نوفه‌آلود است", size=15,
        bold=True, color=TITLE)
textbox(s, 8.6, 4.43, 3.8, 1.2,
        "کلاس Assault تنها سه ویدیوی آزمون دارد. یک ویدیو یعنی یک‌سوم "
        "یادآوری آن کلاس، و ماکرو F1 به هر ۱۴ کلاس وزن یکسان می‌دهد.",
        size=11.5, spacing=18)
card(s, 8.3, 5.9, 4.4, 1.05, BLUSH)
textbox(s, 8.6, 6.08, 3.8, 0.75,
        "پیش از ارائه این اعداد را با فایل تقسیم خودتان بسنجید.", size=11,
        color=RED, spacing=17)
stat(s, 0.6, 6.2, 2.4, "290", "کل مجموعه آزمون")
stat(s, 3.1, 6.2, 2.4, "1610", "کل مجموعه آموزش")
stat(s, 5.6, 6.2, 2.4, "0%", "همپوشانی دو مجموعه", TEAL)
notes(s, "ستون سوم از گزارش آزمایش‌ها بازسازی شده و ستون دوم توزیع "
         "استاندارد است. هر دو با جمع کل می‌خوانند.")

# =====================================================================
# 6 — weak supervision
# =====================================================================
s = light_slide("نظارت ضعیف و یادگیری چندنمونه‌ای", "چارچوب")
table(s, 0.6, 1.8, 6.2, ["سطح نظارت", "برچسب چیست", "هزینه"], [
    ["نظارت کامل", "هر فریم برچسب دارد", "بسیار پرهزینه"],
    ["نظارت ضعیف", "فقط کل ویدیو برچسب دارد", "کم‌هزینه"],
    ["بدون نظارت", "هیچ برچسبی نیست", "رایگان"],
], [1.9, 2.9, 1.4], row_h=0.4, size=12, highlight=(1,))
card(s, 7.1, 1.8, 5.6, 2.0, CREAM)
textbox(s, 7.4, 1.98, 5.0, 0.35, "مسئله ما", size=16, bold=True, color=AMBER)
textbox(s, 7.4, 2.4, 5.0, 1.2,
        "در یک ویدیوی سه‌دقیقه‌ای، سرقت شاید ۱۵ ثانیه باشد. بیش از ۹۰ "
        "درصد آنچه مدل می‌بیند، با وجود برچسب «سرقت»، عادی است.", size=13,
        spacing=20)
textbox(s, 0.6, 4.15, 12.1, 0.4,
        "راه حل: هر ویدیو یک کیف، هر قطعه یک نمونه", size=18, bold=True,
        color=TITLE)
for index, line in enumerate([
        "ویدیو به ۱۶ قطعه شکسته می‌شود",
        "مدل به هر قطعه نمره می‌دهد",
        "مشکوک‌ترین قطعه‌ها انتخاب می‌شوند",
        "فرض: همان‌ها برچسب کیف را حمل می‌کنند"]):
    x = 9.9 - index * 3.1
    card(s, x, 4.75, 2.85, 1.65)
    dot(s, x + 2.1, 4.95, index + 1, AMBER if index == 3 else NAVY)
    textbox(s, x + 0.2, 5.55, 2.45, 0.8, line, size=12.5, spacing=18)
notes(s, "نظارت ضعیف یک معامله است: داده بیشتر با کیفیت برچسب کمتر.")

# =====================================================================
# 7 — path one
# =====================================================================
s = dark_slide()
shape(s, MSO_SHAPE.OVAL, -1.8, -1.5, 5.6, 5.6, DEEP)
textbox(s, 5.0, 2.4, 7.5, 0.6, "مسیر اول", size=18, bold=True, color=AMBER)
textbox(s, 5.0, 3.0, 7.5, 1.2, "X-CLIP", size=58, bold=True, color=WHITE,
        rtl=False, align=PP_ALIGN.RIGHT)
textbox(s, 5.0, 4.3, 7.5, 0.5, "تبدیل کد UMIL از دودویی به چهارده‌کلاسه",
        size=17, color=ICE)
textbox(s, 5.0, 5.1, 7.5, 0.7, "0.3327  ←  0.3881", size=26, bold=True,
        color=AMBER, rtl=False, align=PP_ALIGN.RIGHT)
notes(s, "مسیر اول روی کد UMIL بنا شد. ماکرو F1 از 0.3327 به 0.3881 رسید.")

# =====================================================================
# 8 — how X-CLIP works
# =====================================================================
s = light_slide("X-CLIP چطور کار می‌کند", "مسیر اول")
for index, (title, body) in enumerate([
        ("CLIP روی جفت عکس و متن آموزش دیده",
         "عکس و توضیحش در یک فضای مشترک نزدیک هم می‌افتند"),
        ("طبقه‌بندی بدون آموزش",
         "اسم هر کلاس به بردار تبدیل می‌شود و نزدیک‌ترین برنده است"),
        ("X-CLIP بخش زمانی اضافه می‌کند",
         "حالا ورودی ویدیو است، نه یک عکس تنها"),
        ("در پروژه ما",
         "بردار ویدیو با ۱۴ بردار متنی مقایسه می‌شود")]):
    y = 1.75 + index * 1.22
    card(s, 0.6, y, 12.1, 1.05)
    dot(s, 11.9, y + 0.27, index + 1, AMBER if index == 3 else NAVY)
    textbox(s, 1.0, y + 0.14, 10.6, 0.35, title, size=15.5, bold=True,
            color=TITLE)
    textbox(s, 1.0, y + 0.52, 10.6, 0.35, body, size=12.5, color=MUTED)
textbox(s, 0.6, 6.65, 12.1, 0.4,
        "آن ۱۴ بردار متنی نقش وزن‌های طبقه‌بند را بازی می‌کنند و هرگز "
        "آموزش نمی‌بینند.", size=13, bold=True, color=AMBER)
notes(s, "در X-CLIP وزن‌های طبقه‌بند از متن ساخته می‌شوند. همان جایی که "
         "بعدا سقف پیدا شد.")

# =====================================================================
# 9 — three bugs
# =====================================================================
s = light_slide("سه اشکال بنیادی که پیدا و اصلاح شد", "مسیر اول")
for index, (title, body, fix) in enumerate([
        ("میانگین‌گیری در ارزیابی",
         "احتمال هر ۱۶ قطعه میانگین گرفته می‌شد. در ویدیوی ناهنجار، "
         "قطعه‌های عادی شواهد را خفه می‌کردند و تقریباً همه چیز عادی "
         "پیش‌بینی می‌شد.",
         "اصلاح: تصمیم دومرحله‌ای با بیشینه"),
        ("فایل نمرات کهنه",
         "کد هر فایل نمرات موجود را بازاستفاده می‌کرد. در یک گزارش، ۲۸۸ "
         "ویدیو از ۲۹۸ کنار گذاشته شده و عدد روی ده ویدیو حساب شده بود.",
         "اصلاح: بازمحاسبه پیش‌فرض"),
        ("انتخاب قطعه برتر به تفکیک کلاس",
         "کانال عادی از عادی‌ترین قطعه‌های یک ویدیوی سرقت ساخته می‌شد. "
         "نویز برچسب سیستماتیک وارد آموزش می‌شد.",
         "اصلاح: یک رتبه‌بندی برای همه کلاس‌ها")]):
    x = 8.85 - index * 4.13
    card(s, x, 1.8, 3.85, 4.45)
    dot(s, x + 3.05, 2.05, index + 1, RED)
    textbox(s, x + 0.25, 2.75, 3.35, 0.75, title, size=15, bold=True,
            color=TITLE, spacing=21)
    textbox(s, x + 0.25, 3.5, 3.35, 1.8, body, size=12, spacing=19)
    textbox(s, x + 0.25, 5.4, 3.35, 0.7, fix, size=12, bold=True,
            color=TEAL, spacing=18)
notes(s, "این سه اصلاح مقدم بر هر ایده جدیدی بود. بدون آن‌ها هیچ "
         "اندازه‌گیری قابل اعتماد نبود.")

# =====================================================================
# 10 — two-stage decision
# =====================================================================
s = light_slide("تصمیم دومرحله‌ای", "مسیر اول")
card(s, 0.6, 1.75, 5.9, 2.5, CREAM)
textbox(s, 0.95, 1.95, 5.2, 0.4, "مرحله اول: آیا ناهنجار است؟", size=17,
        bold=True, color=AMBER)
textbox(s, 0.95, 2.4, 5.2, 0.8,
        "بیشینه روی ۱۶ قطعه از مقدار «۱ منهای احتمال عادی»، در مقایسه با "
        "آستانه ۰٫۵", size=13, spacing=20)
textbox(s, 0.95, 3.2, 5.2, 0.85,
        "چرا بیشینه و نه میانگین: یک قطعه مشکوک کافی است، چون رویداد فقط "
        "چند ثانیه است", size=12, color=MUTED, spacing=19)
card(s, 6.8, 1.75, 5.9, 2.5)
textbox(s, 7.15, 1.95, 5.2, 0.4, "مرحله دوم: کدام نوع؟", size=17,
        bold=True, color=TITLE)
textbox(s, 7.15, 2.4, 5.2, 0.8,
        "رأی‌گیری میانگین چهار قطعه برتر، تنها میان ۱۳ کلاس ناهنجار",
        size=13, spacing=20)
textbox(s, 7.15, 3.2, 5.2, 0.85,
        "کلاس عادی در این مرحله اصلاً وارد رقابت نمی‌شود", size=12,
        color=MUTED, spacing=19)
textbox(s, 0.6, 4.5, 12.1, 0.4,
        "مثال عددی: ویدیوی سرقت، رویداد در قطعه‌های ۴ و ۵", size=16,
        bold=True, color=TITLE)
table(s, 0.6, 5.0, 12.1,
      ["روش", "نمره", "در برابر آستانه ۰٫۵", "تصمیم"], [
          ["میانگین‌گیری روی ۸ قطعه", "0.26", "زیر آستانه", "عادی — غلط"],
          ["بیشینه روی همان ۸ قطعه", "0.85", "بالای آستانه", "ناهنجار — درست"],
      ], [3.8, 2.3, 3.0, 3.0], row_h=0.42, size=13, highlight=(1,))
notes(s, "مدل درست کار می‌کرد و مرحله تجمیع نتیجه را نابود می‌کرد.")

# =====================================================================
# 11 — class collapse
# =====================================================================
s = light_slide("کلاسی که مدل یاد گرفت هرگز نگوید", "مسیر اول")
textbox(s, 0.6, 1.8, 12.1, 0.4,
        "به‌جای شمردن پیش‌بینی‌های درست، شمردیم مدل هر کلاس را چند بار "
        "پیش‌بینی می‌کند.", size=14)
table(s, 0.6, 2.35, 7.3,
      ["کلاس", "ویدیوی واقعی", "چند بار پیش‌بینی شد", "نسبت"], [
          ["Fighting — درگیری", "8", "2", "0.25"],
          ["Assault — تعرض", "8", "3", "0.38"],
          ["Vandalism — تخریب", "8", "5", "0.62"],
          ["Stealing — دزدی", "15", "20", "1.33"],
          ["RoadAccidents — تصادف", "23", "39", "1.70"],
      ], [2.8, 1.5, 1.9, 1.1], row_h=0.4, size=12, highlight=(0,))
card(s, 8.2, 2.35, 4.5, 2.05, CREAM)
textbox(s, 8.5, 2.53, 3.9, 0.35, "تشخیص", size=16, bold=True, color=AMBER)
textbox(s, 8.5, 2.95, 3.9, 1.3,
        "مدل روی درگیری شکست نخورده بود. یاد گرفته بود هرگز نگوید "
        "درگیری، چون گفتنش شرط‌بندی بدی بود.", size=12.5, spacing=19)
textbox(s, 0.6, 4.6, 12.1, 0.4, "علت ریشه‌ای", size=17, bold=True,
        color=TITLE)
bullets(s, 0.6, 5.1, 12.1, [
    "تابع وزن‌دهی کلاس در کد وجود داشت ولی هیچ‌جا صدا زده نمی‌شد.",
    "وزن‌ها همه‌جا ۱٫۰ بودند؛ کلاس عادی حدود ۱۹ برابر کلاس درگیری گرادیان "
    "تولید می‌کرد.",
    "با وزن‌دهی معکوس فراوانی آن نسبت به ۳٫۶ رسید و درگیری ۹ بار پیش‌بینی "
    "شد به‌جای ۲ بار.",
])
notes(s, "با وجود ۹ پیش‌بینی هنوز هیچ‌کدام درست نبود. مشکل دوم بصری بود، "
         "نه وزنی.")

# =====================================================================
# 12 — text prototype ceiling
# =====================================================================
s = light_slide("محدودیت بنیادی: هندسه فضای متنی", "مسیر اول")
textbox(s, 0.6, 1.78, 12.1, 0.4,
        "اسم خام کلاس‌ها در فضای زبانی CLIP تقریباً هم‌راستا هستند و سقفی "
        "مستقل از کیفیت تصویری تحمیل می‌کنند.", size=14)
table(s, 0.6, 2.35, 5.5, ["جفت کلاس", "شباهت کسینوسی"], [
    ["Abuse  /  Stealing", "0.91"],
    ["Assault  /  Arrest", "0.91"],
    ["Fighting  /  Shooting", "0.90"],
    ["میانگین همه جفت‌ها", "0.83"],
], [3.5, 2.0], row_h=0.42, size=13, highlight=(3,))
card(s, 6.4, 2.35, 6.3, 2.05, CREAM)
textbox(s, 6.7, 2.53, 5.7, 0.35, "آزمایش تعیین‌کننده", size=16, bold=True,
        color=AMBER)
textbox(s, 6.7, 2.95, 5.7, 1.3,
        "فرض کنید بخش تصویری کاملاً بی‌نقص باشد، یعنی بردار ویدیو دقیقاً "
        "برابر بردار کلاس خودش. در این حالت آرمانی، احتمال کلاس درست "
        "تنها ۰٫۴۴ درمی‌آید.", size=12.5, spacing=19)
textbox(s, 0.6, 4.6, 12.1, 0.4,
        "راه حل: توصیف بصری به‌جای اسم حقوقی", size=17, bold=True,
        color=TITLE)
card(s, 0.6, 5.1, 6.0, 1.4)
textbox(s, 0.9, 5.25, 5.4, 0.3, "پیش از تغییر", size=12, color=MUTED)
textbox(s, 0.9, 5.6, 5.4, 0.4, "\"Burglary\"", size=16, bold=True,
        color=RED, rtl=False, align=PP_ALIGN.RIGHT)
textbox(s, 0.9, 6.05, 5.4, 0.35, "سقف اندازه‌گیری‌شده: 0.44", size=12.5)
card(s, 6.9, 5.1, 5.8, 1.4, MINT)
textbox(s, 7.2, 5.25, 5.2, 0.3, "پس از تغییر", size=12, color=MUTED)
textbox(s, 7.2, 5.6, 5.2, 0.4,
        "\"a burglar climbing through a broken window\"", size=11.5,
        bold=True, color=TEAL, rtl=False, align=PP_ALIGN.RIGHT)
textbox(s, 7.2, 6.05, 5.2, 0.35, "سقف اندازه‌گیری‌شده: 0.86", size=12.5)
notes(s, "اصطلاحات حقوقی انتزاعی در فضای زبانی خوشه می‌شوند؛ توصیف آنچه "
         "دوربین می‌بیند نه.")

# =====================================================================
# 13 — path one results
# =====================================================================
s = light_slide("نتایج مسیر اول", "مسیر اول")
table(s, 0.6, 1.85, 12.1,
      ["پیکربندی", "AUC دودویی", "ماکرو F1", "صحت نوع", "صحت کلی"], [
          ["اجرای پایه — اسم خام، وزن دستی", "0.9473", "0.3327", "0.3716",
           "0.6385"],
          ["توصیف بصری + وزن معکوس فراوانی", "0.9637", "0.3713", "0.4054",
           "0.6351"],
          ["افزودن چهار نمای زمانی در آزمون", "0.9629", "0.3881", "0.4257",
           "0.6419"],
      ], [4.9, 1.8, 1.8, 1.8, 1.8], row_h=0.44, size=12.5, highlight=(2,))
textbox(s, 0.6, 3.95, 6.0, 0.4, "کلاس‌هایی که از مرگ برگشتند", size=16,
        bold=True, color=TITLE)
table(s, 0.6, 4.45, 6.0, ["کلاس", "F1 پیش از تغییر", "F1 پس از تغییر"], [
    ["Arrest — بازداشت", "0.000", "0.556"],
    ["Abuse — سوءاستفاده", "0.462", "0.714"],
    ["Stealing — دزدی", "0.200", "0.350"],
    ["Vandalism — تخریب", "0.000", "0.167"],
], [2.4, 1.8, 1.8], row_h=0.36, size=11.5)
card(s, 7.0, 3.95, 5.7, 2.55, CREAM)
textbox(s, 7.3, 4.15, 5.1, 0.35, "چرا از این مسیر فاصله گرفتیم", size=16,
        bold=True, color=AMBER)
bullets(s, 7.3, 4.6, 5.1, [
    "هر پیکربندی حدود ۱۶ ساعت طول می‌کشید",
    "با این هزینه، آزمون سیستماتیک فرضیه‌ها ناممکن بود",
    "خطای باقی‌مانده در خوشه دزدی بود و ماهیتش مکانی بود، نه زمانی",
], size=12)
notes(s, "مسیر اول به 0.3881 رسید. دو دیوار ماند: هزینه آزمایش و ماهیت "
         "مکانی خطای خوشه دزدی.")

# =====================================================================
# 14 — path two
# =====================================================================
s = dark_slide()
shape(s, MSO_SHAPE.OVAL, -1.8, -1.5, 5.6, 5.6, DEEP)
textbox(s, 5.0, 2.4, 7.5, 0.6, "مسیر دوم", size=18, bold=True, color=AMBER)
textbox(s, 5.0, 3.0, 7.5, 1.2, "VideoMAE", size=58, bold=True, color=WHITE,
        rtl=False, align=PP_ALIGN.RIGHT)
textbox(s, 5.0, 4.3, 7.5, 0.5, "رمزگذار منجمد و ویژگی‌های ذخیره‌شده",
        size=17, color=ICE)
textbox(s, 5.0, 5.1, 7.5, 0.7, "0.2975  ←  0.4298", size=26, bold=True,
        color=AMBER, rtl=False, align=PP_ALIGN.RIGHT)
notes(s, "مسیر دوم از صفر نوشته شد. ماکرو F1 از 0.2975 به 0.4298 رسید.")

# =====================================================================
# 15 — how VideoMAE learned
# =====================================================================
s = light_slide("VideoMAE چطور یاد گرفت", "مسیر دوم")
for index, (title, body) in enumerate([
        ("پوشاندن", "حدود ۹۰ درصد مکعب‌های فضازمانی پوشانده می‌شوند"),
        ("بازسازی",
         "رمزگذار فقط ۱۰ درصد باقی‌مانده را می‌بیند و رمزگشا جای خالی‌ها "
         "را حدس می‌زند"),
        ("بدون برچسب",
         "خود ویدیو هم سؤال است و هم جواب، پس روی حجم عظیمی از داده ممکن "
         "است"),
        ("دور انداختن رمزگشا",
         "پس از پیش‌آموزش تنها رمزگذار می‌ماند و روی Kinetics تنظیم "
         "می‌شود")]):
    y = 1.8 + index * 1.12
    card(s, 0.6, y, 8.3, 0.95)
    dot(s, 8.1, y + 0.22, index + 1, AMBER if index == 3 else NAVY)
    textbox(s, 1.0, y + 0.1, 6.9, 0.32, title, size=15, bold=True,
            color=TITLE)
    textbox(s, 1.0, y + 0.45, 6.9, 0.42, body, size=12, color=MUTED)
card(s, 9.2, 1.8, 3.5, 4.4, CREAM)
textbox(s, 9.5, 2.0, 2.9, 0.35, "تفاوت با X-CLIP", size=15, bold=True,
        color=AMBER)
textbox(s, 9.5, 2.45, 2.9, 3.5,
        ["X-CLIP از متن یاد گرفت.", "", "VideoMAE از خود ویدیو.", "",
         "پس VideoMAE اصلاً کلمه نمی‌شناسد و آن سقف ۰٫۴۴ اینجا اصلاً وجود "
         "ندارد."], size=12.5, spacing=20)
notes(s, "پوشاندن لوله‌ای است: اگر جایی پوشانده شود، در همه فریم‌ها "
         "پوشانده می‌شود، تا مدل از فریم بغلی کپی نکند.")

# =====================================================================
# 16 — the freeze decision
# =====================================================================
s = light_slide("مهم‌ترین تصمیم پروژه: یخ زدن رمزگذار", "مسیر دوم")
textbox(s, 0.6, 1.78, 12.1, 0.4,
        "چون رمزگذار آموزش نمی‌بیند، خروجی‌اش ثابت است. پس محاسبه مجددش "
        "در هر دوره کار تکراری محض است.", size=14)
stat(s, 8.8, 2.35, 3.9, "16", "ساعت برای هر آزمایش — روش قدیم", RED)
shape(s, MSO_SHAPE.LEFT_ARROW, 7.3, 2.6, 1.3, 0.55, ICE)
stat(s, 3.2, 2.35, 3.9, "80", "ثانیه برای هر آزمایش — روش جدید", TEAL)
stat(s, 0.6, 2.35, 2.3, "200", "ارزیابی انجام‌شده")
table(s, 0.6, 4.05, 12.1, ["مورد", "روش قدیم", "روش جدید"], [
    ["رمزگشایی ویدیو", "هر دوره تکرار می‌شد", "یک بار، هرگز دوباره"],
    ["اجرای رمزگذار", "هر دوره تکرار می‌شد", "یک بار، هرگز دوباره"],
    ["حجم داده ورودی", "۷۰ گیگابایت ویدیو", "۹۳ مگابایت عدد"],
    ["پارامترهای آموزش‌پذیر", "حدود ۳۰۰ میلیون", "۴٫۸ میلیون"],
    ["زمان ۲۰۰ آزمایش", "حدود ۱۳۳ روز", "حدود ۴ ساعت"],
], [3.7, 4.2, 4.2], row_h=0.35, size=12.5, highlight=(4,))
textbox(s, 0.6, 6.45, 12.1, 0.4,
        "هزینه‌ای که پرداختیم: رمزگذار هرگز به دامنه ویدیوی نظارتی تطبیق "
        "نمی‌یابد. این را صریح گزارش کرده‌ایم.", size=13, bold=True,
        color=AMBER)
notes(s, "این تصمیم خودش نتیجه را بهتر نکرد. اجازه داد بقیه چیزها را پیدا "
         "کنیم.")

# =====================================================================
# 17 — pipeline
# =====================================================================
s = light_slide("خط لوله نهایی", "مسیر دوم")
for row, (boxes, caption, top) in enumerate([
        ([("ویدیوی خام", "۹۰۰۰ فریم\n۴۰ مگابایت", TITLE),
          ("۱۶ قطعه", "هر قطعه ۱۶ فریم\n۲۲۴ در ۲۲۴", TITLE),
          ("رمزگذار ViT-L", "منجمد\nیک بار اجرا", AMBER),
          ("ویژگی ذخیره‌شده", "۱۶ در ۱۰۲۴\n۶۴ کیلوبایت", TEAL)],
         "مرحله اول — یک بار برای ۱۸۹۵ ویدیو، ۱۵۰ دقیقه", 1.9),
        ([("سر ترنسفورمر", "دو لایه توجه\nروی محور زمان", TITLE),
          ("نمره هر قطعه", "۱۶ در ۱۴", TITLE),
          ("تصمیم دومرحله‌ای", "وجود، سپس نوع", AMBER),
          ("برچسب نهایی", "یکی از ۱۴ کلاس", TEAL)],
         "مرحله دوم — هر پیکربندی، ۸۰ ثانیه", 4.2)]):
    for index, (title, body, colour) in enumerate(boxes):
        x = 9.95 - index * 3.15
        card(s, x, top, 2.8, 1.5, CREAM if index == 2 else PANEL)
        textbox(s, x + 0.15, top + 0.2, 2.5, 0.35, title, size=14,
                bold=True, color=colour, align=PP_ALIGN.CENTER)
        textbox(s, x + 0.15, top + 0.6, 2.5, 0.7, body, size=11.5,
                color=MUTED, align=PP_ALIGN.CENTER, spacing=17)
        if index < 3:
            shape(s, MSO_SHAPE.LEFT_ARROW, x - 0.28, top + 0.6, 0.25, 0.3,
                  ICE)
    textbox(s, 0.6, top + 1.6, 12.1, 0.35, caption, size=12.5, color=MUTED,
            align=PP_ALIGN.CENTER)
textbox(s, 0.6, 6.35, 12.1, 0.4,
        "تنها بخشی که ما آموزش دادیم، سر طبقه‌بند است: کمتر از دو درصد "
        "پارامترهای کل سیستم.", size=13, bold=True, color=AMBER,
        align=PP_ALIGN.CENTER)
notes(s, "ویدیوی ۴۰ مگابایتی به آرایه ۶۴ کیلوبایتی تبدیل می‌شود.")

# =====================================================================
# 18 — temporal head
# =====================================================================
s = light_slide("سر ترنسفورمر روی محور زمان", "مسیر دوم")
card(s, 0.6, 1.8, 5.9, 2.3)
textbox(s, 0.95, 2.0, 5.2, 0.35, "مشکل: قطعه تنها مبهم است", size=16,
        bold=True, color=MUTED)
textbox(s, 0.95, 2.45, 5.2, 1.4,
        "یک قطعه سه‌ثانیه‌ای داخل مغازه، در دزدی از مغازه و سرقت مسلحانه "
        "ظاهر یکسانی دارد. سر نقطه‌ای اطلاعات لازم را اصولاً در اختیار "
        "ندارد.", size=13, spacing=20)
card(s, 6.8, 1.8, 5.9, 2.3, CREAM)
textbox(s, 7.15, 2.0, 5.2, 0.35, "راه حل: خودتوجهی میان قطعه‌ها", size=16,
        bold=True, color=AMBER)
textbox(s, 7.15, 2.45, 5.2, 1.4,
        "یک جدول ۱۶ در ۱۶ می‌گوید هر قطعه چقدر به هر قطعه دیگر توجه کند. "
        "بردار جدید هر قطعه، جمع وزن‌دار همه قطعه‌هاست.", size=13,
        spacing=20)
textbox(s, 0.6, 4.35, 5.9, 0.35, "پیکربندی", size=16, bold=True, color=TITLE)
table(s, 0.6, 4.8, 5.9, ["پارامتر", "مقدار"], [
    ["بعد پنهان", "512"],
    ["سرهای توجه", "4"],
    ["لایه‌ها", "2"],
    ["حذف تصادفی", "0.3"],
    ["پارامترها", "4,772,366"],
], [3.4, 2.5], row_h=0.3, size=11.5)
textbox(s, 6.8, 4.35, 5.9, 0.35, "نتیجه", size=16, bold=True, color=TITLE)
table(s, 6.8, 4.8, 5.9, ["سر طبقه‌بند", "ماکرو F1، میانگین ۵ بذر"], [
    ["سر نقطه‌ای — خط پایه", "0.3049"],
    ["سر ترنسفورمر زمانی", "0.3480"],
], [3.2, 2.7], row_h=0.42, size=12.5, highlight=(1,))
textbox(s, 6.8, 6.25, 5.9, 0.7,
        "هر پنج بذر سر زمانی از بهترین بذر سر نقطه‌ای بالاتر بود.",
        size=12, spacing=19)
notes(s, "این تغییر از تحلیل خطا آمد، نه از جستجوی ابرپارامتر.")

# =====================================================================
# 19 — backbone upgrade
# =====================================================================
s = light_slide("ارتقای ظرفیت رمزگذار", "مسیر دوم")
table(s, 0.6, 1.85, 7.2, ["مورد", "ViT-B", "ViT-L"], [
    ["تعداد لایه", "12", "24"],
    ["پهنای بردار", "768", "1024"],
    ["پارامترها", "≈ 87 میلیون", "≈ 300 میلیون"],
    ["اندازه وصله", "16 × 16", "16 × 16"],
    ["ماکرو F1، میانگین ۵ بذر", "0.3405", "0.3955"],
    ["ماکرو F1، نتیجه گروهی", "0.3645", "0.4122"],
], [3.2, 2.0, 2.0], row_h=0.4, size=12.5, highlight=(4, 5))
card(s, 8.1, 1.85, 4.6, 2.35, CREAM)
textbox(s, 8.4, 2.05, 4.0, 0.35, "نکته‌ای که نباید اشتباه گفت", size=15,
        bold=True, color=AMBER)
textbox(s, 8.4, 2.5, 4.0, 1.5,
        "اندازه وصله در هر دو یکسان است، پس تعداد وصله‌ها فرق نمی‌کند. "
        "بهبود از ظرفیت مدل می‌آید، نه از وضوح مکانی بیشتر.", size=12.5,
        spacing=19)
textbox(s, 0.6, 4.6, 12.1, 0.4, "چرا این تنها بهبود قطعی پروژه است",
        size=17, bold=True, color=TITLE)
for x, title, value, colour, fill in [
        (0.6, "بازه پنج بذر ViT-B", "0.310 — 0.354", RED, PANEL),
        (4.7, "بازه پنج بذر ViT-L", "0.370 — 0.423", TEAL, MINT),
        (8.8, "آزمون من-ویتنی", "p = 1 / 252", AMBER, CREAM)]:
    card(s, x, 5.1, 3.9, 1.5, fill)
    textbox(s, x + 0.3, 5.3, 3.3, 0.3, title, size=12, color=MUTED)
    textbox(s, x + 0.3, 5.65, 3.3, 0.5, value, size=20, bold=True,
            color=colour, rtl=False, align=PP_ALIGN.RIGHT)
notes(s, "دو بازه هیچ همپوشانی ندارند. بدترین بذر ViT-L از بهترین بذر "
         "ViT-B بالاتر است.")

# =====================================================================
# 20 — mixup and logit adjustment
# =====================================================================
s = light_slide("آمیزش و تنظیم لاجیت", "مسیر دوم")
card(s, 0.6, 1.8, 5.9, 2.15)
textbox(s, 0.95, 1.98, 5.2, 0.35, "آمیزش در فضای ویژگی", size=16,
        bold=True, color=TITLE)
textbox(s, 0.95, 2.42, 5.2, 1.4,
        "مدل تا دوره دهم زیان آموزش را تقریباً صفر می‌کرد، یعنی حفظ "
        "می‌کرد. ترکیب محدب دو ویدیو نمونه‌هایی می‌سازد که در داده نیستند.",
        size=12.5, spacing=19)
card(s, 6.8, 1.8, 5.9, 2.15, CREAM)
textbox(s, 7.15, 1.98, 5.2, 0.35, "تنظیم لاجیت برای دم بلند", size=16,
        bold=True, color=AMBER)
textbox(s, 7.15, 2.42, 5.2, 1.4,
        "مقداری متناسب با لگاریتم احتمال پیشین از لاجیت هر کلاس کسر "
        "می‌شود تا مرز تصمیم به سود کلاس‌های کمیاب جابه‌جا شود.", size=12.5,
        spacing=19)
card(s, 0.6, 4.15, 12.1, 1.15, BLUSH)
textbox(s, 0.95, 4.3, 11.4, 0.35,
        "آزمایشی که شکست خورد و از آن یاد گرفتیم", size=15, bold=True,
        color=RED)
textbox(s, 0.95, 4.68, 11.4, 0.55,
        "اعمال خام فرمول روی هر ۱۴ کلاس، صحت را از ۰٫۶۸۹۷ به ۰٫۴۷۲ "
        "انداخت: مرحله اول تصمیم خراب می‌شد. اصلاح: محدود کردن به ۱۳ کلاس "
        "ناهنجار و مرکزی کردن.", size=12, spacing=18)
table(s, 0.6, 5.5, 12.1,
      ["پیکربندی", "میانگین بذرها", "نتیجه گروهی", "صحت", "صحت نوع"], [
          ["پایه ViT-L", "0.3955", "0.4122", "0.6897", "0.4786"],
          ["آمیزش ۰٫۲", "0.4203", "0.4161", "0.6793", "0.4714"],
          ["آمیزش ۰٫۲ + تنظیم لاجیت", "0.4147", "0.4298", "0.6897",
           "0.5000"],
          ["آمیزش ۰٫۴ + تنظیم لاجیت", "0.4135", "0.4248", "0.6759",
           "0.4929"],
      ], [4.3, 2.0, 2.0, 1.9, 1.9], row_h=0.3, size=11.5, highlight=(2,))
notes(s, "ترکیب روی هر سه معیار بهترین یا هم‌ارز بهترین است. سه ترکیب "
         "مقایسه شد، پس تورم انتخاب حدود ۰٫۰۵ است.")

# =====================================================================
# 21 — progression, drawn with shapes
# =====================================================================
s = light_slide("مسیر بهبود، گام به گام", "مسیر دوم")
BASE, TOP, LO, HI = 5.30, 1.95, 0.25, 0.46


def y_of(value):
    return BASE - (value - LO) / (HI - LO) * (BASE - TOP)


for level in (0.30, 0.35, 0.40, 0.45):
    shape(s, MSO_SHAPE.RECTANGLE, 1.45, y_of(level), 11.25, 0.02,
          GRID)
    textbox(s, 0.6, y_of(level) - 0.14, 0.75, 0.28, f"{level:.2f}", size=10,
            color=MUTED, rtl=False, align=PP_ALIGN.RIGHT)

shape(s, MSO_SHAPE.RECTANGLE, 1.45, BASE, 11.25, 0.025,
      AXIS)

for index, (label, value, colour) in enumerate([
        ("MLP / ViT-B", 0.2975, BAR),
        ("+ Temporal", 0.3444, BAR),
        ("+ lr 3e-4", 0.3645, BAR),
        ("+ ViT-L", 0.4122, AMBER),
        ("+ Mixup", 0.4161, AMBER),
        ("+ Logit adj.", 0.4298, AMBER)]):
    slot = 11.25 / 6
    x = 1.45 + index * slot + (slot - 1.30) / 2
    top = y_of(value)
    shape(s, MSO_SHAPE.RECTANGLE, x, top, 1.30, BASE - top, colour)
    textbox(s, x - 0.25, top - 0.34, 1.80, 0.3, f"{value:.4f}", size=11,
            bold=True, rtl=False, align=PP_ALIGN.CENTER)
    textbox(s, x - 0.35, BASE + 0.1, 2.0, 0.3, label, size=11, rtl=False,
            align=PP_ALIGN.CENTER)

stat(s, 0.6, 5.85, 3.9, "+0.1323", "بهبود مطلق ماکرو F1", TEAL)
stat(s, 4.7, 5.85, 3.9, "+44%", "بهبود نسبی", TEAL)
stat(s, 8.8, 5.85, 3.9, "0.075", "کف نوفه این مجموعه آزمون", AMBER)
notes(s, "دو گام از شش گام بیشترین سهم را داشتند: توجه زمانی و ارتقای "
         "رمزگذار. هر دو از تحلیل خطا آمدند.")

# =====================================================================
# 22 — negative results
# =====================================================================
s = light_slide("آزمایش‌هایی که نتیجه ندادند", "صداقت روش‌شناختی")
table(s, 0.6, 1.85, 12.1, ["آزمایش", "نتیجه", "مرجع", "داوری"], [
    ["نمونه‌برداری سبک Kinetics با گام ۴ فریم", "0.3089", "0.3405", "بدتر"],
    ["دو برابر کردن تعداد قطعه‌ها به ۳۲", "0.3875", "0.3955", "بدتر"],
    ["الحاق ویژگی دو رمزگذار", "0.3974", "0.3955", "درون نوفه"],
    ["آموزش با یک قطعه برتر", "—", "—", "بدون بهبود"],
    ["تنظیم لاجیت به‌تنهایی", "0.4114", "0.3955", "درون نوفه"],
    ["میانگین‌گیری بذرها برای ماکرو F1", "—", "—", "بدون بهبود"],
    ["بازنویسی توصیف‌های متنی کلاس‌ها", "0.74", "0.86", "بدتر"],
], [6.1, 2.0, 2.0, 2.0], row_h=0.33, size=12)
card(s, 0.6, 4.75, 12.1, 1.95, CREAM)
textbox(s, 0.95, 4.95, 11.4, 0.35,
        "چرا نمونه‌برداری استاندارد بدتر شد — مهم‌ترین یافته منفی", size=15,
        bold=True, color=AMBER)
textbox(s, 0.95, 5.35, 11.4, 1.25,
        ["قطعه‌های ما کشیده بودند و با نرخ نمونه‌برداری وزن‌های "
         "پیش‌آموزش‌دیده نمی‌خواندند. اصلاح آن نتیجه را بدتر کرد: "
         "قطعه‌های کشیده کل ویدیو را می‌پوشانند، در حالی که سی‌ودو پنجره "
         "دو ثانیه‌ای تنها حدود ۲۱ درصد یک ویدیوی طولانی را لمس می‌کنند.",
         "نتیجه: پوشش بر اعتبار قطعه غلبه دارد."], size=12, spacing=18,
        space_after=6)
notes(s, "گزارش نتایج منفی بخشی از نتیجه است، نه حاشیه آن.")

# =====================================================================
# 23 — noise floor
# =====================================================================
s = light_slide("چقدر از این اعداد نوفه است؟", "اعتبارسنجی آماری")
textbox(s, 0.6, 1.78, 12.1, 0.4,
        "مجموعه آزمون با ۴۰۰۰ بار بازنمونه‌گیری خودگردان تحلیل شد تا "
        "پراکندگی مورد انتظار تخمین زده شود.", size=14)
table(s, 0.6, 2.35, 7.3,
      ["معیار", "مقدار نقطه‌ای", "انحراف معیار", "بازه اطمینان ۹۵ درصد"], [
          ["ماکرو F1", "0.4298", "0.037", "[0.339 , 0.503]"],
          ["صحت چندکلاسه", "0.6897", "0.027", "[0.635 , 0.745]"],
          ["صحت نوع ناهنجاری", "0.5000", "0.041", "[0.420 , 0.585]"],
          ["AUC دودویی", "0.9610", "0.011", "[0.938 , 0.980]"],
      ], [2.2, 1.7, 1.7, 1.7], row_h=0.42, size=12, highlight=(0,))
card(s, 8.2, 2.35, 4.5, 2.15, CREAM)
textbox(s, 8.5, 2.53, 3.9, 0.35, "کف نوفه", size=16, bold=True, color=AMBER)
textbox(s, 8.5, 2.9, 3.9, 0.6, "0.075", size=32, bold=True, color=AMBER,
        rtl=False, align=PP_ALIGN.RIGHT)
textbox(s, 8.5, 3.6, 3.9, 0.8,
        "اختلاف کمتر از این مقدار در ماکرو F1 روی این تقسیم از نوفه قابل "
        "تفکیک نیست.", size=12, spacing=18)
textbox(s, 0.6, 4.7, 12.1, 0.4, "سقف تنظیم روی یک مجموعه آزمون کوچک",
        size=17, bold=True, color=TITLE)
table(s, 0.6, 5.2, 4.6, ["تعداد ارزیابی", "تورم مورد انتظار"], [
    ["۸", "+0.069"],
    ["۵۰", "+0.095"],
    ["۲۰۰", "+0.111"],
], [2.3, 2.3], row_h=0.33, size=12, highlight=(2,))
card(s, 5.6, 5.2, 7.1, 1.45)
textbox(s, 5.9, 5.4, 6.5, 1.1,
        "با حدود ۲۰۰ ارزیابی روی همین تقسیم، گزارش بهترین عدد "
        "مشاهده‌شده نتیجه را بیش از اندازه خود اثر بزرگ نشان می‌دهد. همه "
        "اعداد این ارائه از پیکربندی‌هایی می‌آیند که پیش از اجرا تثبیت "
        "شده بودند.", size=12, spacing=18)
notes(s, "این تحلیل تفسیر کل پروژه را عوض کرد. ارتقای رمزگذار از کف نوفه "
         "عبور می‌کند؛ بقیه بهبودها با احتیاط گزارش می‌شوند.")

# =====================================================================
# 24 — full results
# =====================================================================
s = light_slide("جدول کامل نتایج", "جمع‌بندی نتایج")
table(s, 0.6, 1.8, 12.1,
      ["پیکربندی", "رمزگذار", "ماکرو F1", "صحت نوع", "صحت کلی", "AUC"], [
          ["مسیر اول — اسم خام کلاس، وزن دستی", "X-CLIP ViT-B/32", "0.3327",
           "0.3716", "0.6385", "0.9473"],
          ["مسیر اول — توصیف بصری + وزن معکوس", "X-CLIP ViT-B/32", "0.3713",
           "0.4054", "0.6351", "0.9637"],
          ["مسیر اول — چهار نمای زمانی در آزمون", "X-CLIP ViT-B/32",
           "0.3881", "0.4257", "0.6419", "0.9629"],
          ["مسیر دوم — سر نقطه‌ای، خط پایه", "VideoMAE ViT-B", "0.2975",
           "0.3860", "0.5660", "—"],
          ["مسیر دوم — سر ترنسفورمر زمانی", "VideoMAE ViT-B", "0.3444",
           "0.3930", "0.6350", "—"],
          ["مسیر دوم — نرخ یادگیری ۰٫۰۰۰۳", "VideoMAE ViT-B", "0.3645",
           "0.4210", "0.6550", "—"],
          ["مسیر دوم — ارتقا به رمزگذار بزرگ", "VideoMAE ViT-L", "0.4122",
           "0.4786", "0.6897", "—"],
          ["مسیر دوم — افزودن آمیزش", "VideoMAE ViT-L", "0.4161", "0.4714",
           "0.6793", "—"],
          ["مسیر دوم — آمیزش + تنظیم لاجیت", "VideoMAE ViT-L", "0.4298",
           "0.5000", "0.6897", "0.9610"],
      ], [4.5, 2.2, 1.4, 1.4, 1.4, 1.2], row_h=0.36, size=11,
      head_size=11.5, highlight=(8,))
textbox(s, 0.6, 5.78, 12.1, 0.5,
        "مسیر اول روی تقسیمی با ۲۹۶ ویدیوی آزمون و مسیر دوم روی تقسیم "
        "استاندارد با ۲۹۰ ویدیو اجرا شده‌اند.", size=11.5, color=MUTED,
        spacing=17)
stat(s, 0.6, 6.3, 3.9, "0.4298", "ماکرو F1 نهایی", AMBER)
stat(s, 4.7, 6.3, 3.9, "0.5000", "صحت نوع ناهنجاری")
stat(s, 8.8, 6.3, 3.9, "0.6897", "صحت چندکلاسه")
notes(s, "عدد قابل دفاع نهایی 0.4298 با بودجه ثابت و بدون انتخاب روی "
         "مجموعه آزمون است.")

# =====================================================================
# 25 — conclusions
# =====================================================================
s = dark_slide()
shape(s, MSO_SHAPE.OVAL, 10.2, -1.4, 5.0, 5.0, DEEP)
textbox(s, 0.7, 0.55, 11.9, 0.7, "جمع‌بندی و کارهای آینده", size=29,
        bold=True, color=WHITE)
for index, (label, body, colour) in enumerate([
        ("دستاورد",
         "اصلاح سه اشکال بنیادی، بازطراحی معماری با رمزگذار منجمد، سر "
         "ترنسفورمر زمانی، و پروتکل ارزیابی با کف نوفه", GOOD_TINT),
        ("محدودیت",
         "رمزگذار به دامنه نظارتی تطبیق نمی‌یابد؛ مجموعه آزمون کوچک است "
         "و بازه‌های اطمینان پهن", ACCENT_TINT),
        ("کار آینده",
         "افزایش وضوح مکانی برای خوشه دزدی، تنظیم دقیق رمزگذار روی داده "
         "نظارتی، و تقسیم داده با سهم بزرگ‌تر برای آزمون", ICE)]):
    y = 1.75 + index * 1.25
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, y, 11.9, 0.95, DEEP)
    textbox(s, 10.6, y + 0.16, 1.8, 0.4, label, size=15, bold=True,
            color=colour)
    textbox(s, 1.0, y + 0.2, 9.4, 0.75, body, size=13, color=WHITE,
            spacing=20)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 5.7, 11.9, 1.1, AMBER)
textbox(s, 1.0, 5.92, 11.3, 0.7,
        "بیشترین اثر را نه یک ایده الگوریتمی، بلکه کاهش هزینه هر آزمایش "
        "داشت؛ و کمّی کردن نوفه پیش از تفسیر، از ادعاهایی جلوگیری کرد که "
        "داده پشتیبان آن‌ها نبود.", size=14, bold=True,
        color=ONACCENT, spacing=21)
notes(s, "سوال‌های محتمل: چرا رمزگذار را تنظیم دقیق نکردید، چرا عدد از "
         "مقالات پیشرفته پایین‌تر است، و از کجا معلوم آستانه روی تست "
         "تنظیم نشده.")


prs.save(OUTPUT)
print(f"wrote {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB, "
      f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
