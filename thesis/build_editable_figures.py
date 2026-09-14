# -*- coding: utf-8 -*-
"""Rebuild every thesis figure as native PowerPoint objects.

    python thesis/build_editable_figures.py

make_figures.py draws with matplotlib, so what reaches the document is a
picture: to change a label you have to edit Python and re-run. This writes
the same eight figures as things PowerPoint owns.

Four are flow diagrams, built from rounded rectangles, connectors and text
boxes -- click any box and type. Four are data plots, built as real charts
with their numbers in an embedded worksheet, so right-click then Edit Data
opens the table and the chart redraws itself.

Charts are written by python-pptx rather than by hand. An earlier attempt
with a different library emitted three c:axId children against two declared
axes, which PowerPoint refuses to open; the numbers here all come from
REVIEW_14CLASS.md and match the figures in the report.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "Thesis_Figures_Editable.pptx"

INK = RGBColor(0x1B, 0x22, 0x36)
MUTED = RGBColor(0x6A, 0x74, 0x8C)
ACCENT = RGBColor(0x2E, 0x5E, 0x8C)
ACCENT2 = RGBColor(0xC1, 0x6B, 0x2E)
GREEN = RGBColor(0x5B, 0x7D, 0x55)
PALE = RGBColor(0xEE, 0xF3, 0xF7)
SAND = RGBColor(0xF5, 0xEF, 0xE6)
LEAF = RGBColor(0xEA, 0xF1, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FA = "B Nazanin"
EN = "Calibri"


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------

def deck_new():
    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)

    return deck


def slide_new(deck, caption):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25),
                                   Inches(12.3), Inches(0.5))
    frame = box.text_frame
    paragraph = frame.paragraphs[0]
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.set("rtl", "1")
    p_pr.set("algn", "r")
    run = paragraph.add_run()
    run.text = caption
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = INK
    run.font.name = FA

    return slide


def box(slide, x, y, w, h, text, face=PALE, edge=ACCENT, size=11):
    """A rounded rectangle carrying centred text -- click it and type."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
        Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = face
    shape.line.color.rgb = edge
    shape.line.width = Pt(1.25)
    shape.shadow.inherit = False

    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.05)

    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = INK
        run.font.name = EN

    return shape


def label(slide, x, y, w, text, size=10, color=MUTED, italic=False,
          align=PP_ALIGN.CENTER, rtl=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w),
                                     Inches(0.3))
    frame = shape.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]

    if rtl:
        p_pr = paragraph._p.get_or_add_pPr()
        p_pr.set("rtl", "1")

    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = FA if rtl else EN

    return shape


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    """A straight connector with a head, drawn as a thin rotated shape."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW if y1 == y2 else MSO_SHAPE.DOWN_ARROW,
        Inches(min(x1, x2)), Inches(min(y1, y2)),
        Inches(max(abs(x2 - x1), 0.28)), Inches(max(abs(y2 - y1), 0.22)),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False

    return shape


def line(slide, x1, y1, x2, y2, color=MUTED):
    from pptx.util import Emu as _Emu

    connector = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    connector.line.color.rgb = color
    connector.line.width = Pt(1.1)

    return connector


def chart(slide, kind, x, y, w, h, categories, series, *, colors=None,
          legend=False, number_format="0.0000", font=9):
    """A real chart: its numbers live in a worksheet Edit Data can open."""
    data = CategoryChartData()
    data.categories = categories

    for name, values in series:
        data.add_series(name, values, number_format)

    graphic = slide.shapes.add_chart(
        kind, Inches(x), Inches(y), Inches(w), Inches(h), data
    )
    plot = graphic.chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format = number_format
    labels.number_format_is_linked = False
    labels.font.size = Pt(font - 1)
    labels.font.color.rgb = INK

    if colors:
        for index, colour in enumerate(colors):
            if index < len(plot.series):
                fill = plot.series[index].format.fill
                fill.solid()
                fill.fore_color.rgb = colour

    graphic.chart.has_legend = legend

    if legend:
        graphic.chart.legend.position = XL_LEGEND_POSITION.TOP
        graphic.chart.legend.include_in_layout = False
        graphic.chart.legend.font.size = Pt(font)

    for axis in (graphic.chart.category_axis, graphic.chart.value_axis):
        axis.tick_labels.font.size = Pt(font)
        axis.tick_labels.font.color.rgb = INK

    return graphic.chart


def hide_series(chart_obj, index):
    """Drop a helper series from the labels and the legend, not the data.

    A range is drawn as a transparent bar up to the low end followed by the
    span itself. The first bar has to stay in the worksheet -- moving it is
    how you move the range -- but printing its value and listing it in the
    legend would only confuse the reader.
    """
    from pptx.oxml.ns import qn

    series = chart_obj.plots[0].series[index]._element
    labels = series.find(qn("c:dLbls"))

    if labels is None:
        labels = series.makeelement(qn("c:dLbls"), {})
        marker = series.find(qn("c:cat"))
        series.insert(list(series).index(marker) if marker is not None
                      else len(series), labels)

    for child in list(labels):
        labels.remove(child)

    delete = labels.makeelement(qn("c:delete"), {})
    delete.set("val", "1")
    labels.append(delete)

    legend = chart_obj._chartSpace.find(qn("c:chart")).find(qn("c:legend"))

    if legend is not None:
        entry = legend.makeelement(qn("c:legendEntry"), {})
        idx = entry.makeelement(qn("c:idx"), {})
        idx.set("val", str(index))
        gone = entry.makeelement(qn("c:delete"), {})
        gone.set("val", "1")
        entry.append(idx)
        entry.append(gone)
        legend.insert(0, entry)


# ----------------------------------------------------------------------
# the eight figures
# ----------------------------------------------------------------------

def fig_pipeline(deck):
    slide = slide_new(deck, "شکل ۱-۱  خط لوله‌ی نهایی")

    label(slide, 0.6, 1.0, 12.1,
          "Stage 1  -  run once, 150 min for 1895 videos", italic=True)

    row = 1.45
    box(slide, 0.6, row, 2.6, 1.3,
        "Surveillance video\n(untrimmed, one\nvideo-level label)")
    box(slide, 3.75, row, 2.6, 1.3, "16 clips\n16 frames each\n224x224")
    box(slide, 6.9, row, 2.7, 1.3,
        "VideoMAE ViT-L\n(frozen, Kinetics)\ntoken mean-pool",
        face=SAND, edge=ACCENT2)
    box(slide, 10.15, row, 2.6, 1.3,
        "cached features\n[16, 1024]\none .npy / video")

    for x in (3.28, 6.43, 9.68):
        arrow(slide, x, row + 0.55, x + 0.4, row + 0.55)

    line(slide, 11.45, 2.75, 11.45, 3.2)
    line(slide, 11.45, 3.2, 1.9, 3.2)
    arrow(slide, 1.78, 3.2, 1.78, 3.62)
    label(slide, 3.5, 3.0, 6.5, "features reloaded from disk each epoch",
          size=9)

    row = 3.9
    box(slide, 0.6, row, 2.6, 1.3,
        "LayerNorm + Linear\n+ positional\nembedding")
    box(slide, 3.75, row, 2.6, 1.3,
        "Transformer\nencoder over\nthe 16 clips")
    box(slide, 6.9, row + 0.2, 2.7, 0.9, "clip logits\n[16, 14]")
    box(slide, 10.15, row + 0.2, 2.6, 0.9,
        "top-k MIL pooling\n-> video logits", face=SAND, edge=ACCENT2)

    for x in (3.28, 6.43, 9.68):
        arrow(slide, x, row + 0.6, x + 0.4, row + 0.6)

    label(slide, 0.6, 5.45, 12.1,
          "Stage 2  -  trains in ~80 s, so hyper-parameters can be swept",
          italic=True)


def fig_two_stage(deck):
    slide = slide_new(deck, "شکل ۲-۱  رویه‌ی تصمیم دومرحله‌ای")

    row = 1.3
    box(slide, 0.6, row, 3.3, 1.35, "per-clip softmax\nP(c | clip), c = 0..13")
    box(slide, 4.6, row, 3.5, 1.35,
        "anomaly evidence\ns = max over clips\nof 1 - P(Normal)",
        face=SAND, edge=ACCENT2)
    box(slide, 8.8, row, 3.4, 1.35, "s < threshold ?")

    arrow(slide, 4.0, row + 0.57, 4.5, row + 0.57)
    arrow(slide, 8.2, row + 0.57, 8.7, row + 0.57)

    arrow(slide, 10.4, 2.75, 10.4, 3.9)
    label(slide, 10.55, 3.1, 1.0, "yes", size=11, align=PP_ALIGN.LEFT)
    box(slide, 8.8, 4.05, 3.4, 1.0, "predict Normal", face=LEAF, edge=GREEN)

    line(slide, 9.3, 2.75, 5.2, 3.9)
    label(slide, 6.6, 3.05, 1.2, "no", size=11)
    box(slide, 0.9, 4.05, 6.8, 1.0,
        "average P over the top-k most anomalous clips,\n"
        "argmax over classes 1..13", face=SAND, edge=ACCENT2)

    label(slide, 0.6, 5.5, 12.1,
          'Stage 1 answers "is it anomalous", stage 2 answers '
          '"which anomaly"', italic=True)


def fig_macro_f1_progress(deck):
    slide = slide_new(deck, "شکل ۳-۱  روند ماکرو اف‑یک در طول مراحل پروژه")

    chart(slide, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.9, 1.2, 11.5, 4.9,
          ["MLP head, ViT-B", "temporal head, ViT-B", "+ lr 3e-4",
           "+ ViT-L features", "+ mixup 0.2 + logit adjust"],
          [("macro F1", (0.2975, 0.3444, 0.3645, 0.4122, 0.4298))],
          colors=[ACCENT])

    label(slide, 0.9, 6.25, 11.5,
          "X-CLIP fine-tuned reached 0.3881 on its own split",
          italic=True, color=ACCENT2)


def fig_backbone_compare(deck):
    slide = slide_new(deck, "شکل ۳-۲  مقایسه‌ی دو رمزگذار")

    chart(slide, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.9, 1.15, 11.5, 5.2,
          ["binary AUC", "accuracy", "type accuracy", "macro F1"],
          [("X-CLIP fine-tuned", (0.9629, 0.6419, 0.4257, 0.3881)),
           ("VideoMAE ViT-B (frozen)", (0.9649, 0.6552, 0.4214, 0.3645)),
           ("VideoMAE ViT-L (frozen, final)",
            (0.9610, 0.6897, 0.5000, 0.4298))],
          colors=[MUTED, ACCENT, ACCENT2], legend=True,
          number_format="0.000")


def fig_seed_ranges(deck):
    slide = slide_new(deck, "شکل ۳-۳  دامنه‌ی نتایج پنج بذر")

    # a range is a bar from lo to hi: an invisible bar to the start, then
    # the span itself, which is what Edit Data lets you move
    span = chart(slide, XL_CHART_TYPE.BAR_STACKED, 0.9, 1.3, 11.5, 3.6,
                 ["VideoMAE ViT-B (5 seeds)", "VideoMAE ViT-L (5 seeds)"],
                 [("start", (0.310, 0.370)),
                  ("macro F1 range", (0.044, 0.053))],
                 colors=[WHITE, ACCENT2], legend=False,
                 number_format="0.000")
    hide_series(span, 0)

    label(slide, 0.9, 5.1, 11.5,
          "ViT-B spans 0.310-0.354 with mean 0.3405   |   "
          "ViT-L spans 0.370-0.423 with mean 0.3955", size=11)
    label(slide, 0.9, 5.5, 11.5,
          "the two ranges do not overlap  (Mann-Whitney p = 1/252)",
          italic=True, color=ACCENT2, size=11)


def fig_mixup_logit(deck):
    slide = slide_new(deck, "شکل ۳-۴  اثر آمیزش و تنظیم لاجیت")

    chart(slide, XL_CHART_TYPE.COLUMN_CLUSTERED, 0.9, 1.15, 11.5, 5.2,
          ["baseline", "mixup 0.2", "mixup 0.2 + adjust",
           "mixup 0.4 + adjust", "mixup 0.4"],
          [("macro F1", (0.4122, 0.4161, 0.4298, 0.4248, 0.4025)),
           ("anomaly-type accuracy",
            (0.4786, 0.4714, 0.5000, 0.4929, 0.4643))],
          colors=[ACCENT, MUTED], legend=True)


def fig_bootstrap_ci(deck):
    slide = slide_new(deck, "شکل ۳-۵  بازه‌های اطمینان ۹۵ درصد")

    names = ["binary AUC", "accuracy", "type accuracy", "macro F1"]
    lo = (0.938, 0.635, 0.420, 0.339)
    hi = (0.980, 0.745, 0.585, 0.503)
    width = tuple(round(b - a, 3) for a, b in zip(lo, hi))

    interval = chart(slide, XL_CHART_TYPE.BAR_STACKED, 0.9, 1.25, 11.5, 3.9,
                     names,
                     [("lower bound", lo),
                      ("95% confidence interval", width)],
                     colors=[WHITE, ACCENT], legend=False,
                     number_format="0.000")
    hide_series(interval, 0)

    label(slide, 0.9, 5.35, 11.5,
          "point estimates:  AUC 0.9610   |   accuracy 0.6897   |   "
          "type accuracy 0.5000   |   macro F1 0.4298", size=11)
    label(slide, 0.9, 5.75, 11.5,
          "4000 bootstrap resamples of the 290 test videos",
          italic=True, size=11)


def fig_noise_floor(deck):
    slide = slide_new(deck, "شکل ۳-۶  کف نوفه در برابر بهبودهای گزارش‌شده")

    import math

    sigma = 0.034
    counts = [2, 5, 8, 15, 25, 50, 75, 100, 150, 200, 250]
    inflation = tuple(round(sigma * math.sqrt(2 * math.log(k)), 4)
                      for k in counts)

    chart(slide, XL_CHART_TYPE.LINE_MARKERS, 0.9, 1.2, 11.5, 4.6,
          [str(k) for k in counts],
          [("expected inflation of the best score", inflation)],
          colors=[ACCENT], legend=False)

    label(slide, 0.9, 6.0, 11.5,
          "evaluations against the same test split  -  "
          "8 runs inflate by 0.070, 200 runs by 0.110", size=11)
    label(slide, 0.9, 6.4, 11.5,
          "which is why every number in the report comes from a "
          "configuration fixed in advance", italic=True, size=11)


FIGURES = [
    fig_pipeline,
    fig_two_stage,
    fig_macro_f1_progress,
    fig_backbone_compare,
    fig_seed_ranges,
    fig_mixup_logit,
    fig_bootstrap_ci,
    fig_noise_floor,
]


def main():
    deck = deck_new()

    for figure in FIGURES:
        figure(deck)

    deck.save(OUTPUT)
    size = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT}  ({size:.0f} KB, {len(FIGURES)} slides)")
    print("Boxes: click and type. Charts: right-click, Edit Data.")


if __name__ == "__main__":
    main()
