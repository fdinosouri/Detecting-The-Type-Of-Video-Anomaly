# -*- coding: utf-8 -*-
"""Put every thesis figure on its own slide, as editable vector art.

    python thesis/make_figures.py        # first, to draw them
    python thesis/build_figure_deck.py

PowerPoint stores an SVG picture as two parts: the vector itself and a
raster fallback for older readers. python-pptx only knows how to add the
raster, so the SVG is attached by hand through the a:svgBlip extension --
the same structure PowerPoint writes when you insert an SVG yourself.

The point of the vector is that the figure stays editable: right-click a
picture, Convert to Shape, and every box, arrow and label becomes a
PowerPoint shape whose text and colour can be changed. Editing the
generating code in make_figures.py remains the other route, and the one
that keeps the document and the deck in step.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
OUTPUT = HERE / "Thesis_Figures_Editable.pptx"

SVG_URI = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
SVG_TYPE = "image/svg+xml"

INK = RGBColor(0x1B, 0x22, 0x36)
MUTED = RGBColor(0x6A, 0x74, 0x8C)

# figure number in the report -> caption
CAPTIONS = [
    ("fig_pipeline", "شکل ۱-۱  خط لوله‌ی نهایی"),
    ("fig_two_stage", "شکل ۲-۱  رویه‌ی تصمیم دومرحله‌ای"),
    ("fig_macro_f1_progress", "شکل ۳-۱  روند ماکرو اف‑یک در طول مراحل پروژه"),
    ("fig_backbone_compare", "شکل ۳-۲  مقایسه‌ی دو رمزگذار در پنج بذر مستقل"),
    ("fig_seed_ranges", "شکل ۳-۳  دامنه‌ی نتایج پنج بذر برای هر پیکربندی"),
    ("fig_mixup_logit", "شکل ۳-۴  اثر آمیزش و تنظیم لاجیت"),
    ("fig_bootstrap_ci", "شکل ۳-۵  بازه‌های اطمینان ۹۵ درصد"),
    ("fig_noise_floor", "شکل ۳-۶  کف نوفه در برابر بهبودهای گزارش‌شده"),
]


def attach_svg(slide, picture, svg_path, index):
    """Hang the vector off a picture that already carries its PNG.

    The SVG is added as a part by hand rather than through
    get_or_add_image_part, which runs every image through Pillow to read
    its size and raises UnidentifiedImageError on vector input.
    """
    part = Part(
        PackURI(f"/ppt/media/figure{index}.svg"),
        SVG_TYPE,
        slide.part.package,
        svg_path.read_bytes(),
    )
    rId = slide.part.relate_to(part, RT.IMAGE)

    blip = picture._element.blipFill.find(qn("a:blip"))
    ext_lst = blip.makeelement(qn("a:extLst"), {})
    ext = blip.makeelement(qn("a:ext"), {"uri": SVG_URI})
    svg_blip = blip.makeelement(f"{{{SVG_NS}}}svgBlip", {})
    svg_blip.set(qn("r:embed"), rId)

    ext.append(svg_blip)
    ext_lst.append(ext)
    blip.append(ext_lst)


def text_box(slide, left, top, width, height, text, size, bold=False,
             color=INK):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]

    # the captions are Persian, so the line has to read right to left
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.set("rtl", "1")
    p_pr.set("algn", "r")

    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "B Nazanin"

    return box


def build():
    missing = [name for name, _ in CAPTIONS
               if not (FIGURES / f"{name}.png").exists()]

    if missing:
        raise SystemExit(
            f"missing figures: {missing}. Run: python thesis/make_figures.py"
        )

    deck = Presentation()
    deck.slide_width = Inches(13.333)
    deck.slide_height = Inches(7.5)
    blank = deck.slide_layouts[6]

    for index, (name, caption) in enumerate(CAPTIONS, start=1):
        slide = deck.slides.add_slide(blank)

        text_box(slide, Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.5),
                 caption, 20, bold=True)
        text_box(slide, Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.4),
                 f"برای ویرایش: راست‌کلیک روی شکل ← Convert to Shape   "
                 f"|   فایل برداری: figures/{name}.svg", 11, color=MUTED)

        png = FIGURES / f"{name}.png"
        svg = FIGURES / f"{name}.svg"

        # fit inside the frame while keeping the aspect ratio
        from PIL import Image

        with Image.open(png) as image:
            ratio = image.width / image.height

        box_w, box_h = Inches(11.5), Inches(5.7)
        width = min(box_w, Emu(int(box_h * ratio)))
        height = Emu(int(width / ratio))

        picture = slide.shapes.add_picture(
            str(png),
            Emu(int((deck.slide_width - width) / 2)),
            Inches(1.0),
            width=width,
            height=height,
        )

        if svg.exists():
            attach_svg(slide, picture, svg, index)

    deck.save(OUTPUT)

    return OUTPUT


def main():
    path = build()
    size = path.stat().st_size / 1024
    print(f"wrote {path}  ({size:.0f} KB, {len(CAPTIONS)} slides)")
    print("In PowerPoint: right-click a figure, Convert to Shape, and every "
          "box, arrow and label becomes editable.")


if __name__ == "__main__":
    main()
