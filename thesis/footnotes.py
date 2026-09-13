# -*- coding: utf-8 -*-
"""Footnote support for the thesis and report builders.

python-docx has no footnote API: it can neither create `word/footnotes.xml`
nor write a `<w:footnoteReference>` into a run. Both are done here by hand.

The Persian writing guide wants every foreign term given in Persian in the
body with the original spelling in a footnote on the same page, so the
content modules write the Latin form inline with the FOOTNOTE_MARK
delimiter and `Builder.para` turns it into a reference:

    "سر ترنسفورمر⟨Transformer head⟩ روی محور زمان"

Word numbers footnotes per page or per section on its own; the ids here
only have to be unique and ascending. Ids -1 and 0 are reserved by the
format for the separator lines Word draws above the notes.
"""

from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml.ns import qn
from docx.shared import Pt

from docx_style import _el, _set

FOOTNOTE_MARK = "⟨"   # ⟨
FOOTNOTE_END = "⟩"    # ⟩

CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument"
    ".wordprocessingml.footnotes+xml"
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Ids -1 and 0 carry the separator rules, so real notes start at 1.
FIRST_ID = 1


def split_footnotes(text):
    """Split body text into (chunk, note-or-None) pairs.

    "الف⟨A⟩ب" -> [("الف", "A"), ("ب", None)]
    """
    parts = []
    rest = text

    while FOOTNOTE_MARK in rest:
        before, _, after = rest.partition(FOOTNOTE_MARK)
        note, _, rest = after.partition(FOOTNOTE_END)
        parts.append((before, note))

    if rest:
        parts.append((rest, None))

    return parts


class FootnoteStore:
    """Collects note text and writes the footnotes part at save time."""

    def __init__(self):
        self._notes = []

    def __bool__(self):
        return bool(self._notes)

    def add(self, text):
        """Register one note and return the id to reference it by."""
        self._notes.append(text)

        return FIRST_ID + len(self._notes) - 1

    # ------------------------------------------------------------------

    def reference(self, paragraph, note_id, size):
        """Append the superscript reference mark to a paragraph."""
        run = paragraph.add_run()
        rpr = run._r.get_or_add_rPr()
        _set(rpr, "w:rStyle", val="FootnoteReference")
        _set(rpr, "w:vertAlign", val="superscript")
        _set(rpr, "w:rtl", val="0")
        run._r.append(_el("w:footnoteReference", id=str(note_id)))
        run.font.size = Pt(size)

        return run

    # ------------------------------------------------------------------

    def _xml(self):
        from lxml import etree

        root = etree.Element(qn("w:footnotes"), nsmap={"w": W_NS})

        for note_id, kind in ((-1, "separator"), (0, "continuationSeparator")):
            note = etree.SubElement(root, qn("w:footnote"))
            note.set(qn("w:type"), kind)
            note.set(qn("w:id"), str(note_id))

            paragraph = etree.SubElement(note, qn("w:p"))
            ppr = etree.SubElement(paragraph, qn("w:pPr"))
            spacing = etree.SubElement(ppr, qn("w:spacing"))
            spacing.set(qn("w:after"), "0")
            spacing.set(qn("w:line"), "240")
            spacing.set(qn("w:lineRule"), "auto")

            run = etree.SubElement(paragraph, qn("w:r"))
            etree.SubElement(run, qn(f"w:{kind}"))

        for index, text in enumerate(self._notes):
            note = etree.SubElement(root, qn("w:footnote"))
            note.set(qn("w:id"), str(FIRST_ID + index))

            paragraph = etree.SubElement(note, qn("w:p"))
            ppr = etree.SubElement(paragraph, qn("w:pPr"))
            style = etree.SubElement(ppr, qn("w:pStyle"))
            style.set(qn("w:val"), "FootnoteText")
            bidi = etree.SubElement(ppr, qn("w:bidi"))
            bidi.set(qn("w:val"), "1")
            jc = etree.SubElement(ppr, qn("w:jc"))
            jc.set(qn("w:val"), "right")

            mark = etree.SubElement(paragraph, qn("w:r"))
            mark_pr = etree.SubElement(mark, qn("w:rPr"))
            mark_style = etree.SubElement(mark_pr, qn("w:rStyle"))
            mark_style.set(qn("w:val"), "FootnoteReference")
            etree.SubElement(mark, qn("w:footnoteRef"))

            body = etree.SubElement(paragraph, qn("w:r"))
            body_pr = etree.SubElement(body, qn("w:rPr"))
            rtl = etree.SubElement(body_pr, qn("w:rtl"))
            # The note carries the Latin original, so it reads left to right
            # even though the note itself hangs off a right-aligned line.
            rtl.set(qn("w:val"), "0")

            label = etree.SubElement(body, qn("w:t"))
            label.set(
                "{http://www.w3.org/XML/1998/namespace}space", "preserve"
            )
            label.text = f" {text}"

        return etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )

    def attach(self, doc):
        """Add word/footnotes.xml to the package and relate it."""
        if not self._notes:
            return

        package = doc.part.package
        part = Part(
            PackURI("/word/footnotes.xml"), CONTENT_TYPE, self._xml(), package
        )
        doc.part.relate_to(part, RT.FOOTNOTES)
