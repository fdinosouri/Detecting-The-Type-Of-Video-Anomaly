# -*- coding: utf-8 -*-
"""Repair the two pptxgenjs defects PowerPoint refuses to open.

    python3 fix_pptx.py deck.pptx

Both are invisible to LibreOffice, python-pptx and the XSD, so nothing
catches them before the file reaches PowerPoint:

1. Tables are numbered from a different counter than the other shapes, so
   a slide holding both can emit two `<p:cNvPr>` elements with the same
   id. Shape ids must be unique within a slide.
2. `<c:barChart>` lists three `<c:axId>` children while only two axis
   elements are declared, so the third id resolves to nothing.

The archive is rewritten entry by entry, carrying each original
ZipInfo across unchanged, so the repaired package differs from the one
the library wrote only in the parts that had to change. Rebuilding it
with `zip` instead sets Unix attributes that PowerPoint then cannot
read at all.
"""

import collections
import re
import shutil
import sys
import zipfile
from pathlib import Path

CNVPR = re.compile(rb'<p:cNvPr id="(\d+)"')
AXIS = re.compile(rb'<c:(catAx|valAx|serAx|dateAx)>.*?<c:axId val="(\d+)"/>',
                  re.S)
CHART_BLOCK = re.compile(rb'<c:(\w*Chart)>.*?</c:\1>', re.S)
AX_ID = re.compile(rb'<c:axId val="(\d+)"/>')
OVERRIDE = re.compile(rb'<Override PartName="([^"]+)"[^>]*/>')


def drop_dangling_overrides(xml, parts):
    """Remove `<Override>` entries naming a part the package lacks.

    pptxgenjs writes one slideMaster override per slide while emitting a
    single master, so a 25-slide deck declares 24 parts that do not
    exist. That alone makes PowerPoint refuse to read the file.
    """
    def keep(match):
        name = match.group(1).lstrip(b"/").decode()

        return match.group(0) if name in parts else b""

    return OVERRIDE.sub(keep, xml)


def dedupe_shape_ids(xml):
    """Renumber any `<p:cNvPr>` id already used earlier in the slide."""
    used = set()
    counter = [1]

    def fix(match):
        current = match.group(1)

        if current not in used:
            used.add(current)
            return match.group(0)

        while str(counter[0]).encode() in used:
            counter[0] += 1

        replacement = str(counter[0]).encode()
        used.add(replacement)

        return b'<p:cNvPr id="%s"' % replacement

    return CNVPR.sub(fix, xml)


def strip_orphan_axis_ids(xml):
    """Drop `<c:axId>` references no axis element declares."""
    declared = {m.group(2) for m in AXIS.finditer(xml)}

    def fix_block(block):
        return AX_ID.sub(
            lambda m: m.group(0) if m.group(1) in declared else b"",
            block.group(0),
        )

    return CHART_BLOCK.sub(fix_block, xml)


def repair(path):
    path = Path(path)
    temp = path.with_suffix(".repaired.pptx")
    changed = []

    with zipfile.ZipFile(path) as src:
        entries = src.infolist()
        parts = {e.filename for e in entries if not e.is_dir()}

        with zipfile.ZipFile(temp, "w") as out:
            for item in entries:
                data = src.read(item.filename)

                if item.filename == "[Content_Types].xml":
                    fixed = drop_dangling_overrides(data, parts)
                elif item.filename.startswith("ppt/slides/slide"):
                    fixed = dedupe_shape_ids(data)
                elif item.filename.startswith("ppt/charts/"):
                    fixed = strip_orphan_axis_ids(data)
                else:
                    fixed = data

                if fixed != data:
                    changed.append(item.filename)

                # carry the library's own zip metadata across unchanged
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.internal_attr = item.internal_attr
                info.create_system = item.create_system

                out.writestr(info, fixed)

    shutil.move(str(temp), str(path))

    return changed


def check(path):
    """Report anything PowerPoint would still reject."""
    problems = []

    with zipfile.ZipFile(path) as z:
        parts = {n for n in z.namelist() if not n.endswith("/")}

        for name in OVERRIDE.findall(z.read("[Content_Types].xml")):
            target = name.lstrip(b"/").decode()

            if target not in parts:
                problems.append(f"[Content_Types].xml: no such part {target}")

        for name in z.namelist():
            if not name.startswith("ppt/slides/slide"):
                continue

            ids = CNVPR.findall(z.read(name))
            dupes = [i for i, n in collections.Counter(ids).items() if n > 1]

            if dupes:
                problems.append(f"{name}: duplicate shape ids {dupes}")

        for name in z.namelist():
            if not name.startswith("ppt/charts/") or not name.endswith(".xml"):
                continue

            xml = z.read(name)
            declared = {m.group(2) for m in AXIS.finditer(xml)}

            for block in CHART_BLOCK.finditer(xml):
                for ref in AX_ID.findall(block.group(0)):
                    if ref not in declared:
                        problems.append(f"{name}: undeclared axis id {ref}")

    return problems


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python3 fix_pptx.py deck.pptx")

    target = sys.argv[1]
    changed = repair(target)
    problems = check(target)

    if problems:
        for line in problems:
            print(f"  STILL BROKEN: {line}")
        sys.exit(1)

    print(f"repaired {len(changed)} part(s): {', '.join(changed) or 'none'}")


if __name__ == "__main__":
    main()
