"""Helpers to fill the IEEE conference template (python-docx on the Word-saved transitional copy)."""
import copy
import re
from pathlib import Path
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt, Inches, Cm, RGBColor

import refs as REFS

NS = nsdecls("w", "m")
COL_W_IN = 3.33  # IEEE two-column text width (inches)


FONT_SCALE = 1.14


class Raw(str):
    """Text inserted verbatim (no markup / citation expansion)."""


ORDER_F = ["arch", "flow", "curve", "main", "conf", "kinds", "abl", "cal", "pr"]
ORDER_T = ["rel", "gaps", "edits", "data", "cfg", "main", "pc", "abl", "seed", "sel", "det", "op", "brk", "pq", "err", "demo", "full", "pcf1"]
FIGN = {n: i + 1 for i, n in enumerate(ORDER_F)}
TABN = {n: i + 1 for i, n in enumerate(ORDER_T)}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


class Paper:
    def __init__(self, template):
        self.doc = docx.Document(template)
        self.cache = REFS.fetch()
        self.order = []          # citation keys in order of first use
        self.anchor = None
        self.figs = 0
        self.tabs = 0
        self.eqs = 0

    # ----------------------------------------------------------- citations
    def num(self, key):
        if key not in self.order:
            if key not in self.cache and key not in REFS.ONLINE:
                raise KeyError(f"unknown reference key: {key}")
            self.order.append(key)
        return self.order.index(key) + 1

    def cite_str(self, keys):
        nums = sorted({self.num(k.strip()) for k in keys.split(",")})
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                out.append(f"[{nums[i]}]–[{nums[j]}]")
            else:
                out.extend(f"[{n}]" for n in nums[i:j + 1])
            i = j + 1
        return ", ".join(out)

    def expand(self, text):
        text = re.sub(r"\{T:(\w+)\}", lambda m: "Table " + ROMAN[TABN[m.group(1)]], text)
        text = re.sub(r"\{F:(\w+)\}", lambda m: "Fig. " + str(FIGN[m.group(1)]), text)
        return re.sub(r"\[@([^\]]+)\]", lambda m: self.cite_str(m.group(1).replace("@", "")), text)

    # ----------------------------------------------------------- run markup
    TOK = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")

    def add_runs(self, p, text, size=None, bold=None, italic=None, font=None):
        if isinstance(text, Raw):
            r = p.add_run(str(text))
            if bold is not None: r.bold = bold
            if size: r.font.size = Pt(size)
            return p
        text = self.expand(text)
        for part in self.TOK.split(text):
            if not part:
                continue
            b, i, sub, sup = bold, italic, False, False
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                part, b = part[2:-2], True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                part, i = part[1:-1], True
            elif part.startswith("_{"):
                part, sub = part[2:-1], True
            elif part.startswith("^{"):
                part, sup = part[2:-1], True
            r = p.add_run(part)
            if b is not None: r.bold = b
            if i is not None: r.italic = i
            if sub: r.font.subscript = True
            if sup: r.font.superscript = True
            if size: r.font.size = Pt(size)
            if font:
                r.font.name = font
        return p

    # ----------------------------------------------------------- placement
    def _place(self, el):
        self.anchor.addprevious(el)

    def para(self, style, text="", **kw):
        p = self.doc.add_paragraph(style=style)
        if text:
            self.add_runs(p, text, **kw)
        self._place(p._p)
        return p

    def h1(self, t): return self.para("Heading 1", t)
    def h2(self, t): return self.para("Heading 2", t)
    def h3(self, t): return self.para("Heading 3", t)
    def body(self, t): return self.para("Body Text", t)
    def bullet(self, t): return self.para("bullet list", t)
    def h5(self, t): return self.para("Heading 5", t)

    # ----------------------------------------------------------- figures
    def figure(self, name, path, caption, width_in=COL_W_IN):
        assert self.figs + 1 == FIGN[name], f"figure {name}: expected number {FIGN[name]}, got {self.figs + 1}"
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.first_line_indent = Pt(0)
        p.add_run().add_picture(str(path), width=Inches(width_in))
        self._place(p._p)
        c = self.para("figure caption", caption)
        c.paragraph_format.keep_together = True
        self.figs += 1
        return c

    # ----------------------------------------------------------- tables
    def table(self, name, caption, header, rows, widths_cm, font=7, align=None, bold_rows=(), footnote=None, shade_rows=()):
        font = font * FONT_SCALE
        assert self.tabs + 1 == TABN[name], f"table {name}: expected number {TABN[name]}, got {self.tabs + 1}"
        cap = self.para("table head", caption)
        cap.paragraph_format.keep_with_next = True
        cap.paragraph_format.space_before = Pt(8)
        cap.paragraph_format.space_after = Pt(4)
        ncol = len(header)
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        t.autofit = False
        tblPr = t._tbl.tblPr
        for el in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(el)
        tblPr.append(parse_xml(f'<w:tblBorders {NS}><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
                               f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'))
        tblPr.append(parse_xml(f'<w:tblCellMar {NS}><w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar>'))
        tblPr.append(parse_xml(f'<w:jc {NS} w:val="center"/>'))
        align = align or (["l"] + ["c"] * (ncol - 1))
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
        allrows = [header] + rows
        for ri, row in enumerate(allrows):
            tr = t.rows[ri]
            trPr = tr._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {NS}/>'))
            if ri == 0:
                trPr.append(parse_xml(f'<w:tblHeader {NS}/>'))
            for ci, val in enumerate(row):
                cell = tr.cells[ci]
                cell.width = Cm(widths_cm[ci])
                p = cell.paragraphs[0]
                p.style = self.doc.styles["table copy"]
                p.alignment = amap[align[ci]]
                p.paragraph_format.space_before = Pt(0.5)
                p.paragraph_format.space_after = Pt(0.5)
                p.paragraph_format.keep_with_next = ri < len(allrows) - 1
                self.add_runs(p, str(val), size=font, bold=True if (ri == 0 or (ri - 1) in bold_rows) else None)
                tcPr = cell._tc.get_or_add_tcPr()
                if ri == 0:
                    tcPr.append(parse_xml(f'<w:tcBorders {NS}><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>'))
                if (ri - 1) in shade_rows:
                    tcPr.append(parse_xml(f'<w:shd {NS} w:val="clear" w:color="auto" w:fill="E8F1FA"/>'))
        # set grid widths
        for gi, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths_cm[gi] * 567)))
        self._place(t._tbl)
        if footnote:
            fp = self.doc.add_paragraph()
            fp.paragraph_format.space_before = Pt(2)
            fp.paragraph_format.space_after = Pt(8)
            fp.paragraph_format.first_line_indent = Pt(0)
            fp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            self.add_runs(fp, footnote, size=7)
            self._place(fp._p)
        else:
            sp = self.doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(4)
            sp.paragraph_format.space_before = Pt(0)
            for r in sp.runs: r.font.size = Pt(2)
            sp.paragraph_format.line_spacing = Pt(4)
            self._place(sp._p)
        self.tabs += 1

    # ----------------------------------------------------------- algorithm box
    def algorithm(self, title, lines, font=6.6):
        font = font * FONT_SCALE
        t = self.doc.add_table(rows=1 + len(lines), cols=1)
        t.autofit = False
        tblPr = t._tbl.tblPr
        for el in tblPr.findall(qn("w:tblBorders")):
            tblPr.remove(el)
        tblPr.append(parse_xml(f'<w:tblBorders {NS}><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
                               f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'))
        tblPr.append(parse_xml(f'<w:tblCellMar {NS}><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar>'))
        for gi, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(8.5 * 567)))
        for ri, txt in enumerate([title] + lines):
            cell = t.rows[ri].cells[0]
            cell.width = Cm(8.5)
            p = cell.paragraphs[0]
            p.style = self.doc.styles["table copy"]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(0.5)
            p.paragraph_format.keep_with_next = ri < len(lines)
            was_raw = isinstance(txt, Raw)
            indent = len(txt) - len(txt.lstrip(" "))
            if indent:
                p.paragraph_format.left_indent = Pt(5 * (indent // 2))
            body_txt = Raw(txt.lstrip(" ")) if was_raw else txt.lstrip(" ")
            self.add_runs(p, body_txt, size=font, bold=True if ri == 0 else None)
            if ri == 0:
                cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:tcBorders {NS}><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>'))
        self._place(t._tbl)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = Pt(4)
        self._place(sp._p)

    # ----------------------------------------------------------- equations (native Word math)
    @staticmethod
    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def equation(self, omml_inner, label=None):
        self.eqs += 1
        n = self.eqs
        p = self.doc.add_paragraph(style="equation")
        pPr = p._p.get_or_add_pPr()
        for el in pPr.findall(qn("w:tabs")):
            pPr.remove(el)
        pPr.append(parse_xml(f'<w:tabs {NS}><w:tab w:val="clear" w:pos="2520"/><w:tab w:val="clear" w:pos="5040"/>'
                             f'<w:tab w:val="center" w:pos="2380"/><w:tab w:val="right" w:pos="4760"/></w:tabs>'))
        pPr.append(parse_xml(f'<w:spacing {NS} w:before="80" w:after="80" w:line="240" w:lineRule="auto"/>'))
        r = p.add_run("\t")
        r.font.name = "Times New Roman"
        p._p.append(parse_xml(f'<m:oMath {NS}>{omml_inner}</m:oMath>'))
        r2 = p.add_run(f"\t({n})")
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        self._place(p._p)
        return n

    # ----------------------------------------------------------- references
    def write_references(self, anchor_style="references"):
        for k in self.order:
            self.para(anchor_style, REFS.fmt(k, self.cache))


# ---------------------------------------------------------------- OMML mini-builder
def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def R(t, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t xml:space="preserve">{esc(t)}</m:t></m:r>'


def Sub(b, s):
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def Sup(b, s):
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def SubSup(b, s, u):
    return f"<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{u}</m:sup></m:sSubSup>"


def Frac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def Paren(e, beg="(", end=")"):
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{e}</m:e></m:d>'


def Abs(e):
    return Paren(e, "|", "|")


def Nary(chr_, lo, hi, body):
    hi_x = f"<m:sup>{hi}</m:sup>" if hi else "<m:sup/>"
    hide = "" if hi else '<m:supHide m:val="1"/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
            f"<m:sub>{lo}</m:sub>{hi_x}<m:e>{body}</m:e></m:nary>")


def Func(name, arg):
    return f"<m:func><m:fName>{R(name, True)}</m:fName><m:e>{arg}</m:e></m:func>"


def Hat(e):
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{e}</m:e></m:acc>'
