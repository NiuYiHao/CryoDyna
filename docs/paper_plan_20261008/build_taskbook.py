"""Render the adjacent Markdown taskbook as a standalone Typst document."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PAPER_EXECPLAN_20261008.md"
DEST = ROOT / "PAPER_EXECPLAN_20261008.typ"


def literal(value):
    return json.dumps(value, ensure_ascii=False)


def prose(value):
    pieces = re.split(r"(https?://[^\s]+)", value)
    result = []
    for piece in pieces:
        if piece.startswith("http"):
            url = piece.rstrip("。，；")
            tail = piece[len(url):]
            result.append(f"#link({literal(url)})[#text({literal(url)})]")
            if tail:
                result.append(f"#text({literal(tail)})")
        elif piece:
            result.append(f"#text({literal(piece)})")
    return "".join(result) + "\n\n"


lines = SOURCE.read_text().splitlines()
out = ['''// Generated from PAPER_EXECPLAN_20261008.md. Edit the Markdown source.
#set document(title: "CryoDyna-optpose 30天论文闭环任务书", author: "CryoDyna project")
#set page(paper: "a4", margin: (x: 18mm, y: 17mm),
  header: align(right, text(size: 8pt, fill: rgb("62758a"))[CryoDyna-optpose / v1.1 / 2026-10-09]),
  footer: context align(right, text(size: 8pt)[#counter(page).display("1")]))
#set text(font: ("Microsoft YaHei", "DejaVu Sans"), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.62em, spacing: 0.85em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 22pt, fill: rgb("14395a"))
#show heading.where(level: 2): set text(size: 16pt, fill: rgb("14395a"))
#show raw: set text(font: ("DejaVu Sans Mono", "Microsoft YaHei"), size: 7.3pt)
#show raw: set block(fill: rgb("f3f6f9"), inset: 7pt, radius: 3pt, breakable: true)
''']
i = 0
section = 0
while i < len(lines):
    line = lines[i]
    if line.startswith("# "):
        title = line[2:].split("：", 1)
        out.append("#heading(level: 1)[" + f"#text({literal(title[0])})" + "]\n")
        if len(title) == 2:
            out.append(f"#text(size: 17pt, fill: rgb(\"14395a\"))[#text({literal(title[1])})]\n\n")
    elif line.startswith("## "):
        section += 1
        if section > 1:
            out.append("#pagebreak()\n")
        out.append("#heading(level: 2)[" + f"#text({literal(line[3:])})" + "]\n")
    elif line.startswith("```"):
        buf = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i])
            i += 1
        out.append(f"#raw({literal(chr(10).join(buf))}, block: true)\n\n")
    elif line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            row = [v.strip() for v in lines[i].strip("|").split("|")]
            if not all(re.fullmatch(r"[-: ]+", v) for v in row):
                rows.append(row)
            i += 1
        i -= 1
        n = len(rows[0])
        widths = "(1.2fr, 2fr, 2.8fr)" if n == 3 else "(" + ",".join(["1fr"] * n) + ")"
        out.append("#text(size: 9pt)[#table(\n")
        out.append(f"columns: {widths}, inset: 5pt, stroke: 0.35pt + rgb(\"ccd5df\"),\n")
        out.append("fill: (x, y) => if y == 0 { rgb(\"e8eff6\") } else { none },\n")
        for row in rows:
            for cell in row:
                out.append(f"[#text({literal(cell)})],\n")
        out.append(")]\n\n")
    elif line.strip():
        out.append(prose(line))
    i += 1
DEST.write_text("".join(out))
print(DEST)
