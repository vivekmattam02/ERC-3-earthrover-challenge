#!/usr/bin/env python3
"""Convert all project .tex docs into Obsidian-ready Markdown notes."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "obsidian_vault"
MASTER_TEX = REPO_ROOT / "erc3_full_documentation.tex"


def find_tex_files() -> list[Path]:
    out: list[Path] = []
    for p in sorted(REPO_ROOT.rglob("*.tex")):
        if OUT_ROOT in p.parents:
            continue
        out.append(p)
    return out


def build_maps(tex_files: list[Path]) -> tuple[dict[str, str], dict[str, str | None]]:
    rel_to_note: dict[str, str] = {}
    basename_map: dict[str, str | None] = {}
    basename_counts: dict[str, int] = {}

    for tex in tex_files:
        rel = tex.relative_to(REPO_ROOT).as_posix()
        note = tex.relative_to(REPO_ROOT).with_suffix("").as_posix()
        rel_to_note[rel] = note
        base = tex.name
        basename_counts[base] = basename_counts.get(base, 0) + 1

    for tex in tex_files:
        base = tex.name
        if basename_counts[base] == 1:
            basename_map[base] = tex.relative_to(REPO_ROOT).with_suffix("").as_posix()
        else:
            basename_map[base] = None

    return rel_to_note, basename_map


def convert_title(raw: str) -> str:
    t = raw.replace("\\\\", " - ")
    t = t.replace("\\large", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_comments(line: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", line)


def replace_one_arg(text: str, cmd: str, wrap_left: str, wrap_right: str) -> str:
    pattern = re.compile(rf"\\{cmd}\{{([^{{}}]*)\}}")
    while True:
        new_text = pattern.sub(lambda m: f"{wrap_left}{m.group(1)}{wrap_right}", text)
        if new_text == text:
            return text
        text = new_text


def inline_convert(text: str, rel_to_note: dict[str, str], basename_map: dict[str, str | None]) -> str:
    text = text.replace(r"\_", "_")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\$", "$")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\textdegree", "°")
    text = text.replace("``", "\"").replace("''", "\"")

    text = re.sub(r"\\url\{([^{}]+)\}", lambda m: f"[{m.group(1)}]({m.group(1)})", text)
    text = re.sub(r"\\href\{([^{}]+)\}\{([^{}]+)\}", lambda m: f"[{m.group(2)}]({m.group(1)})", text)

    text = replace_one_arg(text, "textbf", "**", "**")
    text = replace_one_arg(text, "emph", "*", "*")
    text = replace_one_arg(text, "textit", "*", "*")
    text = replace_one_arg(text, "texttt", "`", "`")

    # Convert .tex refs to Obsidian wiki links.
    def link_tex_ref(match: re.Match[str]) -> str:
        raw = match.group(1)
        ref = raw.replace(r"\_", "_").strip().lstrip("./")
        target = rel_to_note.get(ref)
        if target is None:
            target = basename_map.get(Path(ref).name)
        if target is None:
            return raw
        return f"[[{target}]]"

    text = re.sub(r"([A-Za-z0-9_./\\-]+\.tex)", link_tex_ref, text)
    # If a tex reference came from \texttt{...}, avoid wrapping wiki links in code ticks.
    text = re.sub(r"`\[\[([^\]]+)\]\]`", r"[[\1]]", text)

    # Remove a few low-signal latex commands if still present on regular lines.
    text = re.sub(r"\\(toprule|midrule|bottomrule)\b", "", text)
    text = re.sub(r"\\(smallskip|medskip|bigskip)\b", "", text)
    text = re.sub(r"\\vspace\*?\{[^{}]*\}", "", text)
    text = text.replace(r"\newline", "  ")
    text = text.replace(r"\%", "%")
    text = text.strip()
    return text


def convert_tex_to_md(
    tex_path: Path,
    rel_to_note: dict[str, str],
    basename_map: dict[str, str | None],
) -> str:
    source = tex_path.read_text(encoding="utf-8", errors="replace")

    title_match = re.search(r"\\title\{(.*?)\}", source, flags=re.DOTALL)
    title = convert_title(title_match.group(1)) if title_match else tex_path.stem

    body = source
    if r"\begin{document}" in body:
        body = body.split(r"\begin{document}", 1)[1]
    if r"\end{document}" in body:
        body = body.split(r"\end{document}", 1)[0]

    lines = body.splitlines()
    out: list[str] = []
    out.append(f"# {title}")
    rel = tex_path.relative_to(REPO_ROOT).as_posix()
    out.append("")
    out.append(f"> Source: `{rel}`")
    if tex_path != MASTER_TEX:
        out.append("> Master Note: [[erc3_full_documentation]]")
    out.append("")

    list_stack: list[str] = []
    in_code = False
    in_table = False
    in_quote = False
    in_box = False
    box_callout = "note"

    section_re = re.compile(r"\\section\*?\{(.+)\}")
    subsection_re = re.compile(r"\\subsection\*?\{(.+)\}")
    subsubsection_re = re.compile(r"\\subsubsection\*?\{(.+)\}")
    box_start_re = re.compile(r"\\(docbox|warnbox|lessonbox)\{(.+?)\}\{(.*)")
    frame_begin_re = re.compile(r"\\begin\{frame\}(?:\{(.+?)\})?")
    frame_title_re = re.compile(r"\\frametitle\{(.+?)\}")

    for raw_line in lines:
        line = strip_comments(raw_line).rstrip()
        if not line and not in_code and not in_table:
            out.append("")
            continue

        if in_box:
            box_done = line.strip().endswith("}")
            body = line
            if box_done:
                body = re.sub(r"}\s*$", "", body)
                in_box = False
            body = inline_convert(body, rel_to_note, basename_map)
            if body:
                out.append(f"> {body}")
            if not in_box:
                out.append("")
            continue

        # Multi-line raw/code blocks
        if re.search(r"\\begin\{(verbatim|lstlisting)\}", line):
            in_code = True
            out.append("```")
            continue
        if re.search(r"\\end\{(verbatim|lstlisting)\}", line):
            in_code = False
            out.append("```")
            continue
        if re.search(r"\\begin\{(tabular|longtable)\}", line):
            in_table = True
            out.append("```text")
            continue
        if re.search(r"\\end\{(tabular|longtable)\}", line):
            in_table = False
            out.append("```")
            continue

        if in_code or in_table:
            out.append(raw_line.rstrip())
            continue

        if r"\begin{quote}" in line:
            in_quote = True
            continue
        if r"\end{quote}" in line:
            in_quote = False
            out.append("")
            continue

        if re.search(r"\\begin\{itemize\}(\[[^\]]*\])?", line):
            list_stack.append("ul")
            continue
        if re.search(r"\\begin\{enumerate\}(\[[^\]]*\])?", line):
            list_stack.append("ol")
            continue
        if re.search(r"\\begin\{description\}(\[[^\]]*\])?", line):
            list_stack.append("ul")
            continue
        if (
            re.search(r"\\end\{itemize\}", line)
            or re.search(r"\\end\{enumerate\}", line)
            or re.search(r"\\end\{description\}", line)
        ):
            if list_stack:
                list_stack.pop()
            out.append("")
            continue

        if r"\maketitle" in line or r"\tableofcontents" in line:
            continue
        if line.strip() in {r"\titlepage", r"\Large", r"\large", r"\normalsize", r"\small", r"\centering"}:
            continue
        frame_begin = frame_begin_re.search(line)
        if frame_begin:
            frame_title = frame_begin.group(1)
            if frame_title:
                out.append(f"## {inline_convert(frame_title, rel_to_note, basename_map)}")
                out.append("")
            continue
        if r"\end{frame}" in line:
            out.append("")
            continue
        frame_title = frame_title_re.search(line)
        if frame_title:
            out.append(f"## {inline_convert(frame_title.group(1), rel_to_note, basename_map)}")
            out.append("")
            continue
        if line.strip() in {r"\newpage", r"\clearpage"}:
            continue
        if re.match(r"\\label\{[^{}]+\}", line.strip()):
            continue
        if r"\begin{center}" in line or r"\end{center}" in line:
            continue

        box_match = box_start_re.search(line)
        if box_match:
            box_kind, heading, content = box_match.groups()
            box_callout = "warning" if box_kind == "warnbox" else "note"
            heading = inline_convert(heading, rel_to_note, basename_map)
            out.append(f"> [!{box_callout}] {heading}")
            content_done = content.strip().endswith("}")
            content_text = content
            if content_done:
                content_text = re.sub(r"}\s*$", "", content_text)
            content_text = inline_convert(content_text, rel_to_note, basename_map)
            if content_text:
                out.append(f"> {content_text}")
            if content_done:
                out.append("")
            else:
                in_box = True
            continue

        m = section_re.search(line)
        if m:
            out.append(f"# {inline_convert(m.group(1), rel_to_note, basename_map)}")
            out.append("")
            continue
        m = subsection_re.search(line)
        if m:
            out.append(f"## {inline_convert(m.group(1), rel_to_note, basename_map)}")
            out.append("")
            continue
        m = subsubsection_re.search(line)
        if m:
            out.append(f"### {inline_convert(m.group(1), rel_to_note, basename_map)}")
            out.append("")
            continue

        item_match = re.match(r"\s*\\item(?:\[(.*?)\])?\s*(.*)", line)
        if item_match:
            label, rest = item_match.groups()
            item_text = inline_convert(rest, rel_to_note, basename_map)
            if label:
                lbl = inline_convert(label, rel_to_note, basename_map)
                item_text = f"**{lbl}**: {item_text}" if item_text else f"**{lbl}**"
            if list_stack and list_stack[-1] == "ol":
                out.append(f"1. {item_text}")
            else:
                out.append(f"- {item_text}")
            continue

        line = inline_convert(line, rel_to_note, basename_map)
        if not line:
            out.append("")
            continue
        if in_quote:
            out.append(f"> {line}")
        else:
            out.append(line)

    # Normalize whitespace.
    normalized: list[str] = []
    blank_count = 0
    for ln in out:
        if ln.strip() == "":
            blank_count += 1
            if blank_count <= 1:
                normalized.append("")
        else:
            blank_count = 0
            normalized.append(ln.rstrip())

    return "\n".join(normalized).rstrip() + "\n"


def write_notes(tex_files: list[Path], rel_to_note: dict[str, str], basename_map: dict[str, str | None]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for tex in tex_files:
        rel = tex.relative_to(REPO_ROOT)
        out_path = OUT_ROOT / rel.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md = convert_tex_to_md(tex, rel_to_note, basename_map)
        out_path.write_text(md, encoding="utf-8")

    # Vault landing page.
    idx = OUT_ROOT / "INDEX.md"
    lines = [
        "# ERC-3 Obsidian Vault",
        "",
        "Master note:",
        "- [[erc3_full_documentation]]",
        "",
        "Converted notes:",
    ]
    for tex in tex_files:
        note = tex.relative_to(REPO_ROOT).with_suffix("").as_posix()
        lines.append(f"- [[{note}]]")
    lines.append("")
    idx.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    tex_files = find_tex_files()
    rel_to_note, basename_map = build_maps(tex_files)
    write_notes(tex_files, rel_to_note, basename_map)
    print(f"Converted {len(tex_files)} .tex files into {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
