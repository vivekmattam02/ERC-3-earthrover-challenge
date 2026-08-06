#!/usr/bin/env python3
"""Basic structural/style lint for project TeX docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_TEX_FILES = [
    REPO_ROOT / "erc3_full_documentation.tex",
    REPO_ROOT / "competition_results.tex",
    REPO_ROOT / "final_system_overview.tex",
]

ENVIRONMENTS = [
    "itemize",
    "enumerate",
    "description",
    "longtable",
    "tabular",
    "lstlisting",
    "verbatim",
    "quote",
]


def iter_tex_files() -> list[Path]:
    return [p for p in TARGET_TEX_FILES if p.exists()]


def lint_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    has_docclass = "\\documentclass" in text
    begin_doc = text.count("\\begin{document}")
    end_doc = text.count("\\end{document}")
    if has_docclass:
        if begin_doc != 1 or end_doc != 1:
            errors.append(
                f"{path}: expected exactly one \\begin{{document}} and one \\end{{document}}, "
                f"got {begin_doc}/{end_doc}"
            )

    # Balanced core environments.
    for env in ENVIRONMENTS:
        b = text.count(f"\\begin{{{env}}}")
        e = text.count(f"\\end{{{env}}}")
        if b != e:
            errors.append(f"{path}: unbalanced {env}: begin={b}, end={e}")

    # Line length check outside verbatim-like environments.
    in_literal = False
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\\begin\{(lstlisting|verbatim)\}", line):
            in_literal = True
        if not in_literal and len(line) > 320:
            errors.append(f"{path}:{idx}: line longer than 320 chars")
        if re.search(r"\\end\{(lstlisting|verbatim)\}", line):
            in_literal = False

    return errors


def main() -> int:
    errors: list[str] = []
    files = iter_tex_files()
    if not files:
        print("No TeX files found.")
        return 0

    for path in files:
        errors.extend(lint_file(path))

    if errors:
        print("TeX lint failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"TeX lint passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
