#!/usr/bin/env python3
"""Check local Markdown links in canonical docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_MD_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "guide.md",
    REPO_ROOT / "docs" / "index.md",
    REPO_ROOT / "docs" / "INDEX.md",
    REPO_ROOT / "docs" / "DOCUMENTATION_ARCHITECTURE.md",
]

MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def normalize_link(link: str) -> str:
    link = link.strip()
    if link.startswith("<") and link.endswith(">"):
        link = link[1:-1].strip()
    return link


def is_external(link: str) -> bool:
    return (
        link.startswith("http://")
        or link.startswith("https://")
        or link.startswith("mailto:")
    )


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: file missing"]

    text = path.read_text(encoding="utf-8", errors="replace")
    for _, raw_link in MD_LINK_RE.findall(text):
        link = normalize_link(raw_link)
        if not link or is_external(link) or link.startswith("#"):
            continue

        # Markdown URLs encode spaces and other path characters. Decode only
        # the local filesystem component after removing the fragment.
        target = unquote(link.split("#", 1)[0])
        if not target:
            continue

        target_path = (path.parent / target).resolve()
        if not target_path.exists():
            errors.append(f"{path}: broken link target '{link}'")

    return errors


def main() -> int:
    errors: list[str] = []
    for path in TARGET_MD_FILES:
        errors.extend(check_file(path))

    if errors:
        print("Link check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Link check passed for {len(TARGET_MD_FILES)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
