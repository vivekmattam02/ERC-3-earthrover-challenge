#!/usr/bin/env python3
"""Validate run commands embedded in master documentation."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_TEX = REPO_ROOT / "erc3_full_documentation.tex"
INDOOR_RUNTIME = REPO_ROOT / "live_indoor_runtime.py"
OUTDOOR_RUNTIME = REPO_ROOT / "live_outdoor_runtime.py"


def parse_runtime_flags(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Accept both one-line and multi-line argparse calls:
    # parser.add_argument("--flag", ...)
    # parser.add_argument(
    #     "--flag",
    #     ...
    # )
    pattern = re.compile(
        r'parser\.add_argument\(\s*["\'](--[a-zA-Z0-9\-]+)["\']',
        re.MULTILINE,
    )
    return set(pattern.findall(text))


def extract_verbatim_blocks(text: str) -> list[str]:
    pattern = re.compile(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", re.DOTALL)
    return [m.group(1) for m in pattern.finditer(text)]


def normalize_command_block(block: str) -> str:
    # Join line-continuations and normalize whitespace.
    joined = block.replace("\\\n", " ").replace("\n", " ")
    return " ".join(joined.split())


def extract_python_script(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except Exception:
        return None
    if not parts:
        return None
    if parts[0] not in {"python", "python3"}:
        return None
    for part in parts[1:]:
        if part.endswith(".py"):
            return part
    return None


def extract_flags(cmd: str) -> set[str]:
    try:
        parts = shlex.split(cmd)
    except Exception:
        return set()
    return {p for p in parts if p.startswith("--")}


def main() -> int:
    errors: list[str] = []
    text = MASTER_TEX.read_text(encoding="utf-8", errors="replace")
    blocks = extract_verbatim_blocks(text)

    indoor_flags = parse_runtime_flags(INDOOR_RUNTIME)
    outdoor_flags = parse_runtime_flags(OUTDOOR_RUNTIME)

    for idx, raw in enumerate(blocks, start=1):
        cmd = normalize_command_block(raw)
        script = extract_python_script(cmd)
        if script is None:
            continue

        script_path = (REPO_ROOT / script).resolve()
        if not script_path.exists():
            errors.append(f"block {idx}: script '{script}' does not exist")
            continue

        flags = extract_flags(cmd)
        if script == "live_indoor_runtime.py":
            unknown = sorted(flags - indoor_flags)
            if unknown:
                errors.append(f"block {idx}: unknown indoor flags: {', '.join(unknown)}")
        elif script == "live_outdoor_runtime.py":
            unknown = sorted(flags - outdoor_flags)
            if unknown:
                errors.append(f"block {idx}: unknown outdoor flags: {', '.join(unknown)}")

    if errors:
        print("Command reference check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Command reference check passed for {len(blocks)} verbatim blocks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
