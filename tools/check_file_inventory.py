#!/usr/bin/env python3
"""Validate the dated repository inventory without rewriting historical files.

The inventory is a snapshot of main, not a promise to list every new PR file.
Checks: declared folder counts, declared total, duplicate entries, and missing
paths. New files after the snapshot are deliberately not an error.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

HEADER = re.compile(r"^### `(.+?)` — (\d+) files$")
ENTRY = re.compile(r"^- `([^\x60]+)`$")
TOTAL = re.compile(r"^\*\*(\d+) tracked files\*\* across (\d+) top-level groups\.")


def inspect_inventory(text: str, root: Path) -> list[str]:
    errors = []
    declared_total = declared_groups = None
    current = None
    expected = {}
    paths = {}
    in_section = False
    for line in text.splitlines():
        total = TOTAL.fullmatch(line)
        if total:
            declared_total, declared_groups = map(int, total.groups())
        if line == "## Tracked files by folder":
            in_section = True
            continue
        if line.startswith("## ") and in_section:
            in_section = False
            current = None
        if not in_section:
            continue
        header = HEADER.fullmatch(line)
        if header:
            current = header.group(1)
            if current in expected:
                errors.append(f"duplicate folder section: {current}")
            expected[current] = int(header.group(2))
            paths.setdefault(current, [])
            continue
        entry = ENTRY.fullmatch(line)
        if entry:
            if current is None:
                errors.append(f"file entry outside folder: {entry.group(1)}")
                continue
            path = entry.group(1)
            paths[current].append(path)
            if Path(path).is_absolute() or ".." in Path(path).parts:
                errors.append(f"unsafe inventory path: {path}")
                continue
            actual_group = path.split("/", 1)[0] if "/" in path else "(root)"
            if actual_group != current:
                errors.append(f"wrong folder for {path}: {current}")
            if not (root / path).is_file():
                errors.append(f"missing tracked file: {path}")
    flattened = [p for group in paths.values() for p in group]
    for path, count in Counter(flattened).items():
        if count > 1:
            errors.append(f"duplicate file entry: {path}")
    for group, count in expected.items():
        if len(paths[group]) != count:
            errors.append(f"folder {group}: declared {count}, listed {len(paths[group])}")
    if declared_total is None or declared_groups is None:
        errors.append("missing summary count")
    else:
        if len(flattened) != declared_total:
            errors.append(f"total: declared {declared_total}, listed {len(flattened)}")
        if len(expected) != declared_groups:
            errors.append(f"groups: declared {declared_groups}, listed {len(expected)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    path = args.inventory or args.root / "docs/FILE-INVENTORY.md"
    errors = inspect_inventory(path.read_text(encoding="utf-8"), args.root)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: inventory counts and all listed files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
