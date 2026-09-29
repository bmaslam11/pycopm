#!/usr/bin/env python3
"""Prepare the Norne ATW deck for pycopm's generic deck workflow.

This script DOES NOT modify the original deck.

It creates a derived deck that:
  1. Inlines IJK-dependent GRID definitions that pycopm needs in the main
     .DATA file (FAULTS and MULTREGT for this Norne deck).
  2. Removes the original SUMMARY/SCHEDULE content, including the historical
     well schedule, because those IJK well definitions are in include files
     and would otherwise remain on the original 46x112x22 indexing.
  3. Writes an empty SUMMARY and SCHEDULE followed by END.

Run this script from the Norne directory, or pass --root explicitly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENCODING = "iso-8859-1"
DEFAULT_INPUT = "NORNE_ATW2013.DATA"
DEFAULT_OUTPUT = "NORNE_ATW2013_PYCOPM_STATIC.DATA"

# Current pycopm documentation says these GRID-section definitions need to be
# in the main DATA deck rather than hidden behind INCLUDE files when they are
# to be transformed/remapped.
INLINE_INCLUDES = (
    "INCLUDE/FAULT/FAULT_JUN_05.INC",
    "INCLUDE/PETRO/MULTREGT_D_27.prop",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding=ENCODING)


def normalize_include_path(path_text: str) -> str:
    """Normalize Eclipse/OPM include paths for comparison."""
    value = path_text.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def inline_selected_includes(text: str, root: Path) -> tuple[str, list[str]]:
    """Replace selected INCLUDE records with the referenced file content."""
    targets = {p.lower(): p for p in INLINE_INCLUDES}
    found: dict[str, int] = {p.lower(): 0 for p in INLINE_INCLUDES}

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not re.match(r"^\s*INCLUDE\b", line, flags=re.IGNORECASE):
            out.append(line)
            i += 1
            continue

        # Capture the complete INCLUDE record. In this Norne deck the keyword
        # and path are on separate lines, but this also tolerates one-line form.
        record = [line]
        i += 1
        if "/" not in line:
            while i < len(lines):
                record.append(lines[i])
                current = lines[i]
                i += 1
                if "/" in current:
                    break

        record_text = "".join(record)
        match = re.search(r"['\"]([^'\"]+)['\"]", record_text)
        if not match:
            out.extend(record)
            continue

        include_rel = normalize_include_path(match.group(1))
        key = include_rel.lower()
        if key not in targets:
            out.extend(record)
            continue

        include_path = root / include_rel
        if not include_path.is_file():
            raise FileNotFoundError(
                f"Required include file not found: {include_path}\n"
                "Check that the complete Norne INCLUDE directory is present."
            )

        payload = read_text(include_path)
        if payload and not payload.endswith(("\n", "\r")):
            payload += "\n"

        out.append(
            f"\n-- ================================================================\n"
            f"-- BEGIN INLINED BY prepare_norne_for_pycopm.py\n"
            f"-- Original include: ./{include_rel}\n"
            f"-- ================================================================\n"
        )
        out.append(payload)
        out.append(
            f"-- ================================================================\n"
            f"-- END INLINED: ./{include_rel}\n"
            f"-- ================================================================\n\n"
        )
        found[key] += 1

    missing = [targets[k] for k, count in found.items() if count == 0]
    duplicates = [targets[k] for k, count in found.items() if count > 1]

    if missing:
        raise RuntimeError(
            "Could not find these expected INCLUDE records in the input deck: "
            + ", ".join(missing)
        )
    if duplicates:
        raise RuntimeError(
            "Found an expected INCLUDE more than once; refusing to guess: "
            + ", ".join(duplicates)
        )

    return "".join(out), [targets[k] for k in found]


def replace_summary_and_schedule(text: str) -> str:
    """Keep the model through SOLUTION and replace SUMMARY/SCHEDULE with empty sections."""
    match = re.search(r"(?im)^\s*SUMMARY\s*(?:--[^\r\n]*)?(?:\r?\n|$)", text)
    if not match:
        raise RuntimeError("Could not locate the SUMMARY section in the input deck.")

    prefix = text[: match.start()].rstrip() + "\n\n"
    suffix = """-- ============================================================================
-- Static pycopm preparation deck
--
-- Original SUMMARY and SCHEDULE sections intentionally removed.
-- The original Norne history schedule contains WELSPECS/COMPDAT in INCLUDE
-- files using the fine-grid 46 x 112 x 22 IJK indexing. We do not need that
-- history schedule for coarse-grid geometry/property generation; wells and
-- controls will be defined later in the CMG workflow.
-- ============================================================================
SUMMARY

SCHEDULE

END
"""
    return prefix + suffix


def validate_output(text: str) -> None:
    checks = {
        "FAULTS keyword": r"(?im)^\s*FAULTS\b",
        "MULTREGT keyword": r"(?im)^\s*MULTREGT\b",
        "SUMMARY section": r"(?im)^\s*SUMMARY\s*$",
        "SCHEDULE section": r"(?im)^\s*SCHEDULE\s*$",
        "END keyword": r"(?im)^\s*END\s*$",
    }
    missing = [name for name, pattern in checks.items() if not re.search(pattern, text)]
    if missing:
        raise RuntimeError("Output validation failed; missing: " + ", ".join(missing))

    forbidden = (
        "BC0407_HIST01122006.SCH",
        "INCLUDE/FAULT/FAULT_JUN_05.INC",
        "INCLUDE/PETRO/MULTREGT_D_27.prop",
    )
    leftovers = [token for token in forbidden if token.lower() in text.lower()]
    # The original include paths are intentionally present once in comments
    # marking the inlined blocks, so only the history include is truly forbidden.
    if "BC0407_HIST01122006.SCH" in leftovers:
        raise RuntimeError("History schedule include is still present in the output.")

    # The new schedule must not contain legacy completion definitions.
    schedule_match = re.search(r"(?ims)^\s*SCHEDULE\s*$([\s\S]*?)^\s*END\s*$", text)
    if schedule_match and re.search(r"(?im)^\s*(WELSPECS|COMPDAT|COMPDATL)\b", schedule_match.group(1)):
        raise RuntimeError("Legacy well/completion definitions remain in the new SCHEDULE section.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Norne directory containing the DATA file and INCLUDE folder (default: script directory)",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input deck (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output deck (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    root = args.root.resolve()
    input_path = root / args.input
    output_path = root / args.output

    if not input_path.is_file():
        print(f"ERROR: input deck not found: {input_path}", file=sys.stderr)
        return 2
    if output_path.resolve() == input_path.resolve():
        print("ERROR: output must be different from input; original deck will not be overwritten.", file=sys.stderr)
        return 2

    try:
        original = read_text(input_path)
        prepared, inlined = inline_selected_includes(original, root)
        prepared = replace_summary_and_schedule(prepared)
        validate_output(prepared)
        output_path.write_text(prepared, encoding=ENCODING, newline="\n")
    except Exception as exc:  # concise CLI error, original deck remains untouched
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Prepared pycopm static Norne deck successfully.")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print("  Inlined:")
    for item in inlined:
        print(f"    - {item}")
    print("  Removed: original SUMMARY/SCHEDULE content (including Norne history wells)")
    print("  Kept   : GRID/EDIT/PROPS/REGIONS/SOLUTION and static geology/initialization")
    print("\nOriginal deck was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
