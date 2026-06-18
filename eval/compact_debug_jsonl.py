from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

try:
    from codegen_agent.record_compaction import compact_jsonl_file
except ModuleNotFoundError:
    spec = importlib.util.spec_from_file_location(
        "record_compaction",
        Path(__file__).resolve().parents[1] / "record_compaction.py",
    )
    if spec is None or spec.loader is None:
        raise
    record_compaction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(record_compaction)
    compact_jsonl_file = record_compaction.compact_jsonl_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move bulky result JSONL fields into debug artifact JSONL files.")
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path(__file__).resolve().parents[1] / "results" / "main",
        ],
        help="Files or directories to compact. Directories are scanned recursively for *.jsonl files.",
    )
    return parser


def iter_jsonl_files(paths: list[Path]):
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.jsonl"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    for path in iter_jsonl_files(args.roots):
        original_size, compact_size, debug_path = compact_jsonl_file(path)
        if compact_size != original_size:
            logging.info(
                "compacted %s %.2fMB -> %.2fMB debug=%s",
                path,
                original_size / 1024 / 1024,
                compact_size / 1024 / 1024,
                debug_path,
            )
        else:
            logging.info("unchanged %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
