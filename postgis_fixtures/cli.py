"""Command-line entry point for inspecting generated fixtures offline.

The CLI exists so you can see exactly what the plugin will create before wiring
it into a test suite — and so DDL can be diffed in review or piped into ``psql``
to seed a scratch database by hand::

    python -m postgis_fixtures ddl --dataset cities | psql -d gis

Nothing here touches a database.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence, TextIO

import shapely

from .crs import SUPPORTED_SRIDS, WGS84
from .datasets import (
    DEFAULT_ROW_COUNTS,
    build_dataset,
    dataset_names,
    edge_case_specs,
)
from .errors import PostgisFixturesError
from .geometry import GeneratorConfig, to_wkt


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``python -m postgis_fixtures``."""
    parser = argparse.ArgumentParser(
        prog="python -m postgis_fixtures",
        description="Inspect the spatial fixture data and DDL this package generates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GeneratorConfig().seed,
        help="generation seed (default: %(default)s)",
    )
    parser.add_argument(
        "--srid",
        type=int,
        default=WGS84,
        choices=SUPPORTED_SRIDS,
        help="SRID to emit geometry in (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list the available datasets and their row counts")

    ddl = subparsers.add_parser("ddl", help="print CREATE TABLE / CREATE INDEX statements")
    ddl.add_argument("--dataset", action="append", choices=dataset_names(), help="restrict to one dataset (repeatable)")
    ddl.add_argument("--no-indexes", action="store_true", help="omit index statements")

    sample = subparsers.add_parser("sample", help="print sample rows as WKT")
    sample.add_argument("dataset", choices=dataset_names())
    sample.add_argument("--rows", type=int, default=5, help="rows to print (default: %(default)s)")
    sample.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    subparsers.add_parser("edge-cases", help="describe the edge-case catalogue")
    return parser


def _config(args: argparse.Namespace) -> GeneratorConfig:
    """Build a generator config from parsed arguments."""
    return GeneratorConfig(seed=args.seed, srid=args.srid)


def _cmd_list(args: argparse.Namespace, out: TextIO) -> int:
    """Print the dataset catalogue."""
    config = _config(args)
    for name in dataset_names():
        dataset = build_dataset(name, config)
        default = DEFAULT_ROW_COUNTS.get(name, len(dataset))
        out.write(f"{name:<18} {default:>6} rows  {dataset.description}\n")
    return 0


def _cmd_ddl(args: argparse.Namespace, out: TextIO) -> int:
    """Print DDL for the selected datasets."""
    config = _config(args)
    names = args.dataset or list(dataset_names())
    blocks = [
        build_dataset(name, config).ddl(include_indexes=not args.no_indexes)
        for name in names
    ]
    out.write("\n\n".join(blocks) + "\n")
    return 0


def _cmd_sample(args: argparse.Namespace, out: TextIO) -> int:
    """Print sample rows, with geometry rendered as WKT."""
    if args.rows < 1:
        raise PostgisFixturesError(f"--rows must be at least 1, got {args.rows}")
    config = _config(args)
    dataset = build_dataset(args.dataset, config)
    geometry_column = dataset.geometry_column
    records = []
    for row in list(dataset.rows)[: args.rows]:
        record = {key: value for key, value in row.items() if key != geometry_column}
        raw = row.get(geometry_column)
        if raw is None:
            record["wkt"] = None
        else:
            geometry = shapely.from_wkb(bytes.fromhex(str(raw)))
            srid = int(row.get("srid") or dataset.srid or WGS84)
            record["wkt"] = to_wkt(geometry, srid)
            record["srid"] = srid
        records.append(record)
    if args.json:
        out.write(json.dumps(records, indent=2, default=str) + "\n")
        return 0
    for record in records:
        parts = [f"{key}={value}" for key, value in record.items() if key != "wkt"]
        out.write("  ".join(parts) + "\n")
        out.write(f"    {record['wkt']}\n")
    return 0


def _cmd_edge_cases(args: argparse.Namespace, out: TextIO) -> int:
    """Describe the edge-case catalogue."""
    del args
    for spec in edge_case_specs():
        srid = spec.srid if spec.srid is not None else "-"
        out.write(f"{spec.label} (SRID {srid})\n")
        out.write(f"    {spec.note}\n")
        out.write(f"    {spec.wkt if spec.wkt is not None else 'NULL'}\n")
    return 0


_COMMANDS = {
    "list": _cmd_list,
    "ddl": _cmd_ddl,
    "sample": _cmd_sample,
    "edge-cases": _cmd_edge_cases,
}


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout
    try:
        return _COMMANDS[args.command](args, stream)
    except PostgisFixturesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

