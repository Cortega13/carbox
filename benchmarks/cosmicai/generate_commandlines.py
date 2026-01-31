"""Generate MPI command lines for tracer CSV runs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate MPI command lines")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("benchmarks/cosmicai/data/turbulence_tracers_csv"),
        help="Tracer CSV dir (default: benchmarks/cosmicai/data/turbulence_tracers_csv)",
    )
    parser.add_argument(
        "--command-file",
        type=Path,
        default=Path("benchmarks/cosmicai/commandlines.txt"),
        help="Output commandlines file (default: benchmarks/cosmicai/commandlines.txt)",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path("benchmarks/cosmicai/carbox_cosmicai_benchmark.py"),
        help="Benchmark runner path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Benchmark output directory (default: outputs)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tracers with existing output .npy files",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against CARBOX_ROOT when set."""
    if path.is_absolute():
        return path
    root = os.environ.get("CARBOX_ROOT")
    if root:
        return Path(root) / path
    return path


def tracer_id_from_csv(path: Path) -> int:
    """Resolve tracer id from a CSV filename or column."""
    stem = path.stem
    if stem.startswith("tracer_"):
        suffix = stem.split("_", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        if row is None or "tracer" not in row:
            raise ValueError(f"Unable to resolve tracer id from {path}")
        return int(float(row["tracer"]))


def build_command(
    benchmark_script: Path,
    tracer_csv: Path,
    output_dir: Path,
) -> str:
    """Build a single command line for a tracer.

    Note: We intentionally do NOT include any `srun`/MPI prefix here.
    The job-level launcher (e.g., Slurm + pylauncher) is responsible for
    placement; each tracer run is always single-rank.
    """
    return (
        f"python3 {benchmark_script} "
        f"--tracer-csv {tracer_csv} --output-dir {output_dir}"
    )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    args.csv_dir = resolve_path(args.csv_dir)
    args.command_file = resolve_path(args.command_file)
    args.output_dir = resolve_path(args.output_dir)
    args.benchmark_script = resolve_path(args.benchmark_script)

    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    lines = []
    for csv_path in csv_paths:
        tracer_id = tracer_id_from_csv(csv_path)
        if args.skip_existing:
            output_path1 = args.output_dir / f"tracer_{tracer_id}_large.npy"
            output_path2 = args.output_dir / f"tracer_{tracer_id}_small.npy"
            if output_path1.exists() and output_path2.exists():
                continue
        lines.append(
            build_command(
                args.benchmark_script,
                csv_path,
                args.output_dir,
            )
        )

    args.command_file.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"Command lines written to {args.command_file}")
    print(f"Total commands: {len(lines)}")


if __name__ == "__main__":
    main()
