"""Extract tracer CSVs from CosmicAI .npy data.

python benchmarks/cosmicai/npy_to_csv.py \
  --skip-existing \
  --random-count 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
#
# These were previously passed as CLI args. They are now intentionally kept as
# module-level constants so the script can be run without specifying them.
# ---------------------------------------------------------------------------

NPY_PATH = Path("benchmarks/cosmicai/data/M600_seed1_trace_cells.npy")
OUTPUT_DIR = Path("benchmarks/cosmicai/data/turbulence_tracers_csv")
TIMESTEP_KYR = 8.299
CLIP = 400
DISCRETIZATION = 1
SEED = 13


def density_to_number_density(density: np.ndarray) -> np.ndarray:
    """Convert mass density to number density."""
    hydrogen_mass = 1.66053906660e-24
    mean_molecular_mass = 1.4168138025
    return density / (mean_molecular_mass * hydrogen_mass)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert CosmicAI NPY to tracer CSVs")
    parser.add_argument("--benchmark", type=str, default="M600_1", help="Benchmark ID")
    parser.add_argument(
        "--tracers",
        type=int,
        nargs="+",
        default=None,
        help="Tracer indices to export",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=None,
        help="Random tracer count (overrides --tracers)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip CSVs that already exist",
    )
    return parser.parse_args()


def select_tracers(data: np.ndarray, args: argparse.Namespace) -> list[int]:
    """Resolve tracer indices based on selection flags."""
    if args.random_count:
        rng = np.random.default_rng(SEED)
        count = min(args.random_count, data.shape[1])
        return rng.choice(data.shape[1], size=count, replace=False).tolist()
    if args.tracers:
        return args.tracers
    return list(range(data.shape[1]))


def build_frame(
    data: np.ndarray, tracer_index: int, args: argparse.Namespace
) -> pd.DataFrame:
    """Build a tracer dataframe from the NPY array."""
    clip = CLIP if CLIP is not None else data.shape[0]
    tracer_slice = np.array(data[:clip:DISCRETIZATION, tracer_index, :], dtype=float)
    frame = pd.DataFrame(
        tracer_slice,
        columns=[
            "density",
            "gasTemp",
            "av",
            "PI_Rad",
            "radField",
            "NUV_Rad",
            "NIR_Rad",
            "IR_Rad",
        ],
    )
    frame["time"] = np.arange(len(frame)) * TIMESTEP_KYR * DISCRETIZATION
    frame["tracer"] = tracer_index
    frame["benchmark"] = args.benchmark
    frame["density"] = density_to_number_density(frame["density"].to_numpy())
    return frame[["tracer", "time", "gasTemp", "density", "av", "radField"]]


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    data = np.load(NPY_PATH, mmap_mode="r")
    tracers = select_tracers(data, args)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for tracer_index in tracers:
        filename = f"tracer_{tracer_index}.csv"
        output_path = OUTPUT_DIR / filename
        if args.skip_existing and output_path.exists():
            continue
        frame = build_frame(data, tracer_index, args)
        frame.to_csv(output_path, index=False)
        written += 1

    print(f"Selected tracers: {len(tracers)}")
    print(f"CSVs written: {written}")
    print(f"Input npy: {os.fspath(NPY_PATH)}")
    print(f"Output dir: {os.fspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
