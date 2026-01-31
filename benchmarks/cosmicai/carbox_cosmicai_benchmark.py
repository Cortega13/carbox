"""Run a single CosmicAI tracer through Carbox (CSV-only, small→large).

python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --tracer-csv benchmarks/cosmicai/data/turbulence_tracers_csv/tracer_38446.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import time

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.main import run_simulation
from carbox.parsers import NetworkNames, parse_chemical_network

# -----------------
# Globals (no CLI)
# -----------------

SPOOFED_INITIAL_TIME_YR = 1e5
KYR_TO_YR = 1000.0

RADFIELD_FACTOR = 1.7
CR_RATE_DEFAULT = 1.6e-17

ATOL = 1e-15
RTOL = 1e-6
MAX_STEPS = 80000

FRACTION_FLOOR = 1e-24
START_IDX = 1  # drop t=0 snapshot; start from post-spoof state

NETWORK_PATH = Path("network_files/uclchem_small_chemistry.csv")
LARGE_NETWORK_PATH = Path("network_files/uclchem_gas_phase_only.csv")
INITIAL_PATH = Path("benchmarks/initial_conditions/small_chemistry_initial.yaml")


def build_time_years(times_kyr: np.ndarray) -> np.ndarray:
    """Convert kyr grid to years and add spoof step."""
    times_kyr = np.asarray(times_kyr, dtype=float)
    out = np.zeros_like(times_kyr)
    out[0] = 0.0
    out[1] = SPOOFED_INITIAL_TIME_YR
    if len(out) > 2:
        out[2:] = out[1] + np.cumsum(np.diff(times_kyr)[1:] * KYR_TO_YR)
    return out


def load_tracer_csv(path: Path) -> tuple[int, pd.DataFrame]:
    """Load a tracer CSV and return (tracer_id, normalized dataframe)."""
    frame = pd.read_csv(path)
    required = {"time", "gasTemp", "density", "av", "radField"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Tracer CSV missing columns: {sorted(missing)}")

    if "tracer" in frame.columns:
        tracer_id = int(frame["tracer"].iloc[0])
    else:
        stem = path.stem
        tracer_id = int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else 0
        frame = frame.copy()
        frame["tracer"] = tracer_id

    cols = ["tracer", "time", "gasTemp", "density", "av", "radField"]
    return tracer_id, frame[cols]


def make_config(
    time_years: np.ndarray,
    density: np.ndarray,
    temp: np.ndarray,
    av: np.ndarray,
    fuv: np.ndarray,
    initial_abundances: dict[str, float],
) -> SimulationConfig:
    """Create a `SimulationConfig` for a tracer time series."""
    return SimulationConfig(
        time_grid_years=time_years.tolist(),
        number_density_grid=density.tolist(),
        temperature_grid=temp.tolist(),
        visual_extinction_grid=av.tolist(),
        fuv_field_grid=fuv.tolist(),
        cr_rate_grid=(np.ones_like(density) * CR_RATE_DEFAULT).tolist(),
        initial_abundances=initial_abundances,
        solver="kvaerno5",
        atol=ATOL,
        rtol=RTOL,
        max_steps=MAX_STEPS,
    )


def to_fractional(
    ys: np.ndarray, species_names: list[str], number_density_grid: np.ndarray
) -> np.ndarray:
    """Convert absolute abundances to fractional via n_H,nuclei = 2H2 + H."""
    h2_idx = species_names.index("H2") if "H2" in species_names else None
    h_idx = species_names.index("H") if "H" in species_names else None

    if h2_idx is None and h_idx is None:
        denom = np.asarray(number_density_grid, dtype=float)
    else:
        denom = np.zeros(ys.shape[0], dtype=float)
        if h2_idx is not None:
            denom += 2.0 * ys[:, h2_idx]
        if h_idx is not None:
            denom += ys[:, h_idx]

    denom = np.clip(denom, FRACTION_FLOOR, None)
    return np.clip(ys / denom[:, None], FRACTION_FLOOR, None)


def save_tracer_output(
    tracer_id: int,
    time_years: np.ndarray,
    frac_abundances: np.ndarray,
    density: np.ndarray,
    temp: np.ndarray,
    av: np.ndarray,
    fuv: np.ndarray,
    species_names: list[str],
    output_dir: Path,
    label: str,
) -> Path:
    """Save tracer evolution to a `.npy` payload with fractional abundances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["time_years", "density", "temperature", "av", "rad_field"] + list(
        species_names
    )
    data = np.column_stack([time_years, density, temp, av, fuv, frac_abundances])
    payload = {"columns": np.array(columns, dtype=object), "data": data}
    path = output_dir / f"tracer_{tracer_id}_{label}.npy"
    np.save(path, payload, allow_pickle=True)  # type: ignore
    return path


def map_abundances(
    source_names: list[str],
    source_values: np.ndarray,
    target_names: list[str],
    target_base: np.ndarray,
) -> np.ndarray:
    """Map overlapping species abundances from one network ordering to another."""
    lookup = {name: i for i, name in enumerate(source_names)}
    out = np.array(target_base, copy=True)
    for j, name in enumerate(target_names):
        if name in lookup:
            out[j] = source_values[lookup[name]]
    return out


def run_tracer(frame: pd.DataFrame, output_dir: Path) -> float:
    """Run the small→large workflow for one tracer and write outputs."""
    t0 = time()
    with INITIAL_PATH.open() as handle:
        initial = yaml.safe_load(handle)["abundances"]

    times_years = build_time_years(frame["time"].to_numpy())
    dens = frame["density"].to_numpy(dtype=float)
    temp = frame["gasTemp"].to_numpy(dtype=float)
    av = frame["av"].to_numpy(dtype=float)
    fuv = frame["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR

    # ---- small ----
    config_small = make_config(times_years, dens, temp, av, fuv, initial)
    small = run_simulation(
        str(NETWORK_PATH),
        config_small,
        format_type=NetworkNames.uclchem,
        verbose=False,
        save_outputs=False,
        validate_config=False,
    )
    net_small = small["network"]
    ys_small = np.asarray(small["solution"].ys)
    small_names = [s.name for s in net_small.species]
    frac_small = to_fractional(ys_small, small_names, dens)

    save_tracer_output(
        int(frame["tracer"].iloc[0]),
        times_years[START_IDX:],
        frac_small[START_IDX:],
        dens[START_IDX:],
        temp[START_IDX:],
        av[START_IDX:],
        fuv[START_IDX:],
        small_names,
        output_dir,
        label="small",
    )

    # ---- large ----
    times2 = times_years[START_IDX:]
    dens2, temp2, av2, fuv2 = (
        dens[START_IDX:],
        temp[START_IDX:],
        av[START_IDX:],
        fuv[START_IDX:],
    )
    config_large = make_config(times2, dens2, temp2, av2, fuv2, initial)

    net_large = parse_chemical_network(
        str(LARGE_NETWORK_PATH), format_type=NetworkNames.uclchem
    )
    large_names = [s.name for s in net_large.species]
    y0_base = np.asarray(initialize_abundances(net_large, config_large, verbose=False))
    y0_large = map_abundances(small_names, ys_small[START_IDX], large_names, y0_base)

    large = run_simulation(
        str(LARGE_NETWORK_PATH),
        config_large,
        format_type=NetworkNames.uclchem,
        verbose=False,
        network=net_large,
        y0_override=jnp.asarray(y0_large),
        save_outputs=False,
        validate_config=False,
    )
    ys_large = np.asarray(large["solution"].ys)
    frac_large = to_fractional(ys_large, large_names, dens2)

    save_tracer_output(
        int(frame["tracer"].iloc[0]),
        times2,
        frac_large,
        dens2,
        temp2,
        av2,
        fuv2,
        large_names,
        output_dir,
        label="large",
    )

    return time() - t0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Run a single Carbox tracer (CSV-only)")
    p.add_argument("--tracer-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def tracer_outputs_exist(tracer_id: int, output_dir: Path) -> bool:
    """Return True if both small+large outputs for tracer_id already exist."""
    small_path = output_dir / f"tracer_{tracer_id}_small.npy"
    large_path = output_dir / f"tracer_{tracer_id}_large.npy"
    return small_path.exists() and large_path.exists()


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    tracer_id, frame = load_tracer_csv(args.tracer_csv)
    print(f"Processing tracer {tracer_id} from CSV {args.tracer_csv}...")

    # Ensure output directory exists for both binary outputs and runtime metadata.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: skip if this tracer was already processed.
    if tracer_outputs_exist(tracer_id, args.output_dir):
        print(
            "Outputs already exist; skipping: "
            + str(args.output_dir / f"tracer_{tracer_id}_small.npy")
            + " and "
            + str(args.output_dir / f"tracer_{tracer_id}_large.npy")
        )
        return

    started_utc = datetime.now(timezone.utc)
    dt = run_tracer(frame, args.output_dir)
    finished_utc = datetime.now(timezone.utc)
    print(f"Completed tracer {tracer_id} in {dt:.2f} s")

    # Append a single timing row to a shared file (one file total across runs).
    timing_path = args.output_dir / "benchmark_timing.tsv"
    is_new = not timing_path.exists()
    with timing_path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write(
                "\t".join(
                    [
                        "started_utc",
                        "finished_utc",
                        "elapsed_seconds",
                        "tracer_id",
                        "tracer_csv",
                    ]
                )
                + "\n"
            )
        handle.write(
            "\t".join(
                [
                    started_utc.isoformat(),
                    finished_utc.isoformat(),
                    f"{dt:.6f}",
                    str(tracer_id),
                    str(args.tracer_csv),
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
