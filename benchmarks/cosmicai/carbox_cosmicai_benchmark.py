"""Run CosmicAI tracer benchmarks with joblib or single-CSV mode."""

# Examples:
# python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --output-dir outputs --random-count=2
# python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --tracer-csv benchmarks/cosmicai/data/turbulence_tracers_csv/tracer_630.csv --output-dir outputs

import argparse
import os
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, cpu_count, delayed

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.network import Network
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

# Constants ported from run_carbox_benchmark.py
SPOOFED_INITIAL_TIME = 1e2
KYR_TO_YR = 1000.0
YEAR_TO_SEC = 3.15576e7
RADFIELD_FACTOR = 1.7
ELEMENTS = ["H", "HE", "C", "N", "O", "S", "SI", "FE", "MG", "NA", "CL", "P", "F"]
DEFAULT_TRACER_DIR = Path("benchmarks/cosmicai/data/turbulence_tracers_csv")
NETWORK_PATH = Path("network_files/uclchem_small_chemistry.csv")
LARGE_NETWORK_PATH = Path("network_files/uclchem_gas_phase_only.csv")
INITIAL_PATH = Path("benchmarks/initial_conditions/small_chemistry_initial.yaml")
MAX_CHUNK_YEARS = 50.0

# Global cache for worker processes
_WORKER_CACHE = {}
_SEED = 14


@dataclass
class NetworkAssets:
    """Cached network resources."""

    network: Network
    jnetwork: Any
    template: np.ndarray
    species_names: list[str]


def parse_element_counts(name: str, elements: Sequence[str]) -> dict[str, int]:
    """Return elemental counts for a species name."""
    counts: dict[str, int] = {}
    index = 0
    while index < len(name):
        two_letter = name[index : index + 2]
        one_letter = name[index]
        if two_letter in elements:
            value = 1
            if index + 2 < len(name) and name[index + 2].isdigit():
                value = int(name[index + 2])
                index += 1
            counts[two_letter] = counts.get(two_letter, 0) + value
            index += 2
            continue
        if one_letter in elements:
            value = 1
            if index + 1 < len(name) and name[index + 1].isdigit():
                value = int(name[index + 1])
                index += 1
            counts[one_letter] = counts.get(one_letter, 0) + value
        index += 1
    return counts


def build_stoichiometric_matrix(network_species, elements: Sequence[str]) -> np.ndarray:
    """Build stoichiometric matrix mapping species to elements."""
    matrix = np.zeros((len(elements), len(network_species)))
    for column, species in enumerate(network_species):
        name = species.name
        if name in ["E-", "ELECTR"]:
            continue
        composition = parse_element_counts(
            name.replace("+", "").replace("-", ""), elements
        )
        for row, element in enumerate(elements):
            if element in composition:
                matrix[row, column] = composition[element]
    return matrix


def build_time_axis(orig_times: np.ndarray) -> np.ndarray:
    """Return stretched time axis with spoofed initialization."""
    new_times = np.zeros_like(orig_times, dtype=float)
    new_times[0] = 0.0
    new_times[1] = SPOOFED_INITIAL_TIME
    deltas = np.diff(orig_times) * KYR_TO_YR
    if len(new_times) > 2:
        new_times[2:] = new_times[1] + np.cumsum(deltas[1:])
    return new_times


def load_initial_abundances(path: Path) -> dict[str, float]:
    """Load initial abundances from YAML."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    return data["abundances"]


def load_network_assets(
    network_path: Path, initial_abundances: dict[str, float]
) -> NetworkAssets:
    """Load chemical network and compiled ODE system."""
    network = parse_chemical_network(
        str(network_path), format_type=NetworkNames.uclchem
    )
    jnetwork = network.get_ode()

    # Build unit-density abundance template
    config = SimulationConfig(
        number_density=[1.0],
        temperature=[10.0],
        initial_abundances=initial_abundances,
    )
    template = initialize_abundances(network, config)
    species_names = [s.name for s in network.species]

    return NetworkAssets(
        network=network,
        jnetwork=jnetwork,
        template=template,
        species_names=species_names,
    )


def get_cached_assets():
    """Load and cache network assets for the worker process."""
    if "assets" not in _WORKER_CACHE:
        initial_abundances = load_initial_abundances(INITIAL_PATH)
        _WORKER_CACHE["assets"] = {
            "small": load_network_assets(NETWORK_PATH, initial_abundances),
            "large": load_network_assets(LARGE_NETWORK_PATH, initial_abundances),
        }
    return _WORKER_CACHE["assets"]


@dataclass
class BenchmarkSpec:
    """Benchmark source metadata."""

    path: Path
    timestep_kyr: float
    clip: int
    discretization: int


@dataclass
class TracerDataset:
    """Tracer frame with identifier."""

    tracer_id: int
    frame: pd.DataFrame


DEFAULT_BENCHMARK_SPEC = BenchmarkSpec(
    path=Path("benchmarks/cosmicai/data/M600_seed1_trace_cells.npy"),
    timestep_kyr=8.299,
    clip=400,
    discretization=1,
)


def density_to_number_density(density: np.ndarray) -> np.ndarray:
    """Convert mass density to number density."""
    hydrogen_mass = 1.66053906660e-24
    mean_molecular_mass = 1.4168138025
    return density / (mean_molecular_mass * hydrogen_mass)


def get_benchmark_spec(
    benchmark: str,
    npy_path: Path | None,
    timestep: float | None,
    clip: int | None,
    discretization: int,
) -> BenchmarkSpec:
    """Resolve benchmark specification."""
    if benchmark != "M600_1":
        raise ValueError(f"Unsupported benchmark {benchmark}")
    base = DEFAULT_BENCHMARK_SPEC
    return BenchmarkSpec(
        path=npy_path or base.path,
        timestep_kyr=timestep if timestep is not None else base.timestep_kyr,
        clip=clip if clip is not None else base.clip,
        discretization=discretization,
    )


def build_tracer_frame(
    data: np.ndarray, tracer_index: int, spec: BenchmarkSpec
) -> pd.DataFrame:
    """Build a tracer DataFrame from array data."""
    if tracer_index < 0 or tracer_index >= data.shape[1]:
        raise ValueError(f"Tracer index {tracer_index} out of range")
    clip_limit = spec.clip if spec.clip > 0 else data.shape[0]
    tracer_slice = np.array(
        data[: clip_limit : spec.discretization, tracer_index, :], dtype=float
    )
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
    frame["density"] = density_to_number_density(frame["density"].to_numpy())
    frame["time"] = np.arange(len(frame)) * spec.timestep_kyr * spec.discretization
    frame["tracer"] = tracer_index
    return frame[["tracer", "time", "gasTemp", "density", "av", "radField"]]


def load_tracers(args: argparse.Namespace) -> list[TracerDataset]:
    """Load tracer datasets from NPY source."""
    if args.tracer_csv:
        return [load_tracer_csv(args.tracer_csv)]

    spec = get_benchmark_spec(
        args.benchmark, args.npy_path, args.timestep, args.clip, args.discretization
    )
    data = np.load(spec.path, mmap_mode="r")

    if args.random_count:
        rng = np.random.default_rng(_SEED)
        tracer_indices = rng.choice(
            data.shape[1],
            size=min(args.random_count, data.shape[1]),
            replace=False,
        ).tolist()
    else:
        tracer_indices = args.tracers

    datasets = []
    for idx in tracer_indices:
        frame = build_tracer_frame(data, idx, spec)
        datasets.append(TracerDataset(tracer_id=idx, frame=frame))
    return datasets


def load_tracer_csv(path: Path) -> TracerDataset:
    """Load tracer dataset from a CSV file."""
    frame = pd.read_csv(path)
    missing = {"time", "gasTemp", "density", "av", "radField"} - set(frame.columns)
    if missing:
        raise ValueError(f"Tracer CSV missing columns: {sorted(missing)}")

    if "tracer" in frame.columns:
        tracer_id = int(frame["tracer"].iloc[0])
    else:
        stem = path.stem
        tracer_id = int(stem.split("_")[-1]) if stem.split("_")[-1].isdigit() else 0
        frame = frame.copy()
        frame["tracer"] = tracer_id

    frame = frame[["tracer", "time", "gasTemp", "density", "av", "radField"]]
    return TracerDataset(tracer_id=tracer_id, frame=frame)


def compute_fractional_abundances(
    abundances: np.ndarray, network: Network
) -> np.ndarray:
    """Convert species abundances to fractions relative to H nuclei."""
    matrix = build_stoichiometric_matrix(network.species, ELEMENTS)
    elemental = abundances @ matrix.T
    hydrogen_index = ELEMENTS.index("H")
    hydrogen = np.clip(elemental[..., hydrogen_index], 1e-18, None)
    fractions = abundances / hydrogen[..., None]
    return np.clip(fractions, 1e-18, None)


def save_tracer_output(
    tracer_id: int,
    time_grid: np.ndarray,
    abundances: np.ndarray,
    densities: np.ndarray,
    temperatures: np.ndarray,
    avs: np.ndarray,
    rad_fields: np.ndarray,
    species_names: Sequence[str],
    output_dir: Path,
    label: str | None = None,
) -> Path:
    """Save tracer outputs to npy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["time", "density", "temperature", "av", "rad_field"] + list(
        species_names
    )
    matrix = np.column_stack(
        [time_grid, densities, temperatures, avs, rad_fields, abundances]
    )
    payload = {"columns": np.array(columns, dtype=object), "data": matrix}
    suffix = f"_{label}" if label else ""
    output_path = output_dir / f"tracer_{tracer_id}{suffix}.npy"
    np.save(output_path, payload, allow_pickle=True)  # type: ignore
    return output_path


def map_abundances_between_networks(
    source_names: Sequence[str],
    source_values: np.ndarray,
    target_names: Sequence[str],
    base_values: np.ndarray,
) -> np.ndarray:
    """Transfer abundances from source species list into target."""
    lookup = {name: idx for idx, name in enumerate(source_names)}
    mapped = np.array(base_values, copy=True)
    for target_idx, target_name in enumerate(target_names):
        if target_name in lookup:
            mapped[target_idx] = source_values[lookup[target_name]]
    return mapped


def warmup_large_network(
    y0: np.ndarray,
    assets: NetworkAssets,
    density: float,
    temperature: float,
    av: float,
    rad_field: float,
    solver_name: str,
    cr_rate: float = 1.6e-17,
    warmup_seconds: float = 1e5,
) -> np.ndarray:
    """Short pre-relaxation to avoid extreme stiffness at t=0 for large network."""
    config = SimulationConfig(
        number_density=[density, density],
        temperature=[temperature, temperature],
        visual_extinction=[av, av],
        fuv_field=[rad_field, rad_field],
        cr_rate=[cr_rate, cr_rate],
        physics_t=[0.0, warmup_seconds],
        solver=solver_name,
        atol=1e-12,
        rtol=1e-4,
        max_steps=20000,
    )
    solution = solve_network(assets.jnetwork, y0, config)
    return np.asarray(solution.ys)[-1]


def integrate_large_chunked(
    jnetwork,
    y0: np.ndarray,
    densities: np.ndarray,
    temps: np.ndarray,
    avs: np.ndarray,
    rad_fields: np.ndarray,
    physics_t_seconds: np.ndarray,
    solver_name: str,
    atol: float,
    rtol: float,
    per_species_atol: list[float],
    max_steps: int,
    max_chunk_years: float = MAX_CHUNK_YEARS,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate large network in smaller time chunks to avoid hitting max_steps."""
    chunk_seconds = max_chunk_years * YEAR_TO_SEC

    current_y = np.asarray(y0)
    ys = [current_y]
    ts = [physics_t_seconds[0]]
    min_chunk_seconds = 0.01 * YEAR_TO_SEC

    for idx in range(len(physics_t_seconds) - 1):
        t0, t1 = physics_t_seconds[idx], physics_t_seconds[idx + 1]
        span = t1 - t0
        n_sub = max(1, int(np.ceil(span / chunk_seconds)))
        sub_ts = np.linspace(t0, t1, n_sub + 1)

        def lin_interp(a0: float, a1: float) -> np.ndarray:
            return np.linspace(a0, a1, n_sub + 1)

        densities_sub = lin_interp(densities[idx], densities[idx + 1])
        temps_sub = lin_interp(temps[idx], temps[idx + 1])
        avs_sub = lin_interp(avs[idx], avs[idx + 1])
        rad_fields_sub = lin_interp(rad_fields[idx], rad_fields[idx + 1])

        def solve_interval(
            t_start: float,
            t_end: float,
            vals_start: tuple[float, float, float, float],
            vals_end: tuple[float, float, float, float],
            y_start: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            config = SimulationConfig(
                number_density=[vals_start[0], vals_end[0]],
                temperature=[vals_start[1], vals_end[1]],
                visual_extinction=[vals_start[2], vals_end[2]],
                fuv_field=[vals_start[3], vals_end[3]],
                cr_rate=(np.ones(2) * 1.6e-17).tolist(),
                physics_t=[t_start, t_end],
                solver=solver_name,
                atol=atol,
                rtol=rtol,
                per_species_atol=per_species_atol,
                max_steps=max_steps,
            )

            try:
                solution = solve_network(jnetwork, y_start, config)
                return np.asarray(solution.ts)[-1], np.asarray(solution.ys)[-1]
            except Exception as exc:
                message = str(exc).lower()
                if (
                    "maximum number of solver steps" in message
                    and (t_end - t_start) > min_chunk_seconds
                ):
                    midpoint = 0.5 * (t_start + t_end)
                    mid_vals = tuple(
                        0.5 * (vs + ve) for vs, ve in zip(vals_start, vals_end)
                    )
                    _, y_mid = solve_interval(
                        t_start, midpoint, vals_start, mid_vals, y_start
                    )
                    return solve_interval(midpoint, t_end, mid_vals, vals_end, y_mid)
                raise

        for j in range(n_sub):
            vals_start = (
                float(densities_sub[j]),
                float(temps_sub[j]),
                float(avs_sub[j]),
                float(rad_fields_sub[j]),
            )
            vals_end = (
                float(densities_sub[j + 1]),
                float(temps_sub[j + 1]),
                float(avs_sub[j + 1]),
                float(rad_fields_sub[j + 1]),
            )

            t_final, current_y = solve_interval(
                float(sub_ts[j]), float(sub_ts[j + 1]), vals_start, vals_end, current_y
            )
            ys.append(current_y)
            ts.append(t_final)

    return np.asarray(ts), np.asarray(ys)


def process_tracer(tracer: TracerDataset, output_dir: Path, solver_name: str) -> float:
    """Run solver for a single tracer and save results."""
    try:
        start_time = time()

        # Retrieve cached assets (loaded once per worker process)
        assets = get_cached_assets()
        small_assets = assets["small"]
        large_assets = assets["large"]

        # Prepare arrays
        frame = tracer.frame
        orig_times = frame["time"].to_numpy(dtype=float)
        time_grid = build_time_axis(orig_times)
        densities = frame["density"].to_numpy(dtype=float)
        temps = frame["gasTemp"].to_numpy(dtype=float)
        avs = frame["av"].to_numpy(dtype=float)
        rad_fields = frame["radField"].to_numpy(dtype=float)
        rad_fields = rad_fields * RADFIELD_FACTOR

        # Initial state
        # Solver expects fractional abundances; physical density is supplied separately.
        y0_small = small_assets.template

        # Create SimulationConfig for this tracer
        # Convert time grid to seconds for the solver
        physics_t_seconds = time_grid * YEAR_TO_SEC

        config_small = SimulationConfig(
            number_density=densities.tolist(),
            temperature=temps.tolist(),
            visual_extinction=avs.tolist(),
            fuv_field=rad_fields.tolist(),
            cr_rate=(jnp.ones_like(densities) * 1.6e-17).tolist(),
            physics_t=physics_t_seconds.tolist(),
            solver=solver_name,
            atol=1e-10,
            rtol=1e-4,
            max_steps=200000,
        )

        # Solve small network to get post-spoof chemistry
        solution_small = solve_network(small_assets.jnetwork, y0_small, config_small)

        # Process results
        ys_small = np.asarray(solution_small.ys)
        fractional_small = compute_fractional_abundances(ys_small, small_assets.network)

        # Drop the spoofed t=0 snapshot so both networks start from the post-spoof state
        start_idx = 1

        save_tracer_output(
            tracer.tracer_id,
            np.asarray(solution_small.ts)[start_idx:],
            fractional_small[start_idx:],
            densities[start_idx:],
            temps[start_idx:],
            avs[start_idx:],
            rad_fields[start_idx:],
            small_assets.species_names,
            output_dir,
            label="small",
        )

        # Seed the large network with the post-spoof abundances for overlapping species
        post_spoof_abundances = ys_small[start_idx]
        # Keep large network initialization in fractional units as well
        large_y0_base = large_assets.template
        y0_large = map_abundances_between_networks(
            small_assets.species_names,
            post_spoof_abundances,
            large_assets.species_names,
            large_y0_base,
        )

        # Pre-relax new species in the large network to reduce stiffness at t=0
        y0_large = warmup_large_network(
            y0_large,
            large_assets,
            density=densities[start_idx],
            temperature=temps[start_idx],
            av=avs[start_idx],
            rad_field=rad_fields[start_idx],
            solver_name=solver_name,
        )

        # Use relaxed absolute tolerances for extremely tiny species so they don't
        # dominate stepsize control.
        atol_vector = np.full_like(y0_large, 1e-6, dtype=float)
        tiny_mask = y0_large < 1e-9
        atol_vector[tiny_mask] = 1e-2

        config_large = SimulationConfig(
            number_density=densities[start_idx:].tolist(),
            temperature=temps[start_idx:].tolist(),
            visual_extinction=avs[start_idx:].tolist(),
            fuv_field=rad_fields[start_idx:].tolist(),
            cr_rate=(jnp.ones_like(densities[start_idx:]) * 1.6e-17).tolist(),
            physics_t=physics_t_seconds[start_idx:].tolist(),
            solver=solver_name,
            atol=1e-6,
            rtol=1e-2,
            per_species_atol=atol_vector.tolist(),
            max_steps=200000,
        )

        ts_large, ys_large = integrate_large_chunked(
            large_assets.jnetwork,
            y0_large,
            densities[start_idx:],
            temps[start_idx:],
            avs[start_idx:],
            rad_fields[start_idx:],
            physics_t_seconds[start_idx:],
            solver_name=solver_name,
            atol=config_large.atol,
            rtol=config_large.rtol,
            per_species_atol=config_large.per_species_atol or [config_large.atol],
            max_steps=config_large.max_steps,
        )
        fractional_large = compute_fractional_abundances(ys_large, large_assets.network)

        save_tracer_output(
            tracer.tracer_id,
            np.asarray(ts_large),
            fractional_large,
            densities[start_idx:],
            temps[start_idx:],
            avs[start_idx:],
            rad_fields[start_idx:],
            large_assets.species_names,
            output_dir,
            label="large",
        )

        return time() - start_time
    except Exception:
        print(f"\nTracer {tracer.tracer_id} crashed; skipping.")
        traceback.print_exc()
        try:
            frame = tracer.frame
            print(
                "Physical ranges "
                f"dens={frame['density'].min():.3e}/{frame['density'].max():.3e} "
                f"T={frame['gasTemp'].min():.3e}/{frame['gasTemp'].max():.3e} "
                f"Av={frame['av'].min():.3e}/{frame['av'].max():.3e} "
                f"radField={frame['radField'].min():.3e}/{frame['radField'].max():.3e}"
            )
        except Exception:
            pass
        if os.getenv("RAISE_ON_TRACER_ERROR"):
            raise
        return 0.0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch run Carbox tracers")

    # Benchmark args
    parser.add_argument("--benchmark", type=str, default="M600_1", help="Benchmark ID")
    parser.add_argument("--discretization", type=int, default=1, help="Stride")
    parser.add_argument("--clip", type=int, default=None, help="Max timesteps")
    parser.add_argument("--timestep", type=float, default=None, help="Timestep kyr")
    parser.add_argument("--npy-path", type=Path, default=None, help="Source NPY path")
    parser.add_argument(
        "--tracer-csv",
        type=Path,
        default=None,
        help="Run a single tracer CSV instead of loading from NPY",
    )

    # Tracer selection
    parser.add_argument(
        "--tracers", type=int, nargs="+", default=[7650], help="Indices"
    )
    parser.add_argument("--random-count", type=int, default=None, help="Random count")

    # Execution
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output dir"
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument(
        "--solver",
        type=str,
        default="kvaerno5",
        help="ODE solver: dopri5, kvaerno5, tsit5",
    )

    return parser.parse_args()


def main() -> None:
    """Entrypoint for batch benchmark runs."""
    args = parse_args()
    tracers = load_tracers(args)

    if not tracers:
        print("No tracers found to process.")
        return

    if args.tracer_csv:
        print(f"Processing tracer {tracers[0].tracer_id} from CSV...")
        start_wall_time = time()
        durations = [process_tracer(tracers[0], args.output_dir, args.solver)]
    else:
        workers = args.workers or cpu_count() or 1
        print(f"Processing {len(tracers)} tracers with {workers} workers...")
        start_wall_time = time()
        durations = []
        parallel_generator = Parallel(n_jobs=workers, return_as="generator")(
            delayed(process_tracer)(tracer, args.output_dir, args.solver)
            for tracer in tracers
        )

        for i, duration in enumerate(parallel_generator, 1):
            durations.append(duration)
            print(f"Completed {i}/{len(tracers)} tracers", end="\r", flush=True)
        print("")  # Newline after progress bar

    end_wall_time = time()
    wall_time = end_wall_time - start_wall_time

    print(f"Completed {len(tracers)} tracers.")

    total_cpu_time = sum(durations)
    throughput = len(tracers) / wall_time if wall_time > 0 else 0.0

    print("-" * 40)
    print(f"Wall Time:       {wall_time:.2f} s")
    print(f"Throughput:      {throughput:.2f} tracers/s")
    print(f"Total CPU Time:  {total_cpu_time:.2f} s")
    print("-" * 40)

    if durations:
        print(f"Max time: {max(durations):.2f}s")
        print(f"Average time: {total_cpu_time / len(durations):.2f}s")


if __name__ == "__main__":
    main()
