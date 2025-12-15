"""Batch run Carbox benchmark tracers with joblib parallelism.

python benchmarks/cosmicai/batch_run_carbox_benchmark.py --output-dir outputs --random-count=40 --batch-size=4
"""

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time

# Set JAX flags for CPU optimization (must be set before jax import)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true"
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

from benchmarks.cosmicai.run_carbox_benchmark import (
    DEFAULT_TRACER_DIR,
    ELEMENTS,
    INITIAL_PATH,
    RADFIELD_FACTOR,
    build_stoichiometric_matrix,
    build_time_axis,
    load_initial_abundances,
    load_network,
    load_tracer_csv,
)
from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.network import Network
from carbox.solver import solve_network_core

# Global cache for worker processes
_WORKER_CACHE = {}


def get_cached_assets():
    """Load and cache network assets for the worker process."""
    if "assets" not in _WORKER_CACHE:
        network, jnetwork = load_network()
        initial_abundances = load_initial_abundances(INITIAL_PATH)

        # Build unit-density abundance template
        config = SimulationConfig(
            number_density=1.0,
            temperature=10.0,
            initial_abundances=initial_abundances,
        )
        template = initialize_abundances(network, config)

        species_names = [s.name for s in network.species]
        _WORKER_CACHE["assets"] = (network, jnetwork, template, species_names)
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
    """Load tracer datasets from CSV paths or NPY source."""
    if args.csv_paths:
        paths = [Path(p) for p in args.csv_paths]
        datasets = []
        for path in paths:
            frame = load_tracer_csv(path)
            tracer_id = int(frame["tracer"].iloc[0])
            datasets.append(TracerDataset(tracer_id=tracer_id, frame=frame))
        return datasets

    spec = get_benchmark_spec(
        args.benchmark, args.npy_path, args.timestep, args.clip, args.discretization
    )
    data = np.load(spec.path, mmap_mode="r")

    if args.random_count:
        rng = np.random.default_rng()
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
    output_path = output_dir / f"tracer_{tracer_id}.npy"
    np.save(output_path, payload, allow_pickle=True)  # type: ignore
    return output_path


def process_tracer(tracer: TracerDataset, output_dir: Path) -> float:
    """Run solver for a single tracer and save results."""
    start_time = time()

    # Retrieve cached assets (loaded once per worker process)
    network, jnetwork, template, species_names = get_cached_assets()

    # Prepare arrays
    frame = tracer.frame
    orig_times = frame["time"].to_numpy(dtype=float)
    time_grid = build_time_axis(orig_times)
    densities = frame["density"].to_numpy(dtype=float)
    temps = frame["gasTemp"].to_numpy(dtype=float)
    avs = frame["av"].to_numpy(dtype=float)
    rad_fields = frame["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR

    # JAX arrays
    t_eval = jnp.array(time_grid, dtype=float)
    dens_jnp = jnp.array(densities, dtype=float)
    temp_jnp = jnp.array(temps, dtype=float)
    avs_jnp = jnp.array(avs, dtype=float)
    rad_jnp = jnp.array(rad_fields, dtype=float)
    cr_rates = jnp.ones_like(dens_jnp) * 1.6e-17

    # Initial state
    y0 = template * densities[0]

    # Solve
    solution = solve_network_core(
        jnetwork,
        y0,
        t_eval,
        t_eval,
        dens_jnp,
        temp_jnp,
        cr_rates,
        rad_jnp,
        avs_jnp,
        solver_name="kvaerno5",
        atol=1e-14,
        rtol=1e-6,
        max_steps=500000,
    )

    # Process results
    ys = np.asarray(solution.ys)
    fractional = compute_fractional_abundances(ys, network)

    save_tracer_output(
        tracer.tracer_id,
        np.asarray(solution.ts),
        fractional,
        densities,
        temps,
        avs,
        rad_fields,
        species_names,
        output_dir,
    )

    return time() - start_time


def process_tracer_batch(tracers: list[TracerDataset], output_dir: Path) -> float:
    """Run solver for a batch of tracers."""
    start_time = time()

    if not tracers:
        return 0.0

    # Retrieve cached assets (loaded once per worker process)
    network, jnetwork, template, species_names = get_cached_assets()

    # Pre-allocate arrays
    # Get time grid from first tracer (assumed constant for the batch)
    first_frame = tracers[0].frame
    orig_times = first_frame["time"].to_numpy(dtype=float)
    time_grid = build_time_axis(orig_times)
    t_eval = jnp.array(time_grid, dtype=float)

    y0_list = []
    dens_list = []
    temp_list = []
    av_list = []
    rad_list = []

    for tracer in tracers:
        frame = tracer.frame
        dens = frame["density"].to_numpy(dtype=float)
        temp = frame["gasTemp"].to_numpy(dtype=float)
        av = frame["av"].to_numpy(dtype=float)
        rad = frame["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR

        y0_list.append(template * dens[0])
        dens_list.append(dens)
        temp_list.append(temp)
        av_list.append(av)
        rad_list.append(rad)

    y0_batch = jnp.stack(y0_list)
    dens_batch = jnp.stack(dens_list)
    temp_batch = jnp.stack(temp_list)
    av_batch = jnp.stack(av_list)
    rad_batch = jnp.stack(rad_list)
    cr_rates_batch = jnp.ones_like(dens_batch) * 1.6e-17

    def _solve_batch_wrapper(y0, nd, temp, cr, fuv, av):
        return solve_network_core(
            jnetwork,
            y0,
            t_eval,
            t_eval,
            nd,
            temp,
            cr,
            fuv,
            av,
            solver_name="kvaerno5",
            atol=1e-14,
            rtol=1e-6,
            max_steps=500000,
        )

    # Vmap the solver wrapper
    # We map over all arguments (0)
    batch_solver = jax.vmap(_solve_batch_wrapper)

    # Solve
    solution = batch_solver(
        y0_batch, dens_batch, temp_batch, cr_rates_batch, rad_batch, av_batch
    )

    # Process results
    ys_batch = np.asarray(solution.ys)
    ts_batch = np.asarray(solution.ts)

    # Compute fractional abundances
    # ys_batch is (batch, time, species)
    fractional_batch = compute_fractional_abundances(ys_batch, network)

    for i, tracer in enumerate(tracers):
        save_tracer_output(
            tracer.tracer_id,
            ts_batch[i],
            fractional_batch[i],
            dens_list[i],
            temp_list[i],
            av_list[i],
            rad_list[i],
            species_names,
            output_dir,
        )

    return time() - start_time


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch run Carbox tracers")

    # Benchmark args
    parser.add_argument("--benchmark", type=str, default="M600_1", help="Benchmark ID")
    parser.add_argument("--discretization", type=int, default=1, help="Stride")
    parser.add_argument("--clip", type=int, default=None, help="Max timesteps")
    parser.add_argument("--timestep", type=float, default=None, help="Timestep kyr")
    parser.add_argument("--npy-path", type=Path, default=None, help="Source NPY path")

    # Tracer selection
    parser.add_argument(
        "--tracers", type=int, nargs="+", default=[7650], help="Indices"
    )
    parser.add_argument("--random-count", type=int, default=None, help="Random count")
    parser.add_argument("--csv-paths", type=Path, nargs="*", help="Direct CSV paths")
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_TRACER_DIR, help="Base dir"
    )

    # Execution
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output dir"
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size per worker"
    )

    return parser.parse_args()


def main() -> None:
    """Entrypoint for batch benchmark runs."""
    args = parse_args()
    tracers = load_tracers(args)

    if not tracers:
        print("No tracers found to process.")
        return

    workers = args.workers or cpu_count() or 1

    # Determine batch size
    # If not provided, try to balance between core saturation and vectorization
    if args.batch_size:
        batch_size = args.batch_size
    else:
        # Default strategy: ensure at least 1 batch per worker, then fill up
        # For small tracer counts (like 40) and high cores (20), batch_size=2 is good.
        # For large tracer counts, larger batch_size (e.g. 64) is better for SIMD.
        # Let's cap batch size at 64 but ensure we use workers.
        min_batches = workers
        calculated_size = max(1, len(tracers) // min_batches)
        batch_size = min(64, calculated_size)
        # Ensure at least 1
        batch_size = max(1, batch_size)

    print(
        f"Processing {len(tracers)} tracers with {workers} workers (batch size: {batch_size})..."
    )

    # Create batches
    batches = [tracers[i : i + batch_size] for i in range(0, len(tracers), batch_size)]

    start_wall_time = time()
    durations: list[float] = Parallel(n_jobs=workers)(
        delayed(process_tracer_batch)(batch, args.output_dir) for batch in batches
    )  # type: ignore
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
        print(f"Max batch time: {max(durations):.2f}s")
        print(f"Average batch time: {total_cpu_time / len(durations):.2f}s")
        print(f"Average time per tracer: {total_cpu_time / len(tracers):.2f}s")


if __name__ == "__main__":
    main()


"""
python benchmarks/cosmicai/batch_run_carbox_benchmark.py --output-dir outputs --random-count=80 --batch-size=4
Completed 80 tracers.
----------------------------------------
Wall Time:       412.39 s
Throughput:      0.19 tracers/s
Total CPU Time:  7391.24 s
----------------------------------------
Max batch time: 397.98s
Average batch time: 369.56s
Average time per tracer: 92.39s

python benchmarks/cosmicai/batch_run_carbox_benchmark.py --output-dir outputs --random-count=20 --batch-size=1
----------------------------------------
Wall Time:       102.74 s
Throughput:      0.19 tracers/s
Total CPU Time:  1339.42 s
----------------------------------------
Max batch time: 87.35s
Average batch time: 66.97s
Average time per tracer: 66.97s


"""
