"""Batch run Carbox benchmark tracers with vectorized mapping.

python benchmarks/cosmicai/batch_run_carbox_benchmark.py --engine multiprocessing --output-dir outputs --random-count 20
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time

import diffrax as dx
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
    resolve_csv_path,
)
from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.network import JNetwork, Network
from carbox.solver import solve_network_core


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


@dataclass
class TracerArrays:
    """Tracer arrays for solver input."""

    time_grid: np.ndarray
    densities: np.ndarray
    temperatures: np.ndarray
    avs: np.ndarray
    rad_fields: np.ndarray


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


def load_benchmark_array(spec: BenchmarkSpec) -> np.memmap:
    """Load benchmark npy source."""
    return np.load(spec.path, mmap_mode="r")


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


def load_tracers_from_npy(
    spec: BenchmarkSpec, tracer_indices: Sequence[int]
) -> list[TracerDataset]:
    """Load tracer datasets from npy benchmark source."""
    data = load_benchmark_array(spec)
    datasets: list[TracerDataset] = []
    for tracer_index in tracer_indices:
        frame = build_tracer_frame(data, tracer_index, spec)
        datasets.append(TracerDataset(tracer_id=tracer_index, frame=frame))
    return datasets


def load_tracers(paths: Sequence[Path]) -> list[TracerDataset]:
    """Load tracer datasets from CSV paths."""
    datasets: list[TracerDataset] = []
    for path in paths:
        frame = load_tracer_csv(path)
        tracer_id = int(frame["tracer"].iloc[0])
        datasets.append(TracerDataset(tracer_id=tracer_id, frame=frame))
    return datasets


def ensure_common_time(tracers: Sequence[TracerDataset]) -> np.ndarray:
    """Ensure all tracers share the same time grid."""
    times = [tracer.frame["time"].to_numpy(dtype=float) for tracer in tracers]
    base = times[0]
    for current in times[1:]:
        if not np.allclose(current, base):
            raise ValueError("Tracer time arrays differ")
    return base


def build_batch_parameters(
    tracers: Sequence[TracerDataset],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stacked parameter arrays across tracers."""
    orig_times = ensure_common_time(tracers)
    new_times = build_time_axis(orig_times)
    densities = np.stack(
        [tracer.frame["density"].to_numpy(dtype=float) for tracer in tracers]
    )
    temperatures = np.stack(
        [tracer.frame["gasTemp"].to_numpy(dtype=float) for tracer in tracers]
    )
    avs = np.stack([tracer.frame["av"].to_numpy(dtype=float) for tracer in tracers])
    rad_fields = (
        np.stack([tracer.frame["radField"].to_numpy(dtype=float) for tracer in tracers])
        * RADFIELD_FACTOR
    )
    return new_times, densities, temperatures, avs, rad_fields


def build_template(
    network: Network, initial_abundances: dict[str, float]
) -> jnp.ndarray:
    """Build unit-density abundance template."""
    config = SimulationConfig(
        number_density=1.0,
        temperature=10.0,
        fuv_field=1.0,
        visual_extinction=1.0,
        cr_rate=1.6e-17,
        t_start=0.0,
        t_end=[1.0],
        initial_abundances=initial_abundances,
        solver="kvaerno5",
        atol=1e-18,
        rtol=1e-6,
        max_steps=500000,
        n_snapshots=3,
    )
    return initialize_abundances(network, config)


def build_initial_states(template: jnp.ndarray, densities: np.ndarray) -> jnp.ndarray:
    """Scale template by tracer initial densities."""
    starts = jnp.array(densities[:, 0], dtype=float)
    return starts[:, None] * template[None, :]


def build_solver_fn(
    jnetwork: JNetwork,
    t_eval: jnp.ndarray,
    time_grid: jnp.ndarray,
    use_jit: bool,
):
    """Construct solver callable for a single tracer."""

    def _solve(y0, nd, temp, cr, fuv, av):
        return solve_network_core(
            jnetwork,
            y0,
            t_eval,
            time_grid,
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

    return jax.jit(_solve) if use_jit else _solve


def run_vmap(
    jnetwork: JNetwork,
    t_eval: jnp.ndarray,
    time_grid: jnp.ndarray,
    y0_batch: jnp.ndarray,
    densities: jnp.ndarray,
    temperatures: jnp.ndarray,
    rad_fields: jnp.ndarray,
    avs: jnp.ndarray,
    use_jit: bool,
) -> dx.Solution:
    """Run vectorized solver across tracers."""
    cr_rates = jnp.ones_like(densities) * 1.6e-17
    solver_fn = build_solver_fn(jnetwork, t_eval, time_grid, use_jit)
    vectorized = jax.vmap(solver_fn)
    return vectorized(y0_batch, densities, temperatures, cr_rates, rad_fields, avs)


def block_solution_tree(solutions: dx.Solution) -> dx.Solution:
    """Block JAX execution and materialize solution tree."""
    ys = solutions.ys
    ts = solutions.ts
    if isinstance(ys, jax.Array):
        ys = jax.device_get(jax.block_until_ready(ys))
    else:
        ys = np.asarray(ys)
    if isinstance(ts, jax.Array):
        ts = jax.device_get(jax.block_until_ready(ts))
    else:
        ts = np.asarray(ts)
    return solutions.__replace__(ys=ys, ts=ts)


def summarize_batch(
    solutions: dx.Solution,
    network: Network,
    elements: Sequence[str],
    tracer_ids: Sequence[int],
) -> None:
    """Print final elemental fractions per tracer."""
    matrix = jnp.asarray(build_stoichiometric_matrix(network.species, elements))
    final_abundances = jnp.maximum(solutions.ys[:, -1, :], 1e-20)
    elemental = jnp.einsum("es,bs->be", matrix, final_abundances)
    hydrogen_index = elements.index("H")
    totals = elemental[:, hydrogen_index][:, None]
    fractions = elemental / totals
    for index, tracer_id in enumerate(tracer_ids):
        print(f"\nTracer {tracer_id}")
        for element_index, element in enumerate(elements):
            print(f"{element}: {float(fractions[index, element_index]):.4e}")


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
    np.save(output_path, payload, allow_pickle=True)
    return output_path


def prepare_tracer_arrays(
    tracer: TracerDataset,
) -> TracerArrays:
    """Prepare solver arrays for a tracer."""
    frame = tracer.frame
    orig_times = frame["time"].to_numpy(dtype=float)
    time_grid = build_time_axis(orig_times)
    densities = frame["density"].to_numpy(dtype=float)
    temps = frame["gasTemp"].to_numpy(dtype=float)
    avs = frame["av"].to_numpy(dtype=float)
    rad_fields = frame["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR
    return TracerArrays(time_grid, densities, temps, avs, rad_fields)


def prepare_assets() -> tuple[
    Network, JNetwork, dict[str, float], jnp.ndarray, list[str]
]:
    """Load network assets for solving."""
    network, jnetwork = load_network()
    initial_abundances = load_initial_abundances(INITIAL_PATH)
    template = build_template(network, initial_abundances)
    species_names = [species.name for species in network.species]
    return network, jnetwork, initial_abundances, template, species_names


def choose_random_tracers(spec: BenchmarkSpec, count: int) -> list[int]:
    """Select random tracer indices from npy source."""
    data = load_benchmark_array(spec)
    capped = min(count, data.shape[1])
    rng = np.random.default_rng()
    return rng.choice(data.shape[1], size=capped, replace=False).tolist()


def resolve_tracer_indices(args: argparse.Namespace, spec: BenchmarkSpec) -> list[int]:
    """Determine tracer indices from CLI."""
    if args.random_count is not None:
        return choose_random_tracers(spec, args.random_count)
    return args.tracers


def compute_fractional_abundances(
    abundances: np.ndarray, network: Network
) -> np.ndarray:
    """Convert species abundances to fractions relative to H nuclei."""
    matrix = build_stoichiometric_matrix(network.species, ELEMENTS)
    elemental = abundances @ matrix.T
    hydrogen_index = ELEMENTS.index("H")
    hydrogen = np.clip(elemental[:, hydrogen_index], 1e-30, None)
    fractions = abundances / hydrogen[:, None]
    return np.clip(fractions, 1e-20, None)


def build_tracer_state(
    template: jnp.ndarray, arrays: TracerArrays
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Assemble solver inputs for a tracer."""
    y0_batch = build_initial_states(template, arrays.densities[None, :])
    y0 = y0_batch[0]
    t_eval = jnp.array(arrays.time_grid, dtype=float)
    densities = jnp.array(arrays.densities, dtype=float)
    temps = jnp.array(arrays.temperatures, dtype=float)
    avs = jnp.array(arrays.avs, dtype=float)
    rad_fields = jnp.array(arrays.rad_fields, dtype=float)
    cr_rates = jnp.ones_like(densities) * 1.6e-17
    return y0, t_eval, densities, temps, avs, rad_fields, cr_rates


def solve_with_core(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    densities: jnp.ndarray,
    temps: jnp.ndarray,
    avs: jnp.ndarray,
    rad_fields: jnp.ndarray,
    cr_rates: jnp.ndarray,
) -> tuple[dx.Solution, float]:
    """Run solver and return solution with elapsed time."""
    start = time()
    solution = solve_network_core(
        jnetwork,
        y0,
        t_eval,
        t_eval,
        densities,
        temps,
        cr_rates,
        rad_fields,
        avs,
        solver_name="kvaerno5",
        atol=1e-14,
        rtol=1e-6,
        max_steps=500000,
    )
    return solution, time() - start


def solve_tracer(
    tracer: TracerDataset,
    assets: tuple[Network, JNetwork, dict[str, float], jnp.ndarray, Sequence[str]],
    output_dir: Path,
) -> float:
    """Solve a single tracer and save results."""
    network, jnetwork, _, template, species_names = assets
    arrays = prepare_tracer_arrays(tracer)
    y0, t_eval, densities, temps, avs, rad_fields, cr_rates = build_tracer_state(
        template, arrays
    )
    solution, elapsed = solve_with_core(
        jnetwork, y0, t_eval, densities, temps, avs, rad_fields, cr_rates
    )
    solved = block_solution_tree(solution)
    fractional = compute_fractional_abundances(np.asarray(solved.ys), network)
    save_tracer_output(
        tracer.tracer_id,
        np.asarray(t_eval),
        fractional,
        arrays.densities,
        arrays.temperatures,
        arrays.avs,
        arrays.rad_fields,
        species_names,
        output_dir,
    )
    return elapsed


def run_tracer_in_subprocess(tracer: TracerDataset, output_dir: Path) -> float:
    """Run a tracer with fresh assets in a subprocess."""
    assets = prepare_assets()
    return solve_tracer(tracer, assets, output_dir)


def execute_batch_vmap(
    jnetwork: JNetwork,
    time_grid: jnp.ndarray,
    y0_batch: jnp.ndarray,
    densities: jnp.ndarray,
    temps: jnp.ndarray,
    rad_fields: jnp.ndarray,
    avs: jnp.ndarray,
) -> tuple[dx.Solution, float]:
    """Execute vectorized solver and report runtime."""
    start = time()
    solutions = run_vmap(
        jnetwork,
        time_grid,
        time_grid,
        y0_batch,
        densities,
        temps,
        rad_fields,
        avs,
        use_jit=False,
    )
    return solutions, time() - start


def persist_batch_outputs(
    solutions: dx.Solution,
    time_grid: np.ndarray,
    densities: np.ndarray,
    temps: np.ndarray,
    avs: np.ndarray,
    rad_fields: np.ndarray,
    tracer_ids: Sequence[int],
    species_names: Sequence[str],
    output_dir: Path,
    network: Network,
) -> None:
    """Save batch solutions to npy outputs."""
    for index, tracer_id in enumerate(tracer_ids):
        fractional = compute_fractional_abundances(
            np.asarray(solutions.ys[index]), network
        )
        save_tracer_output(
            tracer_id,
            time_grid,
            fractional,
            densities[index],
            temps[index],
            avs[index],
            rad_fields[index],
            species_names,
            output_dir,
        )


def run_batch_vmap(tracers: Sequence[TracerDataset], output_dir: Path) -> None:
    """Run vectorized benchmark across tracers."""
    if not tracers:
        return
    new_times, densities, temps, avs, rad_fields = build_batch_parameters(tracers)
    network, jnetwork, _, template, species_names = prepare_assets()
    y0_batch = build_initial_states(template, densities)
    time_grid = jnp.array(new_times, dtype=float)
    dens_jnp, temp_jnp, avs_jnp, rad_jnp = [
        jnp.array(arr, dtype=float) for arr in (densities, temps, avs, rad_fields)
    ]
    solutions, elapsed = execute_batch_vmap(
        jnetwork, time_grid, y0_batch, dens_jnp, temp_jnp, rad_jnp, avs_jnp
    )
    solutions = block_solution_tree(solutions)
    tracer_ids = [tracer.tracer_id for tracer in tracers]
    print(f"VMAP execution time: {elapsed:.2f} seconds")
    summarize_batch(solutions, network, ELEMENTS, tracer_ids)
    persist_batch_outputs(
        solutions,
        np.asarray(time_grid),
        densities,
        temps,
        avs,
        rad_fields,
        tracer_ids,
        species_names,
        output_dir,
        network,
    )


def run_sequential(tracers: Sequence[TracerDataset], output_dir: Path) -> None:
    """Run tracers sequentially with single-run pipeline."""
    if not tracers:
        return
    assets = prepare_assets()
    durations = [solve_tracer(tracer, assets, output_dir) for tracer in tracers]
    print(
        f"Processed {len(tracers)} tracers in {sum(durations):.2f} seconds sequentially"
    )


def run_multiprocessing(
    tracers: Sequence[TracerDataset], workers: int | None, output_dir: Path
) -> None:
    """Run tracers in parallel using joblib."""
    if not tracers:
        return
    worker_count = workers if workers is not None else cpu_count() or 1
    durations = Parallel(n_jobs=worker_count)(
        delayed(run_tracer_in_subprocess)(tracer, output_dir) for tracer in tracers
    )
    print(
        f"Processed {len(tracers)} tracers in {max(durations):.2f} seconds using {worker_count} workers"
    )


def resolve_batch_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve tracer CSV paths from arguments."""
    if args.csv_paths:
        return [Path(path) for path in args.csv_paths]
    return [
        resolve_csv_path(args.base_dir, args.benchmark, args.discretization, tracer)
        for tracer in args.tracers
    ]


def load_tracer_sources(args: argparse.Namespace) -> list[TracerDataset]:
    """Load tracer datasets from arguments."""
    if args.csv_paths:
        return load_tracers(resolve_batch_paths(args))
    spec = get_benchmark_spec(
        args.benchmark, args.npy_path, args.timestep, args.clip, args.discretization
    )
    tracer_indices = resolve_tracer_indices(args, spec)
    return load_tracers_from_npy(spec, tracer_indices)


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Register benchmark metadata arguments."""
    parser.add_argument(
        "--benchmark",
        type=str,
        default="M600_1",
        help="Benchmark prefix from npy source",
    )
    parser.add_argument(
        "--discretization",
        type=int,
        default=1,
        help="Discretization stride for npy source",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=None,
        help="Maximum timesteps to read from npy source",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=None,
        help="Timestep in kyr for npy source",
    )


def add_tracer_arguments(parser: argparse.ArgumentParser) -> None:
    """Register tracer selection arguments."""
    parser.add_argument(
        "--tracers", type=int, nargs="+", default=[7650], help="Tracer indices to run"
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=None,
        help="Random tracer count sampled from npy source",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_TRACER_DIR,
        help="Directory containing tracer CSV files",
    )
    parser.add_argument(
        "--csv-paths", type=Path, nargs="*", help="Explicit CSV paths to run"
    )
    parser.add_argument(
        "--npy-path",
        type=Path,
        default=None,
        help="Path to benchmark npy source file",
    )


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Register execution-related CLI arguments."""
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save tracer outputs",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="sequential",
        choices=["vmap", "sequential", "multiprocessing"],
        help="Execution mode",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker process count for multiprocessing",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch run Carbox tracers")
    add_benchmark_arguments(parser)
    add_tracer_arguments(parser)
    add_execution_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Entrypoint for batch benchmark runs."""
    args = parse_args()
    tracers = load_tracer_sources(args)
    if args.engine == "vmap":
        run_batch_vmap(tracers, args.output_dir)
    elif args.engine == "multiprocessing":
        run_multiprocessing(tracers, args.workers, args.output_dir)
    else:
        run_sequential(tracers, args.output_dir)


if __name__ == "__main__":
    main()
