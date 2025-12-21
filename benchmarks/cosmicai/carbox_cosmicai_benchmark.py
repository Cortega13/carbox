"""Run CosmicAI tracer benchmarks with joblib or single-CSV mode."""

# Examples:
# python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --output-dir outputs --random-count=60
# python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --tracer-csv tracer_10.csv --output-dir outputs

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import time

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

# Set JAX flags for CPU optimization (must be set before jax import)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true"
)

# Constants ported from run_carbox_benchmark.py
SPOOFED_INITIAL_TIME = 5.0e6
KYR_TO_YR = 1000.0
YEAR_TO_SEC = 3.15576e7
RADFIELD_FACTOR = 1.7
ELEMENTS = ["H", "HE", "C", "N", "O", "S", "SI", "FE", "MG", "NA", "CL", "P", "F"]
DEFAULT_TRACER_DIR = Path("benchmarks/cosmicai/data/turbulence_tracers_csv")
NETWORK_PATH = Path("network_files/uclchem_gas_phase_only.csv")
INITIAL_PATH = Path("benchmarks/initial_conditions/gas_phase_only_initial.yaml")

# Global cache for worker processes
_WORKER_CACHE = {}


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


def load_network():
    """Load chemical network and compiled ODE system."""
    network = parse_chemical_network(
        str(NETWORK_PATH), format_type=NetworkNames.uclchem
    )
    return network, network.get_ode()


def get_cached_assets():
    """Load and cache network assets for the worker process."""
    if "assets" not in _WORKER_CACHE:
        network, jnetwork = load_network()
        initial_abundances = load_initial_abundances(INITIAL_PATH)

        # Build unit-density abundance template
        config = SimulationConfig(
            number_density=[1.0],
            temperature=[10.0],
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
    """Load tracer datasets from NPY source."""
    if args.tracer_csv:
        return [load_tracer_csv(args.tracer_csv)]

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
    try:
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

        # Initial state
        y0 = template * densities[0]

        # Create SimulationConfig for this tracer
        # Convert time grid to seconds for the solver
        physics_t_seconds = time_grid * YEAR_TO_SEC

        config = SimulationConfig(
            number_density=densities.tolist(),
            temperature=temps.tolist(),
            visual_extinction=avs.tolist(),
            fuv_field=rad_fields.tolist(),
            cr_rate=(jnp.ones_like(densities) * 1.6e-17).tolist(),
            physics_t=physics_t_seconds.tolist(),
            solver="kvaerno5",
            atol=1e-14,
            rtol=1e-6,
            max_steps=100000,
        )

        # Solve
        solution = solve_network(jnetwork, y0, config)

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
    except Exception as e:
        print(f"Tracer {tracer.tracer_id} failed: {e}")
        raise


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
        durations = [process_tracer(tracers[0], args.output_dir)]
    else:
        workers = args.workers or cpu_count() or 1
        print(f"Processing {len(tracers)} tracers with {workers} workers...")
        start_wall_time = time()
        durations = []
        parallel_generator = Parallel(n_jobs=workers, return_as="generator")(
            delayed(process_tracer)(tracer, args.output_dir) for tracer in tracers
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
