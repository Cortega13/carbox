"""Run Carbox benchmark for CosmicAI tracer outputs."""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

SPOOFED_INITIAL_TIME = 1.0e6
KYR_TO_YR = 1000.0
RADFIELD_FACTOR = 1.7
ELEMENTS = ["H", "HE", "C", "N", "O", "S", "SI", "FE", "MG", "NA", "CL", "P", "F"]
DEFAULT_TRACER_DIR = Path("benchmarks/cosmicai/data/turbulence_tracers_csv")
NETWORK_PATH = Path("data/uclchem_gas_phase_only.csv")
INITIAL_PATH = Path(
    "benchmarks/gijs_whitepaper/initial_conditions/gas_phase_only_initial.yaml"
)


def clean_species_name(name: str) -> str:
    """Strip charge indicators from a species name."""
    return name.replace("+", "").replace("-", "")


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
        composition = parse_element_counts(clean_species_name(name), elements)
        for row, element in enumerate(elements):
            if element in composition:
                matrix[row, column] = composition[element]
    return matrix


def compute_conservation_vectors(
    solution, network, elements: Sequence[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return elemental abundance vectors at post-init and final states."""
    if solution.ys is None or len(solution.ys) < 2:
        return None
    matrix = build_stoichiometric_matrix(network.species, elements)
    start = np.maximum(solution.ys[1], 1e-20)
    end = np.maximum(solution.ys[-1], 1e-20)
    return matrix @ start, matrix @ end


def print_conservation(
    abund_start: np.ndarray, abund_end: np.ndarray, elements: Sequence[str]
) -> None:
    """Print elemental conservation summary."""
    print("\n" + "=" * 60)
    print("Elemental Conservation Check (Post-Init vs Final)")
    print("=" * 60)
    print(f"{'Element':<10} {'Post-Init':>15} {'Final':>15} {'Change %':>12}")
    print("-" * 60)
    for index, element in enumerate(elements):
        start_val = abund_start[index]
        end_val = abund_end[index]
        pct_change = (
            (end_val - start_val) / start_val * 100 if start_val > 1e-20 else 0.0
        )
        print(f"{element:<10} {start_val:15.6e} {end_val:15.6e} {pct_change:12.6f}%")
    print("=" * 60 + "\n")


def check_conservation(solution, network, elements: Sequence[str]) -> None:
    """Calculate and print elemental conservation."""
    vectors = compute_conservation_vectors(solution, network, elements)
    if vectors is None:
        print("Simulation failed or too few steps for conservation check.")
        return
    abund_start, abund_end = vectors
    print_conservation(abund_start, abund_end, elements)


def build_time_axis(orig_times: np.ndarray) -> np.ndarray:
    """Return stretched time axis with spoofed initialization."""
    new_times = np.zeros_like(orig_times, dtype=float)
    new_times[0] = 0.0
    new_times[1] = SPOOFED_INITIAL_TIME
    deltas = np.diff(orig_times) * KYR_TO_YR
    if len(new_times) > 2:
        new_times[2:] = new_times[1] + np.cumsum(deltas[1:])
    return new_times


def load_tracer_csv(path: Path) -> pd.DataFrame:
    """Load tracer CSV data."""
    return pd.read_csv(path)


def resolve_csv_path(
    base_dir: Path, benchmark: str, discretization: int, tracer: int
) -> Path:
    """Build tracer CSV path using npy_to_csv naming."""
    filename = f"{benchmark}_{discretization}_Tracer_{tracer}.csv"
    return base_dir / filename


def load_initial_abundances(path: Path) -> dict[str, float]:
    """Load initial abundances from YAML."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    return data["abundances"]


def create_config(
    df: pd.DataFrame, new_times: np.ndarray, initial_abundances: dict[str, float]
) -> SimulationConfig:
    """Build simulation configuration from tracer data."""
    rad_fields = df["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR
    return SimulationConfig(
        number_density=list(df["density"].values),
        temperature=list(df["gasTemp"].values),
        fuv_field=list(rad_fields),
        visual_extinction=list(df["av"].values),
        cr_rate=1.6e-17,
        t_start=0.0,
        t_end=list(new_times[1:]),
        initial_abundances=initial_abundances,
        solver="kvaerno5",
        atol=1e-14,
        rtol=1e-6,
        max_steps=500000,
        n_snapshots=3,
    )


def load_network():
    """Load chemical network and compiled ODE system."""
    network = parse_chemical_network(
        str(NETWORK_PATH), format_type=NetworkNames.uclchem
    )
    return network, network.get_ode()


def simulate(network, jnetwork, config: SimulationConfig):
    """Run the solver and return the solution with runtime."""
    y0 = initialize_abundances(network, config)
    start_time = time.time()
    solution = solve_network(jnetwork, y0, config)
    return solution, time.time() - start_time


def print_final_fractionals(solution, network, elements: Sequence[str]) -> None:
    """Print final fractional elemental abundances."""
    if solution.ys is None:
        return
    matrix = build_stoichiometric_matrix(network.species, elements)
    final_abundances = np.maximum(solution.ys[-1], 1e-30)
    elemental_abundances = matrix @ final_abundances
    hydrogen_index = elements.index("H")
    total_h = elemental_abundances[hydrogen_index]
    print("\nFinal Fractional Elemental Abundances (relative to H nuclei):")
    for index, element in enumerate(elements):
        fraction = elemental_abundances[index] / total_h
        print(f"{element}: {fraction:.4e}")


def run_benchmark(csv_path: Path) -> None:
    """Run the benchmark simulation."""
    print(f"Loading benchmark data from: {csv_path}")
    df = load_tracer_csv(csv_path)
    new_times = build_time_axis(df["time"].to_numpy(dtype=float))
    print(f"Total time points: {len(new_times)}")
    print(f"Simulation duration: {new_times[-1]:.2e} years")
    print(f"Initialization phase: {new_times[1]:.2e} years")
    network, jnetwork = load_network()
    initial_abundances = load_initial_abundances(INITIAL_PATH)
    config = create_config(df, new_times, initial_abundances)
    print("Initial Abundances:", config.initial_abundances)
    solution, elapsed = simulate(network, jnetwork, config)
    print(f"Simulation complete in {elapsed:.2f} seconds.")
    check_conservation(solution, network, ELEMENTS)
    print_final_fractionals(solution, network, ELEMENTS)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Carbox Benchmark")
    parser.add_argument("--csv-path", type=Path, help="Path to a tracer CSV file")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="M600_1",
        help="Benchmark prefix from npy_to_csv outputs",
    )
    parser.add_argument(
        "--tracer", type=int, default=7650, help="Tracer index from npy_to_csv outputs"
    )
    parser.add_argument(
        "--discretization",
        type=int,
        default=1,
        help="Discretization value from npy_to_csv outputs",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_TRACER_DIR,
        help="Directory containing tracer CSV files",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark entrypoint."""
    args = parse_args()
    csv_path = args.csv_path or resolve_csv_path(
        args.base_dir, args.benchmark, args.discretization, args.tracer
    )
    run_benchmark(csv_path)


if __name__ == "__main__":
    main()
