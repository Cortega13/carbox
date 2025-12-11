"""Run carbox benchmark with time-spoofed initialization and conservation checks."""

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

# Constants
SPOOFED_INITIAL_TIME = 1.0e6  # years
KYR_TO_YR = 1000.0
RADFIELD_FACTOR = 1.7  # Scaling factor from original script
ELEMENTS = ["H", "HE", "C", "N", "O", "S", "SI", "FE", "MG", "NA", "CL", "P", "F"]


def build_stoichiometric_matrix(network_species, elements):
    """Build stoichiometric matrix mapping species to elemental composition."""
    elemental_matrix = np.zeros((len(elements), len(network_species)))
    for i, species in enumerate(network_species):
        name = species.name
        composition = {}
        if name not in ["E-", "ELECTR"]:
            clean_name = name.replace("+", "").replace("-", "")
            j = 0
            while j < len(clean_name):
                # Try to match 2-letter elements first
                matched = False
                if j + 1 < len(clean_name):
                    two_letter = clean_name[j : j + 2]
                    if two_letter in elements:
                        count = 1
                        if j + 2 < len(clean_name) and clean_name[j + 2].isdigit():
                            count = int(clean_name[j + 2])
                            j += 3
                        else:
                            j += 2
                        composition[two_letter] = composition.get(two_letter, 0) + count
                        matched = True
                        continue

                if not matched:
                    one_letter = clean_name[j]
                    if one_letter in elements:
                        count = 1
                        if j + 1 < len(clean_name) and clean_name[j + 1].isdigit():
                            count = int(clean_name[j + 1])
                            j += 2
                        else:
                            j += 1
                        composition[one_letter] = composition.get(one_letter, 0) + count
                    else:
                        j += 1
        for elem_idx, element in enumerate(elements):
            if element in composition:
                elemental_matrix[elem_idx, i] = composition[element]
    return elemental_matrix


def check_conservation(solution, network, elements):
    """Calculate and print elemental conservation between step 1 and final step."""
    if solution.ys is None or len(solution.ys) < 2:
        print("Simulation failed or too few steps for conservation check.")
        return

    # Build matrix
    elemental_matrix = build_stoichiometric_matrix(network.species, elements)

    # We want to compare the state after initialization (index 1) to final state (index -1)
    # The 'ys' array corresponds to the output time points.
    # Index 0 is t=0 (initial abundances).
    # Index 1 is t=1e6 (after spoofed evolution).
    # Index -1 is final time.

    # Clip negative abundances for physical meaningfulness in check
    y_start = np.maximum(solution.ys[1], 1e-20)
    y_end = np.maximum(solution.ys[-1], 1e-20)

    abund_start = elemental_matrix @ y_start
    abund_end = elemental_matrix @ y_end

    # Normalize by number density to get fractional abundances
    # For now, we just print the raw values which are number densities [cm^-3]
    # To compare with initial fractional abundances, we would need to know the density at each step.
    # However, conservation check is valid on absolute number densities.

    print("\n" + "=" * 60)
    print("Elemental Conservation Check (Post-Init vs Final)")
    print("=" * 60)
    print(f"{'Element':<10} {'Post-Init':>15} {'Final':>15} {'Change %':>12}")
    print("-" * 60)

    for i, element in enumerate(elements):
        start_val = abund_start[i]
        end_val = abund_end[i]

        if start_val > 1e-20:
            pct_change = (end_val - start_val) / start_val * 100
        else:
            pct_change = 0.0

        print(f"{element:<10} {start_val:15.6e} {end_val:15.6e} {pct_change:12.6f}%")
    print("=" * 60 + "\n")


def run_benchmark(csv_path):
    """Run the benchmark simulation."""
    print(f"Loading benchmark data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Extract parameters
    # Note: density in CSV is already number density [cm^-3] per npy_to_csv.py
    densities = df["density"].values
    temperatures = df["gasTemp"].values
    avs = df["av"].values
    # Apply factor to radiation field as in original script
    rad_fields = df["radField"].to_numpy(dtype=float) * RADFIELD_FACTOR

    # Construct Time Array
    # Original time is in kyrs
    orig_times = df["time"].to_numpy(dtype=float)

    # We construct a new time axis where the first interval is stretched to SPOOFED_INITIAL_TIME
    # t[0] = 0
    # t[1] = SPOOFED_INITIAL_TIME
    # t[i] = t[i-1] + (orig_times[i] - orig_times[i-1]) for i > 1

    new_times = np.zeros_like(orig_times)
    new_times[0] = 0.0
    new_times[1] = SPOOFED_INITIAL_TIME

    # Calculate subsequent intervals from original data
    dt_orig = np.diff(orig_times) * KYR_TO_YR
    # Add cumulative sum of original deltas starting from t[1]
    new_times[2:] = new_times[1] + np.cumsum(dt_orig[1:])

    # Prepare configuration lists
    # We pass the full lists. Carbox will interpolate between them.
    # The time points for integration output (t_end) should be the constructed times (excluding t=0)
    t_end_points = list(new_times[1:])

    print(f"Total time points: {len(new_times)}")
    print(f"Simulation duration: {new_times[-1]:.2e} years")
    print(f"Initialization phase: {new_times[1]:.2e} years")

    # Load Network
    print("Loading chemical network...")
    network = parse_chemical_network(
        "data/uclchem_gas_phase_only.csv", format_type=NetworkNames.uclchem
    )
    jnetwork = network.get_ode()

    # Load initial abundances
    with open(
        "benchmarks/gijs_whitepaper/initial_conditions/gas_phase_only_initial.yaml"
    ) as f:
        initial_data = yaml.safe_load(f)
        initial_abundances = initial_data["abundances"]

    # Configure Simulation
    config = SimulationConfig(
        number_density=list(densities),
        temperature=list(temperatures),
        fuv_field=list(rad_fields),
        visual_extinction=list(avs),
        cr_rate=1.6e-17,  # Standard cosmic ray rate
        t_start=0.0,
        t_end=t_end_points,
        initial_abundances=initial_abundances,
        solver="kvaerno5",
        atol=1e-14,  # Tight tolerances for conservation
        rtol=1e-6,
        max_steps=500000,
        n_snapshots=3,
    )

    print("Initial Abundances: ", config.initial_abundances)

    # Initialize Abundances
    y0 = initialize_abundances(network, config)

    # Run Simulation
    print("Starting simulation...")
    start_time = time.time()
    solution = solve_network(jnetwork, y0, config)
    elapsed = time.time() - start_time
    print(f"Simulation complete in {elapsed:.2f} seconds.")

    # Check Conservation
    check_conservation(solution, network, ELEMENTS)

    # Print fractional abundances at the end
    if solution.ys is not None:
        print("\nFinal Fractional Elemental Abundances (relative to H nuclei):")
        elemental_matrix = build_stoichiometric_matrix(network.species, ELEMENTS)
        final_abundances = np.maximum(solution.ys[-1], 1e-30)
        elemental_abundances = elemental_matrix @ final_abundances

        # Calculate total H nuclei density
        # H nuclei = H + 2*H2 + 3*H3 + ... (all hydrogen containing species)
        # This is equivalent to the 'H' row in elemental_matrix @ final_abundances
        h_idx = ELEMENTS.index("H")
        total_h_nuclei = elemental_abundances[h_idx]

        for i, element in enumerate(ELEMENTS):
            frac = elemental_abundances[i] / total_h_nuclei
            print(f"{element}: {frac:.4e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Carbox Benchmark")
    # parser.add_argument("csv_path", type=str, help="Path to the input tracer CSV file")
    args = parser.parse_args()
    args.csv_path = "/workspace/carbox/benchmarks/cosmicai/data/turbulence_tracers_csv/M600_1_1_Tracer_7650.csv"

    run_benchmark(args.csv_path)
