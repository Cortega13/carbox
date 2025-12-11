"""Demonstration of time-dependent physical parameters with sinusoidal evolution.

This script shows the new capability to use list inputs for physical parameters,
with linear interpolation between time points during integration.
Additionally, it verifies solver accuracy by tracking elemental conservation.
"""

import datetime

import numpy as np

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

# Hyperparameters
N_INTERPOLATION_POINTS = 300
INTEGRATION_TIME = 1.0e6  # years

# Parameter ranges for sinusoidal evolution
T_MIN = 100.0
T_MAX = 400.0
FUV_MIN = 0.1
FUV_MAX = 2.0
AV_MIN = 0.5
AV_MAX = 4.0
N_DENSITY_MIN = 1e3
N_DENSITY_MAX = 1e5

print("=" * 60)
print("Time-Dependent Parameters Demo")
print("=" * 60)
print(f"  N_INTERPOLATION_POINTS: {N_INTERPOLATION_POINTS}")
print(f"  INTEGRATION_TIME: {INTEGRATION_TIME:.1e} years")
print("=" * 60)
print()

# Load network
print("Loading and compiling network...")
network_file = "data/uclchem_gas_phase_only.csv"
network = parse_chemical_network(network_file, format_type=NetworkNames.uclchem)
jnetwork = network.get_ode()
print(f"  Loaded {len(network.species)} species, {len(network.reactions)} reactions")
print()

# Generate time points
t_end_list = list(np.linspace(0, INTEGRATION_TIME, N_INTERPOLATION_POINTS + 1)[1:])
print(f"Time intervals: {len(t_end_list)}")
print(f"  First: 0 -> {t_end_list[0]:.1e}")
print(f"  Last: {t_end_list[-2]:.1e} -> {t_end_list[-1]:.1e}")
print()

# Generate sinusoidal parameter evolution
t_points_for_params = np.linspace(0, 2 * np.pi, N_INTERPOLATION_POINTS + 1)

# Temperature: sinusoidal
temperature_list = [
    T_MIN + (T_MAX - T_MIN) * (0.5 * (1 + np.sin(t))) for t in t_points_for_params
]

# FUV field: sinusoidal with phase shift
fuv_list = [
    FUV_MIN + (FUV_MAX - FUV_MIN) * (0.5 * (1 + np.sin(t + np.pi / 2)))
    for t in t_points_for_params
]

# Visual extinction: sinusoidal with different phase
av_list = [
    AV_MIN + (AV_MAX - AV_MIN) * (0.5 * (1 + np.sin(t + np.pi)))
    for t in t_points_for_params
]

# Number density: sinusoidal (affects rates only)
n_density_list = [
    N_DENSITY_MIN
    + (N_DENSITY_MAX - N_DENSITY_MIN) * (0.5 * (1 + np.sin(t + np.pi / 4)))
    for t in t_points_for_params
]

print("Parameter ranges:")
print(f"  Temperature: {min(temperature_list):.1f} - {max(temperature_list):.1f} K")
print(f"  FUV field: {min(fuv_list):.2f} - {max(fuv_list):.2f}")
print(f"  Visual extinction: {min(av_list):.2f} - {max(av_list):.2f} mag")
print(f"  Number density: {min(n_density_list):.1e} - {max(n_density_list):.1e} cm^-3")
print()

# Create config with time-dependent parameters
config = SimulationConfig(
    number_density=n_density_list,
    temperature=temperature_list,
    fuv_field=fuv_list,
    visual_extinction=av_list,
    cr_rate=1e-17,  # Keep constant
    t_start=0.0,
    t_end=t_end_list,
    n_snapshots=3,  # snapshots per interval
    solver="kvaerno5",
    max_steps=500000,
    atol=1e-12,
    rtol=1e-6,
    run_name="complex_parameters",
)

# Initialize abundances
y0 = initialize_abundances(network, config)

# Run simulation
print("Starting integration...")
start_time = datetime.datetime.now()
solution = solve_network(jnetwork, y0, config)
duration = (datetime.datetime.now() - start_time).total_seconds()

print(f"Integration complete in {duration:.2f} seconds")
if solution.stats:
    print(f"Number of steps: {solution.stats['num_steps']}")
if solution.ts is not None:
    print(f"Number of output points: {len(solution.ts)}")
print()

print("=" * 60)
print("Simulation Complete")
print("=" * 60)
print()

# ============================================================================
# CONSERVATION CHECK
# ============================================================================
print("=" * 60)
print("Elemental Abundance Conservation Check")
print("=" * 60)
print()


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
                        continue
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


ELEMENTS = ["H", "He", "C", "N", "O", "S", "Si", "Fe", "Mg", "Na", "Cl", "P", "F"]

if solution.ys is not None and len(solution.ys) > 0:
    assert solution.ts is not None
    assert solution.ys is not None

    # Clip abundances
    clipped_ys = np.array([np.maximum(y, 1e-20) for y in solution.ys])

    # Build stoichiometric matrix
    elemental_matrix = build_stoichiometric_matrix(network.species, ELEMENTS)

    # Calculate elemental abundances
    initial_abundances = elemental_matrix @ clipped_ys[0]
    final_abundances = elemental_matrix @ clipped_ys[-1]

    # Calculate % differences for elements
    print()
    print("ELEMENTAL CONSERVATION:")
    print(f"{'Element':<10} {'Δ%':>12}")
    print("-" * 25)
    for elem_idx, element in enumerate(ELEMENTS):
        if abs(initial_abundances[elem_idx]) > 1e-30:
            pct_change = (
                (final_abundances[elem_idx] - initial_abundances[elem_idx])
                / initial_abundances[elem_idx]
                * 100
            )
            print(f"{element:<10} {pct_change:+12.6f}%")
    print("-" * 25)
    print()

    # Calculate electron-ion balance
    electron_indices = [
        i for i, s in enumerate(network.species) if s.name in ["E-", "ELECTR"]
    ]
    positive_ions = [
        (i, s.name.count("+")) for i, s in enumerate(network.species) if "+" in s.name
    ]

    total_electrons = sum(clipped_ys[-1][i] for i in electron_indices)
    total_positive_ions = sum(charge * clipped_ys[-1][i] for i, charge in positive_ions)

    if total_positive_ions > 1e-30:
        electron_ion_diff = (
            (total_positive_ions - total_electrons) / total_positive_ions * 100
        )
    else:
        electron_ion_diff = 0.0

    print("CHARGE NEUTRALITY:")
    print(f"Electron-Ion Imbalance: {electron_ion_diff:+.6f}%")
    print()


print("=" * 60)
print("Verification Complete")
print("=" * 60)
