"""Testing the umist network with sinusoidal temperature evolution.

This script demonstrates multi-phase simulation with temperature varying
sinusoidally over N steps, maintaining efficiency through network reuse.
"""

import datetime

import numpy as np

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.output import save_abundances, save_metadata, save_summary_report
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

N_STEPS = 100
TOTAL_DURATION = 1.0e5
T_MIN = 50.0
T_MAX = 650.0
INITIAL_PHASE_DURATION = 1.0e5
INITIAL_PHASE_TEMPERATURE = 50.0

print("=" * 60)
print("Setup: Loading and Compiling Network")
print("=" * 60)

network_file = "data/uclchem_gas_phase_only.csv"
print(f"Loading reaction network from {network_file}...")
network = parse_chemical_network(network_file, format_type=NetworkNames.uclchem)
print(f"  Loaded {len(network.species)} species")
print(f"  Loaded {len(network.reactions)} reactions")

print("Compiling JAX network...")
jnetwork = network.get_ode()
print("  Network compiled successfully")
print()

print("=" * 60)
print("Generating Sinusoidal Temperature Profile")
print(f"  N_STEPS: {N_STEPS}")
print(f"  Total duration: {TOTAL_DURATION:.1e} years")
print(f"  Temperature range: {T_MIN:.1f}K - {T_MAX:.1f}K")
print("=" * 60)
print()

t_points = np.linspace(0, 2 * np.pi, N_STEPS)
temperatures = T_MIN + (T_MAX - T_MIN) * (0.5 * (1 + np.sin(t_points)))
step_duration = TOTAL_DURATION / N_STEPS

steps = []

steps.append(
    {
        "name": "phase_00_initial",
        "duration": INITIAL_PHASE_DURATION,
        "temperature": INITIAL_PHASE_TEMPERATURE,
    }
)

for i, temp in enumerate(temperatures):
    steps.append(
        {
            "name": f"phase_{i + 1:03d}_sinusoidal",
            "duration": step_duration,
            "temperature": float(temp),
        }
    )

# -----------------------------------------------------------------------------
# Simulation Loop
# -----------------------------------------------------------------------------
previous_abundances_fractional = None
total_duration = 0.0

for i, step in enumerate(steps):
    config = SimulationConfig(
        number_density=1e4,
        temperature=step["temperature"],
        t_end=step["duration"],
        solver="kvaerno5",
        max_steps=500000,
        n_snapshots=3,
        atol=1e-14,
        rtol=1e-8,
        run_name=step["name"],
        initial_abundances=previous_abundances_fractional
        if previous_abundances_fractional
        else {},
    )

    y0 = initialize_abundances(network, config)

    start_time = datetime.datetime.now()
    solution = solve_network(jnetwork, y0, config)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    total_duration += duration
    print(f"  Integration complete in {duration:.2f} seconds")

    if solution.ys is None:
        raise RuntimeError(f"{step['name']} simulation failed (no solution returned).")

    if i < len(steps) - 1:
        final_abundances_absolute = solution.ys[-1]
        previous_abundances_fractional = {}
        final_density = config.get_final_number_density()
        for idx, species in enumerate(network.species):
            fractional = float(final_abundances_absolute[idx]) / final_density
            previous_abundances_fractional[species.name] = fractional

print("=" * 60)
print("All Simulation Steps Complete")
print(f"Total Computation Time: {total_duration:.2f} seconds")
print(f"Average Time per Step: {total_duration / len(steps):.2f} seconds")
print("=" * 60)
