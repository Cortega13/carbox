"""Demonstration of time-dependent physical parameters with sinusoidal evolution.

This script shows the new capability to use list inputs for physical parameters,
with linear interpolation between time points during integration.
"""

import datetime

import numpy as np

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

# Hyperparameters
N_INTERPOLATION_POINTS = 50
INTEGRATION_TIME = 1.0e5  # years

# Parameter ranges for sinusoidal evolution
T_MIN = 100.0
T_MAX = 650.0
FUV_MIN = 0.1
FUV_MAX = 10.0
AV_MIN = 0.5
AV_MAX = 5.0
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
