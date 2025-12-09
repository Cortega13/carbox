"""Testing the umist network with heating phase.

This script demonstrates how to run a multi-phase simulation efficiently
by compiling the chemical network once and reusing it.
"""

import datetime

from carbox.config import SimulationConfig
from carbox.initial_conditions import initialize_abundances
from carbox.output import save_abundances, save_metadata, save_summary_report
from carbox.parsers import NetworkNames, parse_chemical_network
from carbox.solver import solve_network

# -----------------------------------------------------------------------------
# Setup: Load and Compile Network (Once)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Simulation Loop
# -----------------------------------------------------------------------------
# Define simulation steps
# Step 1: 100k years at 50K
# Steps 2-10: 1000 years at 100K (Heating phase)
steps = []

# Initial phase
steps.append(
    {
        "name": "phase_01_initial",
        "duration": 1.0e5,  # 100,000 years
        "temperature": 50.0,
    }
)

# Heating phases (9 steps of 1000 years each)
current_temp = 50.0
for i in range(20):
    current_temp += 30.0
    steps.append(
        {
            "name": f"phase_{i + 2:02d}_heating",
            "duration": 1000.0,  # 1,000 years
            "temperature": current_temp,
        }
    )

previous_abundances_fractional = None
total_duration = 0.0

for i, step in enumerate(steps):
    print("=" * 60)
    print(f"Running {step['name']}")
    print(f"  Duration: {step['duration']:.1e} years")
    print(f"  Temperature: {step['temperature']:.1f} K")
    print("=" * 60)

    # Configure simulation
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

    # Initialize abundances
    # If previous_abundances_fractional is provided in config, initialize_abundances uses it
    print("Initializing abundances...")
    y0 = initialize_abundances(network, config)

    # Solve
    print(f"Solving ODE system with {config.solver}...")
    start_time = datetime.datetime.now()
    solution = solve_network(jnetwork, y0, config)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    total_duration += duration
    print(f"  Integration complete in {duration:.2f} seconds")

    if solution.ys is None:
        raise RuntimeError(f"{step['name']} simulation failed (no solution returned).")

    # Save results
    print(f"Saving results for {step['name']}...")
    save_abundances(solution, network, config)
    save_metadata(config, network, solution, duration)
    save_summary_report(solution, network, config)
    print()

    # Extract final abundances for next step
    if i < len(steps) - 1:
        print("Extracting final abundances for next phase...")
        final_abundances_absolute = solution.ys[-1]
        previous_abundances_fractional = {}
        for idx, species in enumerate(network.species):
            fractional = float(final_abundances_absolute[idx]) / config.number_density
            previous_abundances_fractional[species.name] = fractional
        print("  Abundances extracted.")
        print()

print("=" * 60)
print("All Simulation Steps Complete")
print(f"Total Computation Time: {total_duration:.2f} seconds")
print("=" * 60)
