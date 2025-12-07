#!/usr/bin/env python3
"""Estimate speedups from QSSA by analyzing eigenvalue spectrum.

Computes the Jacobian eigenvalues and determines the stiffness reduction
achieved by removing the fastest modes (QSSA candidates).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.parsers import UCLCHEMParser

# Enable JAX 64-bit
jax.config.update("jax_enable_x64", True)


def analyze_spectrum(network_name: str = "gas_phase_only"):
    print(f"Analyzing QSSA potential for: {network_name}")

    # --- 1. Load Network ---
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    network_file = data_dir / "uclchem_gas_phase_only.csv"
    ic_file = (
        base_dir
        / "benchmarks/gijs_whitepaper/initial_conditions/gas_phase_only_initial.yaml"
    )

    parser = UCLCHEMParser(cloud_radius_pc=1.0, number_density=1e4)
    network = parser.parse_network(str(network_file))
    jnetwork = network.get_ode()

    # Load initial abundances
    with open(ic_file) as f:
        data = yaml.safe_load(f)
    initial_abundances = data["abundances"]
    y0 = jnp.array([initial_abundances.get(sp.name, 1e-30) for sp in network.species])

    # Define ODE function
    def ode_func(y):
        # Using typical parameters
        return jnetwork(0.0, y, 100.0, 1.3e-17, 1.0, 2.0)

    print("Computing Jacobian and Eigenvalues...")
    jac_func = jax.jit(jax.jacfwd(ode_func))
    J = jac_func(y0)

    # Compute Eigenvalues
    eigvals = np.linalg.eigvals(J)

    # Filter and Sort
    # We care about the magnitude of the real part (decay rates)
    abs_real_evals = np.abs(np.real(eigvals))
    # Filter small values (conserved quantities / zeros)
    valid_evals = abs_real_evals[abs_real_evals > 1e-30]
    sorted_evals = np.sort(valid_evals)[::-1]  # Descending order (Fastest to Slowest)

    print(f"\nTotal valid modes: {len(valid_evals)}")
    print(f"Fastest timescale (tau_0): {1.0 / sorted_evals[0]:.2e} s")
    print(f"Slowest timescale: {1.0 / sorted_evals[-1]:.2e} s")
    print(f"Initial Stiffness Ratio: {sorted_evals[0] / sorted_evals[-1]:.2e}")

    # --- 2. QSSA Analysis ---
    print("\n--- QSSA Candidate Analysis ---")

    # Check what happens if we remove top k species
    k_values = [1, 2, 5, 10, 20, 50]

    results = []

    print(
        f"{'Removed':<8} | {'New Fastest (s)':<15} | {'New Stiffness':<15} | {'Max Explicit Step (yr)':<20}"
    )
    print("-" * 65)

    for k in k_values:
        if k >= len(sorted_evals):
            break

        new_fastest_rate = sorted_evals[k]
        new_fastest_tau = 1.0 / new_fastest_rate
        new_stiffness = new_fastest_rate / sorted_evals[-1]

        # Explicit stability limit roughly proportional to tau
        # Assuming we want step ~ tau (very conservative) or slightly larger
        max_step_yr = new_fastest_tau / 3.154e7

        print(
            f"{k:<8} | {new_fastest_tau:<15.2e} | {new_stiffness:<15.2e} | {max_step_yr:<20.2e}"
        )

        results.append({"k": k, "tau": new_fastest_tau, "stiffness": new_stiffness})

    # --- 3. Conclusion ---
    print("\n--- Conclusion ---")

    # Find k required for 100 year step
    target_step_yr = 100.0
    target_tau = target_step_yr * 3.154e7

    required_k = -1
    for i, rate in enumerate(sorted_evals):
        tau = 1.0 / rate
        if tau > target_tau:
            required_k = i
            break

    if required_k != -1:
        print(f"To achieve an explicit time step of > {target_step_yr} years,")
        print(f"you must QSSA the top {required_k} species.")
        print(
            f"Species count would reduce from {len(network.species)} to {len(network.species) - required_k}."
        )
    else:
        print(
            f"Even removing all species doesn't reach {target_step_yr} years (impossible)."
        )

    # Plot Spectrum
    output_dir = Path("research/speedups")
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 6))
    plt.semilogy(1.0 / sorted_evals, "o-", markersize=3)
    plt.axhline(y=100 * 3.154e7, color="r", linestyle="--", label="100 Years")
    plt.xlabel("Mode Index (Sorted by Speed)")
    plt.ylabel("Timescale (seconds)")
    plt.title("Chemical Network Timescale Spectrum")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plot_file = output_dir / "timescale_spectrum.png"
    plt.savefig(plot_file)
    print(f"\nSpectrum plot saved to: {plot_file}")


if __name__ == "__main__":
    analyze_spectrum()
