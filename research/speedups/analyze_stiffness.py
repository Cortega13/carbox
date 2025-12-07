#!/usr/bin/env python3
"""Analyze the stiffness of the chemical network ODE system.

Computes Jacobian eigenvalues and identifies limiting species.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add Carbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import NetworkNames
from carbox.solver import solve_network

# Enable JAX 64-bit
jax.config.update("jax_enable_x64", True)


def analyze_stiffness(network_name: str = "gas_phase_only"):
    """Run stiffness analysis on the specified network."""
    # --- 1. Setup & Simulation ---
    print(f"Analyzing stiffness for network: {network_name}")

    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    if network_name == "gas_phase_only":
        network_file = data_dir / "uclchem_gas_phase_only.csv"
        ic_file = base_dir / "benchmarks/initial_conditions/gas_phase_only_initial.yaml"
    else:
        raise ValueError(f"Unknown network: {network_name}")

    # Load initial abundances
    with open(ic_file) as f:
        data = yaml.safe_load(f)
    initial_abundances = data["abundances"]

    # Configure simulation
    config = SimulationConfig(
        number_density=1e4,
        temperature=250.0,
        cr_rate=1e-17,
        fuv_field=1.0,
        visual_extinction=2.0,
        t_start=0.0,
        t_end=1e6,
        n_snapshots=20,  # Enough snapshots to see evolution
        rtol=1e-8,
        atol=1e-14,
        solver="kvaerno5",
        max_steps=8192,
        save_abundances=True,
        initial_abundances=initial_abundances,
        run_name=f"stiffness_{network_name}",
        output_dir="stiffness_results",
    )

    print("Running simulation to generate trajectory...")
    results = run_simulation(
        network_file=str(network_file),
        config=config,
        format_type="uclchem",
        verbose=True,
    )

    solution = results["solution"]
    jnetwork = results["jnetwork"]
    network = results["network"]
    params = config.get_physical_params_jax()

    species_names = [s.name for s in network.species]

    # --- 2. Stiffness Analysis ---
    print("\nComputing Jacobians and Eigenvalues...")

    # Define Jacobian function using JAX
    # jnetwork(t, y, args...) -> dy/dt
    def ode_func(y, t):
        return jnetwork(
            t,
            y,
            params["temperature"],
            params["cr_rate"],
            params["fuv_field"],
            params["visual_extinction"],
        )

    jac_func = jax.jit(jax.jacfwd(ode_func))

    analysis_data = []

    # Analyze each snapshot
    ts = solution.ts
    ys = solution.ys

    for i in range(len(ts)):
        t = ts[i]
        y = ys[i]

        # Compute Jacobian
        J = jac_func(y, t)

        # Compute Eigenvalues
        eigvals, eigvecs = jnp.linalg.eig(J)

        # Filter for stable modes (Re(lambda) < 0)
        # Positive eigenvalues indicate instability (explosive growth),
        # but in chemical kinetics we mostly care about the decay timescales.
        # We take absolute value of real part.
        abs_real_evals = jnp.abs(jnp.real(eigvals))

        # Remove zeros to avoid division by zero (conserved quantities have 0 eigenvalue)
        valid_mask = abs_real_evals > 1e-30
        valid_evals = abs_real_evals[valid_mask]

        if len(valid_evals) == 0:
            continue

        max_eval = jnp.max(valid_evals)
        min_eval = jnp.min(valid_evals)

        stiffness_ratio = max_eval / min_eval
        fastest_timescale = 1.0 / max_eval
        slowest_timescale = 1.0 / min_eval

        # Identify fastest species
        # The eigenvector corresponding to max_eval points to the species involved
        max_eval_idx = jnp.argmax(abs_real_evals)
        relevant_evec = eigvecs[:, max_eval_idx]

        # Find species with largest component in this eigenvector
        fastest_species_idx = jnp.argmax(jnp.abs(relevant_evec))
        fastest_species_name = species_names[fastest_species_idx]
        fastest_species_abund = y[fastest_species_idx]

        print(
            f"Time: {t / 3.154e7:.2e} yr | Stiffness: {stiffness_ratio:.2e} | "
            f"Fastest: {fastest_timescale:.2e}s ({fastest_species_name}, y={fastest_species_abund:.2e})"
        )

        analysis_data.append(
            {
                "time_years": float(t / 3.154e7),
                "stiffness_ratio": float(stiffness_ratio),
                "fastest_timescale_sec": float(fastest_timescale),
                "slowest_timescale_sec": float(slowest_timescale),
                "fastest_species": str(fastest_species_name),
                "fastest_species_abundance": float(fastest_species_abund),
                "fastest_species_idx": int(fastest_species_idx),
            }
        )

    # --- 3. Save Results ---
    df = pd.DataFrame(analysis_data)
    output_file = Path("stiffness_results") / f"stiffness_analysis_{network_name}.csv"
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nAnalysis saved to {output_file}")

    # Analyze tiny species correlation
    tiny_fast_species = df[df["fastest_species_abundance"] < 1e-30]
    if not tiny_fast_species.empty:
        print(
            f"\nWARNING: Found {len(tiny_fast_species)} snapshots where fastest species has y < 1e-30!"
        )
        print(
            tiny_fast_species[
                ["time_years", "fastest_species", "fastest_species_abundance"]
            ]
        )
    else:
        print("\nGood news: Fastest species always had abundance > 1e-30.")


if __name__ == "__main__":
    analyze_stiffness()
