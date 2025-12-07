#!/usr/bin/env python3
"""Analyze the sparsity of the chemical network Jacobian.

Computes the Jacobian matrix and calculates its sparsity pattern.
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

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import UCLCHEMParser

# Enable JAX 64-bit
jax.config.update("jax_enable_x64", True)


def analyze_sparsity(network_name: str = "gas_phase_only"):
    """Run sparsity analysis on the specified network."""
    print(f"Analyzing Jacobian sparsity for network: {network_name}")

    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    if network_name == "gas_phase_only":
        network_file = data_dir / "uclchem_gas_phase_only.csv"
        ic_file = base_dir / "benchmarks/initial_conditions/gas_phase_only_initial.yaml"
    else:
        raise ValueError(f"Unknown network: {network_name}")

    # Load initial abundances just to get species list and structure
    # We will use random abundances for the actual sparsity check
    with open(ic_file) as f:
        data = yaml.safe_load(f)

    # Parse network
    print(f"Loading reaction network from {network_file}...")
    parser = UCLCHEMParser(
        cloud_radius_pc=1.0,
        number_density=1e4,
    )
    network = parser.parse_network(str(network_file))
    jnetwork = network.get_ode()

    species_names = [s.name for s in network.species]
    n_species = len(species_names)
    print(f"Network has {n_species} species and {len(network.reactions)} reactions.")

    # --- Compute Jacobian ---

    # Create random abundances to ensure all possible reaction pathways are active
    # (avoiding zero derivatives due to zero abundances)
    key = jax.random.PRNGKey(42)
    y_random = jax.random.uniform(key, shape=(n_species,), minval=1e-10, maxval=1e-5)

    # Dummy physical parameters
    temp = 100.0
    cr_rate = 1.3e-17
    fuv = 1.0
    av = 2.0
    t = 0.0

    # Define ODE function for Jacobian
    def ode_func(y):
        return jnetwork(t, y, temp, cr_rate, fuv, av)

    print("Computing Jacobian with JAX...")
    jac_func = jax.jit(jax.jacfwd(ode_func))
    J = jac_func(y_random)

    # --- Analyze Sparsity ---

    # Identify non-zero elements (using a small tolerance for numerical noise if any)
    # Since this is analytical differentiation of a computational graph, exact zeros should be 0.
    # But let's be safe with a tiny epsilon.
    non_zeros = jnp.abs(J) > 0.0
    n_non_zeros = jnp.sum(non_zeros)
    total_elements = n_species * n_species
    sparsity = 1.0 - (n_non_zeros / total_elements)

    print("\nSparsity Analysis Results:")
    print(f"Matrix Size: {n_species} x {n_species}")
    print(f"Total Elements: {total_elements}")
    print(f"Non-zero Elements: {n_non_zeros}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Fill-in: {100 * (n_non_zeros / total_elements):.2%}")

    # --- Visualization ---

    output_dir = Path("stiffness_results")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.spy(non_zeros, markersize=1)
    plt.title(f"Jacobian Sparsity Pattern ({network_name})\nSparsity: {sparsity:.2%}")
    plt.xlabel("Species Index")
    plt.ylabel("Species Index")

    plot_file = output_dir / f"sparsity_pattern_{network_name}.png"
    plt.savefig(plot_file, dpi=300)
    print(f"\nSparsity plot saved to: {plot_file}")

    # Save raw data
    data_file = output_dir / f"sparsity_data_{network_name}.npz"
    np.savez(data_file, jacobian=J, non_zeros=non_zeros)
    print(f"Raw data saved to: {data_file}")

    # Optional: Identify most connected species
    # Row sum = number of species that affect the rate of species i
    # Col sum = number of species whose rates are affected by species j
    row_counts = jnp.sum(non_zeros, axis=1)
    col_counts = jnp.sum(non_zeros, axis=0)

    print("\nMost Connected Species (Column Sums - affects most other species):")
    top_indices = jnp.argsort(col_counts)[::-1][:10]
    for idx in top_indices:
        print(f"  {species_names[idx]}: affects {col_counts[idx]} species")

    print("\nMost Sensitive Species (Row Sums - affected by most other species):")
    top_indices = jnp.argsort(row_counts)[::-1][:10]
    for idx in top_indices:
        print(f"  {species_names[idx]}: affected by {row_counts[idx]} species")


if __name__ == "__main__":
    analyze_sparsity()
