"""Output management for simulation results.

Handles saving of abundance trajectories, derivatives, rates, and metadata.
"""

import json
from datetime import datetime
from pathlib import Path

import diffrax as dx
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import SimulationConfig
from .network import Network
from .solver import SPY, compute_dnh_dt_slopes


def prepare_output_directory(config: SimulationConfig) -> Path:
    """Create output directory if it doesn't exist.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with output_dir

    Returns:
    -------
    output_path : Path
        Path to output directory
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_abundances(
    solution: dx.Solution,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save abundance time series to CSV.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network (for species names)
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file

    Notes:
    -----
    Output format:
    - Columns: time, physical parameters, then species abundances
    - Values: fractional abundances relative to H nuclei (x_i = n_i / n_{H,nuclei})
    - n_{H,nuclei} = 2*n(H2) + n(H)
    - Physical parameters repeated for each row (for easy filtering/grouping)
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    number_density_grid = jnp.array(config.number_density_grid)
    temperature_grid = jnp.array(config.temperature_grid)
    cr_rate_grid = jnp.array(config.cr_rate_grid)
    fuv_field_grid = jnp.array(config.fuv_field_grid)
    visual_extinction_grid = jnp.array(config.compute_visual_extinction_grid())

    h2_idx = species_names.index("H2") if "H2" in species_names else None
    h_idx = species_names.index("H") if "H" in species_names else None
    if solution.ys is not None and (h2_idx is not None or h_idx is not None):
        n_h_nuclei = jnp.zeros_like(solution.ys[:, 0])
        if h2_idx is not None:
            n_h_nuclei = n_h_nuclei + 2.0 * solution.ys[:, h2_idx]
        if h_idx is not None:
            n_h_nuclei = n_h_nuclei + solution.ys[:, h_idx]
    else:
        n_h_nuclei = number_density_grid

    # Create DataFrame with time and physical parameter columns
    data = {
        "time_seconds": solution.ts,
        "time_years": solution.ts / SPY,  # type:ignore
        "number_density": number_density_grid,
        "temperature": temperature_grid,
        "cr_rate": cr_rate_grid,
        "fuv_field": fuv_field_grid,
        "visual_extinction": visual_extinction_grid,
    }

    # Add species fractional abundances (relative to H nuclei)
    if solution.ys is not None:
        species_data = {
            name: solution.ys[:, i] / n_h_nuclei for i, name in enumerate(species_names)
        }
        df = pd.DataFrame({**data, **species_data})
    else:
        df = pd.DataFrame(data)

    filepath = output_path / f"{config.run_name}_abundances.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved abundances to: {filepath}")
    return filepath


def save_derivatives(
    derivatives: jnp.ndarray,
    times: jnp.ndarray,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save time derivatives to CSV.

    Parameters
    ----------
    derivatives : jnp.ndarray
        Time derivatives [n_times, n_species]
    times : jnp.ndarray
        Time array [s]
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    # Create DataFrame with time and physical parameter columns
    data = {
        "time_seconds": times,
        "time_years": times / SPY,
        "number_density": jnp.array(config.number_density_grid),
        "temperature": jnp.array(config.temperature_grid),
        "cr_rate": jnp.array(config.cr_rate_grid),
        "fuv_field": jnp.array(config.fuv_field_grid),
        "visual_extinction": jnp.array(config.compute_visual_extinction_grid()),
    }

    # Add derivatives
    derivatives_data = {
        f"d{name}_dt": derivatives[:, i] for i, name in enumerate(species_names)
    }
    df = pd.DataFrame({**data, **derivatives_data})

    filepath = output_path / f"{config.run_name}_derivatives.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved derivatives to: {filepath}")
    return filepath


def save_reaction_rates(
    rates: jnp.ndarray,
    times: jnp.ndarray,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save reaction rates to CSV.

    Parameters
    ----------
    rates : jnp.ndarray
        Reaction rates [n_times, n_reactions]
    times : jnp.ndarray
        Time array [s]
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    # Use reaction type as column names (could be more descriptive)
    reaction_names = [f"{r.reaction_type}_{i}" for i, r in enumerate(network.reactions)]

    # Create DataFrame with time and physical parameter columns
    data = {
        "time_seconds": times,
        "time_years": times / SPY,
        "number_density": jnp.array(config.number_density_grid),
        "temperature": jnp.array(config.temperature_grid),
        "cr_rate": jnp.array(config.cr_rate_grid),
        "fuv_field": jnp.array(config.fuv_field_grid),
        "visual_extinction": jnp.array(config.compute_visual_extinction_grid()),
    }

    # Add reaction rates
    rates_data = {name: rates[:, i] for i, name in enumerate(reaction_names)}
    df = pd.DataFrame({**data, **rates_data})

    filepath = output_path / f"{config.run_name}_rates.csv"
    df.to_csv(filepath, index=False)

    print(f"Saved reaction rates to: {filepath}")
    return filepath


def save_metadata(
    config: SimulationConfig,
    network: Network,
    solution: dx.Solution,
    computation_time: float | None = None,
) -> Path:
    """Save simulation metadata to JSON.

    Parameters
    ----------
    config : SimulationConfig
        Configuration used
    network : Network
        Reaction network
    solution : dx.Solution
        Integration solution (for stats)
    computation_time : float, optional
        Wall-clock time [s]

    Returns:
    -------
    filepath : Path
        Path to saved file

    Notes:
    -----
    Metadata includes:
    - Configuration parameters
    - Network statistics (# species, # reactions)
    - Solver statistics
    - Timestamp and computation time
    """
    output_path = prepare_output_directory(config)

    # Derived physical parameters for reproducibility.
    t_grid_sec = jnp.array(config.time_grid_years) * SPY
    nH_grid = jnp.array(config.number_density_grid)
    dnh_dt_slopes = compute_dnh_dt_slopes(t_grid_sec, nH_grid)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_name": config.run_name,
        "computation_time_seconds": computation_time,
        # Configuration
        "config": {
            "physical_params": {
                "number_density_grid": config.number_density_grid,
                # Interval slopes dn_h/dt for each interval [t_i, t_{i+1}).
                "dnh_dt_slopes": [float(v) for v in jnp.asarray(dnh_dt_slopes)],
                "temperature_grid": config.temperature_grid,
                "cr_rate_grid": config.cr_rate_grid,
                "fuv_field_grid": config.fuv_field_grid,
                "visual_extinction_grid": config.compute_visual_extinction_grid(),
                "visual_extinction_config": config.visual_extinction_grid,
                "use_self_consistent_av": config.use_self_consistent_av,
                "cloud_radius_pc": config.cloud_radius_pc,
                "base_av": config.base_av,
            },
            "integration": {
                "time_grid_years": config.time_grid_years,
                "solver": config.solver,
                "atol": config.atol,
                "rtol": config.rtol,
                "max_steps": config.max_steps,
            },
            "initial_abundances": config.initial_abundances,
        },
        # Network info
        "network": {
            "n_species": len(network.species),
            "n_reactions": len(network.reactions),
            "species_names": [s.name for s in network.species],
            "use_sparse": network.use_sparse,
            "vectorize_reactions": network.vectorize_reactions,
        },
        # Solver statistics
        "solver_stats": {
            "num_steps": int(solution.stats["num_steps"]),
            "num_accepted_steps": int(solution.stats["num_accepted_steps"]),
            "num_rejected_steps": int(solution.stats["num_rejected_steps"]),
        }
        if hasattr(solution, "stats")
        else {},
    }

    filepath = output_path / f"{config.run_name}_metadata.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {filepath}")
    return filepath


def save_summary_report(
    solution: dx.Solution,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save human-readable summary report.

    Parameters
    ----------
    solution : dx.Solution
        Integration solution
    network : Network
        Reaction network
    config : SimulationConfig
        Configuration

    Returns:
    -------
    filepath : Path
        Path to saved file
    """
    output_path = prepare_output_directory(config)

    species_names = [s.name for s in network.species]

    lines = []
    lines.append("=" * 60)
    lines.append(f"Carbox Simulation Summary: {config.run_name}")
    lines.append("=" * 60)
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("Physical Parameters:")
    density_min = min(config.number_density_grid)
    density_max = max(config.number_density_grid)
    temp_min = min(config.temperature_grid)
    temp_max = max(config.temperature_grid)
    cr_min = min(config.cr_rate_grid)
    cr_max = max(config.cr_rate_grid)
    fuv_min = min(config.fuv_field_grid)
    fuv_max = max(config.fuv_field_grid)
    av_grid = config.compute_visual_extinction_grid()
    av_min = min(av_grid)
    av_max = max(av_grid)
    lines.append(f"  Total density: {density_min:.2e} - {density_max:.2e} cm^-3")
    lines.append(f"  Temperature: {temp_min:.1f} - {temp_max:.1f} K")
    lines.append(f"  CR ionization rate: {cr_min:.2e} - {cr_max:.2e} s^-1")
    lines.append(f"  FUV field: {fuv_min:.2e} - {fuv_max:.2e} Draine")
    lines.append(f"  Visual extinction: {av_min:.1f} - {av_max:.1f} mag")
    lines.append("")

    lines.append("Integration:")
    lines.append(
        f"  Time range: {config.time_grid_years[0]:.2e} - {config.time_grid_years[-1]:.2e} years"
    )
    lines.append(f"  Snapshots: {len(config.time_grid_years)}")
    lines.append(f"  Solver: {config.solver}")
    lines.append(f"  Tolerances: atol={config.atol:.2e}, rtol={config.rtol:.2e}")
    lines.append("")

    lines.append("Network:")
    lines.append(f"  Species: {len(network.species)}")
    lines.append(f"  Reactions: {len(network.reactions)}")
    lines.append("")

    if hasattr(solution, "stats"):
        lines.append("Solver Statistics:")
        lines.append(f"  Total steps: {solution.stats['num_steps']}")
        lines.append(f"  Accepted: {solution.stats['num_accepted_steps']}")
        lines.append(f"  Rejected: {solution.stats['num_rejected_steps']}")
        lines.append("")

    # Final abundances (top 10)
    lines.append("Final Abundances (top 10):")
    final_abundances = solution.ys[-1]  # type:ignore
    sorted_indices = jnp.argsort(final_abundances)[::-1]
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        lines.append(f"  {species_names[idx]:<10} {final_abundances[idx]:.3e} cm^-3")

    lines.append("=" * 60)

    report = "\n".join(lines)

    filepath = output_path / f"{config.run_name}_summary.txt"
    with open(filepath, "w") as f:
        f.write(report)

    print(f"Saved summary to: {filepath}")
    return filepath


def _select_impactful_species(
    species_names: list[str],
    abundances: jnp.ndarray,
    max_species: int = 8,
) -> list[str]:
    """Pick impactful species for MHD-oriented evolution plots.

    Prefer key ionization/coupling agents plus common cooling/chemistry tracers.
    """
    priority = [
        "H2",
        "CO",
        "C",
        "C+",
        "O",
        "H3+",
        "HCO+",
        "H3O+",
        "E-",
        "MG+",
        "H2O",
        "OH",
    ]
    chosen: list[str] = []
    for name in priority:
        if name in species_names:
            chosen.append(name)
        if len(chosen) >= max_species:
            return chosen

    # Fill remaining with the most abundant species at final time.
    final_abundances = np.asarray(abundances)[-1]
    sorted_indices = np.argsort(final_abundances)[::-1]
    for idx in sorted_indices:
        name = species_names[int(idx)]
        if name not in chosen:
            chosen.append(name)
        if len(chosen) >= max_species:
            break

    return chosen


def save_evolution_plot(
    solution: dx.Solution,
    network: Network,
    config: SimulationConfig,
) -> Path:
    """Save a two-panel plot of physical parameters and key species evolution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = prepare_output_directory(config)

    time_years = np.asarray(solution.ts) / SPY  # type:ignore
    number_density = np.asarray(config.number_density_grid)
    temperature = np.asarray(config.temperature_grid)
    cr_rate = np.asarray(config.cr_rate_grid)
    fuv_field = np.asarray(config.fuv_field_grid)
    visual_extinction = np.asarray(config.compute_visual_extinction_grid())

    species_names = [s.name for s in network.species]
    h2_idx = species_names.index("H2") if "H2" in species_names else None
    h_idx = species_names.index("H") if "H" in species_names else None
    if solution.ys is None:
        raise ValueError("Solution missing abundances for plotting.")
    if h2_idx is not None or h_idx is not None:
        n_h_nuclei = np.zeros_like(solution.ys[:, 0])
        if h2_idx is not None:
            n_h_nuclei = n_h_nuclei + 2.0 * solution.ys[:, h2_idx]
        if h_idx is not None:
            n_h_nuclei = n_h_nuclei + solution.ys[:, h_idx]
    else:
        n_h_nuclei = number_density
    abundances = np.asarray(solution.ys) / n_h_nuclei[:, None]

    key_species = _select_impactful_species(species_names, abundances)
    key_indices = [species_names.index(name) for name in key_species]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    # Top: physical parameter evolution
    av_plot = np.clip(visual_extinction, 1e-6, None)
    ax_top.plot(time_years, number_density, label="n_H [cm^-3]")
    ax_top.plot(time_years, temperature, label="T [K]")
    ax_top.plot(time_years, cr_rate, label="CR rate [s^-1]")
    ax_top.plot(time_years, fuv_field, label="FUV [Draine]")
    ax_top.plot(time_years, av_plot, label="A_V [mag]")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Physical Parameters (log)")
    ax_top.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_top.legend(loc="best")

    # Bottom: abundances of key species
    for name, idx in zip(key_species, key_indices):
        ax_bottom.plot(time_years, abundances[:, idx], label=name)
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlabel("Time [years]")
    ax_bottom.set_ylabel("Fractional Abundance (log)")
    ax_bottom.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_bottom.legend(loc="best", ncol=2)

    if time_years.min() > 0:
        ax_top.set_xscale("log")
        ax_bottom.set_xscale("log")

    filepath = output_path / f"{config.run_name}_evolution.png"
    fig.savefig(filepath, dpi=200)
    plt.close(fig)

    print(f"Saved evolution plot to: {filepath}")
    return filepath
