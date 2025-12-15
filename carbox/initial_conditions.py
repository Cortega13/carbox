"""Initial conditions setup for chemical kinetics simulations.

Handles initialization of abundance vectors from configuration.
"""

import jax.numpy as jnp

from .config import SimulationConfig
from .network import Network


def initialize_abundances(network: Network, config: SimulationConfig) -> jnp.ndarray:
    """Initialize abundance vector from configuration.

    Sets up y0 with:
    - Floor abundance for all species
    - Specified initial abundances from config
    - Validates against network species

    Parameters
    ----------
    network : Network
        Reaction network containing species list
    config : SimulationConfig
        Configuration with initial abundances (fractional, x_i)

    Returns:
    -------
    y0 : jnp.ndarray
        Initial abundance vector [# species]
        Values in fractional abundance: x_i = n_i / n_H

    Notes:
    -----
    Abundance Convention:
    - **Input (config.initial_abundances)**: Fractional abundances x_i
      (e.g., from UCLCHEM: x_i = n_i / n_H_nuclei)
    - **Output (y0)**: Fractional abundances x_i

    - All species start at abundance_floor
    - Specified species are set to their fractional values
    - Missing species in config are kept at floor
    - Extra species in config trigger warning but don't fail
    """
    n_species = len(network.species)

    # Initialize all to floor (fractional abundance)
    y0 = jnp.ones(n_species) * config.abundance_floor

    # Set specified abundances (fractional)
    species_names = [s.name for s in network.species]
    for species_name, fractional_abundance in config.initial_abundances.items():
        if species_name in species_names:
            idx = species_names.index(species_name)
            y0 = y0.at[idx].set(fractional_abundance)
        else:
            print(f"Warning: Species '{species_name}' in config not found in network")

    return y0


def validate_elemental_conservation(
    network: Network, y0: jnp.ndarray, elements: list[str] = ["C", "H", "O"]
) -> dict[str, float]:
    """Check initial elemental abundances.

    Parameters
    ----------
    network : Network
        Reaction network
    y0 : jnp.ndarray
        Initial abundance vector
    elements : List[str]
        Elements to check

    Returns:
    -------
    elemental_abundances : Dict[str, float]
        Total abundance per element [cm^-3]

    Notes:
    -----
    Useful for verifying setup matches expected elemental ratios.
    Should be conserved throughout integration (chemistry conserves atoms).
    """
    elemental_content = network.get_elemental_contents(elements=elements + ["charge"])
    total_abundances = elemental_content @ y0

    result = {}
    for i, elem in enumerate(elements + ["charge"]):
        result[elem] = float(total_abundances[i])

    return result


def abundance_summary(network: Network, y0: jnp.ndarray, top_n: int = 10) -> str:
    """Generate human-readable summary of initial abundances.

    Parameters
    ----------
    network : Network
        Reaction network
    y0 : jnp.ndarray
        Initial abundance vector
    top_n : int
        Number of top species to show

    Returns:
    -------
    summary : str
        Formatted summary string
    """
    species_names = [s.name for s in network.species]

    # Sort by abundance
    sorted_indices = jnp.argsort(y0)[::-1]

    lines = ["Initial Abundances Summary", "=" * 40]
    lines.append(f"{'Species':<10} {'Abundance [cm^-3]':>18} {'Fractional':>12}")
    lines.append("-" * 40)

    for i in range(min(top_n, len(y0))):
        idx = sorted_indices[i]
        name = species_names[idx]
        abundance = y0[idx]
        fractional = abundance / jnp.sum(y0)
        lines.append(f"{name:<10} {abundance:>18.3e} {fractional:>12.3e}")

    return "\n".join(lines)
