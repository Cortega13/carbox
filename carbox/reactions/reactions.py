"""Defines schemas for reactions."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

REACTION_SKIP_LIST = ["CRPHOT", "CRP", "PHOTON"]


class JReactionRateTerm(eqx.Module):
    """Base class for JAX-compatible reaction rate terms.

    All subclasses must implement __call__ with signature:
        __call__(self, temperature, cr_rate, uv_field, visual_extinction, abundance_vector)

    This ensures consistent signatures for JIT compilation.
    Reactions that don't need abundance_vector can simply ignore it.
    """

    @abstractmethod
    def __call__(  # noqa
        self,
        temperature: Array,
        cr_rate: Array,
        fuv_rate: Array,
        visual_extinction: Array,
        abundances: Array,
    ) -> jnp.ndarray:
        raise NotImplementedError


def valid_species_check(species: str | float) -> bool:
    """Check if the species are valid, i.e., not in the skip list."""
    valid = False
    if isinstance(species, float):
        valid = bool(~np.isnan(species))
    elif isinstance(species, str):
        valid = species not in REACTION_SKIP_LIST
    return valid


@dataclass
class Reaction:
    """Dataclass for invidiual reactions."""

    reaction_type: str
    reactants: list[str]
    products: list[str]
    molecularity: int

    def __init__(self, reaction_type: str, reactants: list[str], products: list[str]):  # noqa
        self.reactants = [r for r in reactants if valid_species_check(r)]
        self.products = [p for p in products if valid_species_check(p)]
        self.reaction_type = reaction_type
        self.molecularity = np.array(self.reactants).shape[-1]

    def __str__(self) -> str:
        """String representation of the reaction."""
        return f"{self.reactants} -> {self.products}"

    def __repr__(self) -> str:
        """String representation of the dataclass."""
        return f"Reaction({self.reaction_type}, {self.reactants}, {self.products})\n"

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        # Abstract function to implement in subclasses
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> JReactionRateTerm:  # noqa
        return self._reaction_rate_factory()
