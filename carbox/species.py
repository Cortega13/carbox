"""Species definitions."""

from dataclasses import dataclass

import equinox as eqx


class JSpecies(eqx.Module):
    """Jax jit compiled Species dataclass."""


@dataclass
class Species:
    """Dataclass for a single species."""

    name: str
    mass: float
    charge: int = 0

    def __str__(self) -> str:  # noqa
        return f"{self.name} ({self.mass}, {self.charge})"

    def __repr__(self) -> str:  # noqa
        return f"Species({self.name}, {self.mass}, {self.charge})"

    def __eq__(self, other: object) -> bool:  # noqa
        if isinstance(other, Species):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        """Hashes the name of this species."""
        return hash(self.name)
