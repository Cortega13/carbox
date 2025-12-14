"""Umist specific reactions."""

import jax.numpy as jnp
from jax import Array

from . import JReactionRateTerm, Reaction


class UMISTPhotoReaction(Reaction):
    """UMIST Photo-reaction.

    Rate equation:
        k = alpha * (uv_field / 1.7) * exp(-gamma * visual_extinction)
    """

    def __init__(  # noqa
        self,
        reaction_type: str,
        reactants: list[str],
        products: list[str],
        alpha: float,
        beta: float,
        gamma: float,
    ):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class PHReactionRateTerm(JReactionRateTerm):
            alpha: Array
            gamma: Array

            def __call__(
                self,
                temperature: Array,
                cr_rate: Array,
                uv_field: Array,
                visual_extinction: Array,
                abundance_vector: Array,
            ) -> Array:
                return (
                    self.alpha
                    * (uv_field / 1.7)
                    * jnp.exp(-self.gamma * visual_extinction)
                )

        return PHReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gamma))
