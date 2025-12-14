"""Umist specific reactions."""

import jax.numpy as jnp
from jax import Array

from . import JReactionRateTerm, Reaction


class UMISTPhotoReaction(Reaction):
    """UMIST Photo-reaction.

    Rate equation:
        k = alpha * (uv_field / 1.7) * exp(-gamma * visual_extinction)
    """

    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):  # noqa
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
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return (
                    self.alpha
                    * (uv_field / 1.7)
                    * jnp.exp(-self.gamma * visual_extinction)
                )

        return PHReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gamma))
