"""Latent_tgas / prizmo specific reactions."""

import jax.numpy as jnp
from jax import Array

from .reactions import JReactionRateTerm, Reaction


class FUVReaction(Reaction):
    """Reaction driven by Far-UV radiation.

    Rate equation:
        k = alpha * uv_field
    """

    def __init__(  # noqa
        self,
        reaction_type: str,
        reactants: list[str],
        products: list[str],
        alpha: float,
    ):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class FUVReactionRateTerm(JReactionRateTerm):
            alpha: Array

            def __call__(
                self,
                temperature: Array,
                cr_rate: Array,
                uv_field: Array,
                visual_extinction: Array,
                abundance_vector: Array,
            ) -> Array:
                return self.alpha * uv_field

        return FUVReactionRateTerm(jnp.array(self.alpha))


class H2FormReaction(Reaction):
    """H2 Formation reaction.

    Rate equation:
        k = 100.0 * gas2dust * alpha
    """

    def __init__(  # noqa
        self,
        reaction_type: str,
        reactants: list[str],
        products: list[str],
        alpha: float,
        gas2dust: float,
    ):
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.gas2dust = gas2dust

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class H2ReactionRateTerm(JReactionRateTerm):
            alpha: Array
            gas2dust: Array

            def __call__(
                self,
                temperature: Array,
                cr_rate: Array,
                uv_field: Array,
                visual_extinction: Array,
                abundance_vector: Array,
            ) -> Array:
                return 100.0 * self.gas2dust * self.alpha

        return H2ReactionRateTerm(jnp.array(self.alpha), jnp.array(self.gas2dust))
