"""Universally Used Reactions."""

import jax.numpy as jnp
from jax import Array

from . import JReactionRateTerm, Reaction


class KAReaction(Reaction):
    """KIDA/UMIST Arrhenius Reaction.

    Rate equation:
        k = alpha * (T / 300)^beta * exp(-gamma / T)
    """

    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):  # noqa
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class KAReactionRateTerm(JReactionRateTerm):
            alpha: Array
            beta: Array
            gamma: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                # α(T/300K​)^βexp(−γ/T)
                return (
                    self.alpha
                    * jnp.power(0.0033333333333333335 * temperature, self.beta)
                    * jnp.exp(-self.gamma / temperature)
                )

        return KAReactionRateTerm(
            jnp.array(self.alpha), jnp.array(self.beta), jnp.array(self.gamma)
        )


class KAFixedReaction(Reaction):
    """Fixed Temperature Arrhenius Reaction.

    Rate equation:
        k = alpha * (T_fixed / 300)^beta * exp(-gamma / T_fixed)
    """

    def __init__(  # noqa
        self, reaction_type, reactants, products, alpha, beta, gamma, temperature
    ):
        super().__init__(reaction_type, reactants, products)
        self.reaction_coeff = (
            alpha
            * jnp.power(0.0033333333333333335 * temperature, beta)
            * jnp.exp(-gamma / temperature)
        )

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class KAFixedReactionRateTerm(JReactionRateTerm):
            reaction_coeff: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return self.reaction_coeff

        return KAFixedReactionRateTerm(jnp.array(self.reaction_coeff))


class CRPReaction(Reaction):
    """Cosmic Ray Proton Reaction.

    Rate equation:
        k = alpha * cr_rate
    """

    def __init__(self, reaction_type, reactants, products, alpha):  # noqa
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CRPReactionRateTerm(JReactionRateTerm):
            alpha: Array

            def __call__(
                self,
                temperature,
                cr_rate,
                uv_field,
                visual_extinction,
                abundance_vector,
            ):
                return cr_rate * self.alpha

        return CRPReactionRateTerm(jnp.array(self.alpha))


class CRPhotoReaction(Reaction):
    """Cosmic Ray induced Photo-reaction.

    Rate equation:
        k = 1.31e-17 * cr_rate * (T / 300)^beta * gamma / (1 - omega)
        where omega = 0.5 (grain albedo)
    """

    def __init__(self, reaction_type, reactants, products, alpha, beta, gamma):  # noqa
        super().__init__(reaction_type, reactants, products)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _reaction_rate_factory(self) -> JReactionRateTerm:
        class CRPReactionRateTerm(JReactionRateTerm):
            alpha: Array
            beta: Array
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
                    1.31e-17
                    * cr_rate
                    * jnp.power(0.0033333333333333335 * temperature, self.beta)
                    * self.gamma
                    / (1 - 0.5)  # hardcoded omega value
                )

        return CRPReactionRateTerm(
            jnp.array(self.alpha),
            jnp.array(self.beta),
            jnp.array(self.gamma),
        )
