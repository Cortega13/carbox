"""Tests for the number-density evolution (``d ln n_h / dt``) term.

These tests disable chemistry so the RHS reduces to the density-evolution term,
allowing us to verify that absolute abundances scale with the supplied
time-dependent number-density grid.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest
from jax import Array


class ZeroRateTerm(eqx.Module):
    """A reaction rate term that always returns zero flux.

    This lets us isolate the density-evolution term in the RHS.
    """

    def __call__(
        self,
        temperature: Array,
        cr_rate: Array,
        fuv_rate: Array,
        visual_extinction: Array,
        abundances: Array,
    ) -> jnp.ndarray:
        """Return a zero reaction flux.

        Parameters are accepted to match the reaction-term interface used by the
        network, but are ignored.
        """
        return jnp.array([0.0])


@pytest.mark.parametrize(
    "nH_grid",
    [
        jnp.array([1e4, 2e4, 4e4]),
        jnp.array([1e5, 5e4, 2e5]),
    ],
)
def test_density_evolution_scales_abundances_with_nH(nH_grid: jnp.ndarray) -> None:
    """With chemistry disabled, dy/dt = y * dlnnH_dt => y ∝ nH."""
    from carbox.network import JNetwork
    from carbox.solver import solve_network_core

    # 2 species, 1 dummy reaction with zero rate, and zero incidence => chemistry term is 0.
    n_species = 2
    n_reactions = 1
    incidence = jnp.zeros((n_species, n_reactions))
    reactions = [ZeroRateTerm()]
    filler = n_species + 1
    reactant_multipliers = jnp.array([[0, filler]], dtype=jnp.int32)
    jnetwork = JNetwork(incidence, reactions, reactant_multipliers)

    # Time grid (years) and constant physical params.
    t_eval_years = jnp.array([0.0, 1.0, 2.0])
    temperature_grid = jnp.array([50.0, 50.0, 50.0])
    cr_rate_grid = jnp.array([1e-17, 1e-17, 1e-17])
    fuv_field_grid = jnp.array([1.0, 1.0, 1.0])
    visual_extinction_grid = jnp.array([2.0, 2.0, 2.0])

    # Initial absolute abundances.
    y0 = jnp.array([3.0e-2, 7.0e-2])

    sol = solve_network_core(
        jnetwork=jnetwork,
        y0=y0,
        t_eval_years=t_eval_years,
        number_density_grid=nH_grid,
        temperature_grid=temperature_grid,
        cr_rate_grid=cr_rate_grid,
        fuv_field_grid=fuv_field_grid,
        visual_extinction_grid=visual_extinction_grid,
        # This is a smooth, non-stiff RHS in this test; use an explicit solver
        # to avoid spending lots of steps on implicit solves.
        solver_name="dopri5",
        atol=1e-10,
        rtol=1e-10,
        max_steps=16384,
    )

    assert sol.ys is not None
    expected = y0[None, :] * (nH_grid / nH_grid[0])[:, None]
    assert jnp.allclose(sol.ys, expected, rtol=1e-5, atol=1e-10)


def test_density_evolution_constant_nH_keeps_abundances_constant() -> None:
    """If nH is constant, dlnnH_dt=0 and the extra term vanishes."""
    from carbox.network import JNetwork
    from carbox.solver import solve_network_core

    n_species = 2
    n_reactions = 1
    incidence = jnp.zeros((n_species, n_reactions))
    reactions = [ZeroRateTerm()]
    filler = n_species + 1
    reactant_multipliers = jnp.array([[0, filler]], dtype=jnp.int32)
    jnetwork = JNetwork(incidence, reactions, reactant_multipliers)

    t_eval_years = jnp.array([0.0, 1.0, 2.0])
    temperature_grid = jnp.array([50.0, 50.0, 50.0])
    cr_rate_grid = jnp.array([1e-17, 1e-17, 1e-17])
    fuv_field_grid = jnp.array([1.0, 1.0, 1.0])
    visual_extinction_grid = jnp.array([2.0, 2.0, 2.0])

    nH_grid = jnp.array([1e4, 1e4, 1e4])
    y0 = jnp.array([3.0e-2, 7.0e-2])

    sol = solve_network_core(
        jnetwork=jnetwork,
        y0=y0,
        t_eval_years=t_eval_years,
        number_density_grid=nH_grid,
        temperature_grid=temperature_grid,
        cr_rate_grid=cr_rate_grid,
        fuv_field_grid=fuv_field_grid,
        visual_extinction_grid=visual_extinction_grid,
        solver_name="dopri5",
        atol=1e-10,
        rtol=1e-10,
        max_steps=16384,
    )

    assert sol.ys is not None
    expected = jnp.tile(y0[None, :], (t_eval_years.shape[0], 1))
    assert jnp.allclose(sol.ys, expected, rtol=1e-5, atol=1e-10)
