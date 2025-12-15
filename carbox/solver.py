"""ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

from typing import Any

import diffrax as dx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork, Network


def get_solver(solver_name: str) -> dx.AbstractSolver:
    """Get Diffrax solver instance from name.

    Parameters
    ----------
    solver_name : str
        Solver identifier: 'dopri5', 'kvaerno5', 'tsit5'

    Returns:
    -------
    solver : diffrax.AbstractSolver
        Configured solver instance

    Notes:
    -----
    - dopri5: Explicit RK method, good for non-stiff
    - kvaerno5: SDIRK method, good for stiff chemistry (recommended)
    - tsit5: Explicit RK method, efficient for moderate stiffness
    """
    solvers = {
        "dopri5": dx.Dopri5,
        "kvaerno5": dx.Kvaerno5,
        "tsit5": dx.Tsit5,
    }

    if solver_name.lower() not in solvers:
        raise ValueError(
            f"Unknown solver: {solver_name}. Available: {list(solvers.keys())}"
        )

    return solvers[solver_name.lower()]()


def build_physics_interpolation(config: SimulationConfig) -> dx.CubicInterpolation:
    """Build interpolated path for time-varying physical parameters.

    Parameters
    ----------
    config : SimulationConfig
        Configuration containing physical parameters

    Returns:
    -------
    physics_path : dx.CubicInterpolation
        Cubic interpolation path over time (seconds) with shape (n_times, 5)
        Parameters: [temperature, cr_rate, fuv_field, visual_extinction, number_density]
    """
    params = config.get_physical_params_jax()
    physics_t = params["physics_t"]
    param_names = [
        "temperature",
        "cr_rate",
        "fuv_field",
        "visual_extinction",
        "number_density",
    ]
    param_arrays = [params[name] for name in param_names]

    # Create cubic interpolation path (time in seconds)
    physics_data = jnp.stack(param_arrays, axis=-1)
    coeffs = dx.backward_hermite_coefficients(physics_t, physics_data)
    return dx.CubicInterpolation(physics_t, coeffs)


@jax.jit(static_argnames=["solver_name", "max_steps"])
def jsolve_network(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    physics_path: dx.AbstractPath,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Core ODE solver with time-varying physical parameters.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial fractional abundance vector [relative to H]
    t_eval : jnp.ndarray
        Time points for evaluation [seconds]
    physics_path : dx.Path
        Interpolated path of physical parameters over time.
        Expected order: [temperature, cr_rate, fuv_field, visual_extinction, number_density]
    solver_name : str
        Solver name ('dopri5', 'kvaerno5', 'tsit5')
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    max_steps : int
        Maximum integration steps

    Returns:
    -------
    solution : diffrax.Solution
        Integration results
    """

    def ode_func(t: Any, y: jax.Array, args: dx.AbstractPath) -> jax.Array:
        """Compute time derivatives of fractional abundances.

        Process:
        1. Extract interpolated physical parameters from path
        2. Convert fractional abundances to absolute densities
        3. Calculate reaction rates using absolute densities
        4. Convert rates back to fractional form

        Ignores dilution effects from changing number density.
        """
        temperature, cr_rate, fuv_field, visual_extinction, number_density = (
            args.evaluate(t)
        )

        # Fractional to absolute: n_i = X_i * n
        y_abs = y * number_density

        # Calculate absolute rates [cm^-3 s^-1]
        dy_abs_dt = jnetwork(
            t, y_abs, temperature, cr_rate, fuv_field, visual_extinction
        )

        # Absolute to fractional: dX_i/dt = (1/n) * dn_i/dt
        return dy_abs_dt / number_density

    ode_term = dx.ODETerm(ode_func)

    solver = get_solver(solver_name)

    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=1e-6,  # Initial timestep [s]
        y0=y0,
        stepsize_controller=dx.PIDController(atol=atol, rtol=rtol, factormax=1e3),
        saveat=dx.SaveAt(ts=t_eval),
        args=physics_path,
        max_steps=max_steps,
    )

    return solution


def solve_network(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    config: SimulationConfig,
) -> dx.Solution:
    """Solve chemical network ODE system.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial fractional abundance vector
    config : SimulationConfig
        Configuration with solver and physical parameters

    Returns:
    -------
    solution : diffrax.Solution
        Integration results with:
        - ts: time array [s]
        - ys: fractional abundance array [n_snapshots, n_species]
        - stats: solver statistics

    Notes:
    -----
    - Uses logarithmic time sampling for astrophysical timescales
    - Physical parameters interpolated using CubicInterpolation
    - JIT compiled for performance (first call compiles)
    - Stiff solver (Kvaerno5) recommended for chemistry
    """
    physics_path = build_physics_interpolation(config)

    t_eval = jnp.array(config.physics_t)

    return jsolve_network(
        jnetwork=jnetwork,
        y0=y0,
        t_eval=t_eval,
        physics_path=physics_path,
        solver_name=config.solver,
        atol=config.atol,
        rtol=config.rtol,
        max_steps=config.max_steps,
    )


def compute_derivatives(
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """Recompute dy/dt at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns:
    -------
    derivatives : jnp.ndarray
        Time derivatives [n_physics_t, n_species]

    Notes:
    -----
    Useful for analyzing formation/destruction rates.
    Evaluated at actual solution points (not interpolated).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    physics_path = build_physics_interpolation(config)

    dy = jnp.zeros_like(solution.ys)

    for i, (t_sec, y_frac) in enumerate(zip(solution.ts, solution.ys, strict=False)):
        temp, cr, fuv, av, density = physics_path.evaluate(t_sec)

        # Convert fractional to absolute
        y_abs = y_frac * density

        # Compute absolute rate
        dy_abs = jnetwork(
            t_sec,
            y_abs,
            temp,
            cr,
            fuv,
            av,
        )

        # Convert to fractional rate
        dy_frac = dy_abs / density

        dy = dy.at[i].set(dy_frac)

    return dy


def compute_reaction_rates(
    network: Network,
    jnetwork: JNetwork,
    solution: dx.Solution,
    config: SimulationConfig,
) -> jnp.ndarray:
    """Compute reaction rates at solution snapshots.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled network
    solution : dx.Solution
        Integration solution
    config : SimulationConfig
        Configuration with physical parameters

    Returns:
    -------
    rates : jnp.ndarray
        Reaction rates [n_physics_t, n_reactions]

    Notes:
    -----
    Raw rate coefficients (not multiplied by abundances).
    Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    physics_path = build_physics_interpolation(config)

    n_snapshots = len(solution.ts)
    n_reactions = len(network.reactions)
    rates = jnp.zeros((n_snapshots, n_reactions))

    for i in range(n_snapshots):
        temp, cr, fuv, av, density = physics_path.evaluate(solution.ts[i])

        # Fractional abundances from solution
        y_frac = solution.ys[i]

        # Convert to absolute for rate calculation
        y_abs = y_frac * density

        rates_i = jnetwork.get_rates(
            temp,
            cr,
            fuv,
            av,
            y_abs,
        )
        rates = rates.at[i].set(rates_i)

    return rates
