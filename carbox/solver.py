"""ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

import diffrax as dx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork, Network

# Seconds per year
SPY = 3600.0 * 24 * 365.0


def compute_dnh_dt_slopes(
    t_grid_sec: jnp.ndarray, number_density_grid: jnp.ndarray
) -> jnp.ndarray:
    """Compute piecewise-constant slopes for dn_h/dt.

    Assumes n_H(t) is linearly interpolated between provided grid points.
    For each interval [t_i, t_{i+1}), define:

        slope_i = (nH_{i+1} - nH_i) / (t_{i+1} - t_i)

    Returns an array of length (n_times - 1), where each entry corresponds to
    one time interval.
    """
    return (number_density_grid[1:] - number_density_grid[:-1]) / (
        t_grid_sec[1:] - t_grid_sec[:-1]
    )


def eval_piecewise_constant(
    t,
    t_grid: jnp.ndarray,
    interval_values: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a piecewise-constant function on intervals defined by t_grid.

    interval_values has length len(t_grid) - 1.
    """
    # Interval boundaries are t_grid[1:], so the interval index is the count of
    # boundaries <= t.
    i = jnp.searchsorted(t_grid[1:], t, side="right")
    i = jnp.clip(i, 0, interval_values.shape[0] - 1)
    return interval_values[i]


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


def solve_network_core(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval_years: jnp.ndarray,
    number_density_grid: jnp.ndarray,
    temperature_grid: jnp.ndarray,
    cr_rate_grid: jnp.ndarray,
    fuv_field_grid: jnp.ndarray,
    visual_extinction_grid: jnp.ndarray,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Core ODE solver with raw JAX array parameters.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3]
    t_eval_years : jnp.ndarray
        Time points for evaluation [years]
    temperature_grid : jnp.ndarray
        Gas temperature grid [K]
    cr_rate_grid : jnp.ndarray
        Cosmic ray ionization rate grid [s^-1]
    fuv_field_grid : jnp.ndarray
        FUV radiation field grid (Draine units)
    visual_extinction_grid : jnp.ndarray
        Visual extinction Av grid [mag]
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
    # Convert time to seconds
    t_eval_sec = t_eval_years * SPY

    # Precompute piecewise-constant dn_H/dt slopes consistent with linear interpolation.
    dnh_dt_slopes = compute_dnh_dt_slopes(t_eval_sec, number_density_grid)

    # Define ODE term
    ode_term = dx.ODETerm(
        lambda t, y, args: jnetwork(
            t,
            y,
            eval_piecewise_constant(t, t_eval_sec, args["dnh_dt_slopes"])
            / jnp.interp(t, t_eval_sec, args["number_density_grid"]),
            jnp.interp(t, t_eval_sec, args["temperature_grid"]),
            jnp.interp(t, t_eval_sec, args["cr_rate_grid"]),
            jnp.interp(t, t_eval_sec, args["fuv_field_grid"]),
            jnp.interp(t, t_eval_sec, args["visual_extinction_grid"]),
        )
    )

    # Get solver
    solver = get_solver(solver_name)

    # Physical parameters
    params = {
        "number_density_grid": number_density_grid,
        "dnh_dt_slopes": dnh_dt_slopes,
        "temperature_grid": temperature_grid,
        "cr_rate_grid": cr_rate_grid,
        "fuv_field_grid": fuv_field_grid,
        "visual_extinction_grid": visual_extinction_grid,
    }

    # Solve
    # Pick a reasonable initial timestep relative to the smallest grid interval.
    # (Starting too small can cause us to hit `max_steps` unnecessarily even for
    # simple RHS terms like the density evolution.)
    min_dt = float(jnp.min(t_eval_sec[1:] - t_eval_sec[:-1]))
    dt0 = max(1e-6, min_dt / 10.0)

    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=t_eval_sec[0],
        t1=t_eval_sec[-1],
        dt0=dt0,  # Initial timestep [s]
        y0=y0,
        stepsize_controller=dx.PIDController(atol=atol, rtol=rtol),
        saveat=dx.SaveAt(ts=t_eval_sec),
        args=params,
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
        Initial abundance vector [cm^-3]
    config : SimulationConfig
        Configuration with solver and physical parameters

    Returns:
    -------
    solution : diffrax.Solution
        Integration results with:
        - ts: time array [s]
        - ys: abundance array [n_times, n_species]
        - stats: solver statistics

    Notes:
    -----
    - Uses user-supplied time grid for evaluation
    - Physical parameters interpolated over time grid in ODE function
    - JIT compiled for performance (first call compiles)
    - Stiff solver (Kvaerno5) recommended for chemistry
    """
    # Get physical parameter grids as JAX arrays
    params = config.get_physical_param_grids_jax()

    return solve_network_core(
        jnetwork=jnetwork,
        y0=y0,
        t_eval_years=params["time_grid_years"],
        number_density_grid=params["number_density_grid"],
        temperature_grid=params["temperature_grid"],
        cr_rate_grid=params["cr_rate_grid"],
        fuv_field_grid=params["fuv_field_grid"],
        visual_extinction_grid=params["visual_extinction_grid"],
        solver_name=config.solver,
        atol=config.atol,
        rtol=config.rtol,
        max_steps=config.max_steps,
    )


def solve_network_batch(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval_years: jnp.ndarray,
    temperature_grids: jnp.ndarray,
    cr_rate_grids: jnp.ndarray,
    fuv_field_grids: jnp.ndarray,
    visual_extinction_grids: jnp.ndarray,
    number_density_grids: jnp.ndarray | None = None,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
) -> dx.Solution:
    """Batch solve chemical network ODE system for parameter sweeps.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3] (same for all simulations)
    t_eval_years : jnp.ndarray
        Time points for evaluation [years] (same for all simulations)
    temperature_grids : jnp.ndarray
        Gas temperatures [K], shape (batch_size, n_times)
    cr_rate_grids : jnp.ndarray
        Cosmic ray ionization rates [s^-1], shape (batch_size, n_times)
    fuv_field_grids : jnp.ndarray
        FUV radiation fields (Draine units), shape (batch_size, n_times)
    visual_extinction_grids : jnp.ndarray
        Visual extinctions Av [mag], shape (batch_size, n_times)
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
    solutions : diffrax.Solution
        Batch of integration results, shape (batch_size, ...)
    """
    if number_density_grids is None:
        batch_size = temperature_grids.shape[0]
        n_times = t_eval_years.shape[0]
        number_density_grids = jnp.ones((batch_size, n_times))

    return jax.vmap(
        lambda temp_grid, cr_grid, fuv_grid, av_grid, nH_grid: solve_network_core(
            jnetwork,
            y0,
            t_eval_years,
            nH_grid,
            temp_grid,
            cr_grid,
            fuv_grid,
            av_grid,
            solver_name,
            atol,
            rtol,
            max_steps,
        ),
        in_axes=(0, 0, 0, 0, 0),
    )(
        temperature_grids,
        cr_rate_grids,
        fuv_field_grids,
        visual_extinction_grids,
        number_density_grids,
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
        Time derivatives [n_times, n_species]

    Notes:
    -----
    Useful for analyzing formation/destruction rates.
    Evaluated at actual solution points (not interpolated).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    params = config.get_physical_param_grids_jax()
    t_grid_sec = params["time_grid_years"] * SPY

    dnh_dt_slopes = compute_dnh_dt_slopes(t_grid_sec, params["number_density_grid"])

    dy = jnp.zeros_like(solution.ys)

    for i, (t, y) in enumerate(zip(solution.ts, solution.ys, strict=False)):
        dy_i = jnetwork(
            t,
            y,
            eval_piecewise_constant(t, t_grid_sec, dnh_dt_slopes)
            / jnp.interp(t, t_grid_sec, params["number_density_grid"]),
            jnp.interp(t, t_grid_sec, params["temperature_grid"]),
            jnp.interp(t, t_grid_sec, params["cr_rate_grid"]),
            jnp.interp(t, t_grid_sec, params["fuv_field_grid"]),
            jnp.interp(t, t_grid_sec, params["visual_extinction_grid"]),
        )
        dy = dy.at[i].set(dy_i)

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
        Reaction rates [n_times, n_reactions]

    Notes:
    -----
    Raw rate coefficients (not multiplied by abundances).
    Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    params = config.get_physical_param_grids_jax()
    t_grid_sec = params["time_grid_years"] * SPY

    n_times = len(solution.ts)
    n_reactions = len(network.reactions)
    rates = jnp.zeros((n_times, n_reactions))

    for i in range(n_times):
        rates_i = jnetwork.get_rates(
            jnp.interp(solution.ts[i], t_grid_sec, params["temperature_grid"]),
            jnp.interp(solution.ts[i], t_grid_sec, params["cr_rate_grid"]),
            jnp.interp(solution.ts[i], t_grid_sec, params["fuv_field_grid"]),
            jnp.interp(solution.ts[i], t_grid_sec, params["visual_extinction_grid"]),
            solution.ys[i],  # Load abundances from solution at snapshot i
        )
        rates = rates.at[i].set(rates_i)

    return rates
