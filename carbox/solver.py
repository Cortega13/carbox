"""ODE solver wrapper for chemical kinetics integration.

Wraps Diffrax solvers with appropriate settings for stiff chemistry ODEs.
"""

from functools import partial

import diffrax as dx
import jax
import jax.numpy as jnp

from .config import SimulationConfig
from .network import JNetwork, Network

# Seconds per year
SPY = 3600.0 * 24 * 365.2422222


def get_solver(solver_name: str):
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


def compute_interp_left_and_weight(
    t: jnp.ndarray, time_grid: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute left index and interpolation weight."""
    right = jnp.searchsorted(time_grid, t, side="right")
    max_left = max(int(time_grid.shape[0]) - 2, 0)
    left = jnp.clip(right - 1, 0, max_left)
    t0 = time_grid[left]
    t1 = time_grid[left + 1]
    denom = jnp.where(t1 == t0, 1.0, t1 - t0)
    weight = jnp.clip((t - t0) / denom, 0.0, 1.0)
    return left, weight


def interpolate_param_with_left_and_weight(
    param: jnp.ndarray, left: jnp.ndarray, weight: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate a parameter using a precomputed bracket."""
    return param[left] * (1.0 - weight) + param[left + 1] * weight


@partial(jax.jit, static_argnames=["solver_name", "max_steps"])
def solve_network_core(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    time_grid: jnp.ndarray,
    number_density: jnp.ndarray,
    temperature: jnp.ndarray,
    cr_rate: jnp.ndarray,
    fuv_field: jnp.ndarray,
    visual_extinction: jnp.ndarray,
    solver_name: str = "kvaerno5",
    atol: float = 1e-18,
    rtol: float = 1e-12,
    max_steps: int = 4096,
    pcoeff: float = 0.4,
    icoeff: float = 0.3,
    dcoeff: float = 0.0,
    factormax: float = 1000.0,
) -> dx.Solution:
    """Core ODE solver with raw JAX array parameters.

    Parameters
    ----------
    jnetwork : JNetwork
        Compiled JAX network with reaction rates
    y0 : jnp.ndarray
        Initial abundance vector [cm^-3]
    t_eval : jnp.ndarray
        Time points for evaluation [years]
    time_grid : jnp.ndarray
        Time grid for parameter interpolation [years]
    number_density : jnp.ndarray
        Number density [cm^-3] (scalar or array for time-dependent)
    temperature : jnp.ndarray
        Gas temperature [K] (scalar or array for time-dependent)
    cr_rate : jnp.ndarray
        Cosmic ray ionization rate [s^-1] (scalar or array for time-dependent)
    fuv_field : jnp.ndarray
        FUV radiation field (Draine units) (scalar or array for time-dependent)
    visual_extinction : jnp.ndarray
        Visual extinction Av [mag] (scalar or array for time-dependent)
    solver_name : str
        Solver name ('dopri5', 'kvaerno5', 'tsit5')
    atol : float
        Absolute tolerance for solution accuracy
    rtol : float
        Relative tolerance for solution accuracy
    max_steps : int
        Maximum integration steps
    pcoeff : float
        Proportional coefficient for PID step size controller
    icoeff : float
        Integral coefficient for PID step size controller
    dcoeff : float
        Derivative coefficient for PID step size controller
    factormax : float
        Maximum step size growth factor

    Returns:
    -------
    solution : diffrax.Solution
        Integration results
    """
    # Convert time to seconds
    t_eval_sec = t_eval * SPY
    time_grid_sec = time_grid * SPY

    def _get_params(t, args):
        """Get physical parameters at time t."""
        number_density = args["number_density"]
        temperature = args["temperature"]
        cr_rate = args["cr_rate"]
        fuv_field = args["fuv_field"]
        visual_extinction = args["visual_extinction"]

        needs_interp = (
            number_density.shape[0] > 1
            or temperature.shape[0] > 1
            or cr_rate.shape[0] > 1
            or fuv_field.shape[0] > 1
            or visual_extinction.shape[0] > 1
        )
        if not needs_interp:
            return (
                number_density[0],
                temperature[0],
                cr_rate[0],
                fuv_field[0],
                visual_extinction[0],
            )

        left, weight = compute_interp_left_and_weight(t, args["time_grid"])

        def _maybe_interp(param):
            """Interpolate a parameter if needed."""
            if param.shape[0] == 1:
                return param[0]
            return interpolate_param_with_left_and_weight(param, left, weight)

        return (
            _maybe_interp(number_density),
            _maybe_interp(temperature),
            _maybe_interp(cr_rate),
            _maybe_interp(fuv_field),
            _maybe_interp(visual_extinction),
        )

    # Define ODE term with parameter interpolation
    ode_term = dx.ODETerm(
        lambda t, y, args: jnetwork(
            t,
            y,
            *_get_params(t, args),
        )
    )

    # Get solver
    solver = get_solver(solver_name)

    # Physical parameters (include time_grid for interpolation)
    params = {
        "time_grid": time_grid_sec,
        "number_density": number_density,
        "temperature": temperature,
        "cr_rate": cr_rate,
        "fuv_field": fuv_field,
        "visual_extinction": visual_extinction,
    }

    # Step size controller from config (uses atol/rtol from solver)
    stepsize_controller = dx.PIDController(
        rtol=rtol,
        atol=atol,
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        factormax=factormax,
    )

    # Solve
    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=t_eval_sec[0],
        t1=t_eval_sec[-1],
        dt0=1e-6,  # Initial timestep [s]
        y0=y0,
        stepsize_controller=stepsize_controller,  # dx.PIDController(atol=atol, rtol=rtol),
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
        - ys: abundance array [n_snapshots, n_species]
        - stats: solver statistics

    Notes:
    -----
    - Uses linear time sampling per interval for time-dependent parameters
    - Physical parameters passed as args to ODE function
    - JIT compiled for performance (first call compiles)
    - Stiff solver (Kvaerno5) recommended for chemistry
    """
    # Get physical parameters as JAX arrays
    params = config.get_physical_params_jax()
    time_grid = config.get_time_grid()

    # Construct time evaluation points based on t_end structure
    if isinstance(config.t_end, list):
        # Multi-interval case: linear spacing within each interval
        t_intervals = []
        t_points = [config.t_start] + config.t_end

        for i in range(len(t_points) - 1):
            t_start_interval = t_points[i]
            t_end_interval = t_points[i + 1]

            # Linear spacing for this interval
            if i == 0:
                # First interval: include start point
                t_interval = jnp.linspace(
                    t_start_interval, t_end_interval, config.n_snapshots
                )
            else:
                # Subsequent intervals: exclude start (already included from previous)
                t_interval = jnp.linspace(
                    t_start_interval, t_end_interval, config.n_snapshots + 1
                )[1:]

            t_intervals.append(t_interval)

        t_snapshots = jnp.concatenate(t_intervals)
    else:
        # Single interval case: use linear spacing (backward compatible)
        t_snapshots = jnp.linspace(config.t_start, config.t_end, config.n_snapshots)

    return solve_network_core(
        jnetwork=jnetwork,
        y0=y0,
        t_eval=t_snapshots,
        time_grid=time_grid,
        number_density=params["number_density"],
        temperature=params["temperature"],
        cr_rate=params["cr_rate"],
        fuv_field=params["fuv_field"],
        visual_extinction=params["visual_extinction"],
        solver_name=config.solver,
        atol=config.atol,
        rtol=config.rtol,
        max_steps=config.max_steps,
        pcoeff=config.pcoeff,
        icoeff=config.icoeff,
        dcoeff=config.dcoeff,
        factormax=config.factormax,
    )


def solve_network_batch(
    jnetwork: JNetwork,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
    temperatures: jnp.ndarray,
    cr_rates: jnp.ndarray,
    fuv_fields: jnp.ndarray,
    visual_extinctions: jnp.ndarray,
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
    t_eval : jnp.ndarray
        Time points for evaluation [years] (same for all simulations)
    temperatures : jnp.ndarray
        Gas temperatures [K], shape (batch_size,)
    cr_rates : jnp.ndarray
        Cosmic ray ionization rates [s^-1], shape (batch_size,)
    fuv_fields : jnp.ndarray
        FUV radiation fields (Draine units), shape (batch_size,)
    visual_extinctions : jnp.ndarray
        Visual extinctions Av [mag], shape (batch_size,)
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
    # For batch solving, assume constant parameters (no time dependence)
    time_grid = jnp.array([t_eval[0], t_eval[-1]])
    number_densities = jnp.ones(len(temperatures)) * 1e4  # Default value

    return jax.vmap(
        lambda nd, temp, cr, fuv, av: solve_network_core(
            jnetwork,
            y0,
            t_eval,
            time_grid,
            jnp.atleast_1d(nd),
            jnp.atleast_1d(temp),
            jnp.atleast_1d(cr),
            jnp.atleast_1d(fuv),
            jnp.atleast_1d(av),
            solver_name,
            atol,
            rtol,
            max_steps,
        ),
        in_axes=(0, 0, 0, 0, 0),
    )(number_densities, temperatures, cr_rates, fuv_fields, visual_extinctions)


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
        Time derivatives [n_snapshots, n_species]

    Notes:
    -----
    Useful for analyzing formation/destruction rates.
    Evaluated at actual solution points (not interpolated).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    params = config.get_physical_params_jax()
    time_grid = config.get_time_grid()
    time_grid_sec = time_grid * SPY

    dy = jnp.zeros_like(solution.ys)

    def _get_param(t, param_array):
        """Get parameter value at time t."""
        if param_array.shape[0] == 1:
            return param_array[0]
        else:
            return jnp.interp(t, time_grid_sec, param_array)

    for i, (t, y) in enumerate(zip(solution.ts, solution.ys, strict=False)):
        dy_i = jnetwork(
            t,
            y,
            _get_param(t, params["number_density"]),
            _get_param(t, params["temperature"]),
            _get_param(t, params["cr_rate"]),
            _get_param(t, params["fuv_field"]),
            _get_param(t, params["visual_extinction"]),
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
        Reaction rates [n_snapshots, n_reactions]

    Notes:
    -----
    Raw rate coefficients (not multiplied by abundances).
    Units depend on reaction type (typically cm^3/s for bimolecular).
    """
    if not (solution.ys and solution.ts):
        raise Exception("Missing solution.ys or solution.ts.")

    params = config.get_physical_params_jax()
    time_grid = config.get_time_grid()
    time_grid_sec = time_grid * SPY

    n_snapshots = len(solution.ts)
    n_reactions = len(network.reactions)
    rates = jnp.zeros((n_snapshots, n_reactions))

    def _get_param(t, param_array):
        """Get parameter value at time t."""
        if param_array.shape[0] == 1:
            return param_array[0]
        else:
            return jnp.interp(t, time_grid_sec, param_array)

    for i in range(n_snapshots):
        t = solution.ts[i]
        rates_i = jnetwork.get_rates(
            _get_param(t, params["number_density"]),
            _get_param(t, params["temperature"]),
            _get_param(t, params["cr_rate"]),
            _get_param(t, params["fuv_field"]),
            _get_param(t, params["visual_extinction"]),
            solution.ys[i],  # Load abundances from solution at snapshot i
        )
        rates = rates.at[i].set(rates_i)

    return rates
