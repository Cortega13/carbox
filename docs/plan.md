# Plan: Time-Dependent Physical Parameters

## Objective
Enable time-dependent physical parameters (`temperature`, `number_density`, `fuv_field`, `visual_extinction`) in Carbox simulations. The implementation must support list inputs for these parameters corresponding to a time grid, using linear interpolation during integration. The solution must maintain current performance levels.

## Implementation Steps

### 1. Configuration Updates (`carbox/config.py`)
- Modify `SimulationConfig` to allow `List[float]` (or `jnp.ndarray`) for:
    - `temperature`
    - `number_density`
    - `fuv_field`
    - `visual_extinction`
    - `t_end`: Now accepts `List[float]` to define time intervals.
- **Time Grid Logic**:
    - The full time grid is constructed as `[t_start] + t_end` (if `t_end` is a list).
    - Parameter lists must have length equal to `len(t_end) + 1` (values at start + values at each end point).
    - `n_snapshots` applies to *each interval* defined by `t_end`.
- Add validation to ensure all list inputs have consistent lengths.
- Update `get_physical_params_jax` to handle conversion of lists to JAX arrays.

### 2. Network and Reaction Updates
We need to propagate `number_density` through the network execution chain, as it is now a dynamic parameter requested by the user.

- **`carbox/reactions/reactions.py`**:
    - Update `JReactionRateTerm.__call__` signature to include `number_density`.
- **`carbox/network.py`**:
    - Update `JNetwork.get_rates` and `JNetwork.__call__` to accept and pass `number_density`.
- **`carbox/reactions/uclchem_reactions.py`**:
    - Update all `__call__` methods of reaction rate terms to accept `number_density`.

### 3. Solver Updates (`carbox/solver.py`)
- Update `solve_network` to construct the evaluation time points (`t_eval`) based on the multi-interval `t_end` list.
    - `t_eval` will be a concatenation of points for each interval.
- Update `solve_network_core` signature to accept `time_grid` (the defining points for interpolation) and parameter arrays.
- Modify the `ode_term` lambda function to handle both scalar and time-dependent parameters:
    - **Optimization**: Check if `param` is a scalar (or size 1 array). If so, use it directly.
    - If `param` is an array (size > 1), perform linear interpolation:
        ```python
        current_param = jnp.interp(t, time_grid, param_array)
        ```
    - This ensures zero overhead for the standard constant-parameter case.
- Pass the interpolated `number_density` to `jnetwork`.

### 4. New Usage Script (`notebooks/usage/uclchem_complex_parameters.py`)
- Create a barebones script to demonstrate the new functionality.
- **Features**:
    - "Hyperparameter" for number of interpolation points.
    - "Hyperparameter" for integration time.
    - Sinusoidal evolution for parameters.
    - Print integration time to verify performance.

## Verification
- Run `notebooks/usage/uclchem_complex_parameters.py` and ensure it runs successfully and quickly.
- Verify that `number_density` changes are accepted (even if they only affect rates as requested).
