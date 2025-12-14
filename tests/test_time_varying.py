import jax.numpy as jnp
import pytest

from carbox.config import SimulationConfig
from carbox.main import run_simulation


def test_time_varying_physics():
    """Test that time-varying physical parameters are handled correctly."""
    # Create a dummy network file
    network_file = "network_files/uclchem_small_chemistry.csv"

    # Define time-varying parameters
    # t: 0 -> 1e5 seconds
    t_start = 0.0
    t_end = 1e5
    times = [t_start, t_end]

    # Density increases: 1e4 -> 1e5
    density = [1e4, 1e5]

    # Temperature constant
    temp = [10.0, 10.0]

    # CR constant
    cr = [1.3e-17, 1.3e-17]

    # FUV constant
    fuv = [1.0, 1.0]

    # Av constant
    av = [1.0, 1.0]

    config = SimulationConfig(
        physics_t=times,
        number_density=density,
        temperature=temp,
        cr_rate=cr,
        fuv_field=fuv,
        visual_extinction=av,
        t_start=t_start,
        t_end=t_end,
        n_snapshots=10,
        initial_abundances={"H": 1.0},  # H is inert in small chemistry? Maybe not.
        solver="kvaerno5",
    )

    # Run simulation
    results = run_simulation(network_file, config, verbose=True)

    solution = results["solution"]

    # Check that solution ran
    assert solution.ys is not None

    # Check last density used implicitly
    # Since we solve for fractional abundances, X_H should be roughly conserved if H is not reacting much
    # In uclchem_small_chemistry, H might react.

    # Let's check dimensions
    assert solution.ys.shape[0] == config.n_snapshots

    print("Simulation successful with time-varying parameters.")


def test_constant_physics_compatibility():
    """Test that constant parameters (length 1 lists) still work."""
    network_file = "network_files/uclchem_small_chemistry.csv"

    config = SimulationConfig(
        physics_t=[0.0],
        number_density=[1e4],
        temperature=[10.0],
        cr_rate=[1.3e-17],
        fuv_field=[1.0],
        visual_extinction=[1.0],
        t_end=1e4,  # 10,000 seconds
    )

    results = run_simulation(network_file, config, verbose=False)
    assert results["solution"].ys is not None
    print("Simulation successful with constant parameters.")


if __name__ == "__main__":
    test_time_varying_physics()
    test_constant_physics_compatibility()
