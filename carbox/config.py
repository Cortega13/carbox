"""Configuration management for Carbox simulations.

Simple dataclass-based config for chemical kinetics simulations.
Supports loading from YAML/JSON and programmatic setup.
"""

import json
from dataclasses import dataclass, field

import jax.numpy as jnp
import yaml


@dataclass
class SimulationConfig:
    """Configuration for astrochemical kinetics simulation.

    Attributes:
    ----------
    Physical Parameters (can be scalars or lists for time-dependent sims):
        number_density : float | list[float]
            Total hydrogen number density [cm^-3]. Range: [1e2, 1e6]
        temperature : float | list[float]
            Gas temperature [K]. Range: [10, 1e5]
        cr_rate : float | list[float]
            Cosmic ray ionization rate [s^-1]. Range: [1e-17, 1e-14]
        fuv_field : float | list[float]
            FUV radiation field (Draine units). Range: [1e0, 1e5]
        visual_extinction : float | list[float]
            Visual extinction Av [mag]. Range: [0, 10]
        gas_to_dust_ratio : float
            Gas-to-dust mass ratio. Typical: 100 (= 0.01 dust/gas)

    Initial Abundances:
        initial_abundances : Dict[str, float]
            Species name -> fractional abundance (relative to number_density)
            Example: {"H2": 1.0, "O": 2e-4, "C": 1e-4}
        abundance_floor : float
            Minimum abundance for all species (numerical stability)

    Integration Parameters:
        t_start : float
            Start time [years]
        t_end : float | list[float]
            End time [years]. If list, defines time intervals for parameter evolution.
        n_snapshots : int
            Number of output snapshots per interval (linear-spaced within each interval)
        solver : str
            Solver name: 'dopri5', 'kvaerno5', 'tsit5'
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        max_steps : int
            Maximum integration steps
        pcoeff : float
            Proportional coefficient for PID step size controller
        icoeff : float
            Integral coefficient for PID step size controller
        dcoeff : float
            Derivative coefficient for PID step size controller
        factormax : float
            Maximum step size growth factor per step

    Output Settings:
        output_dir : str
            Directory for output files
        save_abundances : bool
            Save abundance time series
        save_derivatives : bool
            Save dy/dt at each snapshot
        save_rates : bool
            Save reaction rates at each snapshot
        run_name : str
            Identifier for this run
    """

    # Physical parameters (can be scalars or lists)
    number_density: float | list[float] = 1e4
    temperature: float | list[float] = 50.0
    cr_rate: float | list[float] = 1e-17
    fuv_field: float | list[float] = 1.0
    visual_extinction: float | list[float] = 2.0
    gas_to_dust_ratio: float = 100.0

    # Cloud geometry (for photoreaction shielding and self-consistent Av)
    cloud_radius_pc: float = 1.0  # Cloud radius in parsecs
    base_av: float = 0.0  # Base Av before column density contribution
    use_self_consistent_av: bool = False  # Compute Av from column density

    # Initial abundances (fractional relative to number_density)
    initial_abundances: dict[str, float] = field(
        default_factory=lambda: {
            "H2": 1.0,
            "O": 2e-4,
            "C": 1e-4,
        }
    )
    abundance_floor: float = 1e-30

    # Integration parameters
    t_start: float = 0.0
    t_end: float | list[float] = 1e6  # years
    n_snapshots: int = 1000
    solver: str = "kvaerno5"
    atol: float = 1e-18
    rtol: float = 1e-12
    max_steps: int = 4096

    # Step size controller parameters (PID controller for adaptive time stepping)
    pcoeff: float = 0.4  # Proportional coefficient
    icoeff: float = 0.3  # Integral coefficient
    dcoeff: float = 0.0  # Derivative coefficient
    factormax: float = 1000.0  # Maximum step size growth factor

    # Output settings
    output_dir: str = "outputs"
    save_abundances: bool = True
    save_derivatives: bool = False
    save_rates: bool = False
    run_name: str = "carbox_run"

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str) -> "SimulationConfig":
        """Load configuration from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def compute_visual_extinction(self) -> float | list[float]:
        """Compute self-consistent visual extinction from column density.

        Formula: Av = base_Av + N_H / 1.6e21
        where N_H = cloud_radius_pc * number_density (converted to cm)

        Returns:
        -------
        float | list[float]
            Visual extinction [mag]
        """
        if not self.use_self_consistent_av:
            return self.visual_extinction

        # Convert parsec to cm: 1 pc = 3.086e18 cm
        pc_to_cm = 3.086e18
        cloud_radius_cm = self.cloud_radius_pc * pc_to_cm

        # Handle both scalar and list number_density
        if isinstance(self.number_density, list):
            return [
                self.base_av + (cloud_radius_cm * nd) / 1.6e21
                for nd in self.number_density
            ]
        else:
            column_density = cloud_radius_cm * self.number_density
            av = self.base_av + column_density / 1.6e21
            return av

    def get_physical_params_jax(self):
        """Get JAX arrays for physical parameters (for solver args)."""
        # Compute Av (either fixed or self-consistent)
        visual_extinction = self.compute_visual_extinction()

        return {
            "number_density": jnp.atleast_1d(jnp.array(self.number_density)),
            "temperature": jnp.atleast_1d(jnp.array(self.temperature)),
            "cr_rate": jnp.atleast_1d(jnp.array(self.cr_rate)),
            "fuv_field": jnp.atleast_1d(jnp.array(self.fuv_field)),
            "visual_extinction": jnp.atleast_1d(jnp.array(visual_extinction)),
        }

    def get_time_grid(self) -> jnp.ndarray:
        """Get time grid for parameter interpolation."""
        if isinstance(self.t_end, list):
            return jnp.array([self.t_start] + self.t_end)
        else:
            return jnp.array([self.t_start, self.t_end])

    def get_initial_number_density(self) -> float:
        """Get initial number_density value (for backward compatibility)."""
        if isinstance(self.number_density, list):
            return self.number_density[0]
        return self.number_density

    def get_final_number_density(self) -> float:
        """Get final number_density value."""
        if isinstance(self.number_density, list):
            return self.number_density[-1]
        return self.number_density

    def validate(self):
        """Basic validation of parameter ranges."""
        assert self.n_snapshots > 2, "n_snapshots must be 3 or greater"

        # Handle list parameters
        def _validate_param(
            param, name, min_val=None, max_val=None, check_positive=False
        ):
            """Validate parameter (scalar or list)."""
            values = param if isinstance(param, list) else [param]
            for val in values:
                if min_val is not None and val < min_val:
                    raise AssertionError(f"{name} out of range: {val} < {min_val}")
                if max_val is not None and val > max_val:
                    raise AssertionError(f"{name} out of range: {val} > {max_val}")
                if check_positive and val < 0:
                    raise AssertionError(f"{name} must be non-negative: {val}")

        _validate_param(self.number_density, "number_density", 1e2, 1e8)
        _validate_param(self.temperature, "temperature", 10, 1e5)
        _validate_param(
            self.visual_extinction, "visual_extinction", check_positive=True
        )

        # Validate time grid
        if isinstance(self.t_end, list):
            assert len(self.t_end) > 0, "t_end list must not be empty"
            for t in self.t_end:
                assert t > self.t_start, (
                    f"t_end values must be > t_start: {t} <= {self.t_start}"
                )
            # Check monotonically increasing
            for i in range(len(self.t_end) - 1):
                assert self.t_end[i] < self.t_end[i + 1], (
                    f"t_end must be monotonically increasing: {self.t_end}"
                )
        else:
            assert self.t_end > self.t_start, "t_end must be > t_start"

        # Validate list lengths match time grid
        time_grid_len = len(self.t_end) + 1 if isinstance(self.t_end, list) else 2

        def _check_length(param, name):
            if isinstance(param, list):
                assert len(param) == time_grid_len, (
                    f"{name} list length ({len(param)}) must match time grid length ({time_grid_len})"
                )

        _check_length(self.number_density, "number_density")
        _check_length(self.temperature, "temperature")
        _check_length(self.cr_rate, "cr_rate")
        _check_length(self.fuv_field, "fuv_field")
        _check_length(self.visual_extinction, "visual_extinction")

        assert self.solver in ["dopri5", "kvaerno5", "tsit5"], (
            f"Unknown solver: {self.solver}"
        )
