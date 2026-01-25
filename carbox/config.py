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
    Physical Parameters (time-grid):
        time_grid_years : List[float]
            Simulation time grid [years], strictly increasing
        number_density_grid : List[float]
            Total hydrogen number density [cm^-3]
        temperature_grid : List[float]
            Gas temperature [K]
        cr_rate_grid : List[float]
            Cosmic ray ionization rate [s^-1]
        fuv_field_grid : List[float]
            FUV radiation field (Draine units)
        visual_extinction_grid : List[float]
            Visual extinction Av [mag] (ignored if use_self_consistent_av is True)
        gas_to_dust_ratio : float
            Gas-to-dust mass ratio. Typical: 100 (= 0.01 dust/gas)

    Initial Abundances:
        initial_abundances : Dict[str, float]
            Species name -> fractional abundance (relative to number_density_grid[0])
            Example: {"H2": 1.0, "O": 2e-4, "C": 1e-4}
        abundance_floor : float
            Minimum abundance for all species (numerical stability)

    Integration Parameters:
        solver : str
            Solver name: 'dopri5', 'kvaerno5', 'tsit5'
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        max_steps : int
            Maximum integration steps

    Output Settings:
        output_dir : str
            Directory for output files
        save_abundances : bool
            Save abundance time series
        save_derivatives : bool
            Save dy/dt at each snapshot
        save_rates : bool
            Save reaction rates at each snapshot
        save_plots : bool
            Save evolution plots (physical parameters and key species)
        run_name : str
            Identifier for this run
    """

    # Physical parameters (time-grid)
    time_grid_years: list[float] = field(default_factory=lambda: [0.0, 1e6])
    number_density_grid: list[float] = field(default_factory=lambda: [1e4, 1e4])
    temperature_grid: list[float] = field(default_factory=lambda: [50.0, 50.0])
    cr_rate_grid: list[float] = field(default_factory=lambda: [1e-17, 1e-17])
    fuv_field_grid: list[float] = field(default_factory=lambda: [1.0, 1.0])
    visual_extinction_grid: list[float] = field(default_factory=lambda: [2.0, 2.0])
    gas_to_dust_ratio: float = 100.0

    # Cloud geometry (for photoreaction shielding and self-consistent Av)
    cloud_radius_pc: float = 1.0  # Cloud radius in parsecs
    base_av: float = 0.0  # Base Av before column density contribution
    use_self_consistent_av: bool = False  # Compute Av from column density

    # Initial abundances (fractional relative to number_density_grid[0])
    initial_abundances: dict[str, float] = field(
        default_factory=lambda: {
            "H2": 1.0,
            "O": 2e-4,
            "C": 1e-4,
        }
    )
    abundance_floor: float = 1e-30

    # Integration parameters
    solver: str = "kvaerno5"
    atol: float = 1e-18
    rtol: float = 1e-12
    max_steps: int = 4096

    # Output settings
    output_dir: str = "outputs"
    save_abundances: bool = True
    save_derivatives: bool = False
    save_rates: bool = False
    save_plots: bool = True
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

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def compute_visual_extinction_grid(self) -> list[float]:
        """Compute visual extinction grid.

        Formula: Av = base_Av + N_H / 1.6e21
        where N_H = cloud_radius_pc * number_density (converted to cm)

        Returns:
        -------
        list[float]
            Visual extinction grid [mag]
        """
        if not self.use_self_consistent_av:
            return list(self.visual_extinction_grid)

        # Convert parsec to cm: 1 pc = 3.086e18 cm
        pc_to_cm = 3.086e18
        cloud_radius_cm = self.cloud_radius_pc * pc_to_cm

        # Column density: N_H = n_H * L [cm^-2]
        column_density = cloud_radius_cm * jnp.array(self.number_density_grid)

        # Av = base_Av + N_H / 1.6e21
        av = self.base_av + column_density / 1.6e21

        return [float(value) for value in jnp.asarray(av)]

    def get_physical_param_grids_jax(self) -> dict[str, jnp.ndarray]:
        """Get JAX arrays for physical parameter grids (for solver args)."""
        return {
            "time_grid_years": jnp.array(self.time_grid_years),
            "number_density_grid": jnp.array(self.number_density_grid),
            "temperature_grid": jnp.array(self.temperature_grid),
            "cr_rate_grid": jnp.array(self.cr_rate_grid),
            "fuv_field_grid": jnp.array(self.fuv_field_grid),
            "visual_extinction_grid": jnp.array(self.compute_visual_extinction_grid()),
        }

    def get_initial_number_density(self) -> float:
        """Get number density at the first timepoint."""
        return float(self.number_density_grid[0])

    def validate(self) -> None:
        """Basic validation of parameter ranges."""
        if len(self.time_grid_years) < 2:
            raise ValueError("time_grid_years must have at least 2 points")
        if any(
            t2 <= t1
            for t1, t2 in zip(self.time_grid_years, self.time_grid_years[1:])
        ):
            raise ValueError("time_grid_years must be strictly increasing")

        grid_lengths = {
            "number_density_grid": len(self.number_density_grid),
            "temperature_grid": len(self.temperature_grid),
            "cr_rate_grid": len(self.cr_rate_grid),
            "fuv_field_grid": len(self.fuv_field_grid),
            "visual_extinction_grid": len(self.visual_extinction_grid),
        }
        expected_len = len(self.time_grid_years)
        for name, length in grid_lengths.items():
            if length != expected_len:
                raise ValueError(f"{name} must match time_grid_years length")

        if any(n < 1e2 or n > 1e8 for n in self.number_density_grid):
            raise ValueError("number_density_grid out of physical range")
        if any(t < 10 or t > 1e5 for t in self.temperature_grid):
            raise ValueError("temperature_grid out of range")
        if any(av < 0 for av in self.visual_extinction_grid):
            raise ValueError("visual_extinction_grid out of range")
        assert self.solver in ["dopri5", "kvaerno5", "tsit5"], (
            f"Unknown solver: {self.solver}"
        )
