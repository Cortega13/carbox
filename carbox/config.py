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
    Physical Parameters:
        number_density : float
            Total hydrogen number density [cm^-3]. Range: [1e2, 1e6]
        temperature : float
            Gas temperature [K]. Range: [10, 1e5]
        cr_rate : float
            Cosmic ray ionization rate [s^-1]. Range: [1e-17, 1e-14]
        fuv_field : float
            FUV radiation field (Draine units). Range: [1e0, 1e5]
        visual_extinction : float
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
            Start time [seconds]
        t_end : float
            End time [seconds]
        n_snapshots : int
            Number of output snapshots (log-spaced)
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
        run_name : str
            Identifier for this run
    """

    # Physical parameters
    physics_t: list[float] = field(default_factory=lambda: [0.0])
    number_density: list[float] = field(default_factory=lambda: [1e4])
    temperature: list[float] = field(default_factory=lambda: [50.0])
    cr_rate: list[float] = field(default_factory=lambda: [1e-17])
    fuv_field: list[float] = field(default_factory=lambda: [1.0])
    visual_extinction: list[float] = field(
        default_factory=lambda: [2.0]
    )  # Can be overridden by self-consistent calculation
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
    solver: str = "kvaerno5"
    atol: float = 1e-18
    rtol: float = 1e-12
    max_steps: int = 4096

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

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def compute_visual_extinction(self) -> list[float]:
        """Compute self-consistent visual extinction from column density.

        Formula: Av = base_Av + N_H / 1.6e21
        where N_H = cloud_radius_pc * number_density (converted to cm)

        Returns:
        -------
        list[float]
            Visual extinction [mag]
        """
        if not self.use_self_consistent_av:
            return self.visual_extinction

        # Convert parsec to cm: 1 pc = 3.086e18 cm
        pc_to_cm = 3.086e18
        cloud_radius_cm = self.cloud_radius_pc * pc_to_cm

        # Column density: N_H = n_H * L [cm^-2]
        # Handle list of number densities
        number_density_arr = jnp.array(self.number_density)
        column_density = cloud_radius_cm * number_density_arr

        # Av = base_Av + N_H / 1.6e21
        av = self.base_av + column_density / 1.6e21

        return av.tolist()

    def get_physical_params_jax(self) -> dict[str, jnp.ndarray]:
        """Get JAX arrays for physical parameters (for solver args)."""
        # Compute Av (either fixed or self-consistent)
        visual_extinction = self.compute_visual_extinction()

        return {
            "physics_t": jnp.array(self.physics_t),
            "number_density": jnp.array(self.number_density),
            "temperature": jnp.array(self.temperature),
            "cr_rate": jnp.array(self.cr_rate),
            "fuv_field": jnp.array(self.fuv_field),
            "visual_extinction": jnp.array(visual_extinction),
        }

    def validate(self) -> None:
        """Basic validation of parameter ranges."""
        # Validate lengths
        n_points = len(self.physics_t)
        assert n_points >= 2, "physics_t must have at least 2 points"
        assert self.physics_t[0] == 0.0, "physics_t must start at 0.0"

        # Check for strictly increasing time
        physics_t_arr = jnp.array(self.physics_t)
        assert jnp.all(jnp.diff(physics_t_arr) > 0), (
            "physics_t must be strictly increasing"
        )

        assert len(self.number_density) == n_points, "number_density length mismatch"
        assert len(self.temperature) == n_points, "temperature length mismatch"
        assert len(self.cr_rate) == n_points, "cr_rate length mismatch"
        assert len(self.fuv_field) == n_points, "fuv_field length mismatch"
        assert len(self.visual_extinction) == n_points, (
            "visual_extinction length mismatch"
        )

        assert self.solver in ["dopri5", "kvaerno5", "tsit5"], (
            f"Unknown solver: {self.solver}"
        )
