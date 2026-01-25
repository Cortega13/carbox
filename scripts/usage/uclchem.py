"""Testing the umist network."""

import math

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import NetworkNames

time_grid_years = [i * 1e4 for i in range(121)]
cycles = 16.0
phase = [2.0 * math.pi * cycles * (t / time_grid_years[-1]) for t in time_grid_years]
scale = [10 ** (2 * math.sin(p)) for p in phase]
number_density_grid = [1e5 * s for s in scale]
temperature_grid = [1e3 * s for s in scale]
cr_rate_grid = [1e-17 * s for s in scale]
fuv_field_grid = [1.0 * s for s in scale]
visual_extinction_grid = [2.0 * s for s in scale]

config = SimulationConfig(
    time_grid_years=time_grid_years,
    number_density_grid=number_density_grid,
    temperature_grid=temperature_grid,
    cr_rate_grid=cr_rate_grid,
    fuv_field_grid=fuv_field_grid,
    visual_extinction_grid=visual_extinction_grid,
    solver="kvaerno5",
    max_steps=500000,
    atol=1e-15,
    rtol=1e-6,
)

results = run_simulation(
    network_file="network_files/uclchem_gas_phase_only.csv",
    config=config,
    format_type=NetworkNames.uclchem,
)

print("Simulation finished.")
print(
    f"Final abundances stored in: {config.output_dir}/{config.run_name}_abundances.csv"
)
