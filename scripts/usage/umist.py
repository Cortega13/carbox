"""Testing the umist network."""

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import NetworkNames

config = SimulationConfig(
    time_grid_years=[0.0, 1e5],
    number_density_grid=[1e4, 1e4],
    temperature_grid=[50.0, 50.0],
    cr_rate_grid=[1e-17, 1e-17],
    fuv_field_grid=[1.0, 1.0],
    visual_extinction_grid=[2.0, 2.0],
    solver="kvaerno5",
    max_steps=500000,
    atol=1e-14,
    rtol=1e-8,
)

results = run_simulation(
    network_file="network_files/rate22_final.rates",
    config=config,
    format_type=NetworkNames.umist,
)

print("Simulation finished.")
print(
    f"Final abundances stored in: {config.output_dir}/{config.run_name}_abundances.csv"
)
