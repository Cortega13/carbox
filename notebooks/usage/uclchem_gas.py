"""Testing the umist network."""

from carbox.config import SimulationConfig
from carbox.main import run_simulation
from carbox.parsers import NetworkNames

config = SimulationConfig(
    number_density=1e4,
    temperature=100.0,
    t_end=1e6,
    solver="kvaerno5",
    max_steps=500000,
    n_snapshots=3,
    atol=1e-14,
    rtol=1e-8,
)

results = run_simulation(
    network_file="data/uclchem_gas_phase_only.csv",
    config=config,
    format_type=NetworkNames.uclchem,
)

print("Simulation finished.")
print(
    f"Final abundances stored in: {config.output_dir}/{config.run_name}_abundances.csv"
)

"""Outputs
============================================================
Carbox Chemical Kinetics Simulation
============================================================
Network file: data/uclchem_gas_phase_only.csv
Run name: carbox_run

Validating configuration...
Loading reaction network from data/uclchem_gas_phase_only.csv...
✓ Detected 3 special photoreactions
  Loaded 163 species
  Loaded 2227 reactions

Initializing abundances...
Setting initial abundance for H2: 1.000e+00 (fractional)
Setting initial abundance for O: 2.000e-04 (fractional)
Setting initial abundance for C: 1.000e-04 (fractional)
Initial Abundances Summary
========================================
Species     Abundance [cm^-3]   Fractional
----------------------------------------
H2                  1.000e+04    9.997e-01
O                   2.000e+00    1.999e-04
C                   1.000e+00    9.997e-05
SO2+                1.000e-26    9.997e-31
SO2                 1.000e-26    9.997e-31
SO+                 1.000e-26    9.997e-31
SO                  1.000e-26    9.997e-31
SISH+               1.000e-26    9.997e-31

Initial elemental abundances:
  C: 1.000e+00 cm^-3
  H: 2.000e+04 cm^-3
  O: 2.000e+00 cm^-3
  Net charge: 0.000e+00

Compiling JAX network...
  Network compiled successfully

Solving ODE system with kvaerno5...
  Time range: 0.00e+00 - 1.00e+06 years
  Snapshots: 3
  Compiling solver (first call)...
  Integration complete in 6.62 seconds
  Steps: 232 (accepted: 232, rejected: 0)

Saving results...
Saved abundances to: output/carbox_run_abundances.csv
Saved metadata to: output/carbox_run_metadata.json
Saved summary to: output/carbox_run_summary.txt

============================================================
Simulation complete! Total time: 15.65 seconds
Output saved to: output/
============================================================
Simulation finished.
Final abundances stored in: output/carbox_run_abundances.csv
"""
