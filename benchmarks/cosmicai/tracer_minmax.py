"""Print min/max for CosmicAI benchmark physical parameters."""

import argparse
from pathlib import Path

import numpy as np

KEYS = ("density", "temperature", "av", "rad_field")


def density_to_number_density(density: np.ndarray) -> np.ndarray:
    """Convert mass density to number density."""
    hydrogen_mass = 1.66053906660e-24
    mean_molecular_mass = 1.4168138025
    return density / (mean_molecular_mass * hydrogen_mass)


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Print min/max for CosmicAI NPY data")
    parser.add_argument(
        "--npy-path",
        type=Path,
        required=True,
        help="Path to benchmark .npy file",
    )
    args = parser.parse_args()
    data = np.load(args.npy_path, mmap_mode="r")
    density = density_to_number_density(data[..., 0])
    temp = data[..., 1]
    av = data[..., 2]
    rad = data[..., 4]
    values = (density, temp, av, rad)
    for key, series in zip(KEYS, values):
        print(f"{key}: {float(series.min())} {float(series.max())}")


if __name__ == "__main__":
    main()
