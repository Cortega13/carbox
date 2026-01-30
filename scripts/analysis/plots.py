"""Generate tracer plots for physical parameters and abundances."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

PLOT_SPECIES = [
    "H2",
    "CO",
    "C",
    "C+",
    "O",
    "H3+",
    "HCO+",
    "H3O+",
    "E-",
    "MG+",
    "H2O",
    "OH",
]


@dataclass
class TracerData:
    """Container for tracer plot data."""

    tracer_id: int
    time: np.ndarray
    physical: dict[str, np.ndarray]
    species: list[str]
    abundances: np.ndarray


class TracerPair(NamedTuple):
    """Small/large network outputs for one tracer id."""

    tracer_id: int
    small: TracerData
    large: TracerData


def safe_log(values: np.ndarray) -> np.ndarray:
    """Return log10 with floor to avoid zeros."""
    return np.log10(np.clip(values, 1e-30, None))


def parse_tracer_file(path: Path) -> TracerData:
    """Load tracer npy payload into structured data."""
    payload = np.load(path, allow_pickle=True).item()
    columns = list(payload["columns"])
    data = np.asarray(payload["data"], dtype=float)
    tracer_id = int(path.stem.split("_")[1])

    time_key = "time_years" if "time_years" in columns else "time"
    physical_keys = ["density", "temperature", "av", "rad_field"]
    time = data[:, columns.index(time_key)]
    physical_keys = ["density", "temperature", "av", "rad_field"]
    physical = {key: data[:, columns.index(key)] for key in physical_keys}

    # Expected layout from benchmark runner:
    # time_years, density, temperature, av, rad_field, <species...>
    species_start = 5
    species = columns[species_start:]
    abundances = data[:, species_start:]
    return TracerData(tracer_id, time, physical, list(species), abundances)


def build_global_species_list(tracers: Sequence[TracerData], count: int) -> list[str]:
    """Select a global top species list by mean final abundance."""
    if not tracers:
        return []
    species = tracers[0].species
    final_matrix = np.stack([tracer.abundances[-1] for tracer in tracers], axis=0)
    mean_final = final_matrix.mean(axis=0)
    capped = min(count, mean_final.shape[0])
    order = np.argsort(mean_final)[::-1][:capped]
    return [species[int(idx)] for idx in order]


def plot_physical(ax, tracer: TracerData) -> None:
    """Plot physical parameters."""
    for key, label in [
        ("density", "Density"),
        ("temperature", "Temperature"),
        ("av", "Av"),
        ("rad_field", "Rad Field"),
    ]:
        ax.plot(tracer.time[1:], safe_log(tracer.physical[key][1:]), label=label)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("log10(Value)")
    ax.legend()


def plot_abundances(
    ax, tracer: TracerData, species_names: Sequence[str], colors: dict[str, str]
) -> None:
    """Plot species abundances."""
    index_by_name = {name: idx for idx, name in enumerate(tracer.species)}
    for name in species_names:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        ax.plot(
            tracer.time[1:],
            safe_log(tracer.abundances[1:, idx]),
            label=name,
            color=colors.get(name),
        )
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("log10(Abundance)")
    ax.legend(ncol=2, fontsize=8, loc="lower right", framealpha=0.85)


def build_color_map(species: Sequence[str]) -> dict[str, str]:
    """Assign colors to species."""
    cmap = plt.get_cmap("tab20")
    colors: dict[str, str] = {}
    for index, name in enumerate(species):
        colors[name] = cmap(index % cmap.N)  # type:ignore
    return colors


def render_tracer_plot(
    tracer_pair: TracerPair,
    species_names: Sequence[str],
    output_dir: Path,
    colors: dict[str, str],
) -> Path:
    """Create and save tracer plots (physical + small + large abundances)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Top: physical parameters (use small output; should match large in practice).
    plot_physical(axes[0], tracer_pair.small)
    axes[0].set_title(f"Tracer {tracer_pair.tracer_id}")

    # Middle: small network
    plot_abundances(axes[1], tracer_pair.small, species_names, colors)
    axes[1].set_title("Small network")

    # Bottom: large network
    plot_abundances(axes[2], tracer_pair.large, species_names, colors)
    axes[2].set_title("Large network")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tracer_{tracer_pair.tracer_id}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def gather_tracer_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Return matched (small_path, large_path) pairs for each tracer id."""
    small_paths = sorted(input_dir.glob("tracer_*_small.npy"))
    large_by_id: dict[int, Path] = {}
    for path in input_dir.glob("tracer_*_large.npy"):
        tracer_id = int(path.stem.split("_")[1])
        large_by_id[tracer_id] = path

    pairs: list[tuple[Path, Path]] = []
    for small_path in small_paths:
        tracer_id = int(small_path.stem.split("_")[1])
        large_path = large_by_id.get(tracer_id)
        if large_path is None:
            continue
        pairs.append((small_path, large_path))
    return pairs


def process_tracers(input_dir: Path, output_dir: Path, count: int) -> None:
    """Generate plots for all tracers found."""
    pairs = gather_tracer_pairs(input_dir)
    if not pairs:
        return

    tracer_pairs: list[TracerPair] = []
    for small_path, large_path in pairs:
        small = parse_tracer_file(small_path)
        large = parse_tracer_file(large_path)
        tracer_pairs.append(TracerPair(small.tracer_id, small, large))

    # Species choice: keep fixed list for consistency across plots.
    species_names = PLOT_SPECIES
    colors = build_color_map(species_names)
    for pair in tracer_pairs:
        render_tracer_plot(pair, species_names, output_dir, colors)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot tracer physical parameters and abundances"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing tracer_*.npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/analysis/plots"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--species-count",
        type=int,
        default=40,
        help="Number of top species to plot",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for tracer plotting."""
    args = parse_args()
    process_tracers(args.input_dir, args.output_dir, args.species_count)


if __name__ == "__main__":
    main()
