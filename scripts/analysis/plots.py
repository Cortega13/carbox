"""Generate tracer plots for physical parameters and abundances."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TracerData:
    """Container for tracer plot data."""

    tracer_id: int
    time: np.ndarray
    physical: dict[str, np.ndarray]
    species: list[str]
    abundances: np.ndarray


def safe_log(values: np.ndarray) -> np.ndarray:
    """Return log10 with floor to avoid zeros."""
    return np.log10(np.clip(values, 1e-30, None))


def parse_tracer_file(path: Path) -> TracerData:
    """Load tracer npy payload into structured data."""
    payload = np.load(path, allow_pickle=True).item()
    columns = list(payload["columns"])
    data = np.asarray(payload["data"], dtype=float)
    tracer_id = int(path.stem.split("_")[-1])
    physical_keys = ["density", "temperature", "av", "rad_field"]
    physical = {key: data[:, columns.index(key)] for key in physical_keys}
    species = columns[5:]
    abundances = data[:, 5:]
    return TracerData(tracer_id, data[:, 0], physical, list(species), abundances)


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
        colors[name] = cmap(index % cmap.N)
    return colors


def render_tracer_plot(
    tracer: TracerData,
    species_names: Sequence[str],
    output_dir: Path,
    colors: dict[str, str],
) -> Path:
    """Create and save tracer plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_physical(axes[0], tracer)
    plot_abundances(axes[1], tracer, species_names, colors)
    axes[0].set_title(f"Tracer {tracer.tracer_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tracer_{tracer.tracer_id}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def gather_tracer_paths(input_dir: Path) -> list[Path]:
    """List tracer npy files."""
    return sorted(input_dir.glob("tracer_*.npy"))


def process_tracers(input_dir: Path, output_dir: Path, count: int) -> None:
    """Generate plots for all tracers found."""
    paths = gather_tracer_paths(input_dir)
    if not paths:
        return
    tracers = [parse_tracer_file(path) for path in paths]
    species_names = build_global_species_list(tracers, count)
    colors = build_color_map(species_names)
    for tracer in tracers:
        render_tracer_plot(tracer, species_names, output_dir, colors)


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
