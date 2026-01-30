"""Print final-time abundance order-of-magnitude differences (small vs large)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TracerSnapshot:
    """Parsed tracer snapshot data (species list + final abundance row)."""

    tracer_id: int
    species: list[str]
    final_abundances: np.ndarray


def safe_log10(values: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Compute log10(values) with a floor to avoid -inf."""
    return np.log10(np.clip(values, floor, None))


def parse_tracer_snapshot(path: Path) -> TracerSnapshot:
    """Load a tracer `.npy` payload and return final abundances + species list."""
    payload = np.load(path, allow_pickle=True).item()
    columns = list(payload["columns"])
    data = np.asarray(payload["data"], dtype=float)

    tracer_id = int(path.stem.split("_")[1])
    species_start = 5  # time, density, temperature, av, rad_field
    species = [str(s) for s in columns[species_start:]]
    final_abundances = data[-1, species_start:]
    return TracerSnapshot(
        tracer_id=tracer_id, species=species, final_abundances=final_abundances
    )


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


def print_tracer_diffs(
    tracer_id: int,
    small: TracerSnapshot,
    large: TracerSnapshot,
    *,
    threshold_oom: float,
    top_k: int | None,
    floor: float,
) -> None:
    """Print per-species order-of-magnitude differences for one tracer."""
    small_idx = {name: i for i, name in enumerate(small.species)}
    large_idx = {name: i for i, name in enumerate(large.species)}
    common = sorted(set(small_idx).intersection(large_idx))
    if not common:
        print(f"Tracer {tracer_id}: no overlapping species between small/large")
        return

    small_vals = np.array(
        [small.final_abundances[small_idx[name]] for name in common], dtype=float
    )
    large_vals = np.array(
        [large.final_abundances[large_idx[name]] for name in common], dtype=float
    )

    small_log = safe_log10(small_vals, floor=floor)
    large_log = safe_log10(large_vals, floor=floor)
    delta = large_log - small_log
    mask = np.abs(delta) >= float(threshold_oom)

    if not np.any(mask):
        return

    # Sort by absolute delta descending.
    idxs = np.where(mask)[0]
    idxs = idxs[np.argsort(np.abs(delta[idxs]))[::-1]]
    if top_k is not None:
        idxs = idxs[: int(top_k)]

    print(f"Tracer {tracer_id} (|Δlog10| >= {threshold_oom}):")
    for i in idxs:
        name = common[int(i)]
        print(
            f"  {name:<12}  Δlog10={delta[i]:+7.2f}  "
            f"small_log10={small_log[i]:+8.2f}  large_log10={large_log[i]:+8.2f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Print per-tracer final abundance differences between small and large networks "
            "in orders-of-magnitude (Δlog10)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing tracer_*_small.npy and tracer_*_large.npy files",
    )
    parser.add_argument(
        "--threshold-oom",
        type=float,
        default=1e-10,
        help="Minimum |Δlog10| to print (1.0 = 1 order-of-magnitude)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Max species to print per tracer (sorted by |Δlog10|); use 0 for unlimited",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=1e-20,
        help="Floor for abundances before taking log10",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    pairs = gather_tracer_pairs(args.input_dir)
    if not pairs:
        return

    top_k = None if args.top_k == 0 else int(args.top_k)

    for small_path, large_path in pairs:
        small = parse_tracer_snapshot(small_path)
        large = parse_tracer_snapshot(large_path)
        print_tracer_diffs(
            small.tracer_id,
            small,
            large,
            threshold_oom=float(args.threshold_oom),
            top_k=top_k,
            floor=float(args.floor),
        )


if __name__ == "__main__":
    main()
