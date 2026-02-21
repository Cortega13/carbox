"""Compress tracer `.npy` payloads into one float32 `(M, R, C)` `.npy`.

Inputs: `tracer_<id>_<model>.npy` (default models: `small large`). Each input file
contains a dict with `columns` and 2D `data`.

Outputs:
- `output_path`: a single float32 array shaped `(n_models, n_rows, n_cols)` where
  each model is a stacked-row table across all tracers, and the first column is
  a `tracer_id` column.
- `columns_json`: a JSON mapping of column-index -> column-name for axis2.

This assumes identical `columns` *within each model* across all tracers.

If models have different columns (e.g. `small` has fewer species than `large`), we
output the union of all columns and fill missing values with NaN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _tracer_id(path: Path) -> int:
    """Parse tracer id from `tracer_<id>_<model>.npy`."""
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "tracer":
        raise ValueError(f"Unrecognized tracer filename: {path.name}")
    return int(parts[1])


def _matched_tracer_ids(input_dir: Path, models: list[str]) -> list[int]:
    """Tracer ids that have *all* model outputs."""
    ids: set[int] | None = None
    for m in models:
        these = {_tracer_id(p) for p in input_dir.glob(f"tracer_*_{m}.npy")}
        ids = these if ids is None else (ids & these)
    return sorted(ids or set())


def compress_tracers_to_npy(
    input_dir: Path, output_path: Path, columns_json: Path, models: list[str]
) -> None:
    """Write `output_path` array and `columns_json` mapping."""
    input_dir, output_path = Path(input_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracer_ids = _matched_tracer_ids(input_dir, models)
    if not tracer_ids:
        raise FileNotFoundError(
            f"No matched tracer sets found in {input_dir} for models={models} (expected tracer_*_<model>.npy)"
        )

    def load(tracer_id: int, model: str) -> tuple[list[str], np.ndarray]:
        payload = np.load(
            input_dir / f"tracer_{tracer_id}_{model}.npy", allow_pickle=True
        ).item()
        return [str(x) for x in payload["columns"]], np.asarray(
            payload["data"], np.float32
        )

    model_cols = {m: load(tracer_ids[0], m)[0] for m in models}
    union: list[str] = []
    for m in models:
        union += [c for c in model_cols[m] if c not in union]
    union_idx = {c: i for i, c in enumerate(union)}
    dst_cols = {m: [union_idx[c] + 1 for c in model_cols[m]] for m in models}

    tables: list[np.ndarray] = []
    for model in models:
        blocks: list[np.ndarray] = []
        for tracer_id in tqdm(tracer_ids, desc=f"model={model}"):
            cols, data = load(tracer_id, model)
            if cols != model_cols[model]:
                raise ValueError(
                    f"Column mismatch within model={model} (tracer_id={tracer_id})"
                )
            block = np.full((data.shape[0], 1 + len(union)), np.nan, np.float32)
            block[:, 0] = tracer_id
            block[:, dst_cols[model]] = data
            blocks.append(block)
        tables.append(np.concatenate(blocks, axis=0))

    max_rows = max(t.shape[0] for t in tables)
    out = np.full((len(models), max_rows, tables[0].shape[1]), np.nan, np.float32)
    for i, table in enumerate(tables):
        out[i, : table.shape[0]] = table
    np.save(output_path, out)

    columns = ["tracer_id", *union]
    columns_json.parent.mkdir(parents=True, exist_ok=True)
    columns_json.write_text(
        json.dumps({str(i): c for i, c in enumerate(columns)}, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(
        description="Compress tracer_*_<model>.npy outputs into one stacked 3D .npy."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing tracer_*_<model>.npy",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/tracers.npy"),
        help="Output .npy file path (float32 array with shape MxRxC)",
    )
    p.add_argument(
        "--columns-json",
        type=Path,
        default=None,
        help="Output JSON mapping column-index -> column-name (default: <output-path>.columns.json)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["small", "large"],
        help="Model labels to include (must exist for each tracer id)",
    )
    return p.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    columns_json = (
        args.columns_json
        if args.columns_json is not None
        else Path(args.output_path).with_suffix(".columns.json")
    )
    compress_tracers_to_npy(
        args.input_dir, args.output_path, columns_json, list(args.models)
    )


if __name__ == "__main__":
    main()
