# CosmicAI MPI Benchmark (Vista/TACC)

This folder contains scripts to extract per-tracer CSVs from the CosmicAI `.npy` file and generate MPI commandlines for running the simplified benchmark on a single Vista node (144 cores) via Slurm + pylauncher.

## Prereqs
- Vista/TACC node (1 node, 144 cores)
- Slurm + pylauncher modules available
- Virtualenv with carbox dependencies

## 1) Extract tracer CSVs

Example for the default M600 dataset:

```bash
python benchmarks/cosmicai/npy_to_csv.py \
  --npy-path benchmarks/cosmicai/data/M600_seed1_trace_cells.npy \
  --output-dir benchmarks/cosmicai/data/turbulence_tracers_csv \
  --timestep-kyr 8.299 \
  --clip 400 \
  --discretization 1 \
  --skip-existing \
  --random-count 40 \
  --seed 123
```

Notes:
- Use `--tracers 1 2 3` to select specific tracers.
- Use `--random-count 100 --seed 123` to sample tracers.

## 2) Generate MPI commandlines

```bash
python benchmarks/cosmicai/generate_commandlines.py \
  --csv-dir benchmarks/cosmicai/data/turbulence_tracers_csv \
  --command-file benchmarks/cosmicai/commandlines.txt \
  --output-dir outputs/cosmicai \
  --mpi-launcher srun \
  --ranks 1 \
  --skip-existing
```

This creates one line per tracer CSV, e.g.:

```bash
srun -n 1 python benchmarks/cosmicai/carbox_cosmicai_benchmark.py --tracer-csv ... --output-dir outputs/cosmicai
```

## 3) Run with Slurm + pylauncher

```bash
sbatch benchmarks/cosmicai/run_pylauncher.slurm
```

The Slurm script expects:
- a `venv` in the repo root
- commandlines at `benchmarks/cosmicai/commandlines.txt`

Adjust `benchmarks/cosmicai/run_pylauncher.slurm` if your environment differs.

## Single-tracer manual run

```bash
python benchmarks/cosmicai/carbox_cosmicai_benchmark.py \
  --tracer-csv benchmarks/cosmicai/data/turbulence_tracers_csv/tracer_10.csv \
  --output-dir outputs/cosmicai
```
