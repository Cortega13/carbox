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
  --skip-existing \
  --random-count 40
```

## 2) Generate MPI commandlines

For local
```bash
python benchmarks/cosmicai/generate_commandlines.py \
  --csv-dir benchmarks/cosmicai/data/turbulence_tracers_csv \
  --command-file benchmarks/cosmicai/commandlines.txt \
  --output-dir outputs \
  --skip-existing
```

For vista
```bash
python3 /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/generate_commandlines.py \
  --csv-dir /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/data/turbulence_tracers_csv \
  --benchmark-script /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/carbox_cosmicai_benchmark.py \
  --command-file /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/commandlines.txt \
  --output-dir /work/09338/carlos9/vista/carbox/outputs \
  --skip-existing
```

This creates one line per tracer CSV, e.g.:

```bash
python3 benchmarks/cosmicai/carbox_cosmicai_benchmark.py --tracer-csv ... --output-dir outputs
```

## 3) Run with Slurm + pylauncher

```bash
sbatch benchmarks/cosmicai/run_pylauncher.slurm
```

The Slurm script expects:
- a `venv` in the repo root
- commandlines at `benchmarks/cosmicai/commandlines.txt`

Adjust `benchmarks/cosmicai/run_pylauncher.slurm` if your environment differs.
