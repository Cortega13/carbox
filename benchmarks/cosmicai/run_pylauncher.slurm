#!/bin/bash
#SBATCH -N 1
#SBATCH -p gg
#SBATCH -t 01:00:00
#SBATCH -A AST24021
#SBATCH -J carbox_cosmicai_benchmark
#SBATCH -o benchmarks/cosmicai/job.o%j
module load gcc/13.2.0
source /work/09338/carlos9/vista/carbox/.venv/bin/activate
python /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/carbox_cosmicai_benchmark.py --output-dir /work/09338/carlos9/vista/carbox/benchmarks/cosmicai/outputs --random-count=8000
