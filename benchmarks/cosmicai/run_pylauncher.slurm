#!/bin/bash
#SBATCH -N 1
#SBATCH -p gg
#SBATCH -t 01:00:00
#SBATCH -A AST24021
#SBATCH -J carbox_cosmicai_benchmark
#SBATCH -o job.o%j
module load gcc
module load python3_mpi/3.11.8

CARBOX_ROOT="/work/09338/carlos9/vista/carbox"

source $CARBOX_ROOT/venv/bin/activate

export PYTHONPATH=$CARBOX_ROOT:$PYTHONPATH

python3 $CARBOX_ROOT/benchmarks/cosmicai/carbox_cosmicai_benchmark.py --output-dir $CARBOX_ROOT/benchmarks/cosmicai/outputs --random-count=8000
