#!/bin/bash
#PBS -l storage=gdata/ub4+gdata/hh5
#PBS -l wd
#PBS -l jobfs=100gb
#PBS -l walltime=10:00
#PBS -j oe

eval "$('/g/data/hh5/public/apps/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

conda activate dev

set -eu

python bench_climtas.py
