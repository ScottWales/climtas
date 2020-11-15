#!/bin/bash
#PBS -l storage=gdata/ub4
#PBS -l wd
#PBS -l walltime=10:00

conda activate dev

python run_xarray.sh
