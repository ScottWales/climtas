#!/bin/bash

set -eu

function run(){
    subt=$1
    ncpus=$2

    qsub -l ncpus=$ncpus,mem=$(( ncpus * 4 ))gb -o $TMPDIR run_${subt}.sh
}

#run xarray 4
#run xarray 8
#run xarray 16
#run xarray 32
#run xarray 48

#run climtas 4
#run climtas 8
#run climtas 16
#run climtas 32
#run climtas 48

run xarray 2
run climtas 2
run xarray 2
run climtas 2
run xarray 2
run climtas 2
run xarray 2
run climtas 2
