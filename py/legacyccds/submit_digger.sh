#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:03:00
#SBATCH -J digger
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

module load mpi4py
#module load mpi4py-hpcp


if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let cores*=${SLURM_JOB_NUM_NODES}


date
echo Using nodes=${SLURM_JOB_NUM_NODES} on $NERSC_HOST, for total tasks=$cores
#srun -n ${cores} -N ${SLURM_JOB_NUM_NODES} python digger.py --file_list test_mosaic.txt --type cpimages_mosaic  
srun -n ${cores} -N ${SLURM_JOB_NUM_NODES} python hello.py
date
echo DONE
