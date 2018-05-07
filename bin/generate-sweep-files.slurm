#! /bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:60:00
#SBATCH -o generate-sweep-files.slurm.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

# FIXME: always update this to the dr being processed.
source dr6-envs.sh

source /project/projectdirs/desi/software/desi_environment.sh

# avoid potential MKL thread oversubscribing
export OMP_NUM_THREADS=1

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}                                                                                    
export NUMPROC=$(($SLURM_CPUS_ON_NODE / 2))

# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.
time srun -u --cpu_bind=no -n 1 python $LEGACYPIPE_DIR/bin/generate-sweep-files.py -v \
        --numproc $NUMPROC \
     -I -f fits -F $TRACTOR_FILELIST --schema blocks -d $BRICKSFILE $TRACTOR_INDIR $SWEEP_OUTDIR
