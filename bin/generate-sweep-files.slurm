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

# faster python start-up
module load python/3.5-anaconda
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}                                                                                    

# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.
time srun -u --cpu_bind=no -n 1 python-mpi $LEGACYPIPE_DIR/bin/generate-sweep-files.py -v --numproc 24 \
     -I -f fits -F $TRACTOR_FILELIST --schema blocks -d $BRICKSFILE $TRACTOR_INDIR $SWEEP_OUTDIR
