#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o generate-sweep-files.slurm.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

# faster python start-up
module load python/3.5-anaconda

source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# allow the python modules
export PYTHONPATH=$PWD/../py:$PYTHONPATH

# generate filelist (faster than iterating in python)
ROOT=/global/project/projectdirs/cosmo/work/legacysurvey/dr5/DR5_out
TRACTOR=$ROOT/tractor
find $TRACTOR -name 'tractor-*.fits' > filelist

# define the survey-bricks.fits file (speeds up scanning) and the output directory
BRICKSFILE=/global/cscratch1/sd/desiproc/dr5/survey-bricks.fits.gz

DESTINATION=$CSCRATCH/sweep
/usr/bin/mkdir -p $DESTINATION

# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.

time srun -u --cpu_bind=no -n 1 python-mpi generate-sweep-files.py -v --numproc 24 \
     -I -f fits -F filelist --schema blocks -d $BRICKSFILE $TRACTOR $DESTINATION
