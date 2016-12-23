#! /bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o match-redmapper-v5.10.slurm.cori.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

module load python/3.5-anaconda
# faster python start-up
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# allow the python modules
export PYTHONPATH=$PWD/../py:$PYTHONPATH

# generate filelist (faster than iterating in python)
ROOT=/project/projectdirs/cosmo/data/legacysurvey/dr1
ROOT=/project/projectdirs/desiproc/dr2p
ROOT=/project/projectdirs/desiproc/dr2p
ROOT=/project/projectdirs/desiproc/dr3

ROOT=/project/projectdirs/cosmo/data/legacysurvey/dr3
TRACTOR=$ROOT/tractor

FILELIST=`mktemp`
find $ROOT/tractor -name 'tractor-*.fits' > $FILELIST

# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.

#srun -u --cpu_bind=no -n 1 python-mpi generate-sweep-files.py --numproc 64 -F filelist -v -d /project/projectdirs/cosmo/data/legacysurvey/dr1/decals-bricks.fits /project/projectdirs/cosmo/data/legacysurvey/dr1/tractor $SCRATCH/sandbox/sweep-dr1-stripes

# unzip because .gz crashes fitsio
mkdir -p $SCRATCH/ioannis-redmapper/
gunzip -c /global/cscratch1/sd/ioannis/redmapper_isedfit_v5.10_centrals.fits.gz > $SCRATCH/ioannis-redmapper/redmapper_isedfit_v5.10_centrals.fits

time srun -u --cpu_bind=no -N 1 python-mpi match-boss-catalogue.py -v \
--numproc 64 \
-I \
-f fits hdf5 \
-F $FILELIST \
$SCRATCH/ioannis-redmapper/redmapper_isedfit_v5.10_centrals.fits \
$TRACTOR \
$SCRATCH/sandbox/decals-dr3/tractor-redmapper_isedfit_v5.10_centrals.fits

