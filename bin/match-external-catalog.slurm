#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -t 00:30:00
#SBATCH -o match-external-catalog.slurm.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

# faster python start-up
module load python/3.5-anaconda

source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# allow the python modules
export PYTHONPATH=$PWD/../py:$PYTHONPATH

# generate filelist (faster than iterating in python)
ROOT=/project/projectdirs/cosmo/data/legacysurvey/dr1
ROOT=/project/projectdirs/desiproc/dr2p
ROOT=/project/projectdirs/desiproc/dr2p
ROOT=/project/projectdirs/desiproc/dr3
ROOT=/project/projectdirs/cosmo/work/legacysurvey/dr3
ROOT=/global/cscratch1/sd/desiproc/dr4/data_release/dr4c
TRACTOR=$ROOT/tractor+tags

find $ROOT/tractor -name 'tractor-*.fits' > filelist

# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.

#export incatfile=/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/DR12Q.fits 
#export outcatfile=$SCRATCH/dr4/external/survey-dr4-dr12Q.fits

#export incatfile=/global/project/projectdirs/desi/users/yfeng1/boss-dr7q/dr7qso.fit.gz
#export outcatfile=$SCRATCH/dr4/external/survey-dr4-dr7Q.fits

#export incatfile=/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits
#export outcatfile=$SCRATCH/dr4/external/survey-dr4-superset-dr12Q.fits

export incatfile=/global/projecta/projectdirs/sdss/staging/dr13/sdss/spectro/redux/specObj-dr13.fits
export outcatfile=$SCRATCH/dr4/external/survey-dr4-specObj-dr13.fits

time srun -u --cpu_bind=no -N 1 python-mpi match-external-catalog.py -v \
--numproc 24 \
-I \
-f fits \
-F filelist \
$incatfile \
$TRACTOR \
$outcatfile
     

