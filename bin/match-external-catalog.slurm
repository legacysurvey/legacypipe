#! /bin/bash

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:60:00
#SBATCH -o match-external-catalog.slurm.%j
#SBATCH --profile=all

set -x
export ATP_ENABLED=0

# faster python start-up
module load python/3.5-anaconda
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}                                                                                    
# use python-mpi. 'python' from anaconda fails with libpython.so.0.1 not found
# error when ran via aprun, at least on ~yfeng1's environments.

time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc 24 \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/DR12Q.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-dr12Q.fits

time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc 24 \
     -f fits -F $TRACTOR_FILELIST \
     /global/project/projectdirs/desi/users/yfeng1/boss-dr7q/dr7qso.fit.gz \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-dr7Q.fits

time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc 24 \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-superset-dr12Q.fits

time srun -u --cpu_bind=no -N 1 python-mpi $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc 24 \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/staging/dr14/sdss/spectro/redux/specObj-dr14.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-specObj-dr14.fits
