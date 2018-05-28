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

# FIXME: always update this to the dr being processed.
source dr6-envs.sh
source /project/projectdirs/desi/software/desi_environment.sh

# avoid potential MKL thread oversubscribing
export OMP_NUM_THREADS=1

export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}                                                                                    
export NUMPROC=$(($SLURM_CPUS_ON_NODE / 2))

time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc $NUMPROC \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/DR12Q.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-dr12Q.fits --copycols MJD PLATE FIBERID RERUN_NUMBER 

time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc $NUMPROC \
     -f fits -F $TRACTOR_FILELIST \
     /global/project/projectdirs/cosmo/staging/sdss/dr7/dr7qso.fit.gz \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-dr7Q.fits --copycols SMJD PLATE FIBER RERUN 

time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc $NUMPROC \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/data/sdss/dr12/boss/qso/DR12Q/Superset_DR12Q.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-superset-dr12Q.fits --copycols MJD PLATE FIBERID 

time srun -u --cpu_bind=no -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py -v --numproc $NUMPROC \
     -f fits -F $TRACTOR_FILELIST \
     /global/projecta/projectdirs/sdss/data/sdss/dr14/sdss/spectro/redux/specObj-dr14.fits \
     $TRACTOR_INDIR \
     $EXTERNAL_OUTDIR/survey-$dr-specObj-dr14.fits --copycols MJD PLATE FIBERID RUN2D 
