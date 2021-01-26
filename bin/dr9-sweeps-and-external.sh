#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -L SCRATCH,project
#SBATCH -C haswell

# ADM an all-in-one shell script for running the sweeps and external-match
# ADM files for DR9. Although you can slurm this, it also usually runs to
# ADM completion within 4 hours on an interactive node, e.g.
#   salloc -N 1 -C haswell -t 04:00:00 --qos interactive -L SCRATCH,project

# ADM you may need to change the top-level environment variables from
# -------------------------------here--------------------------------

# ADM set the data release and hence the main input directory.
dr=dr9
drdir=/global/cfs/cdirs/cosmo/work/legacysurvey/$dr

# ADM write to scratch.
droutdir=$CSCRATCH/$dr

# ADM example set-ups for using custom code are commented out!
# export LEGACYPIPE_DIR=$HOME/git/legacypipe/
# source /project/projectdirs/desi/software/desi_environment.sh
export LEGACYPIPE_DIR=/src/legacypipe
# export LEGACYPIPE_DIR=$drdir/code/legacypipe
export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:$LEGACYPIPE_DIR/py

# ADM if the bricks and matching files are common to all surveys you will
# ADM need to uncomment the next line and comment the line in the do block.
# export BRICKSFILE=$drdir/survey-bricks.fits.gz

# ADM location of external-match files.
export SDSSDIR=/global/cfs/cdirs/sdss/data/sdss/

# ------------------------------to here------------------------------

# ADM a sensible number of processors on which to run.
export NUMPROC=$(($SLURM_CPUS_ON_NODE / 2))
# ADM the sweeps need more memory since we started to write three files.
export SWEEPS_NUMPROC=$(($SLURM_CPUS_ON_NODE / 5))

# ADM run once for each of the DECaLS and MzLS/BASS surveys.
for survey in north south
do
    # ADM if the bricks and matching files are NOT common to all surveys.
    export BRICKSFILE=$drdir/$survey/survey-bricks.fits.gz

    # ADM set up the per-survey input and output directories.
    export INDIR=$drdir/$survey
    echo working on input directory $INDIR
    export TRACTOR_INDIR=$INDIR/tractor

    export OUTDIR=$droutdir/$survey
    echo writing to output directory $OUTDIR
    export SWEEP_OUTDIR=$OUTDIR/sweep
    export EXTERNAL_OUTDIR=$OUTDIR/external
    export TRACTOR_FILELIST=$OUTDIR/tractor_filelist

    mkdir -p $SWEEP_OUTDIR
    mkdir -p $EXTERNAL_OUTDIR

    # ADM write the bricks of interest to the output directory.
    find $TRACTOR_INDIR -name 'tractor-*.fits' > $TRACTOR_FILELIST
    echo wrote list of tractor files to $TRACTOR_FILELIST

    # ADM run the sweeps. Should never have to use the --ignore option here,
    # ADM which usually means tthere are some discrepancies in the data model!
    echo running sweeps for the $survey
    time srun -N 1 python $LEGACYPIPE_DIR/bin/generate-sweep-files.py \
         -v --numproc $SWEEPS_NUMPROC -f fits -F $TRACTOR_FILELIST --schema blocks \
         -d $BRICKSFILE $TRACTOR_INDIR $SWEEP_OUTDIR
    echo done running sweeps for the $survey

    # ADM run each of the external matches.
    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr7Q.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr7/dr7qso.fit.gz \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-dr7Q.fits --copycols SMJD PLATE FIBER RERUN
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr7Q.fits

    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr12Q.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr12/boss/qso/DR12Q/DR12Q.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-dr12Q.fits --copycols MJD PLATE FIBERID RERUN_NUMBER
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr12Q.fits

    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr12Q.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr12/boss/qso/DR12Q/Superset_DR12Q.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr12Q.fits --copycols MJD PLATE FIBERID
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr12Q.fits

    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-specObj-dr16.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr16/sdss/spectro/redux/specObj-dr16.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-specObj-dr16.fits --copycols MJD PLATE FIBERID RUN2D
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-specObj-dr16.fits

    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr16Q-v4.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
         $SDSSDIR/dr16/eboss/qso/DR16Q/DR16Q_v4.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-dr16Q-v4.fits --copycols MJD PLATE FIBERID
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-dr16Q-v4.fits

    echo making $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr16Q-v3.fits
    time srun -N 1 python $LEGACYPIPE_DIR/bin/match-external-catalog.py \
         -v --numproc $NUMPROC -f fits -F $TRACTOR_FILELIST \
	 $SDSSDIR/dr16/eboss/qso/DR16Q/DR16Q_Superset_v3.fits \
         $TRACTOR_INDIR \
         $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr16Q-v3.fits --copycols MJD PLATE FIBERID
    echo done making $EXTERNAL_OUTDIR/survey-$dr-$survey-superset-dr16Q-v3.fits
done

wait
echo done writing sweeps and externals for all surveys
