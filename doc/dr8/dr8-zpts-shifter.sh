#! /bin/bash
# Run legacy_zeropoints on a single image with (optionally) the burst buffer
# within a Shifter container at NERSC.

# Variables to be sure are correct: dr, LEGACY_SURVEY_DIR, and CODE_DIR.  Also
# note that LEGACY_SURVEY_DIR has to be defined here, *before* the env script is
# sourced.
dr=dr8b

export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/$dr
source $LEGACY_SURVEY_DIR/dr8-env-shifter.sh
CODE_DIR=$LEGACY_SURVEY_DIR/code

# Use local check-outs of legacypipe and legacyzpts.
export LEGACYPIPE_DIR=$CODE_DIR/legacypipe
export LEGACYZPTS_DIR=$CODE_DIR/legacyzpts
export PYTHONPATH=$PYTHONPATH:$LEGACYPIPE_DIR/py
export PYTHONPATH=$PYTHONPATH:$LEGACYZPTS_DIR/py

# Get the camera from the filename
image_fn="$1"
echo $image_fn
if [[ $image_fn == *"decam"* ]]; then
  camera=decam
  ncores=8 
elif [[ $image_fn == *"90prime"* ]]; then
  camera=90prime
  ncores=4 
elif [[ $image_fn == *"mosaic"* ]]; then
  camera=mosaic
  ncores=4 
else
  echo 'Unable to get camera from file name!'
  exit 1
fi
echo 'Working on camera '$camera

if [ x$DW_PERSISTENT_STRIPED_DR8 == x ]; then
  if [ "$NERSC_HOST" = "edison" ]; then
    outdir=$SCRATCH/$dr
  else
    outdir=$CSCRATCH/$dr
  fi  
  # For writing to project, if necessary.
  outdir=/global/project/projectdirs/cosmo/work/legacysurvey/$dr    
else
  outdir=${DW_PERSISTENT_STRIPED_DR8}$dr
fi
echo 'Writing output to '$outdir
zptsdir=$outdir/zpts
calibdir=$outdir/calib

# Redirect logs to a nested directory.
cpdir=`echo $(basename $(dirname ${image_fn}))`
logdir=$outdir/logs-zpts/$camera/$cpdir
mkdir -p $logdir

log=`echo $(basename ${image_fn} | sed s#.fits.fz#.log#g)`
log=$logdir/$log
echo Logging to: $log

# Limit memory usage.
ncores=8
if [ "$NERSC_HOST" = "edison" ]; then
    # 64 GB / Edison node = 67108864 kbytes
    maxmem=67108864
    let usemem=${maxmem}*${ncores}/24
else
    # 128 GB / Cori Haswell node = 134217728 kbytes
    maxmem=134217728
    let usemem=${maxmem}*${ncores}/32
fi
ulimit -Sv $usemem

# do it!
time python $LEGACYZPTS_DIR/py/legacyzpts/legacy_zeropoints.py --camera ${camera} \
    --image ${image_fn} --outdir ${zptsdir} --calibdir ${calibdir} --threads ${ncores} \
    >> $log 2>&1
