#! /bin/bash

# Run legacy_zeropoints on a single image with the burst buffer.  Assumes the
# file dr8-env.sh exists in the same path as this script is launched.

# To load and launch:
#   qdo load calibs ./image-list.txt
#   qdo launch calibs 256 --cores_per_worker 8 --walltime=00:30:00 --script ./dr8-zpts.sh --batchqueue debug --keep_env --batchopts "--bbf=bb.conf"

# Useful commands:
#   qdo status calibs
#   qdo retry calibs
#   qdo recover calibs --dead
#   qdo tasks calibs --state=Failed

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

source dr8-env.sh

outdir=${DW_PERSISTENT_STRIPED_DR8}dr8b
zptsdir=$outdir/zpts
calibdir=$outdir/calib

# Limit memory usage.
maxmem=134217728 # 128 GB / Cori Haswell node = 134217728 kbytes
let usemem=${maxmem}*${ncores}/32
ulimit -Sv $usemem

# Redirect logs to a nested directory.
cpdir=`echo $(basename $(dirname ${image_fn}))`
logdir=$outdir/logs-zpts/$camera/$cpdir
mkdir -p $logdir

log=`echo $(basename ${image_fn} | sed s#.fits.fz#.log#g)`
log=$logdir/$log
echo Logging to: $log

time python $LEGACYZPTS_DIR/py/legacyzpts/legacy_zeropoints.py --camera ${camera} \
    --image ${image_fn} --outdir ${zptsdir} --calibdir ${calibdir} --threads ${ncores} \
    >> $log 2>&1
