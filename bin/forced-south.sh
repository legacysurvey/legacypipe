#! /bin/bash

DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9m
outdir=$CSCRATCH/dr9-forced

export SKY_TEMPLATE_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9m/calib/sky_pattern
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.2/SGA-ellipse-v3.2.kd.fits
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

camera=$1
expnum=$2

exppre=$(printf %08d $expnum | cut -c 1-5)

logdir=$outdir/logs/$camera/$exppre
mkdir -p $logdir
logfile=$logdir/$expnum.log

ncores=32
# Memory limiting handled by qdo-srun
# 128 GB / Cori Haswell node = 134217728 kbytes
#maxmem=134217728
#let usemem=${maxmem}*${ncores}/32
#ulimit -Sv $usemem
# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes

echo "Logging to $logfile"

python -O "$LEGACYPIPE_DIR/legacypipe/forced_photom.py" \
       --survey-dir "$DIR" \
       --catalog-dir-north "$DIR/north" \
       --catalog-dir-south "$DIR/south" \
       --catalog-resolve-dec-ngc 32.375 \
       --skip \
       --skip-calibs \
       --apphot \
       --derivs \
       --outlier-mask \
       --camera "$camera" \
       --expnum "$expnum" \
       --out-dir "$outdir" \
       --threads "$ncores" \
       >> "$logfile" 2>&1

# eg:
# QDO_BATCH_PROFILE=cori-shifter qdo launch forced-south 1 --cores_per_worker 32 --walltime=30:00 --batchqueue=debug --batchopts "--image=docker:legacysurvey/legacypipe:nersc-dr9.0.1 --license=SCRATCH,project" --script "../bin/forced-south.sh" --keep_env

# Shared queue (--threads 16)
# qdo launch forced-south 1 --cores_per_worker 16 --walltime=1:00:00 --batchqueue=shared --batchopts "--license=SCRATCH,project -a 0-99 --cpus-per-task=32" --script "../bin/forced-south-16.sh" --keep_env -v
