#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC
# with burst buffer!

# Burst-buffer!
if [ x$DW_PERSISTENT_STRIPED_DR9 == x ]; then
  # No burst buffer -- use scratch
  outdir=/global/cscratch1/sd/desiproc/dr9c/south
else
  # Use "DR9" burst buffer.
  outdir=$DW_PERSISTENT_STRIPED_DR9/south
fi

BLOB_MASK_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr8/south

export LEGACY_SURVEY_DIR=/global/cfs/cdirs/cosmo/work/legacysurvey/dr9

export DUST_DIR=/global/cfs/cdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6/fulldepth:/global/cfs/cdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/cfs/cdirs/cosmo/work/wise/outputs/merge/neo6
export UNWISE_MODEL_SKY_DIR=/global/cfs/cdirs/cosmo/work/wise/unwise_catalog/dr3/mod
export GAIA_CAT_DIR=/global/cfs/cdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/cfs/cdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/cfs/cdirs/cosmo/staging/largegalaxies/v7.0/LSLGA-ellipse-v7.0.kd.fits
export PS1CAT_DIR=/global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3

# PYTHONPATH is set in the container.
#export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:/src/unwise_psf/py:.

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY
export KMP_AFFINITY=disabled

# Limit memory to avoid killing the whole MPI job...
ncores=8

# 128 GB / Cori Haswell node = 134217728 kbytes, 32 cores
maxmem=134217728
let usemem=${maxmem}*${ncores}/32

# Can detect Cori KNL node (96 GB) via:
# grep -q "Xeon Phi" /proc/cpuinfo && echo Yes


ulimit -Sv $usemem

cd /src/legacypipe/py

brick="$1"

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

mkdir -p $outdir/metrics/$bri

echo Logging to: $log
echo Running on $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

python -O legacypipe/runbrick.py \
     --run south \
     --brick $brick \
     --skip \
     --skip-calibs \
     --threads ${ncores} \
     --blob-mask-dir ${BLOB_MASK_DIR} \
     --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --outdir $outdir \
     --write-stage srcs \
     >> $log 2>&1

#     --ps "${outdir}/metrics/${bri}/ps-${brick}-${SLURM_JOB_ID}.fits" \
#     --ps-t0 $(date "+%s") \

# QDO_BATCH_PROFILE=cori-shifter qdo launch -v tst 1 --cores_per_worker 8 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/bin/runbrick-shifter.sh"
