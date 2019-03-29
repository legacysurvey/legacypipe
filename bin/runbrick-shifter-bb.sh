#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC
# with burst buffer!

# This merges some contents from legacypipe-env and runbrick.sh

# Burst-buffer!
if [ x$DW_PERSISTENT_STRIPED_DR8 == x ]; then
  # premium job finally running
  BB=$CSCRATCH/
else
  BB=$DW_PERSISTENT_STRIPED_DR8
fi

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr8-depthcut
#export LEGACY_SURVEY_DIR=$BB/dr8-depthcut
#cachedir=${BB}dr8-depthcut

export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/project/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth:/global/project/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4
export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom/
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_DIR=/global/project/projectdirs/cosmo/staging/largegalaxies/v2.0
#UNWISE_PSF_DIR=/global/project/projectdirs/cosmo/staging/unwise_psf/2018-11-25
UNWISE_PSF_DIR=/src/unwise_psf
export WISE_PSF_DIR=${UNWISE_PSF_DIR}/etc
export PS1CAT_DIR=/global/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/

export PYTHONPATH=/usr/local/lib/python:/usr/local/lib/python3.6/dist-packages:.:${UNWISE_PSF_DIR}/py

# Don't add ~/.local/ to Python's sys.path
export PYTHONNOUSERSITE=1

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

# Try limiting memory to avoid killing the whole MPI job...
ncores=8
# Assume cori!
# 128 GB / Cori Haswell node = 134217728 kbytes
maxmem=134217728
let usemem=${maxmem}*${ncores}/32
ulimit -Sv $usemem

cd /src/legacypipe/py

#outdir=$LEGACY_SURVEY_DIR
#outdir=/global/cscratch1/sd/dstn/dr8test010
outdir=${BB}dr8test14

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

python -u legacypipe/runbrick.py \
     --brick $brick \
     --release 7914 \
     --skip \
     --skip-calibs \
     --threads ${ncores} \
     --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --unwise-coadds \
     --survey-dir $LEGACY_SURVEY_DIR \
     --outdir $outdir \
     --ps "${outdir}/metrics/${bri}/ps-${brick}-${SLURM_JOB_ID}.fits" \
     --ps-t0 $(date "+%s") \
     --depth-cut 1 \
    --bail-out \
     >> $log 2>&1

#     --cache-dir $cachedir \


#     --write-stage srcs \

#     --zoom 100 300 100 300 \
#--pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \

# QDO_BATCH_PROFILE=cori-shifter qdo launch -v tst 1 --cores_per_worker 8 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/bin/runbrick-shifter.sh"
