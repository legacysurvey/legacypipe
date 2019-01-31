#! /bin/bash

# Script for running the legacypipe code within a Shifter container at NERSC.

# This merges some contents from legacypipe-env and runbrick.sh

# Where are inputs and outputs going?
export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr8

export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1
export UNWISE_COADDS_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth:/global/project/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth
export UNWISE_COADDS_TIMERESOLVED_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4
export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom/
export GAIA_CAT_VER=2
export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_DIR=/global/project/projectdirs/cosmo/staging/largegalaxies/v2.0
UNWISE_PSF_DIR=/global/project/projectdirs/cosmo/staging/unwise_psf/2018-11-25
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
outdir=$LEGACY_SURVEY_DIR

brick="$1"

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

echo Logging to: $log
echo Running on $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

python3 legacypipe/runbrick.py \
     --skip \
     --skip-calibs \
     --threads ${ncores} \
     --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --write-stage srcs \
     --brick $brick --outdir $outdir \
     --zoom 100 200 100 200 \
     --pickle "${outdir}/pickles/${bri}/runbrick-zoomshifter-%(brick)s-%%(stage)s.pickle" \
     --large-galaxies --unwise-coadds \
     --survey-dir $LEGACY_SURVEY_DIR \
     --outdir $outdir \
     >> $log 2>&1

#--pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \

#python3 legacypipe/runbrick.py --brick 1888p122 --outdir /global/cscratch1/sd/dstn/shifter-out --zoom 100 200 100 200 --large-galaxies --unwise-coadds --pickle '/global/cscratch1/sd/dstn/pickles2/runbrick-zoom%(brick)s-%%(stage)s.pickle' --survey-dir /global/cscratch1/sd/dstn/dr8sub
