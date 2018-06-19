#! /bin/bash

# This script is run (via qdo) to process a single brick in a Legacy
# Surveys reduction.

# Assume we are running in the legacypipe/py directory.

source ../bin/legacypipe-env

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

# Try limiting memory to avoid killing the whole MPI job...
ulimit -S -v 30000000
ulimit -a

outdir=$LEGACY_SURVEY_DIR

# First (only) command-line arg is the brick name.
brick="$1"

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"
echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

python    legacypipe/runbrick.py \
    --skip-calibs \
    --checkpoint $outdir/checkpoints/${bri}/checkpoint-${brick}.pickle \
    --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
    --write-stage srcs \
    --threads 8 \
    --brick $brick --outdir $outdir >> $log 2>&1

#--cache-dir /global/cscratch1/sd/dstn/dr5-new-sky/cache/ \
#--force-all \
#--write-stage coadds \

# qdo launch dr2n 4 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/runbrick.sh --batchqueue regular --verbose --keep_env
