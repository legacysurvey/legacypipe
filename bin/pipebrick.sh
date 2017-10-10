#! /bin/bash

# For modules loaded, see "bashrc" in this directory.

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

# Try limiting memory to avoid killing the whole MPI job...
#ulimit -S -v 15000000
ulimit -S -v 30000000
ulimit -a

export LEGACY_SURVEY_DIR=$CSCRATCH/psf-zpts

outdir=$LEGACY_SURVEY_DIR

brick="$1"
bri=$(echo $brick | head -c 3)

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

python legacypipe/runbrick.py \
    --threads 8 \
    --skip \
    --max-blobsize  250000 \
    --checkpoint $outdir/checkpoints/${bri}/checkpoint-${brick}.pickle \
    --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
    --write-stage srcs \
    --brick $brick --outdir $outdir >> $log 2>&1

# qdo launch dr2n 16 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose
# qdo launch edr0 4 --cores_per_worker 8 --batchqueue regular --walltime 4:00:00 --script ../bin/pipebrick.sh --keep_env --batchopts "--qos=premium -a 0-3"
