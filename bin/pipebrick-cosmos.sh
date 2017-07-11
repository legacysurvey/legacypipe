#! /bin/bash

# For modules loaded, see "bashrc" in this directory.

export LEGACY_SURVEY_DIR=$CSCRATCH/dr5-cosmos

export DUST_DIR=/global/cscratch1/sd/desiproc/dust/v0_0

#export UNWISE_COADDS_DIR=/scratch1/scratchdirs/desiproc/unwise-coadds/fulldepth:/scratch1/scratchdirs/desiproc/unwise-coadds/w3w4
#export UNWISE_COADDS_TIMERESOLVED_DIR=/scratch1/scratchdirs/desiproc/unwise-coadds/time_resolved_neo1

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

brick="$1"
subset="$2"

outdir=$CSCRATCH/cosmos-dr5-${subset}

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n" > $log
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

CHK=${outdir}/checkpoints/${bri}
PIC=${outdir}/pickles/${bri}
mkdir -p $CHK
mkdir -p $PIC

python -u legacypipe/runcosmos.py \
    --subset $subset \
    --threads 24 \
    --skip-calibs \
    --brick $brick --outdir $outdir --nsigma 6 \
    --checkpoint $CHK/checkpoint-${brick}.pickle \
    --pickle "$PIC/cosmos-%(brick)s-%%(stage)s.pickle" \
    --skip \
    --rex \
    --hybrid-psf \
    --no-wise \
    --no-depth-cut \
    --no-blacklist \
     >> $log 2>&1

#    --zoom 500 600 500 600 \

# qdo launch cosmos 1 --cores_per_worker 24 --batchqueue regular --walltime 4:00:00 --keep_env --batchopts "-a 0-19 --qos=premium" --script ../bin/pipebrick-cosmos.sh
