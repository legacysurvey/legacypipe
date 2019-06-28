#! /bin/bash

brick="$1"
subset="$2"

source ../bin/legacypipe-env

export PYTHONPATH=${PYTHONPATH}:.

export LEGACY_SURVEY_DIR=$CSCRATCH/dr7-cosmos
outdir=$CSCRATCH/cosmos-dr7-${subset}

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

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
    --outdir $outdir \
    --brick $brick \
    --skip-calibs \
    --threads 4 \
    --checkpoint $CHK/checkpoint-${brick}.pickle \
    --checkpoint-period 300 \
    --pickle "$PIC/cosmos-%(brick)s-%%(stage)s.pickle" \
    --no-wise \
    --stage image_coadds --blob-image \
     >> $log 2>&1

#    --stage image_coadds \
#    --force-all \

# qdo launch cosmos 1 --cores_per_worker 24 --batchqueue regular --walltime 4:00:00 --keep_env --batchopts "-a 0-19 --qos=premium" --script ../bin/pipebrick-cosmos.sh
