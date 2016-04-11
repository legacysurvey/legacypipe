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

outdir=$SCRATCH/dr3

brick="$1"

logdir=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$logdir
log="$outdir/logs/$logdir/$brick.log"

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

# --no-early-coadds \
# -s tims \
# -P 'pickles/runbrick-dr2p-%(brick)s-%%(stage)s.pickle' \
python -u legacypipe/runbrick.py \
    --force-all --no-write \
    --pipe \
    --threads 8 \
    --on-bricks \
    --brick $brick --outdir $outdir --nsigma 6 >> $log 2>&1

# qdo launch dr2n 16 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose

