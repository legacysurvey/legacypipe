#! /bin/bash

# For modules loaded, see "bashrc" in this directory.

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

# Try limiting memory to avoid killing the whole MPI job...
#ulimit -S -v 15000000
#ulimit -S -v 30000000
ulimit -a

# Make sure we're reading from Edison scratch
#module unload dust
#module load dust/scratch
### argh modules not seeming to work.
export DUST_DIR=/scratch1/scratchdirs/desiproc/dust/v0_0

module unload tractor-hpcp

outdir=$SCRATCH/dr3

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
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

#     --threads 24 \

python legacypipe/runbrick.py \
     --no-write \
     --pipe \
     --skip \
     --threads 6 \
     --skip-calibs \
     --checkpoint checkpoints/${bri}/checkpoint-${brick}.pickle \
     --fitblobs-prereq pickles/${bri}/runbrick-${brick}-srcs.pickle \
     --brick $brick --outdir $outdir --nsigma 6 \
     >> $log 2>&1

#     --ps ps/ps-${brick}.fits \
#     --on-bricks \
#     --allow-missing-brickq 0 \

# -s tims \
# -P 'pickles/runbrick-dr2p-%(brick)s-%%(stage)s.pickle' \
#    --no-early-coadds \
# qdo launch dr2n 16 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose
# qdo launch edr0 4 --cores_per_worker 8 --batchqueue regular --walltime 4:00:00 --script ../bin/pipebrick.sh --keep_env --batchopts "--qos=premium -a 0-3"
