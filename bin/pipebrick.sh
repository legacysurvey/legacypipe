#! /bin/bash

# For modules loaded, see "bashrc" in this directory.

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

outdir=$SCRATCH/dr2i

# git checkout of https://github.com/legacysurvey/legacypipe-dir
# branch cosmos-subset
export DECALS_DIR=/scratch1/scratchdirs/desiproc/decals-cosmos-subset

brick="$1"

logdir=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$logdir
log="$outdir/logs/$logdir/$brick.log"

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n\n\n\n\n\n\n\n" >> $log
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

#python legacypipe/runbrick.py --force-all --no-write --brick $brick --outdir $outdir --threads 6 --nsigma 6 --skip --pipe --pixpsf --splinesky >> $log 2>&1

python -u legacypipe/runbrick.py --force-all --no-write \ #-P 'pickles/runbrick-dr2c-%(brick)s-%%(stage)s.pickle' \
    --brick $brick --outdir $outdir --threads 6 --nsigma 6 --skip --pipe --pixpsf --splinesky --no-sdss >> $log 2>&1

#python -u legacypipe/runbrick.py -P 'pickles/runbrick-fftb-%(brick)s-%%(stage)s.pickle' \
#    --brick $brick --outdir $outdir --threads 6 --nsigma 6 --skip --pipe --pixpsf >> $log 2>&1

# Launch from the 'py' directory;
# qdo launch dr1n 32 --mpack 6 --walltime=48:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose

# Serial queue on edison:
# QDO_BATCH_PROFILE=edison-serial qdo launch dr2c 8 --script ../bin/pipebrick-single.sh --batchopts "-l vmem=10GB" --walltime 24:00:00

# with 8 threads: 3 GB per core * 8 cores (most of the carver nodes have 24 GB)
# qdo launch bricks 1 --batchopts "-l pvmem=3GB -l nodes=1:ppn=8 -A desi -t 1-20 -q regular" --walltime=48:00:00 --script projects/desi/pipebrick.sh

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose

# or maybe
#qdo launch bricks 4 --mpack 2 --batchopts "-l pvmem=6GB -A cosmo -t 1-2" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular
