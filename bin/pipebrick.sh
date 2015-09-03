#! /bin/bash

module load matplotlib-hpcp
module load scipy-hpcp
module load wcslib-hpcp
module load astropy-hpcp
module load photutils-hpcp
module load ceres-hpcp
module load sextractor-hpcp
module load fitsio-hpcp
module load unwise_coadds/2.0

# module load astrometry_net-hpcp
# module load tractor-hpcp

export PYTHONPATH=${PYTHONPATH}:.

## HACK! -- put my PsfEx ahead of anything else!
export PATH=/global/homes/d/dstn/software/psfex-3.17.1/bin:${PATH}

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

outdir=$SCRATCH/fft-b

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

#python -u legacypipe/runbrick.py --force-all --no-write --brick $brick --outdir $outdir --threads 6 --nsigma 6 --skip --pipe --pixpsf >> $log 2>&1

python -u legacypipe/runbrick.py -P 'pickles/runbrick-fftb-%(brick)s-%%(stage)s.pickle' \
    --brick $brick --outdir $outdir --threads 6 --nsigma 6 --skip --pipe --pixpsf >> $log 2>&1

# Launch from the 'py' directory;
# qdo launch dr1n 32 --mpack 6 --walltime=48:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose



# with 8 threads: 3 GB per core * 8 cores (most of the carver nodes have 24 GB)
# qdo launch bricks 1 --batchopts "-l pvmem=3GB -l nodes=1:ppn=8 -A desi -t 1-20 -q regular" --walltime=48:00:00 --script projects/desi/pipebrick.sh

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose

# or maybe
#qdo launch bricks 4 --mpack 2 --batchopts "-l pvmem=6GB -A cosmo -t 1-2" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular
