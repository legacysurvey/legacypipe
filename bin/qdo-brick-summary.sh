#! /bin/bash

# To run the brick-summary code.  As of DR7, this couldn't be run
# on the command line.  Currently set up to be split into 360 tasks
# whose names are 000 to 359.  Code crashes if no bricks at that RA,
# but this failure is without consequence.

# Make sure this has the appropriate paths
source $CSCRATCH/DRcode/legacypipe/bin/legacypipe-env

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

# Try limiting memory to avoid killing the whole MPI job...
ncores=16
if [ "$NERSC_HOST" = "edison" ]; then
    # 64 GB / Edison node = 67108864 kbytes
    maxmem=67108864
    let usemem=${maxmem}*${ncores}/24
else
    # 128 GB / Cori Haswell node = 134217728 kbytes
    maxmem=134217728
    let usemem=${maxmem}*${ncores}/32
fi
ulimit -Sv $usemem

# Need to modify for future release and output directories
drdir=/global/projecta/projectdirs/cosmo/work/legacysurvey/dr7
outdir=$CSCRATCH/dr7bricks
rundir=$CSCRATCH/DRcode

dirname="$1"

log="${outdir}/bs-${dirname}.log"

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

python -u $rundir/legacypipe/py/legacyanalysis/brick-summary.py \
         --dr5 \
         -o ${outdir}/dr7-bricks-summary-${dirname}.fits \
         ${drdir}/coadd/${dirname}/*/*-nexp-*.fits.fz \
         > $log 2>&1

# qdo launch dr7bsum 10 --cores_per_worker 16 --walltime=00:30:00 --script /global/cscratch1/sd/desiproc/DRcode/runmanaging/qdo-brick-summary.sh --batchqueue debug --keep_env --batchopts "-C haswell"

