#! /bin/bash

# Runs bright-neighbour post-processing stage via qdo (example launch command at bottom).
# Qdo tasks are the brick names.
# Martin Landriau, LBNL, July 2018

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

outdir=/global/projecta/projectdirs/cosmo/work/legacysurvey/dr7-attic/post-process
rundir=$CSCRATCH/DRcode

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

python ${rundir}/legacypipe/py/legacypipe/bright-neighbors.py \
     --brick $brick \
     --input-dir /global/projecta/projectdirs/cosmo/work/legacysurvey/dr7 \
     --survey-dir /global/cscratch1/sd/desiproc/dr7 \
     --output-dir $outdir \
     >> $log 2>&1

# qdo launch dr7pp 10 --cores_per_worker 16 --walltime=00:30:00 --script /global/cscratch1/sd/desiproc/DRcode/runmanaging/qdo-post-process.sh --batchqueue debug --keep_env --batchopts "-C haswell"

