#! /bin/bash

source $CSCRATCH/DRcode/legacypipe/bin/legacypipe-env

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

# Try limiting memory to avoid killing the whole MPI job...
ncores=8
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

outdir=$SCRATCH/dr7out
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

python ${rundir}/legacypipe/py/legacypipe/runbrick.py \
     --skip \
     --no-write \
     --skip-calibs \
     --threads ${ncores} \
     --checkpoint ${outdir}/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --write-stage srcs \
     --brick $brick --outdir $outdir \
     >> $log 2>&1

# qdo launch dr7short 60 --cores_per_worker 8 --walltime=12:00:00 --script /scratch1/scratchdirs/desiproc/dr7out/pbcp8.sh --batchqueue regular --keep_env

