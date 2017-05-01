#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J lzp
#SBATCH -L SCRATCH
#-C haswell
#--qos=scavenger
#--mail-user=kburleigh@lbl.gov
#--mail-type=END,FAIL


echo imagefn:$imagefn
echo outdir:$outdir
echo drfn:$drfn
echo camera:$camera

usecores=2
#threads=$usecores
threads=1
# Limit memory to avoid 1 srun killing whole node
if [ "$NERSC_HOST" = "edison" ]; then
    # 62 GB / Edison node = 65000000 kbytes
    maxmem=65000000
    let usemem=${maxmem}*${usecores}/24
else
    # 128 GB / Edison node = 65000000 kbytes
    maxmem=134000000
    let usemem=${maxmem}*${usecores}/32
fi
ulimit -S -v $usemem
ulimit -a
echo usecores=$usecores
echo threads=$threads


export OMP_NUM_THREADS=$threads
mkdir -p $outdir

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

set -x
year=`date|awk '{print $NF}'`
today=`date|awk '{print $3}'`
month=`date +"%F"|awk -F "-" '{print $2}'`
logdir=$outdir/logs/${camera}/${year}_${month}_${today}_${NERSC_HOST}
mkdir -p $logdir
log="$logdir/log.${drfn}_${SLURM_JOB_ID}"
echo Logging to: $log

echo doing srun
date
srun -n 1 -c $usecores python legacyccds/legacy_zeropoints.py \
     --image $imagefn --outdir $outdir \
     >> $log 2>&1 
date


# Bootes
#--run dr4-bootes \

#--no-wise \
#--zoom 1400 1600 1400 1600
#rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#

