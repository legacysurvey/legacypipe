#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH -J trace
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
#-C haswell
#--qos=scavenger

#echo LD_LIBRARY_PATH; echo $LD_LIBRARY_PATH | sed -e 's#:#\n#g'
#echo PYTHONPATH; echo $PYTHONPATH | sed -e 's#:#\n#g'
#echo PATH; echo $PATH | sed -e 's#:#\n#g'

# ./launch_dr4.sh --> brick,outdir,overwrite_tractor
echo outdir:$outdir
echo imagelist:$imagelist
echo camera:$camera

#-p shared
#-n 24
#-p regular
#-N 1

# TO RUN
# 1) haswell if Cori
# set usecores as desired for more mem and set shared n above to 2*usecores, keep threads=6 so more mem per thread!, then --aray equal to number largemmebricks.txt


usecores=24
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
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/large-mem-runs/${usecores}usecores
#mkdir -p $outdir

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
logdir=$outdir/${camera}/logs/${year}_${month}_${today}
if [ "$full_stacktrace" = "yes" ];then
    logdir=${logdir}_stacktrace
fi
mkdir -p $logdir
log="$logdir/log.${SLURM_JOB_ID}"
echo Logging to: $log

echo doing srun
date
srun -n 1 -c $usecores python legacyccds/legacy_zeropoints.py \
     --image_list ${imagelist} --prefix paper --outdir ${outdir} --nproc 1 \
     >> $log 2>&1 
date

