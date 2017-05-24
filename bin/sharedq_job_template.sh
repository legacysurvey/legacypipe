#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 2
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J repack
#SBATCH -L SCRATCH
#SBATCH -C haswell
#--qos=scavenger
#--mail-user=kburleigh@lbl.gov
#--mail-type=END,FAIL

usecores=1
threads=1
if [ "$full_stacktrace" = "yes" ];then
    threads=1
fi
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

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

#set -x
#year=`date|awk '{print $NF}'`
#today=`date|awk '{print $3}'`
#month=`date +"%F"|awk -F "-" '{print $2}'`
#logdir=$outdir/logs/${year}_${month}_${today}_${NERSC_HOST}_remain
#if [ "$full_stacktrace" = "yes" ];then
#    logdir=${logdir}_stacktrace
#fi
#mkdir -p $logdir
#log="$logdir/log.${brick}_${SLURM_JOB_ID}"
#echo Logging to: $log


#export dr4b_dir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
#export dr4c_dir=/global/projecta/projectdirs/cosmo/work/dr4c

