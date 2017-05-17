#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 2
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J repack
#SBATCH -L SCRATCH,projecta
#SBATCH -C haswell
#--qos=scavenger
#--mail-user=kburleigh@lbl.gov
#--mail-type=END,FAIL

echo brick:$brick
echo start_brick=$start_brick
echo end_brick=$end_brick

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


export dr4b_dir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
export dr4c_dir=/global/projecta/projectdirs/cosmo/work/dr4c

while read aline; do
    brick=`echo $aline|awk '{print $1}'`
    echo brick=$brick
    #rsync -av /global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes/coadd/$brick /global/projecta/projectdirs/cosmo/work/dr4b/coadd/
    # New Data Model Catalouge
    bri=`echo $brick|head -c 3`
    in_file=${dr4b_dir}/tractor-i/${bri}/tractor-${brick}.fits
    out_file=${dr4c_dir}/tractor/${bri}/tractor-${brick}.fits
    mkdir -p $(dirname ${out_file})
    echo hey1
    srun -n 1 -c $usecores python legacypipe/format_catalog.py --in ${in_file} --out ${out_file} --dr4 
    echo hey2
    # New Headers
    srun -n 1 -c $usecores python legacypipe/format_headers.py --brick $brick
    echo hey3
done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"


