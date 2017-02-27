#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH --account=desi
#SBATCH -J trace
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
#-C haswell
#--qos=scavenger

export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/dr4_fixes/legacypipe-dir
export CODE_DIR=/scratch1/scratchdirs/desiproc/DRs/code/dr4_fixes/legacypipe

#echo LD_LIBRARY_PATH; echo $LD_LIBRARY_PATH | sed -e 's#:#\n#g'
#echo PYTHONPATH; echo $PYTHONPATH | sed -e 's#:#\n#g'
#echo PATH; echo $PATH | sed -e 's#:#\n#g'

# ./launch_dr4.sh --> brick,outdir,overwrite_tractor
echo brick:$brick
echo outdir:$outdir
echo overwrite_tractor:$overwrite_tractor
echo full_stacktrace:$full_stacktrace
echo early_coadds:$early_coadds
echo dr4_fixes:$dr4_fixes
if [ "$dr4_fixes" = "yes" ]; then
    set -x
    export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/legacypipe-dir
    #export DUST_DIR=adfa
    #export unwise_dir=lakdjf
    set +x
fi 



#-p shared
#-n 24
#-p regular
#-N 1

# TO RUN
# 1) haswell if Cori
# set usecores as desired for more mem and set shared n above to 2*usecores, keep threads=6 so more mem per thread!, then --aray equal to number largemmebricks.txt


usecores=4
#threads=$usecores
threads=4
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

# Extra srun options
export extra_opt="--skip"
if [ "$overwrite_tractor" = "yes" ]; then
    export extra_opt=""
elif [ "$early_coadds" = "yes" ]; then
    export extra_opt="--stage image_coadds --early-coadds"
fi


export OMP_NUM_THREADS=$threads
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/large-mem-runs/${usecores}usecores
mkdir -p $outdir

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1



#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

qdo_table=dr4v2

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

set -x
year=`date|awk '{print $NF}'`
today=`date|awk '{print $3}'`
month=`date +"%F"|awk -F "-" '{print $2}'`
logdir=$outdir/logs/${year}_${month}_${today}
if [ "$full_stacktrace" = "yes" ];then
    logdir=${logdir}_stacktrace
fi
mkdir -p $logdir
log="$logdir/log.${brick}_${SLURM_JOB_ID}"
echo Logging to: $log

echo doing srun
date
module load psfex-hpcp
srun -n 1 -c $usecores python legacypipe/runbrick.py \
     --run $qdo_table \
     --brick $brick \
     --hybrid-psf \
     --threads $threads \
     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --outdir $outdir --nsigma 6 \
     --no-write ${extra_opt} \
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

