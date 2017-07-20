#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH -J mpi
#SBATCH -L SCRATCH,projecta,project
#SBATCH -C haswell
#--qos=scavenger

export hardcores=8
let softcores=2*${hardcores}

source ~/.bashrc_desiconda

#module load mpi4py-hpcp
#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

#export outdir=/global/cscratch1/sd/kaylanb/observing_paper
export outdir=/global/cscratch1/sd/kaylanb/dr5_zpts
export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/zpts_oom.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/err_oom_abs.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/err_other_abs.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/mosaic_combo.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/mosaic_combo_remain.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/bok_project.txt
#export imagelist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/bok_project_remain.txt
#export camera=mosaic
#export camera=90prime
export camera=decam

echo imagelist=$imagelist
echo camera=$camera
if [ ! -e "$imagelist" ]; then
    echo file=$imagelist does not exist, quitting
    exit 999
fi
mkdir -p $outdir

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}/${hardcores}


#export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
#cd $CODE_DIR/legacypipe/py


#set -x
#year=`date|awk '{print $NF}'`
#today=`date|awk '{print $3}'`
#month=`date +"%F"|awk -F "-" '{print $2}'`
#logdir=$outdir/${camera}/logs/${year}_${month}_${today}
#mkdir -p $logdir
#log="$logdir/log.${SLURM_JOB_ID}"
#echo Logging to: $log

#echo doing srun
#date
# OMP_NUM_THREADS is number of hard cores for EACH task
# -c is number of hard/soft cores for EACH task

# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
echo tasks=$tasks softcores=${softcores} hardcores=${hardcores} camera=${camera} image_list=${imagelist} outdir=${outdir} nproc=$tasks
srun -n $tasks -c ${softcores} python legacyccds/legacy_zeropoints_mpiwrapper.py \
     --camera ${camera} --image_list ${imagelist} --outdir ${outdir} --nproc $tasks
## Up to date working command:
#srun -n 80 -c 1 python legacyccds/legacy_zeropoints_mpiwrapper.py --camera decam --image_list bricks25_remain.txt --outdir $CSCRATCH/dr5_zpts --nproc 80
#date


