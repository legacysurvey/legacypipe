#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J trace
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
#-C haswell
#--qos=scavenger

module load mpi4py-hpcp
#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

export camera=mosaic
#export camera=decam

export outdir=/global/cscratch1/sd/kaylanb/observing_paper
export imagelist=${outdir}/${camera}_list.txt
echo imagelist=$imagelist
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
let tasks=${SLURM_JOB_NUM_NODES}*${cores}




export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py


#set -x
#year=`date|awk '{print $NF}'`
#today=`date|awk '{print $3}'`
#month=`date +"%F"|awk -F "-" '{print $2}'`
#logdir=$outdir/${camera}/logs/${year}_${month}_${today}
#mkdir -p $logdir
#log="$logdir/log.${SLURM_JOB_ID}"
#echo Logging to: $log

echo doing srun
date
# OMP_NUM_THREADS is number of hard cores for EACH task
# -c is number of hard/soft cores for EACH task

# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
srun -n $tasks -c 1 python legacyccds/legacy_zeropoints_mpiwrapper.py \
     --image_list ${imagelist} --outdir ${outdir} --nproc $tasks
date


