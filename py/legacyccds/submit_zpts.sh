#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -A desi
#SBATCH -J zpts
#SBATCH -L SCRATCH,CSCRATCH
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

# TO RUN shared
#for i in `find decam_cp_CP*txt`;do export input=$i;echo $input;sbatch legacyccds/submit_zpts.sh --export input;done
#-p regular
#-N 30
# RUN all decam
#for i in `find decam_cp_CP*txt`;do export input=$i;sbatch legacyccds/submit_zpts.sh --export input;done

set -x 
export OMP_NUM_THREADS=1

export camera=mosaicz
#export camera=decam

export outdir=/global/cscratch1/sd/kaylanb/zeropoints
export imagelist=${outdir}/${camera}_imagelist.txt
echo imagelist=$imagelist
if [ ! -e "$imagelist" ]; then
    echo file=$imagelist does not exist, quitting
    exit 999
fi

yyear=`date|awk '{print $NF}'`
today=`date|awk '{print $3}'`
month=`date +"%F"|awk -F "-" '{print $2}'`
logdir=$outdir/${camera}/logs/${year}_${month}_${today}
mkdir -p $logdir
log="$logdir/log.${SLURM_JOB_ID}"

srun -n 1 -c 24 python legacyccds/legacy_zeropoints.py \
     --image_list ${imagelist} --prefix paper --outdir ${outdir} --nproc 1 \
     >> $log 2>&1 


# Yu Feng's bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh
#
## Put legacypipe in path
#export PYTHONPATH=.:${PYTHONPATH}
#
#if [ "$NERSC_HOST" == "cori" ]; then
#    cores=32
#elif [ "$NERSC_HOST" == "edison" ]; then
#    cores=24
#fi
#let tasks=${SLURM_JOB_NUM_NODES}*${cores}

#find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202v2/k4m*ooi*.fits.fz > mosaic_CP20160202v2.txt
#find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160202/ksb*ooi*.fits.fz > 90prime_CP20160202.txt
#input=mosaic_CP20160202v2.txt
#input=90prime_CP20160202.txt
#input=decam_cp_all.txt
#input=mosaic_cp_all.txt
prefix=paper
# Make zpts
#tasks=1
#srun -n $tasks -c 1 python legacyccds/legacy-zeropoints.py --prefix $prefix --image_list $input --nproc $tasks
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints.py --prefix $prefix --image_list $input --nproc $tasks
# Make zpts VERBOSE
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints.py --prefix $prefix --image_list $input --nproc $tasks --verboseplots

# Gather into 1 file
#input=90prime_CP20160202_zpts.txt 
#input=mosaic_CP20160202v2_zpts.txt 
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_$(echo $input| sed s/.txt/.fits/g)
