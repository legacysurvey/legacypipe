#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 20
#SBATCH -t 00:30:00
#SBATCH -A eboss
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

set -x 
export OMP_NUM_THREADS=1

# Yu Feng's bcast
#source /scratch2/scratchdirs/kaylanb/yu-bcase/activate.sh
source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

# Put legacypipe in path
export PYTHONPATH=.:${PYTHONPATH}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

#find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202v2/k4m*ooi*.fits.fz > mosaic_CP20160202v2.txt
#find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160202/ksb*ooi*.fits.fz > 90prime_CP20160202.txt
#input=mosaic_CP20160202v2.txt
#input=90prime_CP20160202.txt
input=decam_cplist_CP20141227.txt
prefix=paper
# Make zpts
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints.py --prefix $prefix --image_list $input --nproc $tasks --verboseplots
# Make zpts VERBOSE
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints.py --prefix $prefix --image_list $input --nproc $tasks --verboseplots

# Gather into 1 file
#input=90prime_CP20160202_zpts.txt 
#input=mosaic_CP20160202v2_zpts.txt 
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_$(echo $input| sed s/.txt/.fits/g)
