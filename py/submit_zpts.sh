#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 15
#SBATCH -t 00:20:00
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

set -x 
export OMP_NUM_THREADS=1

module load mpi4py-hpcp
module unload tractor-hpcp
#bcast
source /scratch2/scratchdirs/kaylanb/yu-bcase/activate.sh
# Set env vars
export PYTHONPATH=.:/scratch2/scratchdirs/kaylanb/dr3/tractor:${PYTHONPATH}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

#find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202v2/k4m*ooi*.fits.fz > mosaic_CP20160202v2.txt
#find /project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160202/ksb*ooi*.fits.fz > 90prime_CP20160202.txt
input=mosaic_CP20160202v2.txt
#input=90prime_CP20160202.txt
prefix=dr4
# Make zpts
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-stars-gain1.8noCorr.py --prefix $prefix --image_list $input --nproc $tasks

# Gather into 1 file
#input=mzls_zpts_v2thruMarch19.txt 
#input=bok_zpts.txt
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_bok_all.fits
