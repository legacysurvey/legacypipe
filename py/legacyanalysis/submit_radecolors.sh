#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J rdcols
#SBATCH -L SCRATCH
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

set -x 
outdir=/scratch2/scratchdirs/kaylanb/dr3/production/obiwan
# cd $PBS_O_WORKDIR
pwd
export OMP_NUM_THREADS=1
#srun -n 1 -N 1 -c 1 python legacyccds/legacy-zeropoints.py --images /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202v2/k4m_160203_025200_ooi_zd_v2.fits.fz
#srun -n 1 -N 1 -c 1 python legacyccds/legacy-zeropoints.py --image_list image_list_CP20160204v2

module load mpi4py-hpcp
module unload tractor-hpcp
#bcast
source /scratch2/scratchdirs/kaylanb/yu-bcase/activate.sh
# Set env vars
#export PYTHONPATH=.:/scratch2/scratchdirs/kaylanb/dr3/tractor:${PYTHONPATH}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

export PYTHONPATH=$HOME/repos:$PYTHONPATH
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyanalysis/decals_sim_radeccolors.py --ra1 205 --ra2 215 --dec1 25.5 --dec 35.5 --ndraws 10000 --outdir $outdir --nproc $tasks

#input=validation_zpt_images.txt
#prefix=stars2
#code=legacy-zeropoints-stars.py 
# Make zpts
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/$code --prefix $prefix --image_list $input --nproc $tasks

# ccd ra,dec only
#input=bok_images.txt
#prefix=radec-only
#code=legacy-zeropoints-stars-gain1.8noCorr.py 
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/$code --only_ccd_centers --prefix $prefix --image_list $input --nproc $tasks
# Gather into 1 file
input=boks_to_gather.txt
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_boks.fits

#input=bok_zpt_files.txt
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_all_mpi.fits

# rm zeropoint files
#for i in `find /scratch2/scratchdirs/kaylanb/cosmo/staging/mosaicz/MZLS_CP/*v2 -type d -name "zpts"`;do rm $i/only_ccd_centers_zeropoint*;done
