#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 5
#SBATCH -t 00:10:00
#SBATCH -J zpts
#SBATCH -L SCRATCH,project
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL

set -x 
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
export PYTHONPATH=.:/scratch2/scratchdirs/kaylanb/dr3/tractor:${PYTHONPATH}

if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

#input=mzls_cps.txt
#srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints.py --image_list $input --nproc $tasks --only_ccd_centers

#input=mzls_zpt_test.txt
#srun -n 2 -N 1 -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc 2 --outname gathered_mpi.fits
input=bok_zpt_files.txt
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python legacyccds/legacy-zeropoints-gather.py --file_list $input --nproc $tasks --outname gathered_all_mpi.fits
