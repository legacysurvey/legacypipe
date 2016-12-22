#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 6
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J bootes-dr3-obiwan
#SBATCH -o bootes-dr3-obiwan.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#-p shared
#-n 6
#-p debug
#-N 1

# Yu Feng's bcast
source /scratch2/scratchdirs/kaylanb/yu-bcase/activate.sh
# Put legacypipe in path
export PYTHONPATH=.:/global/homes/k/kaylanb/repos:${PYTHONPATH}

outdir=/scratch2/scratchdirs/kaylanb/obiwan-eboss-ngc
#cp /project/projectdirs/desi/users/burleigh/obiwan_backup_data/*.pickle $outdir

#source ~/.bashrc_hpcp
#source ~/.bashrc_dr4-bootes
python -c "import tractor;print(tractor)"
python -c "import astrometry;print(astrometry)"

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
set -x
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a


if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}

# eBOSS NGC
prefix=eboss_ngc
ra1=122.
ra2=177.
dec1=12.
dec2=32.
## eBOSS SGC
#prefix=eboss_sgc
#ra1=310.
#ra2=50.
#dec1=-6.
#dec2=6.
# Test
#prefix=test
#ra1=10.
#ra2=11.
#dec1=20.
#dec2=21.

srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python obiwan/decals_sim_radeccolors.py --nproc $tasks --ra1 $ra1 --ra2 $ra2 --dec1 $dec1 --dec2 $dec2 --outdir $outdir --prefix $prefix

