#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 10
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH -J INP_SAMPLE
#SBATCH -o INP_SAMPLE.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#-p shared
#-n 6
#-p debug
#-N 1

# Yu Feng's bcast
source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

set -x
#export LEGACY_SURVEY_DIR=/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir
#/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir #/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir
export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

#outdir=/scratch2/scratchdirs/kaylanb/obiwan/testing
#outdir=/scratch2/scratchdirs/kaylanb/obiwan/eboss_ngc
outdir=/scratch2/scratchdirs/kaylanb/obiwan/eboss_ngc_good

#cp /project/projectdirs/desi/users/burleigh/obiwan_backup_data/*.pickle $outdir

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
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
#prefix=eboss_sgc_
#ra1=310.
#ra2=50.
#dec1=-6.
#dec2=6.
# Test brick
#prefix=testmpi
#ra1=1.
#ra2=5.
#dec1=1.
#dec2=5.

dowhat=sample
#dowhat=bybrick
#dowhat=merge
#dowhat=check
srun -n $tasks -N ${SLURM_JOB_NUM_NODES} -c 1 python obiwan/decals_sim_radeccolors.py --dowhat $dowhat --ra1 $ra1 --ra2 $ra2 --dec1 $dec1 --dec2 $dec2 --prefix $prefix --nproc $tasks --outdir $outdir
