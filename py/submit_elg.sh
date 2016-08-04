#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J elg
#SBATCH -o elg.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#export fold=1.e-5

export OMP_NUM_THREADS=24
cd $SLURM_SUBMIT_DIR
export DECALS_SIM_DIR=/scratch1/scratchdirs/kaylanb/legacypipe/py/elgs
#export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/kaylanb/desi/dr3_brick_2523p355
#export DUST_DIR=/scratch1/scratchdirs/kaylanb/desi/dr3_brick_2523p355/dust/v0_0
srun -n 1 -c ${OMP_NUM_THREADS} python legacyanalysis/decals_sim.py --brick 2523p355 -n 10 --zoom 1600 2000 1600 2000 -o ELG -ic 1 --rmag-range 18 18 --threads ${OMP_NUM_THREADS} 
echo DONE
