#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J decals_sim
#SBATCH -o decals_sim.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH,project

export OMP_NUM_THREADS=24 #edison
cd $SLURM_SUBMIT_DIR   # optional, since this is the default behavior
export seed=1
export DECALS_SIM_DIR=${SLURM_SUBMIT_DIR}/chunk${seed}
echo START TIME:
date
srun -n 1 -c ${OMP_NUM_THREADS} python legacyanalysis/decals_sim.py --brick 2428p117 -n 500 -o STAR -s ${seed} --threads ${OMP_NUM_THREADS}
echo END TIME:
date
echo DONE

