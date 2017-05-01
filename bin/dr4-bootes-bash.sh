#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 12
#SBATCH -t 01:00:00
#SBATCH --account=desi
#SBATCH -J bash-bootes
#SBATCH -o bash-bootes.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#-p shared
#-n 12
#-p debug
#-N 1

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
set -x

# Yu Feng's bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast/activate.sh
# Put legacypipe in path
#export PYTHONPATH=.:${PYTHONPATH}


threads=12
export OMP_NUM_THREADS=$threads

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a

#outdir,statdir,brick,run_name
bri="$(echo $brick | head -c 3)"

log="$outdir/logs/$bri/$brick/log.$SLURM_JOBID"
mkdir -p $(dirname $log)

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log


echo outdir="$outdir", brick="$brick"
#module load psfex-hpcp
srun -n 1 -c $OMP_NUM_THREADS python legacypipe/runbrick.py \
     --run dr4-bootes \
     --brick $brick \
     --skip \
     --threads $OMP_NUM_THREADS \
     --outdir $outdir --nsigma 6 --force-all \
     >> $log 2>&1
#--checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
#--pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
#--no-wise \
#--zoom 1400 1600 1400 1600
rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#
echo $run_name DONE $SLURM_JOBID


