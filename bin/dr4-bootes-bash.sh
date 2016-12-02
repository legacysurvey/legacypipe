#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --account=desi
#SBATCH -J dr4-bootes
#SBATCH -o dr4-bootes.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH


#-p shared
#-n 6
#-t etc

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
set -x
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

threads=24
export OMP_NUM_THREADS=$threads

echo outdir="$outdir", brick="$brick"
module list
module load psfex-hpcp
srun -n 1 -c $OMP_NUM_THREADS python legacypipe/runbrick.py \
     --run bootes-$camera \
     --brick $brick \
     --skip \
     --threads $OMP_NUM_THREADS \
     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --outdir $outdir --nsigma 6 \
     >> $log 2>&1
#--no-wise \
#--zoom 1400 1600 1400 1600
rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#
echo $run_name DONE $SLURM_JOBID


