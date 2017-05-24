#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 12
#SBATCH -t 01:00:00
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

#source ~/.bashrc_hpcp
source ~/.bashrc_dr4-bootes
python -c "import tractor;print(tractor)"
python -c "import astrometry;print(astrometry)"

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
set -x
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a



#outdir,statdir,brick,run_name
bri="$(echo $brick | head -c 3)"

log="$outdir/logs/$bri/$brick/log.objtype$objtype_rowstart$rowstart_$SLURM_JOBID"
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

threads=12
export OMP_NUM_THREADS=$threads

echo outdir="$outdir", brick="$brick"

export LEGACY_SURVEY_DIR=/scratch2/scratchdirs/kaylanb/dr3/desiproc-dr3-template
export DECALS_SIM_DIR=$outdir
srun -n 1 -c $OMP_NUM_THREADS python legacyanalysis/decals_sim.py \
    --objtype $objtype --brick $brick --rowstart $rowstart \
    --add_sim_noise --threads $OMP_NUM_THREADS
    >> $log 2>&1
rm $statdir/inq_$myrun.txt
#     --skip \
#     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
#     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
#     --outdir $outdir --nsigma 6 \
#     --force-all \

#--no-wise \
#--zoom 1400 1600 1400 1600
#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \

echo $run_name DONE $SLURM_JOBID


