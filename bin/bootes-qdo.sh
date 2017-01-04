#!/bin/bash 

set -x

brick="$1"
#export run_name=bootes-qdo
# Bootes
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/90primeTPV_mzlsv2thruMarch19/wisepsf
run_name=dr4-bootes
threads=24
mkdir -p $outdir


# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a

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

export OMP_NUM_THREADS=$threads

#module load psfex-hpcp
#srun -n 1 -c 1 python python_test_qdo.py
python legacypipe/runbrick.py \
     --run $run_name \
     --brick $brick \
     --skip \
     --threads $OMP_NUM_THREADS \
     --outdir $outdir --nsigma 6 --force-all \
     >> $log 2>&1
#--checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
#--pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
# Bootes
#--run dr4-bootes \

#--no-wise \
#--zoom 1400 1600 1400 1600
#rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#
echo $run_name DONE $SLURM_JOBID

# Bootes
#for i in `seq 1 131`;do qdo launch dr4Bootes3 1 --cores_per_worker 12 --batchqueue shared --walltime 1:00:00 --keep_env --script ./bootes-qdo.sh;done
