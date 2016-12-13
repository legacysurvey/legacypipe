#!/bin/bash 

set -x

brick="$1"
export run_name=dr4-bootes-qdo
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/90primeTPV_mzlsv2thruMarch19
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

threads=24
export OMP_NUM_THREADS=$threads

module load psfex-hpcp
#srun -n 1 -c 1 python python_test_qdo.py
python legacypipe/runbrick.py \
     --run dr4-bootes \
     --brick $brick \
     --skip \
     --threads $OMP_NUM_THREADS \
     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --outdir $outdir --nsigma 6 \
     >> $log 2>&1
#--no-wise \
#--zoom 1400 1600 1400 1600
#rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#
echo $run_name DONE $SLURM_JOBID

#qdo launch dr4Bootes2 100 --cores_per_worker 24 --batchqueue debug --walltime 00:30:00 --script ./dr4-bootes-qdo.sh --keep_env
#qdo launch dr4Bootes2 8 --cores_per_worker 24 --batchqueue regular --walltime 01:00:00 --script ./dr4-bootes-qdo.sh --keep_env --batchopts "--qos=premium"
# qdo launch dr2n 16 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose
# qdo launch edr0 4 --cores_per_worker 8 --batchqueue regular --walltime 4:00:00 --script ../bin/pipebrick.sh --keep_env --batchopts "--qos=premium -a 0-3"
