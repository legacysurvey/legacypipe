#!/bin/bash 

set -x

allthreads=24
runbrick_threads=6
export OMP_NUM_THREADS=$runbrick_threads

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

brick="$1"
export run_name=dr4-qdo
# DR4
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
qdo_table=dr4v2
# Bootes
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/90primeTPV_mzlsv2thruMarch19/wisepsf
#qdo_table=dr4-bootes
mkdir -p $outdir


# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
# 67 kbytes is 64GB (mem of Edison node)
ulimit -S -v 65000000
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


#module load psfex-hpcp
#srun -n 1 -c 1 python python_test_qdo.py
python legacypipe/runbrick.py \
     --run $qdo_table \
     --brick $brick \
     --skip \
     --threads $OMP_NUM_THREADS \
     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
     --outdir $outdir --nsigma 6 \
     --no-write \
     >> $log 2>&1
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

# 
# qdo launch DR4 100 --cores_per_worker 24 --batchqueue regular --walltime 00:55:00 --script ./dr4-qdo.sh --keep_env --batchopts "-a 0-11"
# qdo launch DR4 300 --cores_per_worker 8 --batchqueue regular --walltime 00:55:00 --script ./dr4-qdo-threads8 --keep_env --batchopts "-a 0-11"
# qdo launch DR4 300 --cores_per_worker 8 --batchqueue regular --walltime 00:55:00 --script ./dr4-qdo-threads8-vunlimited.sh --keep_env --batchopts "-a 0-5"

#qdo launch mzlsv2_bcast 4 --cores_per_worker 6 --batchqueue debug --walltime 00:10:00 --script ./dr4-qdo.sh --keep_env
# MPI no bcast
#qdo launch mzlsv2 2500 --cores_per_worker 6 --batchqueue regular --walltime 01:00:00 --script ./dr4-qdo.sh --keep_env
# MPI w/ bcast
#uncomment bcast line in: /scratch1/scratchdirs/desiproc/DRs/code/dr4/qdo/qdo/etc/qdojob
#qdo launch mzlsv2_bcast 2500 --cores_per_worker 6 --batchqueue regular --walltime 01:00:00 --script ./dr4-qdo.sh --keep_env

#qdo launch dr4Bootes2 100 --cores_per_worker 24 --batchqueue debug --walltime 00:30:00 --script ./dr4-bootes-qdo.sh --keep_env
#qdo launch dr4Bootes2 8 --cores_per_worker 24 --batchqueue regular --walltime 01:00:00 --script ./dr4-bootes-qdo.sh --keep_env --batchopts "--qos=premium"
# qdo launch dr2n 16 --cores_per_worker 8 --walltime=24:00:00 --script ../bin/pipebrick.sh --batchqueue regular --verbose
# qdo launch edr0 4 --cores_per_worker 8 --batchqueue regular --walltime 4:00:00 --script ../bin/pipebrick.sh --keep_env --batchopts "--qos=premium -a 0-3"
