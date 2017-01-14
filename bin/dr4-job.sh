#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 12
#SBATCH --qos=scavenger
#SBATCH --array=1-3000
#SBATCH -t 08:00:00
#SBATCH --account=desi
#SBATCH -J DR4
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#-o DR4.o%j
#-p shared
#-n 12
#-p debug
#-N 1

#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh


# DR4

export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
#export outdir=/scratch2/scratchdirs/kaylanb/dr4
qdo_table=dr4v2
# Override Use dr4 legacypipe-dr
#export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

export slurm_name=dr4-qdo

# Threads
usecores=6
threads=6
export OMP_NUM_THREADS=$threads

export statdir="${outdir}/progress"
mkdir -p $statdir 

set -x

# Run another srun if current brick finishes
num_runs=0
while true; do
    let num_runs=$num_runs+1
    echo num_runs=$num_runs

    ########## GET BRICK
    echo GETTING BRICK
    date
    bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4.txt
    if [ ! -e "$bricklist" ]; then
        echo file=$bricklist does not exist, quitting
        exit 999
    fi
    # Start at random line, avoids running same brick
    lns=`wc -l $bricklist |awk '{print $1}'`
    rand=`echo $((1 + RANDOM % $lns))`
    # Use <<< to prevent loop from being subprocess where variables get lost
    while read aline; do
        brick="$aline"
        bri=$(echo $brick | head -c 3)
        tractor_fits=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/tractor/$bri/tractor-$brick.fits
        if [ -e "$tractor_fits" ]; then
            continue
        elif [ -e "$statdir/inq_$brick.txt" ]; then
            continue
        else
            # Found a brick to run
            export brick="$brick"
            touch $statdir/inq_$brick.txt
            break
        fi
    done <<< "$(sed -n ${rand},${lns}p $bricklist)"

    echo FOUND BRICK
    date
    ################

    #export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/90primeTPV_mzlsv2thruMarch19/wisepsf
    #qdo_table=dr4-bootes


    # Force MKL single-threaded
    # https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
    export MKL_NUM_THREADS=1
    # Try limiting memory to avoid killing the whole MPI job...
    # 67 kbytes is 64GB (mem of Edison node)
    #ulimit -S -v 65000000
    ulimit -a

    log="$outdir/logs/$brick/log.$SLURM_ARRAY_TASK_ID"
    mkdir -p $(dirname $log)
    echo Logging to: $log
    echo "-----------------------------------------------------------------------------------------" >> $log
    #module load psfex-hpcp
    #srun -n 1 -c 1 python python_test_qdo.py
    echo doing srun
    date
    srun -n 1 -c $usecores python legacypipe/runbrick.py \
         --run $qdo_table \
         --brick $brick \
         --skip \
         --threads $OMP_NUM_THREADS \
         --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
         --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
         --outdir $outdir --nsigma 6 \
         --no-write \
         >> $log 2>&1 & 
    wait 
    date
    rm $statdir/inq_$brick.txt

done
# Bootes
#--run dr4-bootes \

#--no-wise \
#--zoom 1400 1600 1400 1600
#rm $statdir/inq_$brick.txt

#     --radec $ra $dec
#    --force-all --no-write \
#    --skip-calibs \
#
echo $slurm_name DONE $SLURM_ARRAY_TASK_ID

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
