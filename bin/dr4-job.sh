#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 12
#SBATCH -t 24:00:00
#SBATCH --array=1-30
#SBATCH --qos=scavenger
#SBATCH --account=desi
#SBATCH -J dr4
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH
#-C haswell

#-p shared
#-n 24
#-p regular
#-N 1

# TO RUN
# 1) haswell if Cori
# set usecores as desired for more mem and set shared n above to 2*usecores, keep threads=6 so more mem per thread!, then --aray equal to number largemmebricks.txt


rerun_wise=yes

usecores=6
#threads=$usecores
threads=4
# Limit memory to avoid 1 srun killing whole node
if [ "$NERSC_HOST" = "edison" ]; then
    # 62 GB / Edison node = 65000000 kbytes
    maxmem=65000000
    let usemem=${maxmem}*${usecores}/24
else
    # 128 GB / Edison node = 65000000 kbytes
    maxmem=134000000
    let usemem=${maxmem}*${usecores}/32
fi
ulimit -S -v $usemem
ulimit -a
echo usecores=$usecores
echo threads=$threads

# Output dir
if [ "$NERSC_HOST" = "edison" ]; then
    export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
else
    export outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
fi

# Extra srun options
if [ "$NERSC_HOST" = "edison" ]; then
    if [ "$rerun_wise" = "yes" ]; then
        export extra_opt=""
    else
        export extra_opt="--skip"
    fi
else
    export extra_opt="--skip"
fi

#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4.txt
bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}.txt 
if [ "$rerun_wise" = "yes" ]; then
    bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}-nowise.txt 
fi 
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-edison.txt 
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-cori.txt 
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-oom-forced.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-oom-fitblobs-coadd.txt
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

export OMP_NUM_THREADS=$threads
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/large-mem-runs/${usecores}usecores
export statdir="${outdir}/progress"
mkdir -p $statdir $outdir

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1



#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh

qdo_table=dr4v2

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

# Run another srun if current brick finishes
num_runs=0
while true; do
    let num_runs=$num_runs+1
    echo num_runs=$num_runs

    ########## GET BRICK
    echo GETTING BRICK
    date
    # Start at random line, avoids running same brick
    lns=`wc -l $bricklist |awk '{print $1}'`
    rand=`echo $((1 + RANDOM % $lns))`
    # Use <<< to prevent loop from being subprocess where variables get lost
    gotbrick=0
    while read aline; do
        brick=`echo $aline|awk '{print $1}'`
        whyrun=`echo $aline|awk '{print $2}'`
        bri=$(echo $brick | head -c 3)
        tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
        if [ -e "$tractor_fits" ]; then
            if [ "$rerun_wise" = "yes" ]; then
                echo rerun_wise ignoring existing tractor.fits
            else
                continue
            fi
        fi
        if [ -e "$statdir/inq_$brick.txt" ]; then
            continue
        fi
        # If reached this point, run the brick
        gotbrick=1
        # Found a brick to run
        #export brick="$brick"
        touch $statdir/inq_$brick.txt
        break
    done <<< "$(sed -n ${rand},${lns}p $bricklist)"
    if [ "$gotbrick" -eq 0 ]; then
        echo Never found a brick, exiting
        exit
    fi

    echo FOUND BRICK
    date
    ################
    
    set -x
    log="$outdir/logs/$bri/log.${brick}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p $(dirname $log)
    echo Logging to: $log
    echo "-----------------------------------------------------------------------------------------" >> $log
    #module load psfex-hpcp
    #srun -n 1 -c 1 python python_test_qdo.py

    # Copy checkpoints file to new outdir
    # Only happens if outdir is not dr4/, e.g. if running a test
    #check_name=$outdir/checkpoints/$bri/${brick}.pickle
    #if [ ! -e $check_name ]; then
    #    mkdir $outdir/checkpoints/$bri
    #    cp /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/checkpoints/$bri/${brick}.pickle $outdir/checkpoints/$bri/
    #fi 
    # Print why running this brick
    echo Running b/c: $whyrun

    echo doing srun
    date
    srun -n 1 -c $usecores python legacypipe/runbrick.py \
         --run $qdo_table \
         --brick $brick \
         --hybrid-psf \
         --threads $threads \
         --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
         --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
         --outdir $outdir --nsigma 6 \
         --no-write ${extra_opt} \
         >> $log 2>&1 & 
    wait
    date
    rm $statdir/inq_$brick.txt
    set +x 

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
echo dr4 DONE 

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
