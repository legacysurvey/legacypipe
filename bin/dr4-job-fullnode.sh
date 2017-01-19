#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J OneNode
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#--array=1-2
#--qos=premium
#-p shared
#-n 24
#-p regular
#-N 1

# TO RUN
# set usecores as desired for more mem and set shared n above to 2*usecores, keep threads=6 so more mem per thread!, then --aray equal to number largemmebricks.txt


bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-oom-forced.txt
#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-oom-fitblobs-coadd.txt
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
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

threads=6
export OMP_NUM_THREADS=$threads

set -x
# Initialize job pids with bogus
for cnt in `seq 1 4`;do
    declare pid_${cnt}=blahblah 
    #export "pid_${cnt}=blahblah"
    pid=pid_${cnt}
    echo cnt=$cnt, pid=${!pid}
done
set +x
for i in `seq 1 1440`;do
    cnt=0
    for usecores in 12 6 6;do 
        # Which pid is this?
        let cnt=$cnt+1
        pid=pid_${cnt}
        # Check if pid is running
        running=`ps |grep ${!pid} |wc -c`
        if [ "$running" -gt 0 ]; then
            # All sruns are going
            echo Waiting: pid_${cnt}=${!pid} IS running
        else
            # Need to srun a brick
            echo pid_${cnt}=${!pid} has finished, Do a new srun
            # Limit memory
            echo usecores=$usecores
            if [ "$usecores" -eq 24 ]; then
                # full node (~ 62 GB)
                maxmem=65000000
            elif [ "$usecores" -eq 12 ]; then
                maxmem=32000000
            elif [ "$usecores" -eq 6 ]; then
                maxmem=16000000
            else
                echo usecores=$usecores not supported
                exit
            fi
            ulimit -S -v $maxmem
            ulimit -a
            # Get a brick to run
            date
            echo GETTING BRICK
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
                    continue
                elif [ -e "$statdir/inq_$brick.txt" ]; then
                    continue
                else
                    gotbrick=1
                    # Found a brick to run
                    #export brick="$brick"
                    touch $statdir/inq_$brick.txt
                    break
                fi
            done <<< "$(sed -n ${rand},${lns}p $bricklist)"
            if [ "$gotbrick" -eq 0 ]; then
                echo Never found a brick, exiting
                exit
            fi
            echo FOUND BRICK
            date
            
            set -x
            log="$outdir/logs/$bri/log.${brick}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            mkdir -p $(dirname $log)
            echo Logging to: $log
            echo "-----------------------------------------------------------------------------------------" >> $log
            # Copy checkpoints file to new outdir
            # Only happens if outdir is not dr4/, e.g. if running a test
            check_name=$outdir/checkpoints/$bri/${brick}.pickle
            if [ ! -e $check_name ]; then
                mkdir $outdir/checkpoints/$bri
                cp /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/checkpoints/$bri/${brick}.pickle $outdir/checkpoints/$bri/
            fi 
            # Print why running this brick
            echo Running b/c: $whyrun

            echo doing srun
            date
            # Run and store process ID
            srun -n 1 -c $usecores python legacypipe/runbrick.py \
                 --run $qdo_table \
                 --brick $brick \
                 --hybrid-psf \
                 --skip \
                 --threads $OMP_NUM_THREADS \
                 --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
                 --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
                 --outdir $outdir --nsigma 6 \
                 --no-write \
                 >> $log 2>&1 & 
            # KEY LOGIC: Store PID so know when it is running or done
            declare pid_${cnt}=$!
            pid=pid_${cnt}
            echo cnt=$cnt, pid=${!pid}
            set +x 
        fi
    # End loop over usecores / checking if need launch srun
    done
    sleep 60
# End loop over sleep
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
