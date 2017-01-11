#!/bin/bash -l

#SBATCH -p shared
#SBATCH -n 2
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J OBIWAN
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

export runwhat=star
#--array=1-2
#--qos=scavenger
#-o DR4.o%j
#-p shared
#-n 12
#-p debug
#-N 1

#bcast
#source /scratch1/scratchdirs/desiproc/DRs/code/dr4/yu-bcast_2/activate.sh


# DR4
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
#export outdir=/scratch2/scratchdirs/kaylanb/dr4
export outdir=$DECALS_SIM_DIR
qdo_table=dr4v2
# Override Use dr4 legacypipe-dr
#export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir

export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

########## GET OBJTYPE, BRICK, ROWSTART
export statdir="${outdir}/progress"
mkdir -p $statdir $outdir

echo GETTING BRICK
date
bricklist=${LEGACY_SURVEY_DIR}/eboss-ngc-load-${runwhat}.txt
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi
# Start at random line, avoids running same brick
lns=`wc -l $bricklist |awk '{print $1}'`
rand=`echo $((1 + RANDOM % $lns))`
sed -n ${rand},${lns}p $bricklist | while read aline; do
    objtype=`echo $aline|awk '{print $1}'`
    brick=`echo $aline|awk '{print $2}'`
    rowstart=`echo $aline|awk '{print $3}'`
    # Check whether to skip it
    bri=$(echo $brick | head -c 3)
    tractor_fits="$outdir/$objtype/$bri/$brick/rowstart$rowstart/tractor-$objtype-$brick-rowstart$rowstart.fits"
    exceed_rows="$outdir/$objtype/$bri/$brick/rowstart${rowstart}_exceeded.txt"
    if [ -e "$tractor_fits" ]; then
        continue
    elif [ -e "$exceed_rows" ]; then
        continue
    elif [ -e "$statdir/inq_$brick.txt" ]; then
        continue
    else
        # Found something to run
        #export objtype="$objtype"
        #export brick="$brick"
        #export rowstart="$rowstart"
        touch $statdir/inq_$brick.txt
        break
    fi
done

echo FOUND BRICK: $objtype $brick $rowstart
date
################

set -x

export run_name=obiwan_$objtype_$brick_$rowstart
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/90primeTPV_mzlsv2thruMarch19/wisepsf
#qdo_table=dr4-bootes

# Threads
usecores=6
threads=6
export OMP_NUM_THREADS=$threads

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
# 67 kbytes is 64GB (mem of Edison node)
#ulimit -S -v 65000000
ulimit -a

log="$outdir/logs/$brick/log.$SLURM_JOBID"
mkdir -p $(dirname $log)
echo Logging to: $log
echo "-----------------------------------------------------------------------------------------" >> $log
#module load psfex-hpcp
export therun=eboss-ngc
export prefix=eboss_ngc
srun -n 1 -c $usecores python obiwan/decals_sim.py \
    --run $therun --objtype $objtype --brick $brick --rowstart $rowstart \
    --add_sim_noise --prefix $prefix --threads $OMP_NUM_THREADS \
    >> $log 2>&1

rm $statdir/inq_$brick.txt
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
