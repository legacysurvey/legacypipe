#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH --account=desi
#SBATCH -J obiwan
#SBATCH -o obiwan.o%j
#SBATCH --mail-user=kburleigh@lbl.gov
#SBATCH --mail-type=END,FAIL
#SBATCH -L SCRATCH

#--qos=premium
#-p shared
#-n 6
#-p debug
#-N 1
# Run script like this
# for objtype in star elg lrg qso;do for rowstart in `seq 0 500 10000`; do bash decals_sim-bash-manager.sh $rowstart $objtype;done;done

# Yu Feng's bcast
source $CODE_DIR/yu-bcast/activate.sh
set -x
export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
#pwd
cd $CODE_DIR/legacypipe/py
#pwd
python -c "from legacypipe.runbrick import run_brick"
python -c "import sklearn;print(sklearn)"
# Put legacypipe in path

export objtype="$1"
export brick=1220p282
export rowstart=0
export prefix=finaltest
#export brick="$2"
#export rowstart="$3"
if [ "$objtype" = "star" ] || [ "$objtype" = "elg" ] || [ "$objtype" = "lrg" ] || [ "$objtype" = "qso" ] ; then
    echo Script running
else
    echo Crashing
    exit
fi



#source ~/.bashrc_dr4-bootes
#python -c "import tractor;print(tractor)"
#python -c "import astrometry;print(astrometry)"

#source /scratch1/scratchdirs/desiproc/DRs/dr4/legacypipe-dir/bashrc
# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a

threads=24
export OMP_NUM_THREADS=$threads
export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir #/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir
export DECALS_SIM_DIR=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr3-obiwan/$prefix #obiwan-eboss-ngc
export outdir=$DECALS_SIM_DIR
mkdir -p $outdir
export run_name=obiwan_$objtype_$brick_$rowstart
if [ "$NERSC_HOST" == "cori" ]; then
    cores=32
elif [ "$NERSC_HOST" == "edison" ]; then
    cores=24
fi
let tasks=${SLURM_JOB_NUM_NODES}*${cores}/$threads

# Clean up
#for i in `find . -maxdepth 1 -type f -name "slurm.o*"|xargs grep "${run_name} DONE"|cut -d ' ' -f 3|grep "^[0-9]*[0-9]$"`; do mv slurm.o$i $outdir/logs/;done

bri=$(echo $brick | head -c 3)
#let rowend=$rowstart+500
tractor_fits="$outdir/$objtype/$bri/$brick/rowstart$rowstart/tractor-$objtype-$brick-rowstart$rowstart.fits"
exceed_rows="$outdir/$objtype/$bri/$brick/rowstart${rowstart}_exceeded.txt"
export myrun=${objtype}_${brick}_rowst${rowstart}
if [ -e "$tractor_fits" ]; then
    echo skipping $tractor_fits, its done
    # Remove coadd, checkpoint, pickle files they are huge
    #rm $outdir/pickles/${bri}/runbrick-${brick}*.pickle
    #rm $outdir/checkpoints/${bri}/${brick}*.pickle
    #rm $outdir/coadd/${bri}/${brick}/*
elif [ -e "$exceed_rows" ]; then
    echo skipping $myrun, it exceeds total number artifical sources/rows
else
    # Main
    log="$outdir/logs/$bri/$brick/log.obj${objtype}_rowstart${rowstart}_${SLURM_JOBID}"
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
    srun -n $tasks -c $OMP_NUM_THREADS python obiwan/decals_sim.py \
        --objtype $objtype --brick $brick --rowstart $rowstart \
        --add_sim_noise --prefix $prefix --threads $OMP_NUM_THREADS \
        >> $log 2>&1
    #     --skip \
    #     --checkpoint $outdir/checkpoints/${bri}/${brick}.pickle \
    #     --pickle "$outdir/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
    #     --outdir $outdir --nsigma 6 \
    #     --force-all \
    #     --zoom 1400 1600 1400 1600

    echo $run_name DONE $SLURM_JOBID
fi
