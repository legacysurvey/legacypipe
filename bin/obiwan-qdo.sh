#!/bin/bash 

set -x
export PYTHONPATH=$CODE_DIR/legacypipe/py:${PYTHONPATH}
cd $CODE_DIR/legacypipe/py

export objtype="$1"
export brick="$2"
export rowstart="$3"

export therun=eboss-ngc
export prefix=finaltest
if [ "$objtype" = "star" ] || [ "$objtype" = "elg" ] || [ "$objtype" = "lrg" ] || [ "$objtype" = "qso" ] ; then
    echo Script running
else
    echo Crashing
    exit
fi


# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1
# Try limiting memory to avoid killing the whole MPI job...
ulimit -a

threads=8
export OMP_NUM_THREADS=$threads
export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr3-obiwan/legacypipe-dir #/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir
export DECALS_SIM_DIR=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr3-obiwan/$prefix #obiwan-eboss-ngc
export outdir=$DECALS_SIM_DIR
mkdir -p $outdir
export run_name=obiwan_$objtype_$brick_$rowstart

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
    python obiwan/decals_sim.py \
        --run $therun --objtype $objtype --brick $brick --rowstart $rowstart \
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
