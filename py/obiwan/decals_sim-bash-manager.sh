#!/bin/bash

# Run script like this
# for objtype in star elg lrg qso;do for rowstart in `seq 0 500 10000`; do bash decals_sim-bash-manager.sh $rowstart $objtype;done;done
export rowstart="$1"
export objtype="$2"
if [ "$objtype" = "star" ] || [ "$objtype" = "elg" ] || [ "$objtype" = "lrg" ] || [ "$objtype" = "qso" ] ; then
    echo Script running
else
    echo Crashing
    exit
fi


export outdir=/scratch2/scratchdirs/kaylanb/dr3-bootes
#export outdir=/scratch2/scratchdirs/kaylanb/obiwan
#bricklist=bricks_ndwfs.txt
bricklist=bricks-dr3-bootes.txt
export run_name=bootes-dr3-obiwan
export statdir="${outdir}/progress"
mkdir -p $statdir $outdir


# Clean up
for i in `find . -maxdepth 1 -type f -name "${run_name}.o*"|xargs grep "${run_name} DONE"|cut -d ' ' -f 3|grep "^[0-9]*[0-9]$"`; do mv ${run_name}.o$i $outdir/logs/;done

# Cancel ALL jobs and remove all inq.txt files
#for i in `squeue -u desiproc|grep dr3-mzls-b|cut -c10-18`; do scancel $i;done
#rm /scratch1/scratchdirs/desiproc/data-releases/dr3-mzls-bash/progress/*.txt

# Submit jobs
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

for brick in `cat $bricklist`;do
    export brick="$brick"
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
    elif [ -e "$statdir/inq_$myrun.txt" ]; then
        echo skipping $myrun, its queued
    elif [ -e "$exceed_rows" ]; then
        echo skipping $myrun, it exceeds total number artifical sources/rows
    else
        echo submitting $myrun
        touch $statdir/inq_$myrun.txt
        sbatch decals_sim-bash.sh --export objtype,rowstart,outdir,statdir,brick,run_name,myrun
    fi
done
