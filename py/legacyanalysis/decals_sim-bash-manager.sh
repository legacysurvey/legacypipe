#!/bin/bash

# Run script like this
# for i in `seq 0 9`;do objtype=STAR;let rowstart=1+$i*500;bash decals_sim-bash-manager.sh $rowstart $objtype;done 
export rowstart="$1"
export objtype="$2"
export outdir=/scratch2/scratchdirs/kaylanb/obiwan
export statdir="${outdir}/progress"
mkdir -p $statdir $outdir

export run_name=obiwan-testrun

# Clean up
for i in `find . -maxdepth 1 -type f -name "${run_name}.o*"|xargs grep "${run_name} DONE"|cut -d ' ' -f 3|grep "^[0-9]*[0-9]$"`; do mv ${run_name}.o$i $outdir/logs/;done

# Cancel ALL jobs and remove all inq.txt files
#for i in `squeue -u desiproc|grep dr3-mzls-b|cut -c10-18`; do scancel $i;done
#rm /scratch1/scratchdirs/desiproc/data-releases/dr3-mzls-bash/progress/*.txt

# Submit jobs
bricklist="$1"
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

for brick in `cat $bricklist`;do
    export brick="$brick"
    bri=$(echo $brick | head -c 3)
    tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
    if [ -e "$tractor_fits" ]; then
        echo skipping $brick, its done
        # Remove coadd, checkpoint, pickle files they are huge
        #rm $outdir/pickles/${bri}/runbrick-${brick}*.pickle
        #rm $outdir/checkpoints/${bri}/${brick}*.pickle
        #rm $outdir/coadd/${bri}/${brick}/*
    elif [ -e "$statdir/inq_$brick.txt" ]; then
        echo skipping $brick, its queued
    else
        echo submitting $brick
        touch $statdir/inq_$brick.txt
        sbatch ${run_name}.sh --export objtype,rowstart,outdir,statdir,brick,run_name
    fi
done
