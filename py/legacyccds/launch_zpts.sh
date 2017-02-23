#!/bin/bash 

export camera=mosaicz
#export camera=decam

export outdir=/global/cscratch1/sd/kaylanb/zeropoints
export imagelist=${outdir}/${camera}_imagelist.txt
echo imagelist=$imagelist
if [ ! -e "$imagelist" ]; then
    echo file=$imagelist does not exist, quitting
    exit 999
fi

sbatch legacyccds/job_zpts.sh --export outdir,imagelist,camera

#export statdir="${outdir}/progress"
#mkdir -p $statdir 

# Loop over bricks
#start_brick=1
#end_brick=2000
#cnt=0
#while read aline; do
#    export brick=`echo $aline|awk '{print $1}'`
#    if [ "$full_stacktrace" = "yes" ];then
#        stat_file=$statdir/stacktrace_$brick.txt
#    else
#        stat_file=$statdir/submitted_$brick.txt
#    fi
#    bri=$(echo $brick | head -c 3)
#    tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
#    if [ -e "$tractor_fits" ]; then
#        if [ "$overwrite_tractor" = "yes" ]; then
#            echo ignoring existing tractor.fits
#        else
#            continue
#        fi
#    fi
#    if [ -e "$stat_file" ]; then
#        continue
#    fi
#    sbatch ../bin/job_dr4.sh --export brick,outdir,overwrite_tractor,full_stacktrace
#    touch $stat_file
#    let cnt=${cnt}+1
#done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
#echo submitted $cnt bricks
