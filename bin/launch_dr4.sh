#!/bin/bash 

export overwrite_tractor=no
export full_stacktrace=yes

#bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-${NERSC_HOST}.txt
bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-notdone-${NERSC_HOST}.txt
if [ "$overwrite_tractor" = "yes" ]; then
    bricklist=${LEGACY_SURVEY_DIR}/aadsfasdf.txt 
elif [ "$full_stacktrace" = "yes" ]; then
    bricklist=${LEGACY_SURVEY_DIR}/bricks-dr4-full-stacktrace-${NERSC_HOST}.txt 
fi 
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

# Output dir
if [ "$NERSC_HOST" = "edison" ]; then
    export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
else
    export outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
fi


export statdir="${outdir}/progress"
mkdir -p $statdir 

# Loop over bricks
num_bricks=1
cnt=0
while read aline; do
    export brick=`echo $aline|awk '{print $1}'`
    if [ "$full_stacktrace" = "yes" ];then
        stat_file=$statdir/stacktrace_$brick.txt
    else
        stat_file=$statdir/submitted_$brick.txt
    fi
    bri=$(echo $brick | head -c 3)
    tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
    if [ -e "$tractor_fits" ]; then
        if [ "$overwrite_tractor" = "yes" ]; then
            echo ignoring existing tractor.fits
        else
            continue
        fi
    fi
    if [ -e "$stat_file" ]; then
        continue
    fi
    sbatch ../bin/job_dr4.sh --export brick,outdir,overwrite_tractor,full_stacktrace
    touch $stat_file
    let cnt=${cnt}+1
done <<< "$(sed -n 1,${num_bricks}p $bricklist)"
echo submitted $cnt bricks
