#!/bin/bash 

#export camera=mosaic
export camera=decam
export outdir=/global/cscratch1/sd/kaylanb/observing_paper

bricklist=${outdir}/${camera}_list.txt
echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

export statdir="${outdir}/progress"
mkdir -p $statdir 

# Loop over bricks
start_brick=1
end_brick=1
cnt=0
while read aline; do
    export imagefn=`echo $aline|awk '{print $1}'`
    fn=`basename $imagefn`
    dr=`dirname $imagefn|awk -F '/' '{print $NF}'`
    export drfn=${dr}_$fn
    stat_file=$statdir/submitted_$drfn.txt
    if [ -e "$stat_file" ]; then
        continue
    fi
    sbatch legacyccds/job_zpts.sh --export imagefn,outdir,drfn,camera
    touch $stat_file
    let cnt=${cnt}+1
done <<< "$(sed -n ${start_brick},${end_brick}p $bricklist)"
echo submitted $cnt bricks
