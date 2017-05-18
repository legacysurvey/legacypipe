#!/bin/bash 

#export bricklist=$SCRATCH/test/legacypipe/py/coadd_dirs.txt
export bricklist=/global/cscratch1/sd/kaylanb/test/legacypipe/py/dr4_bricks.txt

echo bricklist=$bricklist
if [ ! -e "$bricklist" ]; then
    echo file=$bricklist does not exist, quitting
    exit 999
fi

# Loop over a star and end
step=100
beg=1
max=100
###
cnt=0
for star in `seq ${beg} ${step} ${max}`;do 
    export start_brick=${star}
    let en=${star}+${step}-1
    export end_brick=${en}
    sbatch ../bin/sharedq_job.sh --export bricklist,start_brick,end_brick
    let cnt=${cnt}+1
done
echo submitted $cnt bricks
