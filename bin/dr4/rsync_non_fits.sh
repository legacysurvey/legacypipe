#!/bin/bash

# Run
# ../bin/rsync_non_fits.sh 1 1000

start_ln="$1"
end_ln="$2"

bricklist=dr4_bricks.txt
dr=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
target_dr=/global/cscratch1/sd/desiproc/dr4/data_release/dr4c
# Loop
set -x
n_bricks=`wc -l $bricklist |awk '{print $1}'`
cnt=0
for brick in `sed -n ${start_ln},${end_ln}p $bricklist`;do
    let cnt=$cnt+1
    ith=$(($cnt % 100))
    if [ "$ith" == "0" ];then echo $cnt/$n_bricks;fi
    bri=`echo $brick|head -c 3`
    # Jpegs in coadd dir
    #coadd_fns=`find ${dr}/coadd/$bri/$brick/*.jpg`
    rsync -av ${dr}/coadd/$bri/$brick/*.jpg ${target_dr}/coadd/$bri/$brick/
    # Chkpts
    #find ${dr}/checkpoints/${brick}.pickle
    rsync -av ${dr}/checkpoints/${brick}.pickle ${target_dr}/checkpoints/
    #logs_dr=${dr}/logs
done
