#!/bin/bash

# checkpoints  coadd  metrics  tractor  tractor-i
# ../bin/dr4-rsync-projecta.sh coadd
# checkpoints  coadd  metrics  tractor  tractor-i
#nohup ../bin/dr4-rsync-projecta.sh checkpoints > checkpoints_rsync.out &
#nohup ../bin/dr4-rsync-projecta.sh coadd > coadd_rsync.out &
#nohup ../bin/dr4-rsync-projecta.sh metrics > metrics_rsync.out &
#nohup ../bin/dr4-rsync-projecta.sh tractor > tractor_rsync.out &
#nohup ../bin/dr4-rsync-projecta.sh tractor-i > tractor-i_rsync.out &

whichdir="$1"

if [ "$NERSC_HOST" == "edison" ]; then
    outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
    #bricklist=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir/bricks-dr4-${NERSC_HOST}.txt
else
    outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
    #bricklist=/global/cscratch1/sd/desiproc/dr4/legacypipe-dir/bricks-dr4-${NERSC_HOST}.txt
fi

if [ ! -e "$outdir/$whichdir" ]; then
    echo does not exist: $outdir/$whichdir
    exit 999
fi

proj_dir=/global/projecta/projectdirs/cosmo/work/dr4

# List all files that exists for each completed brick
# use projecta/ permissions and group
#args="-aRv --no-p --no-g --chmod=ugo=rwX"
args="-av --no-p --no-g --chmod=ugo=rwX"
echo copying $outdir/$whichdir to $proj_dir
rsync $args $outdir/$whichdir ${proj_dir}/
#for brick in `cat $bricklist`;do
    #bri="$(echo $brick | head -c 3)"
    #rsync $args $outdir/checkpoints/$bri/$brick.pickle ${proj_dir}/
    #rsync $args $outdir/coadd/$bri/$brick ${proj_dir}/
    #rsync $args $outdir/metrics/$bri/*${brick}* ${proj_dir}/
    #rsync $args $outdir/tractor/$bri/*${brick}* ${proj_dir}/
    #rsync $args $outdir/tractor-i/$bri/*${brick}* ${proj_dir}/
#done
#chown -R cosmo ${proj_dir}/
chgrp -R cosmo ${proj_dir}/
chmod -R ug=rwx,o=rx ${proj_dir}/
#find ${proj_dir}/ -type f | xargs chmod -R ug=rw,o=r ${proj_dir}/
#find ${proj_dir}/ -type d | xargs chmod -R ug=rwx,o=rx ${proj_dir}/
#chmod -R 775 ${proj_dir}/
