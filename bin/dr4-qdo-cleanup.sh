#!/bin/bash

# Run: bash dr4-bootes-bash-manager.sh 90prime; bash dr4-bootes-bash-manager.sh mzlsv2thruMarch19

#export bricks=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir/bricks-dr4.txt
#export bricks="$1"
#if [ "$(basename $bricks)" = "" ] || [ "$camera" = "mzlsv2thruMarch19" ] ; then
#    echo Script running
#else
#    echo Crashing
#    exit
#fi

#for cnt in 1 2 3 4 5; do
export run_name=dr4-qdo
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4

# Clean up
#for i in `find . -maxdepth 1 -type f -name "slurm-309*.out"|xargs grep "${run_name} DONE"|cut -d ' ' -f 3|grep "^[0-9]*[0-9]$"`; do mv dr4-bootes.o$i $outdir/logs/;done

# Identify done bricks and remove pickles for these
#don=bricks_done.txt
#rm $don
#for fn in `find ${outdir}/tractor -name "tractor*.fits"`;do echo $(basename $fn)|sed s/tractor-//g |sed s/.fits//g >> $don;done
#for brick in `cat $don`;do
#    bri=$(echo $brick | head -c 3)
#    tractor_fits=$outdir/tractor/$bri/tractor-$brick.fits
#    if [ -e "$tractor_fits" ]; then
#        echo Brick done: $brick, removing pickles,checkpoints
#        # Remove coadd, checkpoint, pickle files they are huge
#        rm $outdir/pickles/${bri}/runbrick-${brick}*.pickle
#        rm $outdir/checkpoints/${bri}/${brick}.pickle
#    fi
#done

# Size of DR4 dirs
echo Size of DR4 dirs:
for dr in `find $outdir -maxdepth 1 -type d|sed -n 2,8p`;do echo $dr;du -shc $dr;done

# Find pickles having at least first stage, keep only latest stage
#pics=tims_pickles.txt
#rm $pics
#find $outdir/pickles/*/runbrick-*-tims.pickle >> $pics
#wc -l $pics
#for fn in `cat $pics`;do
#    coadds=`echo $fn|sed s/-tims.pickle/-coadds.pickle/g`
#    fitblobs=`echo $fn|sed s/-tims.pickle/-fitblobs.pickle/g`
#    srcs=`echo $fn|sed s/-tims.pickle/-srcs.pickle/g`
#    mask=`echo $fn|sed s/-tims.pickle/-mask_junk.pickle/g`
#    echo $fn
#    # Keep tims so can find them again
#    #if [ -e "$mask" ]; then rm $fn;fi
#    if [ -e "$srcs" ]; then rm $mask;fi
#    if [ -e "$fitblobs" ]; then rm $srcs;fi
#    if [ -e "$coadds" ]; then rm $fitblobs;fi
#    #ls $fn
#    #ls $mask
#    #ls $srcs
#    #ls $fitblobs
#    #ls $coadds
#done

# Remove all pickles except fitblobs and coadds
rm $outdir/pickles/*/runbrick-*-{srcs,tims,mask_junk}.pickle

## Find latest pickle, remove all else
#stag=coadds
#pics=pickles_${stag}.txt;rm $pics
#find $outdir/pickles/*/runbrick-*-${stag}.pickle >> $pics
#for fn in `cat $pics`;do
#    fitblobs=`echo $fn|sed s/-${stag}.pickle/-fitblobs.pickle/g`
#    srcs=`echo $fn|sed s/-${stag}.pickle/-srcs.pickle/g`
#    mask=`echo $fn|sed s/-${stag}.pickle/-mask_junk.pickle/g`
#    tims=`echo $fn|sed s/-${stag}.pickle/-tims.pickle/g`
#    echo $fn
#    rm $fitblobs $srcs $mask $tims
#done
#stag=fitblobs
#pics=pickles_${stag}.txt;rm $pics
#find $outdir/pickles/*/runbrick-*-${stag}.pickle >> $pics
#for fn in `cat $pics`;do
#    srcs=`echo $fn|sed s/-${stag}.pickle/-srcs.pickle/g`
#    mask=`echo $fn|sed s/-${stag}.pickle/-mask_junk.pickle/g`
#    tims=`echo $fn|sed s/-${stag}.pickle/-tims.pickle/g`
#    echo $fn
#    rm $srcs $mask $tims
#done
#stag=srcs
#pics=pickles_${stag}.txt;rm $pics
#find $outdir/pickles/*/runbrick-*-${stag}.pickle >> $pics
#for fn in `cat $pics`;do
#    mask=`echo $fn|sed s/-${stag}.pickle/-mask_junk.pickle/g`
#    tims=`echo $fn|sed s/-${stag}.pickle/-tims.pickle/g`
#    echo $fn
#    rm $mask $tims
#done
#stag=mask_junk
#pics=pickles_${stag}.txt;rm $pics
#find $outdir/pickles/*/runbrick-*-${stag}.pickle >> $pics
#for fn in `cat $pics`;do
#    tims=`echo $fn|sed s/-${stag}.pickle/-tims.pickle/g`
#    echo $fn
#    rm $tims
#done






# Size of DR4 dirs
echo Size of DR4 dirs:
for dr in `find $outdir -maxdepth 1 -type d|sed -n 2,8p`;do echo $dr;du -shc $dr;done

#sleep 7200
#done
