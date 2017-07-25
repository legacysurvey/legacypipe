#!/bin/bash

export camera="$1"
if [ "$camera" = "90prime" ] || [ "$camera" = "mzlsv2thruMarch19" ] ; then
    echo Script running
else
    echo Crashing
    exit
fi

export run_name=dr4-bootes-$camera
export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4-bootes/$camera

# Bricks finished and not
for fn in `find $outdir/tractor -name "tractor-*.fits"`;do echo $(basename $fn)|sed 's/tractor-//g'|sed 's/.fits//g';done > bricks_done_$camera.tmp
python dr4-bootes-status.py --camera $camera --get_bricks_notdone
# log files for those not finished
tmp_dir=temp_$camera
if [ -e "tmp_dir" ]
then
    rm $tmp_dir/*
    rmdir $tmp_dir
fi
mkdir $tmp_dir
for brick in `cat bricks_notdone_$camera.tmp`
do
    bri="$(echo $brick | head -c 3)"
    log=`ls -lrt $outdir/logs/$bri/$brick/log.*|tail -n 1|expand|awk '{print $9}'`
    cp $log $tmp_dir/
done
# Finish
ball=`wc -l bricks_all_$camera.tmp |expand|cut -d " " -f 1`
don=`wc -l bricks_done_$camera.tmp |expand|cut -d " " -f 1`
notdon=`wc -l bricks_notdone_$camera.tmp |expand|cut -d " " -f 1`
nfils=`ls $tmp_dir|wc -l`
echo Bricks total=$ball, finished=$don, not=$notdon
echo Latest stdout files: $tmp_dir/, there are: $nfils

# List errors in stdout files
echo ERRORs
# file to write bricks to resubmit
resubmitfn=bricks_easyresubmit_$camera.txt
if [ -e "$resubmitfn" ]
then 
    rm $resubmitfn
fi
# List errors
for fn in `ls $tmp_dir/log.*`
do
    mem=`grep "Exceeded job memory limit at some point" $fn |wc -c`
    pol=`grep "ValueError: unknown record: POLNAME1" $fn |wc -c`
    tim=`grep "raise TimeoutError" $fn |wc -c`
    goteof=`grep "EOFError" $fn |wc -c`
    gotfitsio=`grep "IOError: FITSIO status" $fn |wc -c`
    notphot=`grep "No photometric CCDs touching brick" $fn |wc -c`
    runt=`grep "RuntimeError" $fn |wc -c`
    if [ "$mem" -gt 0 ]
    then
        echo $fn Exceeded job memory limit at some point
    elif [ "$pol" -gt 0 ]
    then
        echo $fn ValueError: unknown record: POLNAME1
    elif [ "$tim" -gt 0 ]
    then
        echo $fn raise TimeoutError
    elif [ "$goteof" -gt 0 ]
    then
        echo $fn EOFError
    elif [ "$gotfitsio" -gt 0 ]
    then
        echo $fn IOError: FITSIO status
    elif [ "$notphot" -gt 0 ]
    then
        echo $fn No photometric CCDs touching brick
    elif [ "$runt" -gt 0 ]
    then
        echo $fn RuntimeError
    else 
        echo $fn no error
    fi

    # For easy to fix errors write bricknames to txt file
    if [ "$mem" -gt 0 ] || [ "$tim" -gt 0 ]
    then
        brick=`awk '/Command-line args:/ { print $7 }' $fn| sed "s/\,//g"|sed "s/'//g"`
        echo $brick >> $resubmitfn
    fi
done
echo Wrote $resubmitfn

