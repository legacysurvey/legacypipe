#!/bin/bash


# Time spent on each brick
all=logs.txt
echo Appending all logs to: $all 
find /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/logs/*/*/log.* > $all
tim=logs_time.txt
rm $logs_time.txt
echo Looping through: $all 
for fn in `cat $all`;do 
    check=`grep "runbrick.py starting at" $fn|wc -l`
    if [ "$check" -eq 1 ]; then
        start=`grep "runbrick.py starting at" $fn|awk '{print $4}'|sed s/T/\ /g`
        end=`stat $fn|grep Change|awk '{print $2,$3}'`
        echo $fn $start $end >> $tim
    else
        echo "runbrick starting at" either 0 or > 1 times: $fn
    fi
done
out=bricks_time.txt
rm $out
# Get time spent on each log file, identified as each brick
python job_accounting.py --time_per_brick 
# Sum times for bricks that had to be run again and again
# Plot histogram of time spent on each brick
# compute NERSC hours
python job_accounting.py --nersc_time 

# Any multinode slurm file, why did each node stop?
#slurm=slurm-3198976.out
#log=logfil
#rm $log 
#for suffix in fin tim exis eof asser ext other;do
#    rm ${log}_$suffix
#done
#grep "echo Logging to" $slurm |cut -d " " -f 5 > $log
#echo Looping over $log
#i=0
#for fn in `cat $log`;do
#    let i=$i+1
#    echo $i
#    cnt=`grep "Stage writecat finished:" $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_fin
#        continue
#    fi
#    cnt=`grep "TimeoutError" $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_tim
#        continue
#    fi
#    cnt=`grep Read\ .*sources\ from $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_exis
#        continue
#    fi
#    cnt=`grep EOFError $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_eof
#        continue
#    fi
#    cnt=`grep AssertionError $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_asser
#        continue
#    fi
#    cnt=`grep "IOError: extension not found" $fn |wc -l`
#    if [ "$cnt" -gt 0 ]; then 
#        echo $fn >> ${log}_ext
#        continue
#    fi
#    # If got here, other
#    let cnt_oth=$cnt_oth+1
#    echo $fn >> ${log}_other
#done
#echo finished: $(wc -l ${log}_fin |cut -d " " -f 1)
#echo timeout : $(wc -l ${log}_tim |cut -d " " -f 1)
#echo exist   : $(wc -l ${log}_exis |cut -d " " -f 1)
#echo eof pick: $(wc -l ${log}_eof |cut -d " " -f 1)
#echo asser   : $(wc -l ${log}_asser |cut -d " " -f 1)
#echo io ext  : $(wc -l ${log}_ext |cut -d " " -f 1)
#echo other   : $(wc -l ${log}_other |cut -d " " -f 1)
   


#export run_name=dr4-qdo
#export outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
#
## Bricks finished and not
#for fn in `find $outdir/tractor -name "tractor-*.fits"`;do echo $(basename $fn)|sed 's/tractor-//g'|sed 's/.fits//g';done > bricks_done.tmp
#python dr4-qdo-status.py --get_bricks_notdone
## log files for those not finished
#tmp_dir=temp_$camera
#if [ -e "tmp_dir" ]
#then
#    rm $tmp_dir/*
#    rmdir $tmp_dir
#fi
#mkdir $tmp_dir
#for brick in `cat bricks_notdone_$camera.tmp`
#do
#    bri="$(echo $brick | head -c 3)"
#    log=`ls -lrt $outdir/logs/$bri/$brick/log.*|tail -n 1|expand|awk '{print $9}'`
#    cp $log $tmp_dir/
#done
## Finish
#ball=`wc -l bricks_all_$camera.tmp |expand|cut -d " " -f 1`
#don=`wc -l bricks_done_$camera.tmp |expand|cut -d " " -f 1`
#notdon=`wc -l bricks_notdone_$camera.tmp |expand|cut -d " " -f 1`
#nfils=`ls $tmp_dir|wc -l`
#echo Bricks total=$ball, finished=$don, not=$notdon
#echo Latest stdout files: $tmp_dir/, there are: $nfils
#
## List errors in stdout files
#echo ERRORs
## file to write bricks to resubmit
#resubmitfn=bricks_easyresubmit_$camera.txt
#if [ -e "$resubmitfn" ]
#then 
#    rm $resubmitfn
#fi
## List errors
#for fn in `ls $tmp_dir/log.*`
#do
#    mem=`grep "Exceeded job memory limit at some point" $fn |wc -c`
#    pol=`grep "ValueError: unknown record: POLNAME1" $fn |wc -c`
#    tim=`grep "raise TimeoutError" $fn |wc -c`
#    goteof=`grep "EOFError" $fn |wc -c`
#    gotfitsio=`grep "IOError: FITSIO status" $fn |wc -c`
#    notphot=`grep "No photometric CCDs touching brick" $fn |wc -c`
#    runt=`grep "RuntimeError" $fn |wc -c`
#    if [ "$mem" -gt 0 ]
#    then
#        echo $fn Exceeded job memory limit at some point
#    elif [ "$pol" -gt 0 ]
#    then
#        echo $fn ValueError: unknown record: POLNAME1
#    elif [ "$tim" -gt 0 ]
#    then
#        echo $fn raise TimeoutError
#    elif [ "$goteof" -gt 0 ]
#    then
#        echo $fn EOFError
#    elif [ "$gotfitsio" -gt 0 ]
#    then
#        echo $fn IOError: FITSIO status
#    elif [ "$notphot" -gt 0 ]
#    then
#        echo $fn No photometric CCDs touching brick
#    elif [ "$runt" -gt 0 ]
#    then
#        echo $fn RuntimeError
#    else 
#        echo $fn no error
#    fi
#
#    # For easy to fix errors write bricknames to txt file
#    if [ "$mem" -gt 0 ] || [ "$tim" -gt 0 ]
#    then
#        brick=`awk '/Command-line args:/ { print $7 }' $fn| sed "s/\,//g"|sed "s/'//g"`
#        echo $brick >> $resubmitfn
#    fi
#done
#echo Wrote $resubmitfn
#


