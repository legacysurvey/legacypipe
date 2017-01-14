#!/bin/bash

# TO RUN
# ./thisscript.sh {dr4,obiwan} {mpp,status}
# OR
# nohup ./thisscript.sh dr4 status > nohup_ja.txt &
# grep Error nohup_ja.txt

therun="$1"
dowhat="$2"

if [ "$dowhat" = "mpp" ]; then
    # Create as many files as bricks having logs
    #base=/scratch2/scratchdirs/kaylanb/dr4/profiling/mpi_threads8
    base="$1"
    for dr in `find ${base}/logs/[0-9]* -maxdepth 1 -type d`;do 
        brick=`echo $(basename $dr)`
        logs=$dr/logs_$brick.txt
        find $dr/log.*|sort > $logs
        # Write file with name of each log file and its start, end time
        delta=$dr/logs_startend.txt
        rm $delta
        for fn in `cat $logs`;do 
            check=`grep "runbrick.py starting at" $fn|wc -l`
            if [ "$check" -eq 1 ]; then
                start=`grep "runbrick.py starting at" $fn|awk '{print $4}'|sed s/T/\ /g`
                end=`stat $fn|grep Change|awk '{print $2,$3}'`
                echo $fn $start $end >> $delta
            else
                echo "runbrick starting at" either 0 or > 1 times: $fn
            fi
        done
        # Difference the start,end datetime, write new file with deltas
        python job_accounting.py --dowhat time_per_brick --fn $delta   
    done
    # Sum times for bricks that had to be run again and again
    # Plot histogram of time spent on each brick
    # compute NERSC hours
    #python job_accounting.py --nersc_time 

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

elif [ "$dowhat" = "status" ]; then
    if [ "$therun" = "obiwan" ]; then
        objtype=star
        # Counts of how many finished of each objtype
        #for objtype in star qso elg lrg; do
        #    bricklist=/scratch2/scratchdirs/kaylanb/dr3-obiwan/legacypipe-dir/eboss-ngc-load-${objtype}.txt
        #    while read line; do
        #        objtype=`echo $aline|awk '{print $1}'`
        #        brick=`echo $aline|awk '{print $2}'`
        #        rowstart=`echo $aline|awk '{print $3}'`
        #    done <<< "$(cat $bricklist)"
        #done
        outdir=$DECALS_SIM_DIR
        slurmdir=$outdir/$objtype/slurms/
        slurms=${objtype}_slurm_fils.tmp
        tractors=$outdir/$objtype/
        don=${objtype}_done.tmp
        notdon=${objtype}_notdone.tmp
        logdir=${objtype}_logdir
        logs="$outdir/logs/*/log.*"
        forcedoom_fns=${objtype}_forcedoomfns.tmp
        coaddsoom_fns=${objtype}_coaddsoomfns.tmp
        fitoom_fns=${objtype}_fitoomfns.tmp
        assertoom_fns=${objtype}_assertoomfns.tmp
        noccds_fns=${objtype}_noccds.tmp
        missing_fns=${objtype}_missingfns.tmp
        noreadhdr_fns=${objtype}_noreadhdrfns.tmp
    elif [ "$therun" = "dr4" ]; then
        outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
        slurmdir=$outdir/slurms
        slurms=dr4_slurm_fils.tmp
        tractors=$outdir/tractor/
        don=dr4_bricks_done.tmp
        notdon=dr4_bricks_notdone.tmp
        logdir=dr4_logdir
        logs="$outdir/logs//log.*"
        forcedoom_fns=dr4_forcedoomfns.tmp
        coaddsoom_fns=dr4_coaddsoomfns.tmp
        fitoom_fns=dr4_fitoomfns.tmp
        assertoom_fns=dr4_assertoomfns.tmp
        noccds_fns=dr4_noccds.tmp
        missing_fns=dr4_missingfns.tmp
        noreadhdr_fns=dr4_noreadhdrfns.tmp
    else
        echo therun=$therun is not supported
    fi
   
    # Move completedd slurm-*.out files to slurmdir
    echo Moving slurms to: $slurmdir
    mkdir -p $slurmdir/oom $slurmdir/oot $slurmdir/manybrick
    if [ "$therun" = "obiwan" ]; then
        echo not moving slurm files to $slurmdir
    #else
        #find . -name "slurm-*.out" > $slurms
        #echo Moving at most this many slurm logs: `wc -l $slurms`
        #for fn in `cat $slurms`; do
        #    mem=`grep "Exceeded job memory" $fn|wc -c`
        #    val_oot=`grep "CANCELLED AT" $fn|grep "DUE TO TIME LIMIT" |wc -c`
        #    val_manybrick=`grep "slurmstepd: error:" $fn|grep "CANCELLED AT"|wc -c` 
        #    val_dr4=`grep "dr4 DONE" $fn |wc -c`
        #    val_dr4qdo=`grep "dr4-qdo DONE" $fn |wc -c`
        #    if [ "$mem" -gt 0 ]; then
        #        mv $fn $slurmdir/oom
        #    elif [ "$val_oot" -gt 0 ]; then
        #        mv $fn $slurmdir/oot/
        #    elif [ "$val_manybrick" -gt 0 ];then
        #        mv $fn $slurmdir/manybrick/
        #    elif [ "$val_dr4" -gt 0 ]; then
        #        mv $fn $slurmdir/
        #    elif [ "$val_dr4qdo" -gt 0 ]; then
        #        mv $fn $slurmdir/
        #    else
        #        echo this slurm log remains: $fn, is it running?                
        #    fi
        #done
        #echo Number of slurms:
        #echo clean exit: `ls $slurmdir/slurm*|wc -l`
        #echo OOM: `ls $slurmdir/oom/slurm*|wc -l`
        #echo OOT: `ls $slurmdir/oot/slurm*|wc -l`
        #echo many bricks ran: `ls $slurmdir/manybrick/slurmm*|wc -l`
        #for fn in `grep "echo dr4-qdo DONE" slurm-*.out |awk -F ":" '{print $1}'`;do 
        #    echo moving $fn
        #    mv $fn $slurmdir/
        #done
        #echo Found this many slurm logs: `ls $slurmdir|wc -l`
        ## Repeat b/c echo may not be in every slurm log
        #for fn in `grep "dr4-qdo DONE" slurm-*.out |awk -F ":" '{print $1}'`;do 
        #    echo moving $fn
        #    mv $fn $slurmdir/
        #done
    fi
    # Find tractor cats then just leave bricknames
    find $tractors -name "tractor-*.fits" > $don
    awk -F "/" '{print $NF}' $don |sed s/tractor-//g|sed s/.fits//g > junk
    mv junk $don
    # Reduce to just brick names
    python job_accounting.py --therun $therun --dowhat bricks_notdone --fn $don  
    echo Finished Bricks: `wc -l $don`
    echo Unfinished Bricks: `wc -l $notdon`
    # Put logfiles in single dir
    mkdir $logdir
    rm $logdir/*
    # Bricks with at least one log file
    # Get list of log files from slurms in current dir
    echo Looping over all unfinished bricks, grabbing most recent log file if exists
    if [ "$therun" = "obiwan" ]; then
        cp $outdir/logs/*/log.* $logdir/
    else
        echo WARNING: are there log files here? $outdir/logs/brickname
        #for brick in `cat $notdon`; do
        #    bri="$(echo $brick | head -c 3)"
        #    pth=$outdir/logs/$bri/$brick
        #    num=`find $pth -name "log.*"|wc -l`
        #    if [ "$num" -gt 0 ]; then
        #        fn=`ls -lrt $pth/log.*|tail -n 1|awk '{print $NF}'`
        #        cp $fn $logdir/
        #    fi
        #done
        for fn in `grep "echo Logging to:" slurm-*.out |awk '{print $NF}'`;do 
            cp $fn $logdir/
        done
        echo Bricks touched by currnet slurms: `ls $logdir/*|wc -l` 
    fi
    # Tally
    #n_don=`wc -l $don |expand|cut -d " " -f 1`
    #ball=`wc -l ${objtype}_done_all.tmp |expand|cut -d " " -f 1`
    #n_notdon=`wc -l ${objtype}_notdone.tmp |expand|cut -d " " -f 1`
    #n_fils=`ls $logdir|wc -l`
    #echo Brics finished=$n_don, not=$n_notdon
    # Errors in log files
    echo ERRORs
    rm $missing_fns $noreadhdr_fns
    for fn in `ls $logdir/log.*`; do
        forcedoom=`grep "Photometering WISE band 1" $fn -A 10|grep "Exceeded job memory limit"| wc -c`
        forcedoom2=`grep "slurmstepd: error: Exceeded job memory" $fn -A 20|grep "unwise-coadds"| wc -c`
        coaddsoom=`grep "slurmstepd: error: Exceeded job memory" $fn -B 1|grep "Stage coadds finished:"| wc -c`
        fitoom=`grep "slurmstepd: error: Exceeded job memory" $fn -B 10|grep "Got bad log-prob -inf"| wc -c`
        assertoom=`grep "slurmstepd: error: Exceeded job memory" $fn -B 1|grep "AssertionError"| wc -c`
        missing=`grep "IOError: File not found" $fn |wc -c`
        mem=`grep "Exceeded job memory limit at some point" $fn |wc -c`
        pol=`grep "ValueError: unknown record: POLNAME1" $fn |wc -c`
        tim=`grep "raise TimeoutError" $fn |wc -c`
        goteof=`grep "EOFError" $fn |wc -c`
        gotfitsio=`grep "IOError: FITSIO status" $fn |wc -c`
        notphot=`grep "No photometric CCDs touching brick" $fn |wc -c`
        runt=`grep "RuntimeError" $fn |wc -c`
        if [ "$forcedoom" -gt 0 ]; then
            echo $fn Forced Photometry OOM: appending to $forcedoom_fns
            echo $fn >> $forcedoom_fns 
        elif [ "$forcedoom2" -gt 0 ]; then
            echo $fn Forced Photometry OOM: appending to $forcedoom_fns
            echo $fn >> $forcedoom_fns 
        elif [ "$coaddsoom" -gt 0 ]; then
            echo $fn Coadds OOM: appending to $coaddsoom_fns
            echo $fn >> $coaddsoom_fns 
        elif [ "$fitoom" -gt 0 ]; then
            echo $fn Fitblobs OOM: appending to $fitoom_fns
            echo $fn >> $fitoom_fns 
        elif [ "$assertoom" -gt 0 ]; then
            echo $fn AssertionError OOM: appending to $assertoom_fns
            echo $fn >> $assertoom_fns 
        elif [ "$missing" -gt 0 ]; then
            echo $fn File not found:, appended to $missing_fns
            grep "IOError: File not found:" $fn|awk '{print $5}' >> $missing_fns 
        elif [ "$mem" -gt 0 ]
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
            test=`grep "could not interpret primary array header" $fn|wc -c`
            if [ "$test" -gt 0 ];then
                grep "could not interpret primary array header" $fn -A 1 >> $noreadhdr_fns
            fi
        elif [ "$notphot" -gt 0 ]
        then
            echo $fn No photometric CCDs touching brick
            thebrick=`grep "Command-line args" $fn|awk '{print $7}'|sed s/\'//g|sed s/\,//g`
            echo $thebrick $fn >> $noccds_fns
        elif [ "$runt" -gt 0 ]
        then
            echo $fn RuntimeError
        else 
            echo $fn no error
        fi
    done
    echo ${forcedoom_fns}: `wc -l $forcedoom_fns`
    echo ${coaddsoom_fns}: `wc -l $coaddsoom_fns`
    echo ${fitoom_fns}: `wc -l $fitoom_fns`
    echo ${assertoom_fns}: `wc -l $assertoom_fns`
    echo ${noccds_fns}: `wc -l $noccds_fns`
    echo ${missing_fns}: `wc -l $missing_fns`
    echo ${noreadhdr_fns}: `wc -l $noreadhdr_fns`
else
    echo bad argument: $dowhat
fi
  


