#!/bin/bash

# grep file (log) for error (name)
hasError() {
    log=$1
    name=$2
    if [ ${name} == "timeout" ]; then 
        numa=`grep "srun: Job step aborted: Waiting up to" $log -l |wc -l`
        numb=`grep "DUE TO TIME LIMIT" $log -l |wc -l`
        let num=${numa}+${numb}
    elif [ ${name} == "oom" ]; then 
        num=`grep "MemoryError" $log -l |wc -l`
    # keywords not in header
    elif [ ${name} == "isolated" ]; then 
        num=`grep "ValueError: b_isloated failed" $log -l |wc -l`
    elif [ ${name} == "nofwhm" ]; then 
        num=`grep "'unknown record: fwhm'" $log -l |wc -l`
    elif [ ${name} == "noha" ]; then 
        num=`grep "Keyword 'HA' not found" $log -l |wc -l`
    elif [ ${name} == "nora" ]; then 
        num=`grep "unknown record: RA" $log -l |wc -l`
    # Other
    elif [ ${name} == "match" ]; then 
        num=`grep "ValueError: The catalog for coordinate matching" $log -l |wc -l`
    elif [ ${name} == "index" ]; then 
        num=`grep "IndexError:" $log -l |wc -l`
    elif [ ${name} == "notString" ]; then 
        num=`grep "AttributeError: 'float' object has no attribute 'strip'" $log -l |wc -l`
    # eventually remove these
    elif [ ${name} == "a" ]; then 
        num=`grep "can't open file 'legacyccds/legacy_zeropoints.py'" $log -l |wc -l`
    #elif [ ${name} == "b" ]; then 
    #    num=`grep "line 1441, in __init__" -l |wc -l`
    fi

    if [ ${num} -gt 0 ]; then 
        return 0
    else 
        return 1
    fi
}


# Find the error in file (log), write to corresponding stdout
whichError() {
    log=$1
    echo $log
    foundIt=0
    for err in "timeout" "oom" "nofwhm" "noha" "nora" "isolated" "match" "index" "notString" "a" "b"; do 
        if hasError $log ${err}; then
            echo $log >> err_${err}.txt
            foundIt=1
            break    
        fi
    done
    if [ "${foundIt}" == "0" ]; then
        echo $log >> err_other.txt
    fi
    #if hasError $log "timeout"; then
    #    echo $log >> err_timeout.txt
    #elif hasError $log "oom"; then 
    #    echo $log >> err_oom.txt
    #elif hasError $log "notString"; then 
    #    echo $log >> err_notString.txt
    #elif hasError $log "nofwhm"; then 
    #    echo $log >> err_nofwhm.txt
    #elif hasError $log "match"; then 
    #    echo $log >> err_match.txt
    #elif hasError $log "a"; then 
    #    echo $log >> err_a.txt
    #elif hasError $log "b"; then 
    #    echo $log >> err_b.txt
    #else
    #    echo $log >> err_other.txt
    #fi
}

# Abs path to file
# $1 -- file list of /global/cscratch1/sd/kaylanb/dr5_zpts/decam/DECam_CP/CP20141227/c4d_141230_080049_ooi_g_v1.log
# Source this script
# getAbsPath log_list_name.txt
getAbsPath() {
    flist=$1
    fout=`echo ${flist}|sed s#.txt#_abs.txt#g`
    if [ -e ${fout} ]; then echo removing ${fout};rm $fout;fi
    scr=/global/cscratch1/sd/kaylanb/dr5_zpts/
    proja=/global/projecta/projectdirs/cosmo/staging/
    proj=/project/projectdirs/cosmo/staging/
    cnt=0
    for fn in `cat $flist`;do
        let cnt=${cnt}+1
        if [ $((${cnt}%100)) -eq 0 ]; then echo ${cnt}; fi
        fn_proj=`echo $fn|sed s#${scr}#${proj}#g|sed s#.log#.fits.fz#g`
        fn_proja=`echo ${fn_proj}|sed s#${proj}#${proja}#g`
        if [ -e ${fn_proj} ]; then 
            echo $fn_proj >> ${fout}
        elif [ -e ${fn_proja} ]; then
            echo $fn_proja >> ${fout}
        else
            echo DOES NOT EXIST: $fn_proj >> ${fout}
        fi
    done
    echo Wrote $fout
}

# grep file (log) for every possible error
# list of log files for failed jobs
Main() {
    loglist=$1
    # remove existing std files
    rm err_*.txt
    cnt=0
    lines=`wc -l ${loglist} |awk '{print $1}'`
    for log in `cat $loglist`;do 
        let cnt=${cnt}+1
        if [ $((${cnt}%100)) -eq 0 ]; then echo ${cnt}/${lines}; fi
        whichError $log
    done 
}
