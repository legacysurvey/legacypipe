#!/bin/bash

# grep file (log) for error (name)
hasError() {
    log=$1
    name=$2
    if [ ${name} == "timeout" ]; then 
        num=`grep "srun: Job step aborted: Waiting up to" $log -l |wc -l`
    elif [ ${name} == "oom" ]; then 
        num=`grep "MemoryError" $log -l |wc -l`
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
    if hasError $log "timeout"; then
        echo $log >> err_timeout.txt
    elif hasError $log "oom"; then 
        echo $log >> err_oom.txt
    else
        echo $log >> err_other.txt
    fi
}

# grep file (log) for every possible error
# list of log files for failed jobs
loglist=$1
# remove existing std files
rm err_*.txt
for log in `cat $loglist`;do 
    whichError $log
done 
