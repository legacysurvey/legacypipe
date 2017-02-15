#!/bin/bash

# TO RUN
# ../bin/job_accounting.sh /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/logs/2017_02_10
# e.g. ../bin/job_accounting.sh /abs/path/to/logdir

logdir="$1"
slurmdir=${logdir}/slurms
err_dir=${logdir}/errors
mkdir -p $slurmdir $err_dir

if [ "$NERSC_HOST" == "edison" ]; then
    outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
else
    outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
fi
tractors=$outdir/tractor/
don_tractor=dr4_tractors_done.tmp
don=dr4_bricks_done.tmp
notdon=dr4_bricks_notdone.tmp
logs="$logdir/log.*"

# Move slurms to logdir/slurms/
echo moving slurms to $slurmdir
for jobid in `find ${logdir}/log.*|awk -F "_" '{print $NF}'`;do 
    mv slurm-${jobid}.out $slurmdir/
done

# Finished tractor cats
find $tractors -name "tractor-*.fits" > $don_tractor
awk -F "/" '{print $NF}' ${don_tractor} |sed s/tractor-//g|sed s/.fits//g > $don
echo Finished Bricks: `wc -l $don`
# Some tractor cats do not have "wise_flux", why?
# job_accounting.py lists these cats in "sanity_tractors_nowise.txt"
# Put the log file for each tractor cat in "sanity_tractors_nowise_logs.txt"
#tractor_nowise=sanity_tractors_nowise.txt
#tractor_nowise_logs=sanity_tractors_nowise_logs.txt
#if [ -e "${tractor_nowise}" ];then
#    # this cmd goes to where the log files exist, and gets the latest log file written
#    echo Getting logs for Tractor Cats w/out "wise_flux"
#    for fn in `cat ${tractor_nowise} |uniq`;do 
#        brick=`echo $fn|awk -F "-" '{print $NF}'|sed s/.fits//g`
#        bri=`echo $brick|head -c 3`
#        echo $brick $bri
#        ls -lrt `find /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/logs/${bri}/log.${brick}*`|tail -n 1|awk '{print $NF}' >> ${tractor_nowise_logs}
#    done
#fi

# Errors in log files
echo Gathering log errors
for fn in `find ${logdir}/log.*`; do
    bad_wshot=`grep "IOError: extension not found" $fn|wc -c`
    asserterr=`grep "AssertionError" $fn|wc -c`
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
    calib=`grep "RuntimeError: Command failed: modhead" $fn |wc -c`
    # Append Log name for these errors
    if [ "$bad_wshot" -gt 0 ]; then
        echo $fn bad wshot
        echo $fn >> ${err_dir}/badwshot.txt 
    elif [ "$asserterr" -gt 0 ]; then
        echo $fn AsssertionError
        echo $fn >> ${err_dir}/asserterr.txt 
    elif [ "$forcedoom" -gt 0 ]; then
        echo $fn Forced Photometry OOM
        echo $fn >> ${err_dir}/forcedoom.txt 
    elif [ "$forcedoom2" -gt 0 ]; then
        echo $fn Forced Photometry OOM
        echo $fn >> ${err_dir}/forcedoom.txt 
    elif [ "$coaddsoom" -gt 0 ]; then
        echo $fn Coadds OOM
        echo $fn >> ${err_dir}/coaddsoom.txt 
    elif [ "$fitoom" -gt 0 ]; then
        echo $fn Fitblobs OOM
        echo $fn >> ${err_dir}/fitblobsoom.txt 
    elif [ "$assertoom" -gt 0 ]; then
        echo $fn AssertionError OOM
        echo $fn >> ${err_dir}/assertoom.txt 
    elif [ "$missing" -gt 0 ]; then
        echo $fn FILE NOT FOUND
        echo $fn >> ${err_dir}/filenotfound.txt 
    elif [ "$mem" -gt 0 ]; then
        echo $fn Exceeded job memory limit at some point
        echo $fn >> ${err_dir}/generaloom.txt 
    elif [ "$pol" -gt 0 ]; then
        echo $fn ValueError: unknown record: POLNAME1
        echo $fn >> ${err_dir}/polname1.txt 
    elif [ "$tim" -gt 0 ]; then
        echo $fn raise TimeoutError
        echo $fn >> ${err_dir}/timeout.txt 
    elif [ "$goteof" -gt 0 ]; then
        echo $fn EOFError
        echo $fn >> ${err_dir}/eoferror.txt 
    elif [ "$gotfitsio" -gt 0 ]; then
        echo $fn IOError: FITSIO status
        test=`grep "could not interpret primary array header" $fn|wc -c`
        if [ "$test" -gt 0 ];then
            echo $fn >> ${err_dir}/nohdr.txt
        fi
    elif [ "$notphot" -gt 0 ]; then
        echo $fn No photometric CCDs touching brick
        #thebrick=`grep "Command-line args" $fn|awk '{print $7}'|sed s/\'//g|sed s/\,//g`
        echo $fn >> ${err_dir}/noccds.txt
    elif [ "$runt" -gt 0 ]; then
        echo $fn RuntimeError
        echo $fn >> ${err_dir}/runtimeerror.txt
    else 
        echo $fn Unknown
        echo $fn >> ${err_dir}/unknown.txt
    #else 
    #    echo $fn no error
    #    echo $fn >> ${err_dir}/noerror.txt
    fi
done
# Report stats
wc -l ${err_dir}/*.txt 
  


