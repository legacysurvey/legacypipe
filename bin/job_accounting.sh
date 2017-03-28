#!/bin/bash

# TO RUN
# ../bin/job_accounting.sh /scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/logs/2017_02_10
# e.g. ../bin/job_accounting.sh /abs/path/to/logdir

logdir="$1"
slurmdir=${logdir}/slurms
err_dir=${logdir}/errors
mkdir -p $slurmdir $err_dir

if [ "$NERSC_HOST" == "edison" ]; then
    #outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4
    outdir=/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4_fixes
else
    #outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4
    outdir=/global/cscratch1/sd/desiproc/dr4/data_release/dr4_fixes
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
    # primary catching
    fin=`grep "Stage writecat finished" $fn|wc -c`
    mem=`grep MemoryError $fn|wc -c`
    mem2=`grep "exceeded memory limit" $fn|wc -c` 
    psf=`grep "RuntimeError: Command failed: psfex" $fn |wc -c`
    psf2=`grep "RuntimeError: Command failed: modhead" $fn |wc -c`
    noccd=`grep "No CCDs touching brick" $fn |wc -c`
    noccd2=`grep "No photometric CCDs touching brick" $fn |wc -c`
    # other things 
    asserterr=`grep "AssertionError" $fn|wc -c`
    missing=`grep "IOError: File not found" $fn |wc -c`
    pol=`grep "ValueError: unknown record: POLNAME1" $fn |wc -c`
    timeout=`grep "raise TimeoutError" $fn |wc -c`
    goteof=`grep "EOFError" $fn |wc -c`
    gotfitsio=`grep "IOError: FITSIO status" $fn |wc -c`
    runt=`grep "RuntimeError" $fn |wc -c`
    if [ "$fin" -gt 0 ]; then
        echo $fn >> ${err_dir}/fin.txt 
    elif [ "$mem" -gt 0 ]; then
        echo $fn >> ${err_dir}/mem.txt 
    elif [ "$mem2" -gt 0 ]; then
        echo $fn >> ${err_dir}/mem2.txt 
    elif [ "$psf" -gt 0 ]; then
        echo $fn >> ${err_dir}/psf.txt 
    elif [ "$psf2" -gt 0 ]; then
        echo $fn >> ${err_dir}/psf2.txt 
    elif [ "$noccd" -gt 0 ]; then
        echo $fn >> ${err_dir}/noccd.txt 
    elif [ "$noccd2" -gt 0 ]; then
        echo $fn >> ${err_dir}/noccd2.txt 
    elif [ "$asserterr" -gt 0 ]; then
        echo $fn >> ${err_dir}/asserterr.txt 
    elif [ "$missing" -gt 0 ]; then
        echo $fn >> ${err_dir}/missing.txt 
    elif [ "$pol" -gt 0 ]; then
        echo $fn >> ${err_dir}/pol.txt 
    elif [ "$timeout" -gt 0 ]; then
        echo $fn >> ${err_dir}/timeout.txt 
    elif [ "$noccd2" -gt 0 ]; then
        echo $goteof >> ${err_dir}/goteof.txt 
    elif [ "$gotfitsio" -gt 0 ]; then
        echo $fn >> ${err_dir}/gotfitsio.txt 
    elif [ "$runt" -gt 0 ]; then
        echo $fn >> ${err_dir}/runt.txt 
    else
        echo $fn >> ${err_dir}/other.txt 
    fi
done
# Report stats
wc -l ${err_dir}/*.txt 
  


