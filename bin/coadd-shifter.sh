#! /bin/bash

## This script is meant to be called as a "qdo --script" script that collects memory info
## then calls shifter to run the actual processing script.

brick=$1

outdir=/global/cscratch1/sd/dstn/dr10-early-coadds
bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri
log="$outdir/logs/$bri/$brick.log"

shifter --image docker:legacysurvey/legacypipe:DR9.9.1 /global/homes/d/dstn/legacypipe/bin/coadd.sh "$@"
status=$?

echo "max_memory $(cat /sys/fs/cgroup/memory/slurm/uid_$SLURM_JOB_UID/job_$SLURM_JOB_ID/memory.max_usage_in_bytes)" >> $log

for x in /sys/fs/cgroup/memory/slurm/uid_$SLURM_JOB_UID/job_$SLURM_JOB_ID/*/memory.max_usage_in_bytes; do
    echo "max_memory $x $(cat $x)" >> $log;
done

exit $status
