#! /bin/bash

# Run legacypipe/runbrick.py on a single brick with the burst buffer.  Assumes
# the file dr8-env.sh exists in the same path as this script is launched.

# To load and launch:
#   qdo load dr8a ./brick-lists/test-HSC-NGC
#   qdo launch dr8a 256 --cores_per_worker 8 --walltime=00:30:00 --script ./dr8-runbrick.sh --batchqueue debug --keep_env --batchopts "--bbf=bb.conf"

# Useful commands:
#   qdo status dr8a
#   qdo retry dr8a
#   qdo recover dr8a --dead
#   qdo tasks dr8a --state=Failed

brick="$1"
echo 'Working on brick '$brick

source dr8-env-decam.sh

#drdir=${DW_PERSISTENT_STRIPED_DR8}dr8b
drdir=/global/project/projectdirs/cosmo/work/legacysurvey/dr8b
outdir=$drdir/runbrick-decam

bri=$(echo $brick | head -c 3)
logdir=$drdir/logs-runbrick-decam/$bri
log=$logdir/$brick.log
mkdir -p $logdir

echo Logging to: $log

# Limit memory usage.
ncores=8 # 128 GB / Cori Haswell node = 134217728 kbytes
maxmem=134217728
let usemem=${maxmem}*${ncores}/32
ulimit -Sv $usemem

echo -e "\n\n\n" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log

time python $LEGACYPIPE_DIR/py/legacypipe/runbrick.py \
     --brick $brick --outdir $outdir \
     --skip --skip-calibs --threads ${ncores} \
     --depth-cut 1.0 \
     --checkpoint $outdir/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
      >> $log 2>&1
