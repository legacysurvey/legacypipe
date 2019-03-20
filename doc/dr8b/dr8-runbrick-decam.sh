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

dr=dr8b
camera=decam
export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/${dr}/runbrick-${camera}
source $LEGACY_SURVEY_DIR/dr8-env.sh
CODE_DIR=$LEGACY_SURVEY_DIR/code

# Use local check-outs of legacypipe and legacyzpts.
export LEGACYPIPE_DIR=$CODE_DIR/legacypipe
export LEGACYZPTS_DIR=$CODE_DIR/legacyzpts
export PYTHONPATH=$PYTHONPATH:$LEGACYPIPE_DIR/py
export PYTHONPATH=$PYTHONPATH:$LEGACYZPTS_DIR/py

if [ $camera == "decam" ]; then
  release=8000
else
  release=8001
fi    

brick="$1"
echo 'Working on brick '$brick

if [ x$DW_PERSISTENT_STRIPED_DR8 == x ]; then
  if [ "$NERSC_HOST" = "edison" ]; then
    drdir=$SCRATCH/$dr
  else
    drdir=$CSCRATCH/$dr
  fi  
  # For writing to project, if necessary.
  drdir=/global/project/projectdirs/cosmo/work/legacysurvey/${dr}
else
  drdir=${DW_PERSISTENT_STRIPED_DR8}${dr}
fi
outdir=${drdir}/runbrick-${camera}
echo 'Writing output to '$outdir

bri=$(echo $brick | head -c 3)
logdir=${drdir}/logs-runbrick-${camera}/${bri}
log=${logdir}/${brick}.log
mkdir -p $logdir

echo Logging to: $log

# Limit memory usage.
ncores=8
if [ "$NERSC_HOST" = "edison" ]; then
    # 64 GB / Edison node = 67108864 kbytes
    maxmem=67108864
    let usemem=${maxmem}*${ncores}/24
else
    # 128 GB / Cori Haswell node = 134217728 kbytes
    maxmem=134217728
    let usemem=${maxmem}*${ncores}/32
fi
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
     --release 8000 \
     --depth-cut 1.0 \
     --checkpoint $outdir/checkpoints/${bri}/checkpoint-${brick}.pickle \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
      >> $log 2>&1
