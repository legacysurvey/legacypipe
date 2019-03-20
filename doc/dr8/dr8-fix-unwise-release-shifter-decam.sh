#! /bin/bash
# Run legacypipe/runbrick.py on a single brick with (optionally) the burst
# buffer within a Shifter container at NERSC.

# Variables to be sure are correct: dr, camera, release, LEGACY_SURVEY_DIR, and
# CODE_DIR.  Also note that LEGACY_SURVEY_DIR has to be defined here, *before*
# the env script is sourced.

dr=dr8b
camera=decam
export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/${dr}/runbrick-${camera}
source $LEGACY_SURVEY_DIR/dr8-env-shifter.sh
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
echo >> $log
echo "Environment:" >> $log
set | grep -v PASS >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log

time python ${LEGACYPIPE_DIR}/py/legacypipe/runbrick.py \
     --brick ${brick} --outdir $outdir \
     --skip-calibs --threads ${ncores} \
     --no-write \
     -f wise_forced -f writecat \
     --release ${release} \
     --pickle "${outdir}/pickles/${bri}/runbrick-%(brick)s-%%(stage)s.pickle" \
      >> $log 2>&1
