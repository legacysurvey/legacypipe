#! /bin/bash

export OMP_NUM_THREADS=1

img=$1

log=cfis-dir/zpt-logs/$(echo $img | sed s/.fz//g | sed s/.fits/.log/g)
logdir=$(dirname $log)
mkdir -p $logdir
echo "Logging to $log"

export PYTHONPATH=.:${PYTHONPATH}

python -u legacyzpts/legacy_zeropoints.py \
       --camera megaprime \
       --survey-dir cfis-dir \
       --image $img \
       --sdss-photom \
       --threads 40 \
       >> $log 2>&1
