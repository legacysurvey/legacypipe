#!/bin/bash

#export LEGACY_SURVEY_DIR=/global/projecta/projectdirs/cosmo/work/legacysurvey/dr8/DECaLS/
export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr8new

outdir=/global/cscratch1/sd/dstn/dr8-depthcut

brick="$1"

bri=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$bri

# Shifter
cd /src/legacypipe/py

python legacyanalysis/depth-cut.py --outdir $outdir --margin 1 $brick > $outdir/logs/$bri/$brick.log 2>&1
