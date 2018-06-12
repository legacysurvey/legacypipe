#!/bin/bash

export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr7-depthcut-input/

brick="$1"

bri=$(echo $brick | head -c 3)
mkdir -p depthcuts/logs/$bri

python legacyanalysis/depth-cut.py --margin 1 $brick > depthcuts/logs/$bri/$brick.log
