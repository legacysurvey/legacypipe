#! /bin/bash

brick=$1

export LEGACY_SURVEY_DIR=/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/dr11/

outdir=$SCRATCH/depth-cut

mkdir -p "$outdir/logs"

python -u legacypipe/depthcut.py \
       --outdir "$outdir" \
       --margin 1 \
       --max-gb-per-band 5 \
       "$brick" \
       > "$outdir/logs/$brick.log" 2>&1

#       --dr10-propids \
