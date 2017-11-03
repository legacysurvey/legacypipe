#! /bin/bash

###
### These are post-processing commands that need to be run after the
### brick processing for a data release have finished.
###

### Assume that $LEGACY_SURVEY_DIR contains the path to the brick
### results, and that this is also where the outputs should be
### written.

### Assume this script is run from the legacypipe/py directory, ie,
### like ../bin/postprocess.sh


# 1. Brick summary: bricks-summary.fits.gz

for ((r=0; r<36; r++)); do
    R=$(printf %02d $r);
    python -u legacyanalysis/brick-summary.py -o brick-summary-$R.fits $LEGACY_SURVEY_DIR/coadd/$R*/*/*-nexp-*.fits.gz > brick-summary-$R.log 2>&1 &;
done
wait

python legacyanalysis/brick-summary.py --merge -o brick-summary.fits brick-summary-??.fits

# Make some plots
python legacyanalysis/brick-summary.py --plot brick-summary.fits

# 2. Depth summary --> to be merged into brick-summary.py


# 3. Generate sweeps    FILL ME IN!

# 4. Generate matched catalogs     FILL ME IN!

# 5. chmod ??

# 6. consolidate checksum files ??

# 7. validate checksums ??


