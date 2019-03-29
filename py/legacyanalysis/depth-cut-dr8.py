from astrometry.libkd.spherematch import *
from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.plotutils import *
from glob import glob
from collections import Counter

'''
DR8 Depth Cut

Start with CCDs tables / zeropoints files.
Create survey-ccd-*.kd.fits files via
  python legacypipe/create-kdtrees.py

Create $CSCRATCH/dr8new containing:
calib
images
survey-bricks.fits.gz
survey-ccds-decam-g.kd.fits
survey-ccds-decam-r.kd.fits
survey-ccds-decam-z.kd.fits

Create "depthcut" qdo queue:
LEGACY_SURVEY_DIR=$CSCRATCH/dr8new python -u legacypipe/queue-calibs.py --region dr8-decam > bricks-decam.txt

(hand-edit off the first few chatter lines)

qdo load depthcut bricks-decam.txt

Run "depth-cut.py" on each brick:
QDO_BATCH_PROFILE=cori-shifter qdo launch -v depthcut 32 --cores_per_worker 1 --walltime=30:00 --batchqueue=debug --keep_env --batchopts "--image=docker:dstndstn/legacypipe:intel" --script "/src/legacypipe/py/legacyanalysis/depthcut.sh"

Run this script.
'''

fns = glob('/global/cscratch1/sd/dstn/dr8-depthcut/*/ccds-*.fits')
# The depth-cut.py script does the temp-file-rename trick; I also ran 'fitsverify' on the
# 195,128 files (of 199703 total bricks identified by queue-calibs).
fns.sort()
print('Found', len(fns), 'depth-cut files')

bricknames = []
passedccds = set()
for fn in fns:
    C = fits_table(fn)
    if len(C) == 0:
        print('Zero length:', fn)
        continue
    n0 = len(C)
    C.cut((C.ccd_cuts == 0) * C.overlapping * C.passed_depth_cut)
    print(fn, ':', len(C), 'of', n0, 'CCDs pass')
    if len(C) == 0:
        continue

    brickname = fn[-13:-5]
    bricknames.append(brickname)

    for expnum,ccdname in zip(C.expnum, C.ccdname):
        passedccds.add((expnum, ccdname.strip()))

f = open('bricks-good.txt', 'w')
for b in bricknames:
    f.write(b + '\n')
f.close()
print('Wrote', len(bricknames), 'good bricks')

T = merge_tables([
    fits_table('/global/cscratch1/sd/dstn/dr8new/survey-ccds-decam-g.kd.fits'),
    fits_table('/global/cscratch1/sd/dstn/dr8new/survey-ccds-decam-r.kd.fits'),
    fits_table('/global/cscratch1/sd/dstn/dr8new/survey-ccds-decam-z.kd.fits')
    ])
print('Total of', len(T), 'CCDs')
print(np.sum(T.ccd_cuts == 0), 'pass CCD cuts')

print('Total of', len(passedccds), 'pass depth cut')

ccd_cuts = T.ccd_cuts
for i,(expnum,ccdname) in enumerate(zip(T.expnum, T.ccdname)):
    if not (expnum,ccdname.strip()) in passedccds:
        ccd_cuts[i] |= 0x4000
print(np.sum(T.ccd_cuts == 0), 'pass CCD cuts + depth cut')

T.writeto('survey-ccds-depthcut.fits')
