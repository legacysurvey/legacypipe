# Makes backup of original maskbit files before
# bright neighbour post-processing.
# Tractor-i files are archived, so no need to
# backup original tractor files.
#
# Martin Landriau, LBNL, July 2018

import glob
import os
import sys
import shutil

indir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr7/coadd/'
outdir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr7/original_maskbits/'

# Assumes subdirectories already exist.
subdirlist = glob.glob(indir+'*')
for subdir in subdirlist:
    base = subdir.split('/')[-1]
    bricklist = glob.glob(subdir+'/*')
    for brick in bricklist:
        brickname = brick.split('/')[-1]
        filename = 'legacysurvey-'+brickname+'-maskbits.fits.gz'
        shutil.copy(brick+'/'+filename, outdir+base+'/'+filename)
        #sys.exit()

