import os
from astrometry.libkd.spherematch import *
from astrometry.util.fits import fits_table
import numpy as np

#  This script creates the survey-ccd-*.kd.fits kd-trees from
# survey-ccds-*.fits.gz (zeropoints) files
#

indir = '/global/projecta/projectdirs/cosmo/work/legacysurvey/dr8/DECaLS/'
outdir = '/global/cscratch1/sd/dstn/dr8new'

bands = 'grizY'

for band in bands:
    infn = indir + 'survey-ccds-%s.fits.gz' % band
    print('Input:', infn)

    # gunzip
    tfn = '/tmp/survey-ccd-%s.fits' % band
    cmd = 'gunzip -cd %s > %s' % (infn, tfn)
    print(cmd)
    os.system(cmd)

    # startree
    sfn = '/tmp/startree-%s.fits' % band
    cmd = 'startree -i %s -o %s -P -T -k -n ccds' % (tfn, sfn)
    print(cmd)
    os.systemd(cmd)

    # add expnum-tree
    T = fits_table(sfn, columns=['expnum'])
    ekd = tree_build(np.atleast_2d(T.expnum.copy()).T.astype(float),
                     nleaf=60, bbox=False, split=True)
    ekd.set_name('expnum')
    efn = '/tmp/ekd-%s.fits' % band
    ekd.write(efn)

    # merge
    cmd = 'fitsgetext -i %s -o /tmp/ekd-%s-%%02i -a -M' % (efn, band)
    print(cmd)
    os.system(cmd)

    outfn = outdir + '/survey-ccds-%s.kd.fits' % band
    
    cmd = 'cat %s /tmp/ekd-%s-0[123456] > %s' % (sfn, band, outfn)
    os.system(cmd)
