import os
from astrometry.libkd.spherematch import *
from astrometry.util.fits import fits_table
import numpy as np
import tempfile

#  This script creates the survey-ccd-*.kd.fits kd-trees from
# survey-ccds-*.fits.gz (zeropoints) files
#

def create_kdtree(infn, outfn, ccd_cuts):

    tempdir_obj = tempfile.TemporaryDirectory(prefix='create-kdtree')
    tempdir = tempdir_obj.name

    T = fits_table(infn)
    print('Read', len(T), 'from', infn)
    if ccd_cuts:
        T.cut(T.ccd_cuts == 0)
        print('Cut to', len(T), 'on ccd_cuts')
    tfn = os.path.join(tempdir, 'ccds.fits')
    T.writeto(tfn)

    # startree
    sfn = os.path.join(tempdir, 'startree.fits')
    cmd = 'startree -i %s -o %s -P -T -k -n ccds' % (tfn, sfn)
    print(cmd)
    rtn = os.system(cmd)
    assert(rtn == 0)

    # add expnum-tree
    T = fits_table(sfn, columns=['expnum'])
    ekd = tree_build(np.atleast_2d(T.expnum.copy()).T.astype(float),
                     nleaf=60, bbox=False, split=True)
    ekd.set_name('expnum')
    efn = os.path.join(tempdir, 'ekd.fits')
    ekd.write(efn)

    # merge
    cmd = 'fitsgetext -i %s -o %s/ekd-%%02i -a -M' % (efn, tempdir)
    print(cmd)
    rtn = os.system(cmd)
    assert(rtn == 0)

    cmd = 'cat %s %s/ekd-0[123456] > %s' % (sfn, tempdir, outfn)
    rtn = os.system(cmd)
    assert(rtn == 0)


def pre_depthcut():
    indir = '/global/projecta/projectdirs/cosmo/work/legacysurvey/dr8/DECaLS/'
    outdir = '/global/cscratch1/sd/dstn/dr8new'
    bands = 'grizY'
    for band in bands:
        infn = indir + 'survey-ccds-decam-%s.fits.gz' % band
        print('Input:', infn)
        outfn = outdir + '/survey-ccds-decam-%s.kd.fits' % band
        create_kdtree(infn, outfn, True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infn', help='Input filename (CCDs file)')
    parser.add_argument('outfn', help='Output filename (survey-ccds-X.kd.fits file')
    parser.add_argument('--no-cut', dest='ccd_cuts', default=True, action='store_false')

    opt = parser.parse_args()
    create_kdtree(opt.infn, opt.outfn, opt.ccd_cuts)
