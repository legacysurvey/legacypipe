from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import tree_build
import numpy as np
import os
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infn', help='Input filename (eg, skyscales_ccds.fits)')
    parser.add_argument('outfn', help='Output filename (eg, sky-scales.kd.fits)')
    opt = parser.parse_args()
    infn = opt.infn
    outfn = opt.outfn
    S = fits_table(infn)
    ekd = tree_build(np.atleast_2d(S.expnum.copy()).T.astype(float),
                     nleaf=60, bbox=False, split=True)
    ekd.set_name('expnum')
    ekd.write('ekd.fits')
    cmd = 'fitsgetext -i ekd.fits -o ekd-%02i -a -M'
    rtn = os.system(cmd)
    assert(rtn == 0)
    cmd = 'cat %s ekd-0[1-6] > %s' % (infn, outfn)
    rtn = os.system(cmd)
    assert(rtn == 0)
    sys.exit(0)
