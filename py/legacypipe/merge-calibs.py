from __future__ import print_function
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from legacypipe.common import Decals

if __name__ == '__main__':
    decals = Decals()
    ccds = decals.get_ccds()
    print(len(ccds), 'CCDs')

    expnums = np.unique(ccds.expnum)
    print(len(expnums), 'unique exposures')

    for expnum in expnums:
        C = ccds[ccds.expnum == expnum]
        print(len(C), 'CCDs in expnum', expnum)

        psfex = []
        psfhdrvals = []

        splinesky = []
        skyhdrvals = []
        
        for ccd in C:
            im = decals.get_image_object(ccd)
            
            fn = im.splineskyfn
            T = fits_table(fn)
            splinesky.append(T)
            
            hdr = fitsio.read_header(fn)
            skyhdrvals.append([hdr[k] for k in [
                'SKY', 'LEGPIPEV', 'PLVER']])
            
            fn = im.psffn
            T = fits_table(fn)
            hdr = fitsio.read_header(fn, ext=1)
            for k in ['LOADED', 'ACCEPTED', 'CHI2', 'POLNAXIS', 'POLGRP1',
                      'POLNAME1', 'POLZERO1', 'POLSCAL1', 'POLGRP2',
                      'POLNAME2', 'POLZERO2', 'POLSCAL2', 'POLNGRP',
                      'POLDEG1', 'PSF_FWHM', 'PSF_SAMP', 'PSFNAXIS',
                      'PSFAXIS1', 'PSFAXIS2', 'PSFAXIS3']:
                T.set(k.lower(), np.array([hdr[k]]))
            psfex.append(T)

            hdr = fitsio.read_header(fn)
            psfhdrvals.append([hdr[k] for k in [
                'LEGPIPEV', 'PLVER']])

        T = merge_tables(psfex)
        T.expnum = C.expnum
        T.ccdname = C.ccdname
        T.legpipev = np.array([h[0] for h in psfhdrvals])
        T.plver    = np.array([h[1] for h in psfhdrvals])
        expnumstr = '%08i' % expnum
        fn = os.path.join('psfex', expnumstr[:5], 'decam-%s.fits' % expnumstr)
        trymakedirs(fn, dir=True)
        T.writeto(fn)
        print('Wrote', fn)

        T = merge_tables(splinesky)
        T.expnum = C.expnum
        T.ccdname = C.ccdname
        T.skyclass = np.array([h[0] for h in skyhdrvals])
        T.legpipev = np.array([h[1] for h in skyhdrvals])
        T.plver    = np.array([h[2] for h in skyhdrvals])
        expnumstr = '%08i' % expnum
        fn = os.path.join('splinesky', expnumstr[:5], 'decam-%s.fits' % expnumstr)
        trymakedirs(fn, dir=True)
        T.writeto(fn)
        print('Wrote', fn)

        
