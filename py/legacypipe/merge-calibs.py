from __future__ import print_function
import numpy as np
import os
import fitsio

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

        expnumstr = '%08i' % expnum
        skyoutfn = os.path.join('splinesky', expnumstr[:5], 'decam-%s.fits' % expnumstr)
        psfoutfn = os.path.join('psfex', expnumstr[:5], 'decam-%s.fits' % expnumstr)

        if os.path.exists(skyoutfn) and os.path.exists(psfoutfn):
            print('Exposure', expnum, 'is done already')
            continue

        C = ccds[ccds.expnum == expnum]
        print(len(C), 'CCDs in expnum', expnum)

        psfex = []
        psfhdrvals = []

        splinesky = []
        skyhdrvals = []

        for ccd in C:
            im = decals.get_image_object(ccd)

            fn = im.splineskyfn
            if os.path.exists(fn):
                T = fits_table(fn)
                splinesky.append(T)
                # print(fn)
                # T.about()
                hdr = fitsio.read_header(fn)
                skyhdrvals.append([hdr[k] for k in [
                            'SKY', 'LEGPIPEV', 'PLVER']] + [expnum, ccd.ccdname])
            else:
                print('File not found:', fn)

            fn = im.psffn
            if os.path.exists(fn):
                T = fits_table(fn)
                hdr = fitsio.read_header(fn, ext=1)
                for k in ['LOADED', 'ACCEPTED', 'CHI2', 'POLNAXIS', 'POLGRP1',
                          'POLNAME1', 'POLZERO1', 'POLSCAL1', 'POLGRP2',
                          'POLNAME2', 'POLZERO2', 'POLSCAL2', 'POLNGRP',
                          'POLDEG1', 'PSF_FWHM', 'PSF_SAMP', 'PSFNAXIS',
                          'PSFAXIS1', 'PSFAXIS2', 'PSFAXIS3']:
                    T.set(k.lower(), np.array([hdr[k]]))
                psfex.append(T)
                #print(fn)
                #T.about()
    
                hdr = fitsio.read_header(fn)
                psfhdrvals.append([hdr[k] for k in [
                    'LEGPIPEV', 'PLVER']] + [expnum, ccd.ccdname])
            else:
                print('File not found:', fn)

        if len(psfex):
            T = merge_tables(psfex)
            T.legpipev = np.array([h[0] for h in psfhdrvals])
            T.plver    = np.array([h[1] for h in psfhdrvals])
            T.expnum   = np.array([h[2] for h in psfhdrvals])
            T.ccdname  = np.array([h[3] for h in psfhdrvals])
            fn = psfoutfn
            trymakedirs(fn, dir=True)
            T.writeto(fn)
            print('Wrote', fn)

        if len(splinesky):
            gridh,gridw = 0,0
            for t in splinesky:
                h,w = t.gridvals[0].shape
                gridh = max(gridh, h)
                gridw = max(gridw, w)
            print('Max grid size', gridh,gridw)
            for t in splinesky:
                h,w = t.gridvals[0].shape
                t.gridw = np.array([w])
                t.gridh = np.array([h])
                if h == gridh and w == gridw:
                    continue
                val = t.gridvals[0]
                gv = np.zeros((gridh,gridw), val.dtype)
                gv[:h,:w] = val
                t.gridvals = np.array([gv])
                print('Resized gridvals from', (h,w), 'to', (gridh,gridw))
                val = t.ygrid[0]
                yg = np.zeros(gridh, val.dtype)
                yg[:h] = val
                t.ygrid = np.array([yg])
                val = t.xgrid[0]
                xg = np.zeros(gridw, val.dtype)
                xg[:w] = val
                t.xgrid = np.array([xg])
                #print('xgrid', t.xgrid[0].shape)
                #print('ygrid', t.ygrid[0].shape)
    
            T = merge_tables(splinesky)
            T.skyclass = np.array([h[0] for h in skyhdrvals])
            T.legpipev = np.array([h[1] for h in skyhdrvals])
            T.plver    = np.array([h[2] for h in skyhdrvals])
            T.expnum   = np.array([h[3] for h in skyhdrvals])
            T.ccdname  = np.array([h[4] for h in skyhdrvals])
            fn = skyoutfn
            trymakedirs(fn, dir=True)
            T.writeto(fn)
            print('Wrote', fn)

        
