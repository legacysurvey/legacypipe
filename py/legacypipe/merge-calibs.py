from __future__ import print_function
import numpy as np
import os
import fitsio

import matplotlib
matplotlib.use('Agg')

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from legacypipe.common import Decals

def pad_arrays(A):
    '''
    Given a list of numpy arrays [a0,a1,...], zero-pads them all to be the same shape.
    '''
    maxshape = None
    for a in A:
        ashape = np.array(a.shape)
        if maxshape is None:
            maxshape = ashape
        else:
            maxshape = np.maximum(maxshape, ashape)

    padded = []
    for a in A:
        ashape = np.array(a.shape)
        if np.all(ashape == maxshape):
            padded.append(a)
            continue
        p = np.zeros(maxshape, a.dtype)
        s = list((slice(s) for s in ashape))
        p[s] = a
        padded.append(p)
    return padded

def main():
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

                keys = ['LOADED', 'ACCEPTED', 'CHI2', 'POLNAXIS', 
                        'POLNGRP', 'PSF_FWHM', 'PSF_SAMP', 'PSFNAXIS',
                        'PSFAXIS1', 'PSFAXIS2', 'PSFAXIS3',]

                if hdr['POLNAXIS'] == 0:
                    # No polynomials.  Fake it.
                    T.polgrp1 = np.array([0])
                    T.polgrp2 = np.array([0])
                    T.polname1 = np.array(['fake'])
                    T.polname2 = np.array(['fake'])
                    T.polzero1 = np.array([0])
                    T.polzero2 = np.array([0])
                    T.polscal1 = np.array([1])
                    T.polscal2 = np.array([1])
                    T.poldeg1 = np.array([0])
                    T.poldeg2 = np.array([0])
                else:
                    keys.extend([
                            'POLGRP1', 'POLNAME1', 'POLZERO1', 'POLSCAL1',
                            'POLGRP2', 'POLNAME2', 'POLZERO2', 'POLSCAL2',
                            'POLDEG1'])

                for k in keys:
                    T.set(k.lower(), np.array([hdr[k]]))
                psfex.append(T)
                #print(fn)
                #T.about()
    
                hdr = fitsio.read_header(fn)
                psfhdrvals.append([hdr.get(k,'') for k in [
                    'LEGPIPEV', 'PLVER']] + [expnum, ccd.ccdname])
            else:
                print('File not found:', fn)

        if len(psfex):
            padded = pad_arrays([p.psf_mask[0] for p in psfex])
            cols = psfex[0].columns()
            cols.remove('psf_mask')
            T = merge_tables(psfex, columns=cols)
            T.psf_mask = np.concatenate([[p] for p in padded])
            T.legpipev = np.array([h[0] for h in psfhdrvals])
            T.plver    = np.array([h[1] for h in psfhdrvals])
            T.expnum   = np.array([h[2] for h in psfhdrvals])
            T.ccdname  = np.array([h[3] for h in psfhdrvals])
            fn = psfoutfn
            trymakedirs(fn, dir=True)
            T.writeto(fn)
            print('Wrote', fn)

        if len(splinesky):
            T = fits_table()
            T.gridw = np.array([t.gridvals[0].shape[1] for t in splinesky])
            T.gridh = np.array([t.gridvals[0].shape[0] for t in splinesky])

            padded = pad_arrays([t.gridvals[0] for t in splinesky])
            T.gridvals = np.concatenate([[p] for p in padded])
            padded = pad_arrays([t.xgrid[0] for t in splinesky])
            T.xgrid = np.concatenate([[p] for p in padded])
            padded = pad_arrays([t.xgrid[0] for t in splinesky])
            T.ygrid = np.concatenate([[p] for p in padded])
    
            cols = splinesky[0].columns()
            print('Columns:', cols)
            for c in ['gridvals', 'xgrid', 'ygrid']:
                cols.remove(c)

            T.add_columns_from(merge_tables(splinesky, columns=cols))
            T.skyclass = np.array([h[0] for h in skyhdrvals])
            T.legpipev = np.array([h[1] for h in skyhdrvals])
            T.plver    = np.array([h[2] for h in skyhdrvals])
            T.expnum   = np.array([h[3] for h in skyhdrvals])
            T.ccdname  = np.array([h[4] for h in skyhdrvals])
            fn = skyoutfn
            trymakedirs(fn, dir=True)
            T.writeto(fn)
            print('Wrote', fn)

        
if __name__ == '__main__':
    main()
