from __future__ import print_function
import sys
import numpy as np
import os
import fitsio

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import astropy.io.fits
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from legacypipe.survey import LegacySurveyData

'''

This script is for merging per-CCD calibration files (PsfEx,
splinesky) into larger per-exposure files.

'''


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
        s = tuple((slice(s) for s in ashape))
        p[s] = a
        padded.append(p)
    return padded

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expnum', type=str, help='Run specified exposure numbers (can be comma-separated list')
    parser.add_argument('--all-found', action='store_true', default=False, help='Only write output if all required input files are found')
    parser.add_argument('--ccds', help='Set ccds.fits file to load, default is all')
    parser.add_argument('--continue', dest='con',
                        help='Continue even if one exposure is bad',
                        action='store_true', default=False)
    parser.add_argument('--outdir', help='Output directory, default %(default)s',
                        default='calib')

    opt = parser.parse_args()

    survey = LegacySurveyData()
    if opt.ccds:
        ccds = fits_table(opt.ccds)
        ccds = survey.cleanup_ccds_table(ccds)
        survey.ccds = ccds

    if opt.expnum is not None:
        expnums = [(None, int(x, 10)) for x in opt.expnum.split(',')]
    else:
        ccds = survey.get_ccds()
        expnums = set(zip(ccds.camera, ccds.expnum))
        print(len(expnums), 'unique camera+expnums')

    for i,(camera,expnum) in enumerate(expnums):
        print()
        print('Exposure', i+1, 'of', len(expnums), ':', camera, 'expnum', expnum)
        if camera is None:
            C = survey.find_ccds(expnum=expnum)
            print(len(C), 'CCDs with expnum', expnum)
            camera = C.camera[0]
            print('Set camera to', camera)

        C = survey.find_ccds(expnum=expnum, camera=camera)
        print(len(C), 'CCDs with expnum', expnum, 'and camera', camera)

        im0 = survey.get_image_object(C[0])

        skyoutfn = im0.merged_skyfn
        psfoutfn = im0.merged_psffn

        print('Checking for', skyoutfn)
        print('Checking for', psfoutfn)
        if os.path.exists(skyoutfn) and os.path.exists(psfoutfn):
            print('Exposure', expnum, 'is done already')
            continue

        if not os.path.exists(skyoutfn):
            try:
                merge_splinesky(survey, expnum, C, skyoutfn, opt)
            except:
                if not opt.con:
                    raise
                import traceback
                traceback.print_exc()
                print('Exposure failed:', expnum, '.  Continuing...')

        if not os.path.exists(psfoutfn):
            try:
                merge_psfex(survey, expnum, C, psfoutfn, opt)
            except:
                if not opt.con:
                    raise
                import traceback
                traceback.print_exc()
                print('Exposure failed:', expnum, '.  Continuing...')


def merge_psfex(survey, expnum, C, psfoutfn, opt):
    psfex = []
    #psfhdrvals = []
    imobjs = []
    Cgood = []
    fns = []
    for ccd in C:
        im = survey.get_image_object(ccd)
        for fn in [im.psffn, im.old_single_psffn]:
            if os.path.exists(fn):
                break
        if not os.path.exists(fn):
            print('File not found:', fn)
            if opt.all_found:
                return 0
            continue
        imobjs.append(im)
        Cgood.append(ccd)
        fns.append(fn)

    for fn, ccd,im in zip(fns, Cgood, imobjs):
        print('Reading', fn)
        T = fits_table(fn)

        cols = T.get_columns()
        if not 'plver' in cols:
            from legacypipe.image import psfex_single_to_merged
            T = psfex_single_to_merged(fn, ccd.expnum, ccd.ccdname)
            for k in ['plver', 'procdate', 'plprocid']:
                T.set(k, np.array([getattr(ccd, k)]))

        #hdr = fitsio.read_header(fn, ext=1)
        # hdr = astropy.io.fits.getheader(fn, ext=1)
        # 
        # keys = ['LOADED', 'ACCEPTED', 'CHI2', 'POLNAXIS',
        #         'POLNGRP', 'PSF_FWHM', 'PSF_SAMP', 'PSFNAXIS',
        #         'PSFAXIS1', 'PSFAXIS2', 'PSFAXIS3',]
        # 
        # if hdr['POLNAXIS'] == 0:
        #     # No polynomials.  Fake it.
        #     T.polgrp1 = np.array([0])
        #     T.polgrp2 = np.array([0])
        #     T.polname1 = np.array(['fake'])
        #     T.polname2 = np.array(['fake'])
        #     T.polzero1 = np.array([0])
        #     T.polzero2 = np.array([0])
        #     T.polscal1 = np.array([1])
        #     T.polscal2 = np.array([1])
        #     T.poldeg1 = np.array([0])
        # else:
        #     keys.extend([
        #             'POLGRP1', 'POLNAME1', 'POLZERO1', 'POLSCAL1',
        #             'POLGRP2', 'POLNAME2', 'POLZERO2', 'POLSCAL2',
        #             'POLDEG1'])
        # 
        # for k in keys:
        #     try:
        #         v = hdr[k]
        #     except:
        #         print('Did not find key', k, 'in', fn)
        #         sys.exit(-1)
        #     T.set(k.lower(), np.array([hdr[k]]))
        psfex.append(T)
        #print(fn)
        #T.about()

        # #hdr = fitsio.read_header(fn)
        # hdr = astropy.io.fits.getheader(fn)
        # psfhdrvals.append([hdr.get(k,'') for k in [
        #     'LEGPIPEV', 'PLVER', 'PLPROCID', 'IMGDSUM', 'PROCDATE']] + [expnum, ccd.ccdname])

    if len(psfex) == 0:
        return
    padded = pad_arrays([p.psf_mask[0] for p in psfex])
    cols = psfex[0].columns()
    cols.remove('psf_mask')
    T = merge_tables(psfex, columns=cols)
    T.psf_mask = np.concatenate([[p] for p in padded])
    # T.legpipev = np.array([h[0] for h in psfhdrvals])
    # T.plver    = np.array([h[1] for h in psfhdrvals])
    # T.plprocid = np.array([h[2] for h in psfhdrvals])
    # T.imgdsum  = np.array([h[3] for h in psfhdrvals])
    # T.procdate = np.array([h[4] for h in psfhdrvals])
    # T.expnum   = np.array([h[5] for h in psfhdrvals])
    # T.ccdname  = np.array([h[6] for h in psfhdrvals])
    fn = psfoutfn
    trymakedirs(fn, dir=True)
    tmpfn = os.path.join(os.path.dirname(fn), 'tmp-' + os.path.basename(fn))
    T.writeto(tmpfn)
    os.rename(tmpfn, fn)
    print('Wrote', fn)
    return 1

def merge_splinesky(survey, expnum, C, skyoutfn, opt):
    skies = []
    #skyhdrvals = []
    imobjs = []
    Cgood = []
    for ccd in C:
        im = survey.get_image_object(ccd)
        fn = im.skyfn
        if not os.path.exists(fn):
            print('File not found:', fn)
            if opt.all_found:
                return 0
            continue
        imobjs.append(im)
        Cgood.append(ccd)

    for ccd,im in zip(Cgood, imobjs):
        fn = im.skyfn
        print('Reading', fn)
        T = None
        try:
            T = fits_table(fn)
        except KeyboardInterrupt:
            raise
        except:
            print('Failed to read file', fn, ':', sys.exc_info()[1])
        if T is not None:
            skies.append(T)

            # hdr = astropy.io.fits.getheader(fn)
            # s_pcts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # skyhdrvals.append([hdr.get(k, '') for k in [
            #     'SKY', 'LEGPIPEV', 'PLVER', 'PLPROCID', 'IMGDSUM', 'PROCDATE', 'SIG1',
            #     'S_MODE', 'S_MED', 'S_CMED', 'S_JOHN', 'S_FMASKD', 'S_FINE'] +
            #                    ['S_P%i' % p for p in s_pcts]] +
            #                   [expnum, ccd.ccdname])

    if len(skies) == 0:
        return
    T = fits_table()
    T.gridw = np.array([t.gridvals[0].shape[1] for t in skies])
    T.gridh = np.array([t.gridvals[0].shape[0] for t in skies])
    padded = pad_arrays([t.gridvals[0] for t in skies])
    T.gridvals = np.concatenate([[p] for p in padded])
    padded = pad_arrays([t.xgrid[0] for t in skies])
    T.xgrid = np.concatenate([[p] for p in padded])
    padded = pad_arrays([t.ygrid[0] for t in skies])
    T.ygrid = np.concatenate([[p] for p in padded])

    cols = skies[0].columns()
    #print('Columns:', cols)
    for c in ['gridvals', 'xgrid', 'ygrid', 'gridw', 'gridh']:
        cols.remove(c)

    T.add_columns_from(merge_tables(skies, columns=cols))
    # T.skyclass = np.array([h[0] for h in skyhdrvals])
    # T.legpipev = np.array([h[1] for h in skyhdrvals])
    # T.plver    = np.array([h[2] for h in skyhdrvals])
    # T.plprocid = np.array([h[3] for h in skyhdrvals])
    # T.imgdsum  = np.array([h[4] for h in skyhdrvals])
    # T.procdate = np.array([h[5] for h in skyhdrvals])
    # T.sig1     = np.array([h[6] for h in skyhdrvals]).astype(np.float32)
    # T.sky_mode = np.array([h[7] for h in skyhdrvals]).astype(np.float32)
    # T.sky_med  = np.array([h[8] for h in skyhdrvals]).astype(np.float32)
    # T.sky_cmed = np.array([h[9] for h in skyhdrvals]).astype(np.float32)
    # T.sky_john = np.array([h[10] for h in skyhdrvals]).astype(np.float32)
    # T.sky_fmasked = np.array([h[11] for h in skyhdrvals]).astype(np.float32)
    # T.sky_fine    = np.array([h[12] for h in skyhdrvals]).astype(np.float32)
    # 
    # i0 = 13
    # 
    # for i,p in enumerate(s_pcts):
    #     T.set('sky_p%i' % p, np.array([h[i0 + i] for h in skyhdrvals]).astype(np.float32))
    # 
    # i0 += len(s_pcts)
    # T.expnum   = np.array([h[i0+0] for h in skyhdrvals])
    # T.ccdname  = np.array([h[i0+1] for h in skyhdrvals])
    fn = skyoutfn
    trymakedirs(fn, dir=True)
    tmpfn = os.path.join(os.path.dirname(fn), 'tmp-' + os.path.basename(fn))
    T.writeto(tmpfn)
    os.rename(tmpfn, fn)
    print('Wrote', fn)
    return 1

if __name__ == '__main__':
    main()
