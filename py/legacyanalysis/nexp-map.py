from __future__ import print_function
import os
import fitsio
import numpy as np
from glob import glob
from collections import Counter

import matplotlib
#matplotlib.use('Agg')
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('font', family='serif')
import pylab as plt

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.util.miscutils import clip_polygon
from astrometry.util.multiproc import multiproc

from tractor import NanoMaggies
from legacypipe.survey import LegacySurveyData, wcs_for_brick

def one_brick(X):
    (ibrick, brick) = X
    bands = ['g','r','z']

    print('Brick', brick.brickname)
    wcs = wcs_for_brick(brick, W=94, H=94, pixscale=10.)
    BH,BW = wcs.shape
    targetrd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(BW,1),(BW,BH),(1,BH),(1,1)]])
    survey = LegacySurveyData()
    C = survey.ccds_touching_wcs(wcs)
    if C is None:
        print('No CCDs touching brick')
        return None
    I = np.flatnonzero(C.ccd_cuts == 0)
    if len(I) == 0:
        print('No good CCDs touching brick')
        return None
    C.cut(I)
    print(len(C), 'CCDs touching brick')

    depths = {}
    for band in bands:
        d = np.zeros((BH,BW), np.float32)
        depths[band] = d

    npix = dict([(band,0) for band in bands])
    nexps = dict([(band,0) for band in bands])

    # survey.get_approx_wcs(ccd)
    for ccd in C:
        #im = survey.get_image_object(ccd)
        awcs = survey.get_approx_wcs(ccd)
        
        imh,imw = ccd.height,ccd.width
        x0,y0 = 0,0
        x1 = x0 + imw
        y1 = y0 + imh
        imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
        ok,tx,ty = awcs.radec2pixelxy(targetrd[:-1,0], targetrd[:-1,1])
        tpoly = list(zip(tx,ty))
        clip = clip_polygon(imgpoly, tpoly)
        clip = np.array(clip)
        if len(clip) == 0:
            continue
        x0,y0 = np.floor(clip.min(axis=0)).astype(int)
        x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
        #slc = slice(y0,y1+1), slice(x0,x1+1)
        awcs = awcs.get_subimage(x0, y0, x1-x0, y1-y0)
        ah,aw = awcs.shape

        #print('Image', ccd.expnum, ccd.ccdname, ccd.filter, 'overlap', x0,x1, y0,y1, '->', (1+x1-x0),'x',(1+y1-y0))

        # Find bbox in brick space
        r,d = awcs.pixelxy2radec([1,1,aw,aw], [1,ah,ah,1])
        ok,bx,by = wcs.radec2pixelxy(r,d)
        bx0 = np.clip(np.round(bx.min()).astype(int) -1, 0, BW-1)
        bx1 = np.clip(np.round(bx.max()).astype(int) -1, 0, BW-1)
        by0 = np.clip(np.round(by.min()).astype(int) -1, 0, BH-1)
        by1 = np.clip(np.round(by.max()).astype(int) -1, 0, BH-1)

        #print('Brick', bx0,bx1,by0,by1)

        band = ccd.filter[0]
        assert(band in bands)

        ccdzpt = ccd.ccdzpt + 2.5 * np.log10(ccd.exptime)

        psf_sigma = ccd.fwhm / 2.35
        psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        orig_zpscale = zpscale = NanoMaggies.zeropointToScale(ccdzpt)
        sig1 = ccd.sig1 / orig_zpscale
        detsig1 = sig1 / psfnorm
        
        # print('Image', ccd.expnum, ccd.ccdname, ccd.filter,
        #       'PSF depth', -2.5 * (np.log10(5.*detsig1) - 9), 'exptime', ccd.exptime,
        #       'sig1', ccd.sig1, 'zpt', ccd.ccdzpt, 'fwhm', ccd.fwhm,
        #       'filename', ccd.image_filename.strip())

        depths[band][by0:by1+1, bx0:bx1+1] += (1. / detsig1**2)
        npix[band] += (y1+1-y0)*(x1+1-x0)
        nexps[band] += 1

    for band in bands:
        det = np.median(depths[band])
        # compute stats for 5-sigma detection
        with np.errstate(divide='ignore'):
            depth = 5. / np.sqrt(det)
        # that's flux in nanomaggies -- convert to mag
        depth = -2.5 * (np.log10(depth) - 9)
        if not np.isfinite(depth):
            depth = 0.
        
        depths[band] = depth
        #bricks.get('psfdepth_' + band)[ibrick] = depth
        print(brick.brickname, 'median PSF depth', band, ':', depth,
              'npix', npix[band],
              'nexp', nexps[band])
              #'npix', bricks.get('npix_'+band)[ibrick],
              #'nexp', bricks.get('nexp_'+band)[ibrick])

    return (npix, nexps, depths)

def main():
    survey = LegacySurveyData()
    ccds = survey.get_ccds_readonly()
    print(len(ccds), 'CCDs')
    ccds = ccds[ccds.ccd_cuts == 0]
    print(len(ccds), 'good CCDs')
    
    # Find bricks touched by >=1 CCD
    bricks = survey.get_bricks_readonly()
    bricks = bricks[(bricks.dec > -20) * (bricks.dec < 35.)]
    print(len(bricks), 'bricks in Dec range')
    I,J,d = match_radec(bricks.ra, bricks.dec, ccds.ra, ccds.dec, 0.5, nearest=True)
    bricks = bricks[I]
    print(len(bricks), 'bricks')
    
    bands = ['g','r','z']
    
    nexps = {}
    for b in bands:
        ne = np.zeros(len(bricks), np.int16)
        nexps[b] = ne
        bricks.set('nexp_'+b, ne)
    npix = {}
    for b in bands:
        n = np.zeros(len(bricks), np.int64)
        npix[b] = n
        bricks.set('npix_'+b, n)
    
    for b in bands:
        n = np.zeros(len(bricks), np.float32)
        bricks.set('psfdepth_'+b, n)

    args = enumerate(bricks)
    mp = multiproc(8)
    R = mp.map(one_brick, args)

    for ibrick,res in enumerate(R):
        if res is None:
            continue

        (npix, nexps, depths) = res
        for band in bands:
            bricks.get('npix_' + band)[ibrick] = npix[band]
            bricks.get('nexp_' + band)[ibrick] = nexps[band]
            bricks.get('psfdepth_' + band)[ibrick] = depths[band]

    bricks.cut((bricks.nexp_g + bricks.nexp_r + bricks.nexp_z) > 0)
    bricks.writeto('/global/cscratch1/sd/dstn/bricks-nexp.fits')

if __name__ == '__main__':
    main()

