'''
This is a little script for comparing DECaLS to Pan-STARRS magnitudes for
investigating zeropoint and other issues.
'''
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import os

import scipy.ndimage
from scipy.stats import sigmaclip

from tractor.brightness import NanoMaggies
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.miscutils import *
from astrometry.util.plotutils import *

from legacyanalysis.ps1cat import *
from legacypipe.common import *
from astrometry.libkd.spherematch import *
import photutils

def apphot_ps1stars(ccd, ps,
                    apertures,
                    decals,
                    sky_inner_r=40,
                    sky_outer_r=50):
    im = decals.get_image_object(ccd)

    tim = im.get_tractor_image(gaussPsf=True, splinesky=True)
    img = tim.getImage()

    wcs = tim.subwcs
    
    magrange = (15,21)
    ps1 = ps1cat(ccdwcs=wcs)
    ps1 = ps1.get_stars(magrange=magrange)
    print 'Got', len(ps1), 'PS1 stars'


    ok,x,y = wcs.radec2pixelxy(ps1.ra, ps1.dec)
    apxy = np.vstack((x - 1., y - 1.)).T

    ap = []
    aperr = []
    nmasked = []
    with np.errstate(divide='ignore'):
        ie = tim.getInvError()
        imsigma = 1. / ie
        imsigma[ie == 0] = 0
    for rad in apertures:
        aper = photutils.CircularAperture(apxy, rad)
        p = photutils.aperture_photometry(img, aper, error=imsigma)
        aperr.append(p.field('aperture_sum_err'))
        ap.append(p.field('aperture_sum'))
        p = photutils.aperture_photometry((ie == 0), aper)
        nmasked.append(p.field('aperture_sum'))
    ap = np.vstack(ap).T
    aperr = np.vstack(aperr).T
    nmasked = np.vstack(nmasked).T

    print 'Aperture fluxes:', ap[:5]
    print 'Aperture flux errors:', aperr[:5]
    print 'Nmasked:', nmasked[:5]
    
    H,W = img.shape
    sky = []
    skysigma = []
    skymed = []
    skynmasked = []
    for xi,yi in zip(x,y):
        ix = int(np.round(xi))
        iy = int(np.round(yi))
        skyR = sky_outer_r
        xlo = max(0, ix-skyR)
        xhi = min(W, ix+skyR+1)
        ylo = max(0, iy-skyR)
        yhi = min(H, iy+skyR+1)
        xx,yy = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
        r2 = (xx - xi)**2 + (yy - yi)**2
        inannulus = ((r2 >= sky_inner_r**2) * (r2 < sky_outer_r**2))
        unmasked = (ie[ylo:yhi, xlo:xhi] > 0)
        
        #sky.append(np.median(img[ylo:yhi, xlo:xhi][inannulus * unmasked]))

        skypix = img[ylo:yhi, xlo:xhi][inannulus * unmasked]
        # this is the default value...
        nsigma = 4.
        goodpix,lo,hi = sigmaclip(skypix, low=nsigma, high=nsigma)
        # sigmaclip returns unclipped pixels, lo,hi, where lo,hi are
        # mean(goodpix) +- nsigma * sigma
        meansky = np.mean(goodpix)
        sky.append(meansky)
        skysigma.append((meansky - lo) / nsigma)
        skymed.append(np.median(skypix))
        skynmasked.append(np.sum(inannulus * np.logical_not(unmasked)))
    sky = np.array(sky)
    skysigma = np.array(skysigma)
    skymed = np.array(skymed)
    skynmasked = np.array(skynmasked)

    print 'sky', sky[:5]
    print 'median sky', skymed[:5]
    print 'sky sigma', skysigma[:5]

    band = ccd.filter
    piband = ps1cat.ps1band[band]
    print 'band:', band

    psmag = ps1.median[:,piband]

    ap2 = ap - sky[:,np.newaxis] * (np.pi * apertures**2)[np.newaxis,:]
    
    if ps is not None:
        plt.clf()
        nstars,naps = ap.shape
        for iap in range(naps):
            plt.plot(psmag, ap[:,iap], 'b.')
        #for iap in range(naps):
        #    plt.plot(psmag, ap2[:,iap], 'r.')
        plt.yscale('symlog')
        plt.xlabel('PS1 %s mag' % band)
        plt.ylabel('DECam Aperture Flux')
    
        #plt.plot(psmag, nmasked[:,-1], 'ro')
        plt.plot(np.vstack((psmag,psmag)), np.vstack((np.zeros_like(psmag),nmasked[:,-1])), 'r-', alpha=0.5)
        plt.ylim(0, 1e3)
        ps.savefig()    
    
        plt.clf()
        plt.plot(ap.T / np.max(ap, axis=1), '.')
        plt.ylim(0, 1)
        ps.savefig()
    
        plt.clf()
        dimshow(tim.getImage(), **tim.ima)
        ax = plt.axis()
        plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
        plt.axis(ax)
        ps.savefig()

    T = fits_table()
    T.apflux = ap.astype(np.float32)
    T.apflux2 = ap2.astype(np.float32)
    T.expnum = np.array([ccd.expnum] * len(T))
    T.ccdname = np.array([ccd.ccdname] * len(T)).astype('S3')
    T.ps1_objid = ps1.objid
    T.ps1_mag = psmag
    T.ra  = ps1.ra
    T.dec = ps1.dec
    T.tai = np.array([tim.time.toMjd()]).astype(np.float32)

    T.primhdr = tim.primhdr

    #mjds = [tim.time.toMjd() for tim in tims if tim.band == band]
    #import astropy.time
    #tt = [astropy.time.Time(mjd, format='mjd', scale='tai').utc.isot
    #      for mjd in [minmjd, maxmjd]]

    if False:
        plt.clf()
        plt.plot(skymed, sky, 'b.')
        plt.xlabel('sky median')
        plt.ylabel('sigma-clipped sky')
        ax = plt.axis()
        lo,hi = min(ax),max(ax)
        plt.plot([lo,hi],[lo,hi],'k-', alpha=0.25)
        plt.axis(ax)
        ps.savefig()
    
    return T
        
if __name__ == '__main__':

    decals = Decals()

    #ps = PlotSequence('uber')
    #C = fits_table('coadd/000/0001p000/decals-0001p000-ccds.fits')
    #for c in C:
    #    apphot_ps1stars(c, ps, apertures, decals)
    
    C = decals.get_ccds_readonly()

    pixscale = 0.262
    apertures = apertures_arcsec / pixscale
    ps = None

    exps = [ 346352, 347460, 347721 ]

    for e in exps:
        E = C[C.expnum == e]
        TT = []
        for c in E:
            T = apphot_ps1stars(c, ps, apertures, decals)
            TT.append(T)
        T = merge_tables(TT, header=T[0].primhdr)
        T.writeto('apphot-%08i.fits' % e)
