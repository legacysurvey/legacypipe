from __future__ import print_function
from legacypipe.survey import *
from legacypipe.runbrick import run_brick
from legacyanalysis.ps1cat import ps1cat, ps1_to_decam
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import PlotSequence
import pylab as plt
import numpy as np
import fitsio

if __name__ == '__main__':
    expnum, ccdname = 292604, 'N4'
    
    survey = LegacySurveyData(output_dir='onechip-civ')
    ccds = survey.find_ccds(expnum=expnum, ccdname=ccdname)
    print('Found', len(ccds), 'CCD')
    # HACK -- cut to JUST that ccd.
    survey.ccds = ccds

    ccd = ccds[0]
    im = survey.get_image_object(ccd)
    wcs = survey.get_approx_wcs(ccd)
    rc,dc = wcs.radec_center()

    # Read PS-1 catalog to find out which blobs to process.
    ps1 = ps1cat(ccdwcs=wcs)
    stars = ps1.get_stars()
    print('Read', len(stars), 'PS1 stars')

    brickname = ('custom-%06i%s%05i' %
                 (int(1000*rc), 'm' if dc < 0 else 'p', int(1000*np.abs(dc))))

    outfn = survey.find_file('tractor', brick=brickname, output=True)
    print('Output catalog:', outfn)

    if not os.path.exists(outfn):
        run_brick(None, survey, radec=(rc,dc),
                  width=ccd.height, height=ccd.width,    # CCDs are rotated
                  bands=im.band,
                  wise=False,
                  blobradec=zip(stars.ra, stars.dec),
                  do_calibs=False,
                  write_metrics=True,
                  pixPsf=True,
                  constant_invvar=True,
                  ceres=False,
                  splinesky=True,
                  coadd_bw=True,
                  forceAll=True,
                  writePickles=False)

    print('Reading', outfn)
    cat = fits_table(outfn)
    primhdr = fitsio.read_header(outfn)
    
    iband = survey.index_of_band(im.band)
    cat.flux = cat.decam_flux[:, iband]
    cat.apflux = cat.decam_apflux[:, iband, :]
    cat.mag = -2.5 * (np.log10(cat.flux) - 9.)
    cat.apmag = -2.5 * (np.log10(cat.apflux) - 9.)
    
    I,J,d = match_radec(stars.ra, stars.dec, cat.ra, cat.dec, 1./3600.,
                        nearest=True)
    print(len(I), 'matches to PS1 stars')

    colorterm = ps1_to_decam(stars.median, im.band)
    ipsband = ps1cat.ps1band[im.band]
    stars.mag = stars.median[:, ipsband] + colorterm
    stars.flux = 10.**((stars.mag - 22.5) / -2.5)
    
    mstars = stars[I]
    mcat = cat[J]

    print(np.sum(mcat.type == 'PSF '), 'matched sources are PSF')
    mcat.ispsf = (mcat.type == 'PSF ')
    mcat.unmasked = (np.sum(mcat.decam_anymask, axis=1) == 0)
    P = np.flatnonzero(mcat.ispsf * mcat.unmasked)
    print(len(P), 'matched, unmasked, PSFs')

    print('Median mag difference for PSFs:', np.median(mcat.mag[P] - mstars.mag[P]))

    ps = PlotSequence('onechip')

    tt = 'onechip: DECam %i-%s' % (expnum, ccdname)
    
    maglo, maghi = 17, 20.25
    
    plt.clf()
    plt.plot(mstars.mag,    mcat.mag, 'r.')
    plt.plot(mstars.mag[P], mcat.mag[P], 'k.')
    plt.xlabel('Pan-STARRS %s (mag)' % im.band)
    plt.ylabel('DECaLS %s (mag)' % im.band)
    plt.axis([maghi,maglo,maghi,maglo])
    plt.title(tt)
    ps.savefig()
        
    plt.clf()
    plt.plot(mstars.mag,    mcat.mag - mstars.mag, 'r.')
    plt.plot(mstars.mag[P], mcat.mag[P] - mstars.mag[P], 'k.')
    plt.xlabel('Pan-STARRS %s (mag)' % im.band)
    plt.ylabel('DECaLS %s - Pan-STARRS %s (mag)' % (im.band,im.band))
    plt.axis([maghi,maglo,-0.25,0.25])
    plt.axhline(0., color='k', alpha=0.2)
    plt.axhline(0.025, color='r', alpha=0.2)
    plt.title(tt)
    ps.savefig()

    plt.clf()
    t = []
    for iap in range(8):
        rgb = [0., 0., iap/7.]
        rgb[1] = 1. - rgb[2]
        plt.plot(mstars.mag[P], mcat.apmag[P, iap] - mstars.mag[P], '.',
                 color=rgb)
        t.append('%.3f' % np.median(mcat.apmag[P, iap] - mstars.mag[P]))
    plt.xlabel('Pan-STARRS %s (mag)' % im.band)
    plt.ylabel('DECaLS %s - Pan-STARRS %s (mag)' % (im.band,im.band))
    plt.title(' / '.join(t))
    plt.axis([maghi,maglo,-1,1])
    plt.axhline(0., color='k', alpha=0.2)
    plt.axhline(0.025, color='r', alpha=0.2)
    ps.savefig()
    
    plt.clf()
    plt.subplot(2,1,1)
    iap = 5
    plt.plot(mstars.mag[P], mcat.apmag[P, iap] - mstars.mag[P], 'k.')
    plt.axis([maghi,maglo,-0.2,0.2])
    plt.axhline(0., color='k', alpha=0.2)
    plt.title('Aperture mag 5')
    plt.ylabel('DECaLS Ap mag - PS1')
    plt.subplot(2,1,2)
    plt.plot(mstars.mag[P], mcat.mag[P] - mstars.mag[P], 'k.')
    #plt.plot(mstars.mag[K], mcat.mag[K] - mstars.mag[K], 'r.')
    plt.axis([maghi,maglo,-0.2,0.2])
    plt.axhline(0., color='k', alpha=0.2)
    plt.ylabel('DECaLS PSF mag - PS1')
    plt.title('PSF mag')
    plt.suptitle(tt)
    ps.savefig()

    print('Median mag difference for ap mags:', np.median(mcat.apmag[P,iap] - mstars.mag[P]))
    
    plt.clf()
    iap = 5
    plt.plot(mcat.mag[P], mcat.apmag[P, iap] - mcat.mag[P], 'k.')
    plt.axis([maghi,maglo,-0.25,0.25])
    plt.axhline(0., color='k', alpha=0.2)
    plt.ylabel('DECaLS AP mag - DECaLS PSF mag')
    plt.xlabel('DECaLS PSF mag')
    plt.title(tt)
    ps.savefig()

    print('Median AP - PSF mag:', np.median(mcat.apmag[P, iap] - mcat.mag[P]))
    
    plt.clf()
    plt.plot((mcat.apflux[P, :] / mcat.flux[P, np.newaxis]).T, 'k-', alpha=0.1)
    plt.ylabel('APflux / PSF flux')
    plt.xlabel('Aperture diameter (arcsec)')
    plt.xticks(range(8), [primhdr['APRAD%i' % i]*2 for i in range(8)])
    plt.ylim(0.5, 1.5)
    plt.axhline(1., color='b', alpha=0.5)
    #plt.axhline(1.015, color='r', alpha=0.5)
    plt.title(tt)
    ps.savefig()
    

    plt.clf()
    plt.plot((mcat.apflux[P, :] / mstars.flux[P, np.newaxis]).T,
             'k-', alpha=0.1)
    plt.ylabel('DECaLS APflux / PS1 flux')
    plt.xlabel('Aperture diameter (arcsec)')
    plt.xticks(range(8), [primhdr['APRAD%i' % i]*2 for i in range(8)])
    plt.ylim(0.5, 1.5)
    plt.axhline(1., color='b', alpha=0.5)
    #plt.axhline(1.015, color='r', alpha=0.5)
    plt.title(tt)
    ps.savefig()
    
