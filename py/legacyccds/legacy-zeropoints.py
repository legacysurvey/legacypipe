#!/usr/bin/env python

"""Generate a legacypipe-compatible CCD-level zeropoints file for a given set of
(reduced) BASS, MzLS, or DECaLS imaging.

This script borrows liberally from code written by Ian, Kaylan, Dustin, David
S. and Arjun, including rapala.survey.bass_ccds, legacypipe.simple-bok-ccds,
obsbot.measure_raw, and the IDL codes decstat and mosstat.

Although the script was developed to run on the temporarily repackaged BASS data
created by the script legacyccds/repackage-bass.py (which writes out
multi-extension FITS files with a different naming convention relative to what
NAOC delivers), it is largely camera-agnostic, and should therefore eventually
be able to be used to derive zeropoints for all the Legacy Survey imaging.

On edison the repackaged BASS data are located in
/scratch2/scratchdirs/ioannis/bok-reduced with the correct permissions.

Proposed changes to the -ccds.fits file used by legacypipe:
 * Rename arawgain --> gain to be camera-agnostic.
 * The quantities ccdzpta and ccdzptb are specific to DECam, while for 90prime
   these quantities are ccdzpt1, ccdzpt2, ccdzpt3, and ccdzpt4.  These columns
   can be kept in the -zeropoints.fits file but should be removed from the final
   -ccds.fits file.
 * The pipeline uses the SE-measured FWHM (FWHM, pixels) to do source detection
   and to estimate the depth, instead of SEEING (FWHM, arcsec), which is
   measured by decstat in the case of DECam.  We should remove our dependence on
   SExtractor and simply use the seeing/fwhm estimate measured by us (e.g., this
   code).
 * The pixel scale should be added to the output file, although it can be gotten
   from the CD matrix.
 * AVSKY should be converted to electron or electron/s, to account for
   the varying gain of the amplifiers.  It's actually not even clear
   we need this header keyword.
 * We probably shouldn't cross-match against the tiles file in this code (but
   instead in something like merge-zeropoints), but what else from the annotated
   CCDs file should be directly calculated and stored here?
 * Are ccdnum and image_hdu redundant?

"""
from __future__ import division, print_function

import os
import pdb
import argparse

import numpy as np
from glob import glob
from scipy.stats import sigmaclip

import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt

from photutils import CircularAperture, CircularAnnulus, aperture_photometry, daofind

from astrometry.libkd.spherematch import match_radec

from legacyanalysis.ps1cat import ps1cat

def _ccds_table(camera='decam'):
    '''Initialize the output CCDs table.  See decstat.pro and merge-zeropoints.py
    for details.

    '''
    cols = [
        ('image_filename', 'S65'), # image filename, including the subdirectory
        ('image_hdu', '>i2'),      # integer extension number
        ('camera', 'S7'),          # camera name
        ('expnum', '>i4'),         # unique exposure number
        ('ccdname', 'S4'),         # FITS extension name
#       ('ccdnum', '>i2'),         # CCD number 
        ('expid', 'S16'),          # combination of EXPNUM and CCDNAME
        ('object', 'S35'),         # object (field) name
        ('propid', 'S10'),         # proposal ID
        ('filter', 'S1'),          # filter name / bandpass
        ('exptime', '>f4'),        # exposure time (s)
        ('date_obs', 'S10'),       # date of observation (from header)
        ('mjd_obs', '>f8'),        # MJD of observation (from header)
        ('ut', 'S15'),             # UT time (from header)
        ('ha', 'S13'),             # hour angle (from header)
        ('airmass', '>f4'),        # airmass (from header)
        #('seeing', '>f4'),        # seeing estimate (from header, arcsec)
        ('fwhm', '>f4'),           # FWHM (from header, pixels) -- used in Tractor source detection!
        #('arawgain', '>f4'),       
        ('gain', '>f4'),           # average gain (camera-specific, e/ADU) -- remove?
        ('avsky', '>f4'),          # average sky value from CP (from header, ADU) -- remove?
        ('width', '>i2'),          # image width (pixels, NAXIS1, from header)
        ('height', '>i2'),         # image height (pixels, NAXIS2, from header)
        ('ra_bore', '>f8'),        # RA at the center of the field (deg, CRVAL1, from header)
        ('dec_bore', '>f8'),       # Dec at the center of the field (deg, CRVAL2, from header)
        ('crpix1', '>f4'),
        ('crpix2', '>f4'),
        ('crval1', '>f8'),
        ('crval2', '>f8'),
        ('cd1_1', '>f4'),
        ('cd1_2', '>f4'),
        ('cd2_1', '>f4'),
        ('cd2_2', '>f4'),
        # -- derived quantities --
        ('ra', '>f8'),
        ('dec', '>f8'),
        ('zpt', '>f4'),
        ('ccdskymag', '>f4'),      
        ('ccdskycounts', '>f4'),
        ('ccdskyrms', '>f4'),
        ('ccdnstar', '>i2'),
        ('ccdnmatch', '>i2'),
        ('ccdmdncol', '>f4'), 
        ('ccdphoff', '>f4'),
        ('ccdphrms', '>f4'),
        ('ccdzpt', '>f4'),
        ('ccdtransp', '>f4'), 
        ('ccdraoff', '>f4'),
        ('ccddecoff', '>f4')
        ]

    # Add camera-specific keywords to the output table.
    if camera == 'decam':
        cols.extend([('ccdzpta', '>f4'), ('ccdzptb','>f4'), ('ccdnmatcha', '>i2'), ('ccdnmatchb', '>i2'),
                     ('temp', '>f4')])
    elif camera == 'mosaic':
        pass
    elif camera == '90prime':
        cols.extend([('ccdzpt1', '>f4'), ('ccdzpt2','>f4'), ('ccdzpt3', '>f4'), ('ccdzpt4','>f4'),
                     ('ccdnmatcha', '>i2'), ('ccdnmatch2', '>i2'), ('ccdnmatch3', '>i2'), ('ccdnmatch4', '>i2')])
        
    ccds = Table(np.zeros(1, dtype=cols))
    return ccds

def _stars_table(nstars=1):
    '''Initialize the stars table, which will contain information on all the stars
       detected on the CCD, including the PS1 photometry.

    '''
    cols = [('expid', 'S16'), ('filter', 'S1'), ('x', 'f4'), ('y', 'f4'),
            ('ra', 'f8'), ('dec', 'f8'), ('fwhm', 'f4'), ('apmag', 'f4'),
            ('ps1_ra', 'f8'), ('ps1_dec', 'f8'), ('ps1_mag', 'f4'), ('ps1_gicolor', 'f4')]
    stars = Table(np.zeros(nstars, dtype=cols))

    return stars

class Measurer(object):
    def __init__(self, fn, ext, aprad=3.5, skyrad_inner=7.0, skyrad_outer=10.0,
                 sky_global=False, calibrate=False):
        '''This is the work-horse class which operates on a given image regardless of
        its origin (decam, mosaic, 90prime).

        Args:

        aprad: float
        Aperture photometry radius in arcsec

        skyrad_{inner,outer}: floats
        Sky annulus radius in arcsec

        '''
        from astrometry.util.util import wcs_pv2sip_hdr
        
        self.fn = fn
        self.ext = ext

        self.sky_global = sky_global
        self.calibrate = calibrate
        
        self.aprad = aprad
        self.skyrad = (skyrad_inner, skyrad_outer)

        # Set the nominal detection FWHM (in pixels) and detection threshold.
        self.nominal_fwhm = 5.0 # [pixels]
        self.det_thresh = 10    # [S/N] - used to be 20

        self.matchradius = 2.0  # search radius for finding matching PS1 stars [arcsec]

        # Read the primary header and the header for this extension.
        self.primhdr = fitsio.read_header(fn, ext=0)
        self.hdr = fitsio.read_header(fn, ext=ext)

        # Camera-agnostic primary header cards
        self.propid = self.primhdr['PROPID']
        self.exptime = self.primhdr['EXPTIME']
        self.date_obs = self.primhdr['DATE-OBS']
        self.mjd_obs = self.primhdr['MJD-OBS']
        self.airmass = self.primhdr['AIRMASS']
        self.ha = self.primhdr['HA']
        self.seeing = self.primhdr['SEEING']

        if 'EXPNUM' in self.hdr: # temporary hack!
            self.expnum = self.hdr['EXPNUM']
        else:
            self.expnum = np.int32(os.path.basename(self.fn)[11:17])

        self.ccdname = self.hdr['EXTNAME'].strip()
        self.image_hdu = np.int(self.hdr['CCDNUM'])
        #self.ccdnum = self.hdr['CCDNUM']
        #self.image_hdu = np.int(self.ccdnum)

        self.expid = '{:08d}-{}'.format(self.expnum, self.ccdname)
        self.band = self.get_band()

        self.object = self.primhdr['OBJECT']

        self.wcs = wcs_pv2sip_hdr(self.hdr) # PV distortion
        self.pixscale = self.wcs.pixel_scale()

        # Eventually we would like FWHM to not come from SExtractor.
        self.fwhm = 2.35 * self.primhdr['SEEING'] / self.pixscale  # [FWHM, pixels]

    def zeropoint(self, band):
        return self.zp0[band]

    def sky(self, band):
        return self.sky0[band]

    def extinction(self, band):
        return self.k_ext[band]

    def sensible_sigmaclip(self, arr, nsigma = 4.0):
        '''sigmaclip returns unclipped pixels, lo,hi, where lo,hi are the
        mean(goodpix) +- nsigma * sigma

        '''
        from scipy.stats import sigmaclip
        
        goodpix, lo, hi = sigmaclip(arr, low=nsigma, high=nsigma)
        meanval = np.mean(goodpix)
        sigma = (meanval - lo) / nsigma
        return meanval, sigma

    def get_sky_and_sigma(self, img):
        # Spline sky model to handle (?) ghost / pupil?
        from tractor.splinesky import SplineSky

        #sky, sig1 = self.sensible_sigmaclip(img[1500:2500, 500:1000])

        splinesky = SplineSky.BlantonMethod(img, None, 256)
        skyimg = np.zeros_like(img)
        splinesky.addTo(skyimg)

        mnsky, sig1 = self.sensible_sigmaclip(img - skyimg)
        return skyimg, sig1

    def remove_sky_gradients(self, img):
        from scipy.ndimage.filters import median_filter
        # Ugly removal of sky gradients by subtracting median in first x and then y
        H,W = img.shape
        meds = np.array([np.median(img[:,i]) for i in range(W)])
        meds = median_filter(meds, size=5)
        img -= meds[np.newaxis,:]
        meds = np.array([np.median(img[i,:]) for i in range(H)])
        meds = median_filter(meds, size=5)
        img -= meds[:,np.newaxis]

    def match_ps1_stars(self, px, py, fullx, fully, radius, stars):
        from astrometry.libkd.spherematch import match_xy
        #print('Matching', len(px), 'PS1 and', len(fullx), 'detected stars with radius', radius)
        I,J,d = match_xy(px, py, fullx, fully, radius)
        #print(len(I), 'matches')
        dx = px[I] - fullx[J]
        dy = py[I] - fully[J]
        return I,J,dx,dy

    def run(self):

        # Read the image and header.
        img, hdr = self.read_image()

        # Initialize and begin populating the output CCDs table.
        ccds = _ccds_table(self.camera)

        ccds['image_filename'] = self.fn   
        ccds['image_hdu'] = self.image_hdu 
        ccds['camera'] = self.camera
        ccds['expnum'] = self.expnum
        ccds['ccdname'] = self.ccdname
        #ccds['ccdnum'] = self.ccdnum
        ccds['expid'] = self.expid
        ccds['object'] = self.object
        ccds['propid'] = self.propid
        ccds['filter'] = self.band
        ccds['exptime'] = self.exptime
        ccds['date_obs'] = self.date_obs
        ccds['mjd_obs'] = self.mjd_obs
        ccds['ut'] = self.ut
        ccds['ha'] = self.ha
        ccds['airmass'] = self.airmass
        ccds['fwhm'] = self.fwhm
        ccds['gain'] = self.gain

        # Copy some header cards directly.
        hdrkey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                  'cd1_2', 'cd2_1', 'cd2_2', 'crval1', 'crval2',
                  'naxis1', 'naxis2')
        ccdskey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                   'cd1_2', 'cd2_1', 'cd2_2', 'ra_bore', 'dec_bore',
                   'width', 'height')
        for ckey, hkey in zip(ccdskey, hdrkey):
            ccds[ckey] = hdr[hkey]
            
        exptime = ccds['exptime'].data[0]
        airmass = ccds['airmass'].data[0]
        print('Band {}, Exptime {}, Airmass {}'.format(self.band, exptime, airmass))

        # Get the ra, dec coordinates at the center of the chip.
        H, W = img.shape
        ccdra, ccddec = self.wcs.pixelxy2radec((W+1) / 2.0, (H + 1) / 2.0)
        ccds['ra'] = ccdra   # [degree]
        ccds['dec'] = ccddec # [degree]

        # Measure the sky brightness and (sky) noise level.  Need to capture
        # negative sky.
        sky0 = self.sky(self.band)
        zp0 = self.zeropoint(self.band)
        kext = self.extinction(self.band)

        print('Computing the sky background.')
        sky, sig1 = self.get_sky_and_sigma(img)
        sky1 = np.median(sky)
        skybr = zp0 - 2.5*np.log10(sky1 / self.pixscale / self.pixscale / exptime)
        #skybr = zp0 - 2.5*np.log10(sky1 / self.pixscale / self.pixscale)

        print('  Sky brightness: {:.3f} mag/arcsec^2'.format(skybr))
        print('  Fiducial:       {:.3f} mag/arcsec^2'.format(sky0))

        ccds['ccdskyrms'] = sig1    # [electron/pix]
        ccds['ccdskycounts'] = sky1 # [electron/pix]
        ccds['ccdskymag'] = skybr   # [mag/arcsec^2]

        # Detect stars on the image.  
        det_thresh = self.det_thresh
        obj = daofind(img, fwhm=self.fwhm,
                      threshold=det_thresh*sig1,
                      exclude_border=True)
        if len(obj) < 20:
            det_thresh = self.det_thresh / 2.0
            obj = daofind(img, fwhm=self.fwhm,
                          threshold=det_thresh*sig1,
                          exclude_border=True)
        nobj = len(obj)
        print('{} sources detected with detection threshold {}-sigma'.format(nobj, det_thresh))

        if nobj == 0:
            print('No sources detected!  Giving up.')
            return ccds, _stars_table()

        # Do aperture photometry in a fixed aperture but using either local (in
        # an annulus around each star) or global sky-subtraction.
        print('Performing aperture photometry')

        ap = CircularAperture((obj['xcentroid'], obj['ycentroid']), self.aprad / self.pixscale)
        if self.sky_global:
            apphot = aperture_photometry(img - sky, ap)
            apflux = apphot['aperture_sum']
        else:
            skyap = CircularAnnulus((obj['xcentroid'], obj['ycentroid']),
                                    r_in=self.skyrad[0] / self.pixscale, 
                                    r_out=self.skyrad[1] / self.pixscale)
            apphot = aperture_photometry(img, ap)
            skyphot = aperture_photometry(img, skyap)
            apflux = apphot['aperture_sum'] - skyphot['aperture_sum'] / skyap.area() * ap.area()

        # Remove stars with negative flux (or very large photometric uncertainties).
        istar = np.where(apflux > 0)[0]
        if len(istar) == 0:
            print('All stars have negative aperture photometry!')
            return ccds, _stars_table()
        obj = obj[istar]
        apflux = apflux[istar].data
        ccds['ccdnstar'] = len(istar)

        # Now match against (good) PS1 stars with magnitudes between 15 and 22.
        ps1 = ps1cat(ccdwcs=self.wcs).get_stars(magrange=(15, 22))
        good = np.where((ps1.nmag_ok[:, 0] > 0)*(ps1.nmag_ok[:, 1] > 0)*(ps1.nmag_ok[:, 2] > 0))[0]
        ps1.cut(good)
        nps1 = len(ps1)

        if nps1 == 0:
            print('No overlapping PS1 stars in this field!')
            return ccds, _stars_table()

        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        m1, m2, d12 = match_radec(objra, objdec, ps1.ra, ps1.dec, self.matchradius/3600.0)
        nmatch = len(m1)
        ccds['ccdnmatch'] = nmatch
        
        print('{} PS1 stars match detected sources within {} arcsec.'.format(nmatch, self.matchradius))

        # Initialize the stars table and begin populating it.
        stars = _stars_table(nmatch)
        stars['filter'] = self.band
        stars['expid'] = self.expid
        stars['x'] = obj['xcentroid'][m1]
        stars['y'] = obj['ycentroid'][m1]
        stars['ra'] = objra[m1]
        stars['dec'] = objdec[m1]

        stars['apmag'] = - 2.5 * np.log10(apflux[m1]) + zp0 + 2.5 * np.log10(exptime)

        stars['ps1_ra'] = ps1.ra[m2]
        stars['ps1_dec'] = ps1.dec[m2]
        stars['ps1_gicolor'] = ps1.median[m2, 0] - ps1.median[m2, 2]

        ps1band = ps1cat.ps1band[self.band]
        stars['ps1_mag'] = ps1.median[m2, ps1band]

        #plt.scatter(stars['ra'], stars['dec'], color='orange') ; plt.scatter(stars['ps1_ra'], stars['ps1_dec'], color='blue') ; plt.show()
        
        # Unless we're calibrating the photometric transformation, bring PS1
        # onto the photometric system of this camera (we add the color term
        # below).
        if self.calibrate:
            colorterm = np.zeros(nmatch)
        else:
            colorterm = self.colorterm_ps1_to_observed(ps1.median[m2, :], self.band)

        # Compute the astrometric residuals relative to PS1.
        raoff = np.median((stars['ra'] - stars['ps1_ra']) * np.cos(np.deg2rad(ccddec)) * 3600.0)
        decoff = np.median((stars['dec'] - stars['ps1_dec']) * 3600.0)
        ccds['ccdraoff'] = raoff
        ccds['ccddecoff'] = decoff
        print('Median offsets (arcsec) relative to PS1: dra = {}, ddec = {}'.format(raoff, decoff))

        # Compute the photometric zeropoint but only use stars with main
        # sequence g-i colors.
        print('Computing the photometric zeropoint.')
        mskeep = np.where((stars['ps1_gicolor'] > 0.4) * (stars['ps1_gicolor'] < 2.7))[0]
        if len(mskeep) == 0:
            print('Not enough PS1 stars with main sequence colors.')
            return ccds, stars
        ccds['ccdmdncol'] = np.median(stars['ps1_gicolor'][mskeep]) # median g-i color

        # Get the photometric offset relative to PS1 as the observed PS1
        # magnitude minus the observed / measured magnitude.

        stars['ps1_mag'] += colorterm
        #plt.scatter(stars['ps1_gicolor'], stars['apmag']-stars['ps1_mag']) ; plt.show()
        dmagall = stars['ps1_mag'][mskeep] - stars['apmag'][mskeep]
        _, dmagsig = self.sensible_sigmaclip(dmagall, nsigma=2.5)

        dmag, _, _ = sigmaclip(dmagall, low=3, high=3.0)
        dmagmed = np.median(dmag)
        ndmag = len(dmag)

        zptmed = zp0 + dmagmed
        transp = 10.**(-0.4 * (zp0 - zptmed - kext * (airmass - 1.0)))

        print('  Mag offset: {:.3f}'.format(dmagmed))
        print('  Scatter:    {:.3f}'.format(dmagsig))
        
        print('  {} stars used for zeropoint median'.format(ndmag))
        print('  Zeropoint {:.3f}'.format(zptmed))
        print('  Transparency: {:.3f}'.format(transp))

        ccds['ccdphoff'] = dmagmed
        ccds['ccdphrms'] = dmagsig
        ccds['ccdzpt'] = zptmed
        ccds['ccdtransp'] = transp

        # Hack!  Fit each star with Tractor to measure the FWHM seeing, but for
        # now just take the header (SE-measured) values.
        # ccds['seeing'] = 2.35 * self.hdr['seeing'] # FWHM [arcsec]
        ccds['fwhm'] = 2.35 * self.seeing / self.pixscale # FWHM [pixels]
        stars['fwhm'] = np.repeat(ccds['fwhm'].data, len(stars))

        #alse:
        #fwhms = []
        #psf_r = 15
        #if n_fwhm not in [0, None]:
        #    Jf = J[:n_fwhm]
	#        
	#    for i,(xi,yi,fluxi) in enumerate(zip(fx[Jf],fy[Jf],apflux[Jf])):
	#        #print('Fitting source', i, 'of', len(Jf))
	#        ix = int(np.round(xi))
	#        iy = int(np.round(yi))
	#        xlo = max(0, ix-psf_r)
	#        xhi = min(W, ix+psf_r+1)
	#        ylo = max(0, iy-psf_r)
	#        yhi = min(H, iy+psf_r+1)
	#        xx,yy = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
	#        r2 = (xx - xi)**2 + (yy - yi)**2
	#        keep = (r2 < psf_r**2)
	#        pix = img[ylo:yhi, xlo:xhi].copy()
	#        ie = np.zeros_like(pix)
	#        ie[keep] = 1. / sig1
	#        #print('fitting source at', ix,iy)
	#        #print('number of active pixels:', np.sum(ie > 0), 'shape', ie.shape)
	#
	#        psf = tractor.NCircularGaussianPSF([4.], [1.])
	#        tim = tractor.Image(data=pix, inverr=ie, psf=psf)
	#        src = tractor.PointSource(tractor.PixPos(xi-xlo, yi-ylo),
	#                                  tractor.Flux(fluxi))
	#        tr = tractor.Tractor([tim],[src])
	#
	#        #print('Posterior before prior:', tr.getLogProb())
	#        src.pos.addGaussianPrior('x', 0., 1.)
	#        #print('Posterior after prior:', tr.getLogProb())
	#        
	#        doplot = (i < 5) * (ps is not None)
	#        if doplot:
	#            mod0 = tr.getModelImage(0)
	#
	#        tim.freezeAllBut('psf')
	#        psf.freezeAllBut('sigmas')
	#
	#        # print('Optimizing params:')
	#        # tr.printThawedParams()
	#
	#        #print('Parameter step sizes:', tr.getStepSizes())
	#        optargs = dict(priors=False, shared_params=False)
	#        for step in range(50):
	#            dlnp,x,alpha = tr.optimize(**optargs)
	#            #print('dlnp', dlnp)
	#            #print('src', src)
	#            #print('psf', psf)
	#            if dlnp == 0:
	#                break
	#        # Now fit only the PSF size
	#        tr.freezeParam('catalog')
	#        # print('Optimizing params:')
	#        # tr.printThawedParams()
	#
	#        for step in range(50):
	#            dlnp,x,alpha = tr.optimize(**optargs)
	#            #print('dlnp', dlnp)
	#            #print('src', src)
	#            #print('psf', psf)
	#            if dlnp == 0:
	#                break
	#
	#        fwhms.append(psf.sigmas[0] * 2.35 * pixsc)
	#
	#    fwhms = np.array(fwhms)
	#    medfwhm = np.median(fwhms)
	#    print('Median FWHM: {:.3f}'.format(medfwhm))
	#    ccds['seeing'] = medfwhm

        #pdb.set_trace()
        return ccds, stars
    
class DECamMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single DECam CCD.'''
    def __init__(self, *args, **kwargs):
        super(DECamMeasurer, self).__init__(*args, **kwargs)

        self.camera = 'decam'
        self.ut = self.primhdr['TIME-OBS']
        self.gain = self.hdr['ARAWGAIN'] # hack! average gain [electron/sec]

        print('Hack! Using a constant gain!')
        corr = 2.5 * np.log10(self.gain)
        #corr = 2.5 * np.log10(self.gain) - 2.5 * np.log10(self.exptime)
        self.zp0 = dict(z = 26.552 + corr)
        self.sky0 = dict(z = 18.46 + corr)
        self.k_ext = dict(z = 0.06)

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_decam
        return ps1_to_mosaic(ps1stars, band)

    def read_image(self):
        '''Read the image and header.  Convert image from ADU to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        img *= self.gain
        #img *= self.gain / self.exptime
        return img, hdr

class Mosaic3Measurer(Measurer):
    '''Class to measure a variety of quantities from a single Mosaic3 CCD.'''
    def __init__(self, *args, **kwargs):
        super(Mosaic3Measurer, self).__init__(*args, **kwargs)

        self.camera = 'mosaic3'
        self.ut = self.primhdr['TIME-OBS']
        self.gain = self.hdr['GAIN'] # hack! average gain

        print('Hack! Using an average Mosaic3 zeropoint!!')
        corr = 2.5 * np.log10(self.gain)
        self.zp0 = dict(z = 26.552 + corr)
        self.sky0 = dict(z = 18.46 + corr)
        self.k_ext = dict(z = 0.06)

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0][0] # zd --> z
        return band

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

    def read_image(self):
        '''Read the image and header.  Convert image from electrons/sec to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        img *= self.exptime
        return img, hdr

class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.'''
    def __init__(self, *args, **kwargs):
        super(NinetyPrimeMeasurer, self).__init__(*args, **kwargs)
        
        self.camera = '90prime'
        self.ut = self.primhdr['UT']

        # Average (nominal) gain values.  The gain is sort of a hack since this
        # information should be scraped from the headers, plus we're ignoring
        # the gain variations across amplifiers (on a given CCD).
        gaindict = dict(ccd1 = 1.47, ccd2 = 1.48, ccd3 = 1.42, ccd4 = 1.4275)
        self.gain = gaindict[self.ccdname.lower()]

        # Nominal zeropoints, sky brightness, and extinction values (taken from
        # rapala.ninetyprime.boketc.py).  The sky and zeropoints are both in
        # ADU, so account for the gain here.
        corr = 2.5 * np.log10(self.gain)
        self.zp0 = dict(g = 25.55 + corr, r = 25.38 + corr)
        self.sky0 = dict(g = 22.10 + corr, r = 21.07 + corr)
        self.k_ext = dict(g = 0.17, r = 0.10)

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        band.replace('bokr', 'r')
        return band

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

    def read_image(self):
        '''Read the image and header.  Convert image from electrons/sec to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        img *= self.exptime
        return img, hdr
  
def camera_name(primhdr):
    '''
    Returns 'mosaic3', 'decam', or '90prime'
    '''
    camera = primhdr.get('INSTRUME','').strip().lower()
    if camera == '90prime':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
    elif camera == 'mosaic3':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
    elif camera == 'decam':
        extlist = ['S29', 'S31', 'S25', 'S26', 'S27', 'S28', 'S20', 'S21', 'S22',
                   'S23', 'S24', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S8',
                   'S9', 'S10', 'S11', 'S12', 'S13', 'S1', 'S2', 'S3', 'S4', 'S5',
                   'S6', 'S7', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9',
                   'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18',
                   'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27',
                   'N28', 'N29', 'N31']
    else:
        print('Camera {} not recognized!'.format(camera))
        pdb.set_trace()
    
    return camera, extlist
    
def measure_mosaic3(fn, ext='CCD1', **kwargs):
    '''Wrapper function to measure quantities from the Mosaic3 camera.'''
    measure = Mosaic3Measurer(fn, ext, **kwargs)
    ccds, stars = measure.run()
    return ccds, stars

def measure_90prime(fn, ext='CCD1', **kwargs):
    '''Wrapper function to measure quantities from the 90prime camera.'''
    measure = NinetyPrimeMeasurer(fn, ext, **kwargs)
    ccds, stars = measure.run()
    return ccds, stars

def measure_decam(fn, ext='N4', **kwargs):
    '''Wrapper function to measure quantities from the DECam camera.'''
    measure = DecamMeasurer(fn, ext, **kwargs)
    ccds, stars = measure.run()
    return ccds, stars

def _measure_image(args):
    '''Utility function to wrap measure_image function for multiprocessing map.''' 
    return measure_image(*args)

def measure_image(filelist, measureargs={}):
    '''Wrapper on the camera-specific classes to measure the CCD-level data on all
    the FITS extensions for a given set of images.

    '''
    allccds = []
    for fn in filelist:
        print('Working on image {}'.format(fn))
        if not os.path.isfile(fn):
            print('  Image {} not found!'.format(fn))
            continue

        primhdr = fitsio.read_header(fn)
        camera, extlist = camera_name(primhdr)
        nnext = len(extlist)

        if camera == 'decam':
            measure = measure_decam
        elif camera == 'mosaic3':
            measure = measure_mosaic3
        elif camera == '90prime':
            measure = measure_90prime

        ccds = []
        stars = []
        for ext in extlist:
            ccds1, stars1 = measure(fn, ext, **measureargs)
            ccds.append(ccds1)
            stars.append(stars1)

        # Compute the median zeropoint across all the CCDs.
        ccds = vstack(ccds)
        stars = vstack(stars)
        ccds['zpt'] = np.median(ccds['ccdzpt'])

        if len(allccds) == 0:
            allccds = ccds
            allstars = stars
        else:
            allccds = vstack((allccds, ccds))
            allstars = vstack((allstars, stars))
        
    return allccds, allstars

def main():
    '''Generate a legacypipe-compatible CCDs file.

    '''

    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser.add_argument('--prefix', type=str, default='zeropoints', help='Prefix to prepend to the output files.')
    parser.add_argument('--outdir', type=str, default='./', help='Output directory.')
    parser.add_argument('--aprad', type=float, default=3.5, help='Aperture photometry radius (arcsec).')
    parser.add_argument('--skyrad-inner', type=float, default=7.0, help='Radius of inner sky annulus (arcsec).')
    parser.add_argument('--skyrad-outer', type=float, default=10.0, help='Radius of outer sky annulus (arcsec).')
    parser.add_argument('--nproc', type=int, default=1, help='Number of CPUs to use.')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use this option when deriving the photometric transformation equations.')
    parser.add_argument('--sky-global', action='store_true',
                        help='Use a global rather than a local sky-subtraction around the stars.')
    parser.add_argument('images', metavar='*.fits', nargs='+', help='List of images to process')

    args = parser.parse_args()

    # Build a dictionary with the optional inputs.
    measureargs = vars(args)
    images = np.array(measureargs.pop('images'))
    nproc = measureargs.pop('nproc')

    prefix = measureargs.pop('prefix')
    outdir = measureargs.pop('outdir')
    zptsfile = os.path.join(outdir, '{}.fits'.format(prefix))
    zptstarsfile = os.path.join(outdir, '{}-stars.fits'.format(prefix))

    # Process the data, optionally with multiprocessing.
    if nproc > 1:
        import multiprocessing
        splitimages = np.array_split(images, nproc)
        args = list()
        for ii in range(nproc):
            args.append((splitimages[ii], measureargs))
        pool = multiprocessing.Pool(nproc)
        results = pool.map(_measure_image, args)

        # Pack the results back together (should we sort by filename?).
        ccds = []
        stars = []
        for result in results:
            ccds.append(result[0])
            stars.append(result[1])
        ccds = vstack(ccds)
        stars = vstack(stars)
    else:
        ccds, stars = measure_image(images, measureargs)

    # Write out.
    if os.path.isfile(zptsfile):
        os.remove(zptsfile)
    print('Writing {}'.format(zptsfile))
    ccds.write(zptsfile)

    # Also write out the table of stars, although eventually we'll want to only
    # write this out if we're calibrating the photometry (or if the user
    # requests).
    if os.path.isfile(zptstarsfile):
        os.remove(zptstarsfile)
    print('Writing {}'.format(zptstarsfile))
    stars.write(zptstarsfile)
                    
if __name__ == "__main__":
    main()
