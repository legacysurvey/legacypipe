#!/usr/bin/env python

"""Generate a legacypipe-compatible CCDs file for a given set of (reduced) BASS
data.

This script borrows liberally from code written by Ian, Kaylan, Dustin, David
S. and Arjun, including rapala.survey.bass_ccds, legacypipe.simple-bok-ccds,
legacypipe.merge-zeropoints, obsbot.measure_raw, and the IDL code decstat.

The script runs on the temporarily repackaged BASS data created by the script
legacyzeropoints/repackage-bass.py (which writes out multi-extension FITS files
with a different naming convention relative to what NAOC delivers).  On edison
these data are located in /scratch2/scratchdirs/ioannis/bok-reduced

Proposed changes to the -ccds file used by legacypipe:
 * Rename arawgain --> gain
 * The quantities ccdzpta, ccdzptb, ccdnmatcha, and ccdnmatchb are specific to
   DECam and need to be expanded to accommodate the four amplifiers in the
   90prime camera.
 * The pipeline uses the SE-measured FWHM (FWHM, pixels) to do source detection
   and to estimate the depth, instead of SEEING (FWHM, arcsec), which is
   measured by decstat in the case of DECam.  We should probably remove our
   dependence on SExtractor, right?
 * We should store the pixel scale, too although it can be gotten from the CD matrix.

"""
from __future__ import division, print_function

import os
import pdb
import argparse

import numpy as np
from glob import glob

import fitsio
from astropy.io import fits
from astropy.table import Table, hstack
import matplotlib.pyplot as plt

from photutils import CircularAperture, CircularAnnulus, aperture_photometry, daofind

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.fits import merge_tables
from astrometry.util.plotutils import dimshow, plothist
from astrometry.libkd.spherematch import match_radec

from legacyanalysis.ps1cat import ps1cat

default_aprad = 3.5 # default aperture radius [3.5 arcsec]

def _ccds_table():
    '''Initialize the output CCDs table.  See decstat and merge-zeropoints.py for
    details

    '''
    cols = [
        ('object', 'S35'), ('expnum', '>i4'), ('exptime', '>f4'), ('filter', 'S1'),
        ('seeing', '>f4'), ('date_obs', 'S10'), ('mjd_obs', '>f8'), ('ut', 'S15'),
        ('ha', 'S13'), ('airmass', '>f4'), ('propid', 'S10'), ('zpt', '>f4'),
        ('avsky', '>f4'), ('arawgain', '>f4'), ('fwhm', '>f4'), ('crpix1', '>f4'),
        ('crpix2', '>f4'), ('crval1', '>f8'), ('crval2', '>f8'), ('cd1_1', '>f4'),
        ('cd1_2', '>f4'), ('cd2_1', '>f4'), ('cd2_2', '>f4'), ('ccdnum', '>i2'),
        ('ccdname', 'S3'), ('ccdzpt', '>f4'), ('ccdzpta', '>f4'), ('ccdzptb','>f4'),
        ('ccdphoff', '>f4'), ('ccdphrms', '>f4'), ('ccdskyrms', '>f4'),
        ('ccdskymag', '>f4'), ('ccdskycounts', '>f4'), ('ccdraoff', '>f4'),
        ('ccddecoff', '>f4'), ('ccdtransp', '>f4'), ('ccdnstar', '>i2'),
        ('ccdnmatch', '>i2'), ('ccdnmatcha', '>i2'), ('ccdnmatchb', '>i2'),
        ('ccdmdncol', '>f4'), ('temp', '>f4'), ('camera', 'S5'), ('expid', 'S12'),
        ('image_hdu', '>i2'), ('image_filename', 'S61'), ('width', '>i2'),
        ('height', '>i2'), ('ra_bore', '>f8'), ('dec_bore', '>f8'), ('ra', '>f8'),
        ('dec', '>f8')
        ]
    ccds = Table(np.zeros(1, dtype=cols))
    return ccds

def _stars_table(nstars=1):
    '''Initialize the stars table, which will contain information on all the stars
       detected on the CCD, including the PS1 photometry.

    '''

    cols = [('x', 'f4'), ('y', 'f4'), ('ra', 'f8'), ('dec', 'f8'), ('fwhm', 'f4'),
            ('apmag_skylocal', 'f4'), ('apmag_skyglobal', 'f4'), 
            ('ps1_ra', 'f8'), ('ps1_dec', 'f8'), ('ps1_mag', 'f4'), ('ps1_gicolor', 'f4')]
    stars = Table(np.zeros(nstars, dtype=cols))

    return stars

class Measurer(object):
    def __init__(self, fn, ext, aprad=3.5, skyrad_inner=7.0, skyrad_outer=10.0):
        '''This is the work-horse class which operates on a given image regardless of
        its origin (decam, mosaic, 90prime).

        Args:

        aprad: float
        Aperture photometry radius in arcsec

        skyrad_{inner,outer}: floats
        Sky annulus radius in arcsec

        Returns:


        Raises:

        '''

        self.fn = fn
        self.ext = ext
        
        self.aprad = aprad
        self.skyrad = (skyrad_inner, skyrad_outer)

        # Set the nominal detection FWHM (in pixels) and detection threshold.
        self.nominal_fwhm = 5.0 # [pixels]
        self.det_thresh = 10    # [S/N] - used to be 20

        self.matchradius = 3.0  # search radius for finding matching PS1 stars [arcsec]

        # Read the primary header and the header for this extension.
        self.primhdr = fitsio.read_header(fn, ext=0)
        self.hdr = fitsio.read_header(fn, ext=ext)

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band

    def zeropoint(self, band):
        return self.zp0[band]

    def sky(self, band):
        return self.sky0[band]

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

    def run(self, ps=None):

        # Read the image and header.  Not sure what units the images are in... 
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)

        # Initialize and begin populating the output CCDs table.
        ccds = _ccds_table()
        ccds['image_filename'] = self.fn
        ccds['expnum'] = self.expnum
        ccds['filter'] = self.band
        ccds['propid'] = self.propid
        ccds['arawgain'] = self.gain
        ccds['ccdname'] = self.ccdname
        ccds['ccdnum'] = self.ccdnum
        ccds['image_hdu'] = self.image_hdu
        ccds['expid'] = self.expid
        ccds['avsky'] = self.avsky

        # Copy some header cards directly.
        hdrkey = ('object', 'exptime', 'date-obs', 'mjd_obs', 'time-obs', 'ha',
                  'airmass','crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                  'cd1_2', 'cd2_1', 'cd2_2', 'naxis1', 'naxis2', 'crval1', 'crval2')
        ccdskey = ('object', 'exptime', 'date_obs', 'mjd_obs', 'ut', 'ha',
                   'airmass', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                   'cd1_2', 'cd2_1', 'cd2_2', 'width', 'height', 'ra_bore', 'dec_bore')
        for ckey, hkey in zip(ccdskey, hdrkey):
            ccds[ckey] = hdr[hkey]

        print('Band {}, Exptime {}, Airmass {}'.format(self.band,
                                                       ccds['exptime'].data[0],
                                                       ccds['airmass'].data[0]))

        # Get the ra, dec coordinates at the center of the chip.
        H, W = img.shape
        ccdra, ccddec = self.wcs.pixelxy2radec((W+1) / 2.0, (H + 1) / 2.0)
        ccds['ra'] = ccdra
        ccds['dec'] = ccddec

        # Measure the sky brightness and (sky) noise level.  Need to capture
        # negative sky.
        zp0 = self.zeropoint(self.band)
        sky0 = self.sky(self.band)

        print('Computing the sky image.')
        exptime = ccds['exptime'].data[0]
        sky, sig1 = self.get_sky_and_sigma(img)
        sky1 = np.median(sky)
        skybr = zp0 - 2.5*np.log10(sky1/self.pixscale/self.pixscale/exptime)

        print('  Sky brightness: {:.3f} mag/arcsec^2'.format(skybr))
        print('  Fiducial:       {:.3f} mag/arcsec^2'.format(sky0))

        ccds['ccdskyrms'] = sig1    # [electron/s/pix]
        ccds['ccdskycounts'] = sky1 # [electron/s/pix]
        ccds['ccdskymag'] = skybr   # [mag/arcsec^2]

        # Detect stars on the image.  
        det_thresh = self.det_thresh
        obj = daofind(img, fwhm=self.nominal_fwhm,
                      threshold=det_thresh*sig1,
                      exclude_border=True)
        if len(obj) < 20:
            det_thresh = self.det_thresh / 2.0
            obj = daofind(img, fwhm=self.nominal_fwhm,
                          threshold=det_thresh*sig1,
                          exclude_border=True)
        nobj = len(obj)
        ccds['ccdnstar'] = nobj
        print('{} sources detected with detection threshold {}'.format(nobj, det_thresh))

        if nobj == 0:
            print('No sources detected!  Giving up.')
            return ccds, _stars_table()

        # Aperture photometry here?  So we can cut on the uncertainties...?

        # Read and match all PS1 stars on this CCD.
        ps1 = ps1cat(ccdwcs=self.wcs).get_stars()
        nps1 = len(ps1)

        if nps1 == 0:
            print('No overlapping PS1 stars in this field!')
            return ccds, _stars_table()

        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid'], obj['ycentroid'])
        m1, m2, d12 = match_radec(objra, objdec, ps1.ra, ps1.dec, self.matchradius/3600.0)
        nmatch = len(m1)
        ccds['ccdnmatch'] = nmatch
        
        print('{} PS1 stars match detected sources within {} arcsec.'.format(nmatch, self.matchradius))

        # Initialize the stars table and begin populating it.
        stars = _stars_table(nmatch)
        stars['x'] = obj['xcentroid'][m1]
        stars['y'] = obj['ycentroid'][m1]
        stars['ra'] = objra[m1]
        stars['dec'] = objdec[m1]

        stars['ps1_ra'] = ps1.ra[m2]
        stars['ps1_dec'] = ps1.dec[m2]
        stars['ps1_gicolor'] = ps1.median[m2, 0] - ps1.median[m2, 2]

        ps1band = ps1cat.ps1band[self.band]
        stars['ps1_mag'] = ps1.median[m2, ps1band]

        # Compute the astrometric residuals relative to PS1.
        raoff = np.median((stars['ra'] - stars['ps1_ra']) * np.cos(np.deg2rad(ccddec)) * 3600.0)
        decoff = np.median((stars['dec'] - stars['ps1_dec']) * 3600.0)
        ccds['ccdraoff'] = raoff
        ccds['ccddecoff'] = decoff
        print('Median offsets (arcsec) relative to PS1: dra = {}, ddec = {}'.format(raoff, decoff))

        # Do aperture photometry in a fixed aperture but with two different
        # types of sky-subtraction -- local and global.
        print('Aperture photometry')
        aprad_pix = self.aprad / self.pixscale
        sky_inner_pix, sky_outer_pix = [skyr / self.pixscale for skyr in self.skyrad]

        ap = CircularAperture((stars['x'], stars['y']), aprad_pix)
        skyap = CircularAnnulus((stars['x'], stars['y']), r_in=sky_inner_pix, r_out=sky_outer_pix)
        apphot_global = aperture_photometry(img - sky, ap)
        apphot_local = aperture_photometry(img, ap)
        skyphot_local = aperture_photometry(img, skyap)

        apflux_skyglobal = apphot_global['aperture_sum']
        apflux_skylocal = apphot_local['aperture_sum'] - skyphot_local['aperture_sum'] / skyap.area() * ap.area()

        stars['apmag_skyglobal'] = - 2.5 * np.log10(apflux_skyglobal) + zp0 + 2.5 * np.log10(exptime)
        stars['apmag_skylocal'] = - 2.5 * np.log10(apflux_skylocal) + zp0 + 2.5 * np.log10(exptime)

        # Compute the photometric zeropoint.  Only use stars with main sequence
        # g-i colors.
        print('Computing the photometric zeropoint.')
        mskeep = (stars['ps1_gicolor'] > 0.4) * (stars['ps1_gicolor'] < 2.7)
        nms = np.sum(mskeep*1)
        if nms == 0:
            print('Not enough PS1 stars with main sequence colors.')
            return ccds, stars

        # Get the photometric offset relative to PS1 as the observed PS1
        # magnitude minus the observed / measured magnitude.
        dmag_skylocal = stars['ps1_mag'] - stars['apmag_skylocal']
        dmag_skyglobal = stars['ps1_mag'] - stars['apmag_skyglobal']

        pdb.set_trace()

        colorterm = self.colorterm_ps1_to_observed(stars.median, self.band)
        stars.mag += colorterm
        ps1mag = stars.mag[I]

        dm = ps1mag - apmag[J]
        dmag,dsig = sensible_sigmaclip(dm, nsigma=2.5)
        print('Mag offset: %8.3f' % dmag)
        print('Scatter:    %8.3f' % dsig)

        if not np.isfinite(dmag) or not np.isfinite(dsig):
            print('FAILED TO GET ZEROPOINT!')
            meas.update(zp=None)
            return ccds, stars

        from scipy.stats import sigmaclip
        goodpix,lo,hi = sigmaclip(dm, low=3, high=3)
        dmagmed = np.median(goodpix)
        print(len(goodpix), 'stars used for zeropoint median')
        print('Using median zeropoint:')
        zp_med = zp0 + dmagmed
        trans_med = 10.**(-0.4 * (zp0 - zp_med - kx * (airmass - 1.)))
        print('Zeropoint %6.3f' % zp_med)
        print('Transparency: %.3f' % trans_med)
        
        dm = ps1mag - apmag2[J]
        dmag2,dsig2 = sensible_sigmaclip(dm, nsigma=2.5)
        #print('Sky-sub mag offset', dmag2)
        #print('Scatter', dsig2)

        # Median, sky-sub
        goodpix,lo,hi = sigmaclip(dm, low=3, high=3)
        dmagmedsky = np.median(goodpix)

        zp_mean = zp0 + dmag

        zp_obs = zp0 + dmagmedsky
        transparency = 10.**(-0.4 * (zp0 - zp_obs - kx * (airmass - 1.)))
        meas.update(zp=zp_obs, transparency=transparency)

        print('Zeropoint %6.3f' % zp_obs)
        print('Fiducial  %6.3f' % zp0)
        print('Transparency: %.3f' % transparency)


        fwhms = []
        psf_r = 15
        if n_fwhm not in [0, None]:
            Jf = J[:n_fwhm]
            
        for i,(xi,yi,fluxi) in enumerate(zip(fx[Jf],fy[Jf],apflux[Jf])):
            #print('Fitting source', i, 'of', len(Jf))
            ix = int(np.round(xi))
            iy = int(np.round(yi))
            xlo = max(0, ix-psf_r)
            xhi = min(W, ix+psf_r+1)
            ylo = max(0, iy-psf_r)
            yhi = min(H, iy+psf_r+1)
            xx,yy = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
            r2 = (xx - xi)**2 + (yy - yi)**2
            keep = (r2 < psf_r**2)
            pix = img[ylo:yhi, xlo:xhi].copy()
            ie = np.zeros_like(pix)
            ie[keep] = 1. / sig1
            #print('fitting source at', ix,iy)
            #print('number of active pixels:', np.sum(ie > 0), 'shape', ie.shape)

            psf = tractor.NCircularGaussianPSF([4.], [1.])
            tim = tractor.Image(data=pix, inverr=ie, psf=psf)
            src = tractor.PointSource(tractor.PixPos(xi-xlo, yi-ylo),
                                      tractor.Flux(fluxi))
            tr = tractor.Tractor([tim],[src])
    
            #print('Posterior before prior:', tr.getLogProb())
            src.pos.addGaussianPrior('x', 0., 1.)
            #print('Posterior after prior:', tr.getLogProb())
            
            doplot = (i < 5) * (ps is not None)
            if doplot:
                mod0 = tr.getModelImage(0)
    
            tim.freezeAllBut('psf')
            psf.freezeAllBut('sigmas')
    
            # print('Optimizing params:')
            # tr.printThawedParams()
    
            #print('Parameter step sizes:', tr.getStepSizes())
            optargs = dict(priors=False, shared_params=False)
            for step in range(50):
                dlnp,x,alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break
            # Now fit only the PSF size
            tr.freezeParam('catalog')
            # print('Optimizing params:')
            # tr.printThawedParams()
    
            for step in range(50):
                dlnp,x,alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break
    
            fwhms.append(psf.sigmas[0] * 2.35 * pixsc)

        fwhms = np.array(fwhms)
        medfwhm = np.median(fwhms)
        print('Median FWHM: {:.3f}'.format(medfwhm))
        ccds['seeing'] = medfwhm

        return ccds, stars


class DECamMeasurer(Measurer):
    def __init__(self, *args, **kwargs):
        if not 'pixscale' in kwargs:
            import decam
            kwargs.update(pixscale = decam.decam_nominal_pixscale)
        super(DECamMeasurer, self).__init__(*args, **kwargs)
        self.camera = 'decam'

    def read_raw(self, F, ext):
        return read_raw_decam(F, ext)

    def get_sky_and_sigma(self, img):
        sky,sig1 = sensible_sigmaclip(img[1500:2500, 500:1000])
        return sky,sig1

    def get_wcs(self, hdr):
        from astrometry.util.util import wcs_pv2sip_hdr
        # HACK -- convert TPV WCS header to SIP.
        wcs = wcs_pv2sip_hdr(hdr)
        #print('Converted WCS to', wcs)
        return wcs

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_decam
        return ps1_to_decam(ps1stars, band)

class DECamCPMeasurer(DECamMeasurer):
    def read_raw(self, F, ext):
        img = F[ext].read()
        hdr = F[ext].read_header()
        img = img.astype(np.float32)
        return img,hdr

class Mosaic3Measurer(Measurer):
    def __init__(self, *args, **kwargs):
        if not 'pixscale' in kwargs:
            import mosaic
            kwargs.update(pixscale = mosaic.mosaic_nominal_pixscale)
        super(Mosaic3Measurer, self).__init__(*args, **kwargs)
        self.camera = 'mosaic3'

    def get_band(self, primhdr):
        band = super(Mosaic3Measurer,self).get_band(primhdr)
        # "zd" -> "z"
        return band[0]

    def get_sky_and_sigma(self, img):
        # Spline sky model to handle (?) ghost / pupil?
        from tractor.splinesky import SplineSky

        splinesky = SplineSky.BlantonMethod(img, None, 256)
        skyimg = np.zeros_like(img)
        splinesky.addTo(skyimg)
        
        mnsky,sig1 = sensible_sigmaclip(img - skyimg)
        return skyimg,sig1

    def remove_sky_gradients(self, img):
        pass

    def get_wcs(self, hdr):
        # Older images have ZPX, newer TPV.
        if hdr['CTYPE1'] == 'RA---TPV':
            from astrometry.util.util import wcs_pv2sip_hdr
            wcs = wcs_pv2sip_hdr(hdr)
        else:
            from astrometry.util.util import Tan
            hdr['CTYPE1'] = 'RA---TAN'
            hdr['CTYPE2'] = 'DEC--TAN'
            wcs = Tan(hdr)
        return wcs

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.'''

    def __init__(self, *args, **kwargs):
        super(NinetyPrimeMeasurer, self).__init__(*args, **kwargs)
        
        self.camera = '90prime'
        self.propid = 'BASS'
        self.expnum = np.int32(self.fn[2:10])

        self.ccdname = self.hdr['EXTNAME'].strip()
        self.ccdnum = self.ccdname[-1]
        self.image_hdu = np.int(self.ccdnum)
        self.expid = '{:08d}-{}'.format(self.expnum, self.ccdname)

        self.band = self.get_band()
        self.avsky = self.hdr['SKADU'] # Right units?!?

        self.wcs = wcs_pv2sip_hdr(self.hdr) # PV distortion
        self.pixscale = self.wcs.pixel_scale()
        #self.pixscale = 0.445 # Check this!

        # Nominal zeropoints, sky brightness, and gains.  The gain is a hack
        # since this information should be scraped from the headers.
        self.zp0 = dict(g = 25.5, r = 25.5)
        self.sky0 = dict(g = 22.0, r = 21.0)

        gaindict = dict(ccd1 = 1.47, ccd2 = 1.48, ccd3 = 1.42, ccd4 = 1.4275)
        self.gain = gaindict[self.ccdname.lower()]

        # Eventually we would like FWHM to not come from SExtractor.
        self.fwhm = 2.35*self.hdr['SEEING']/self.pixscale  # [FWHM, pixels]

        # Ambient temperature
        self.temp = -999.0 # no data

    def get_sky_and_sigma(self, img):
        '''Consider doing just a simple median sky'''
        from tractor.splinesky import SplineSky
        
        splinesky = SplineSky.BlantonMethod(img, None, 256)
        skyimg = np.zeros_like(img)
        splinesky.addTo(skyimg)
        
        mnsky,sig1 = sensible_sigmaclip(img - skyimg)
        return skyimg, sig1

    def remove_sky_gradients(self, img):
        pass

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

def measure_mosaic3(fn, ext='im4', nom=None, ps=None, **kwargs):
    if nom is None:
        import mosaic
        nom = mosaic.MosaicNominalCalibration()
    meas = Mosaic3Measurer(fn, ext, nom, **measargs)
    results = meas.run(ps, **kwargs)
    return results

def measure_90prime(fn, ext='CCD1', aprad=default_aprad, ps=None):
    measure = NinetyPrimeMeasurer(fn, ext, default_aprad)
    return measure.run(ps)

def camera_name(primhdr):
    '''
    Returns 'mosaic3', 'decam', or '90prime'
    '''
    return primhdr.get('INSTRUME','').strip().lower()
    
def sensible_sigmaclip(arr, nsigma = 4.):
    '''sigmaclip returns unclipped pixels, lo,hi, where lo,hi are the
      mean(goodpix) +- nsigma * sigma

    '''
    from scipy.stats import sigmaclip
    goodpix,lo,hi = sigmaclip(arr, low=nsigma, high=nsigma)
    meanval = np.mean(goodpix)
    sigma = (meanval - lo) / nsigma
    return meanval, sigma

def measure_image(fn, aprad=default_aprad, ps=None):

    primhdr = fitsio.read_header(fn)
    cam = camera_name(primhdr)

    if cam == 'decam':
        # CP-processed DECam image
        import decam
        nom = decam.DecamNominalCalibration()
        ext = kwargs.pop('ext')
        meas = DECamCPMeasurer(fn, ext, nom)
        result = meas.run(**kwargs)

    elif cam == 'mosaic3':
        result = measure_mosaic3(fn)

    elif cam == '90prime':
        results = []
        for ext in ('CCD1', 'CCD1', 'CCD1', 'CCD4'):
            results.append(measure_90prime(fn, ext, ps=ps))
        result = merge_tables(results)

    # Compute the median zeropoint across all the CCDs.
    result['zpt'] = np.median(result['ccdzpt'])
    return result

def main():

    '''

    python ~/repos/git/legacysurvey/py/legacyzeropoints/bass-ccds.py p_7402_0038_bokr.fits

    '''

    parser = argparse.ArgumentParser(description='Generate the legacypipe-compatible CCDs file from reduced BASS data.')
    parser.add_argument('-o', '--outfile', type=str, default='bass-ccds.fits', help='Output file name.')
    parser.add_argument('--aprad', type=float, default=default_aprad, help='Aperture photometry radius (arcsec).')
    parser.add_argument('--plots', help='Make plots with this fn prefix', default=None)
    parser.add_argument('images', metavar='DECam-filename.fits.fz', nargs='+',
                        help='BASS image to process')

    args = parser.parse_args()
    allfiles = args.images

    # Include this, or generate plots after the fact?
    ps = None
    if args.plots is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(args.plots)

    # Initialize the CCDs file here.
    
    # Push this to multiprocessing, one image per core.
    for thisfile in allfiles:
        print('Working on image {}'.format(thisfile))
        res = measure_image(thisfile, aprad=args.aprad, ps=ps)

        # Pack the dictionary results into the output table... 

    # Write out.
    pdb.set_trace()

    T = fits_table()
    for k,v in vals.items():
        if k == 'wcs':
            continue
        T.set(k, v)
    T.to_np_arrays()
    for k in T.columns():
        v = T.get(k)
        if v.dtype == np.float64:
            T.set(k, v.astype(np.float32))
    print('Writing {}'.format(outfile))
    T.writeto(args.outfile)
                    
if __name__ == "__main__":
    main()
