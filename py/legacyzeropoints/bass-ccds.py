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

"""
from __future__ import division, print_function

import os
import pdb
import argparse

import numpy as np
from glob import glob

import fitsio
import photutils
from astropy.io import fits
import matplotlib.pyplot as plt

from photutils import CircularAperture, CircularAnnulus, aperture_photometry

from astrometry.util.fits import merge_tables
from astrometry.util.plotutils import dimshow, plothist

from legacyanalysis.ps1cat import ps1_to_decam, ps1cat
# Color terms -- no MOSAIC specific ones yet:
ps1_to_mosaic = ps1_to_decam

default_aprad = 3.5 # default aperture radius [3.5 arcsec]

class Measurer(object):
    def __init__(self, fn, ext, aprad=default_aprad, skyrad_inner=7.0, skyrad_outer=10.0,
                 minstar=5, pixscale=0.262, maxshift=120.0):
        '''
        aprad: float
        Aperture photometry radius in arcsec

        skyrad_{inner,outer}: floats
        Sky annulus radius in arcsec

        nom: nominal calibration, eg NominalCalibration object
        '''

        self.fn = fn
        self.ext = ext
        
        self.aprad = aprad
        self.skyrad = (skyrad_inner, skyrad_outer)
        self.minstar = minstar
        self.pixscale = pixscale
        self.maxshift = maxshift

        self.nominal_fwhm = 5.0
        # Detection threshold
        self.det_thresh = 20

        self.debug = True

        self.camera = 'camera'
        
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

    def read_raw(self, F, ext):
        img = F[ext].read()
        hdr = F[ext].read_header()
        return img,hdr

    def get_band(self, primhdr):
        band = primhdr['FILTER']
        band = band.strip().split()[0]
        return band

    def match_ps1_stars(self, px, py, fullx, fully, radius, stars):
        from astrometry.libkd.spherematch import match_xy
        #print('Matching', len(px), 'PS1 and', len(fullx), 'detected stars with radius', radius)
        I,J,d = match_xy(px, py, fullx, fully, radius)
        #print(len(I), 'matches')
        dx = px[I] - fullx[J]
        dy = py[I] - fully[J]
        return I,J,dx,dy

    def detection_map(self, img, sig1, psfsig):
        '''Generate a detection map.'''
        from scipy.ndimage.filters import gaussian_filter

        psfnorm = 1.0/(2.0 * np.sqrt(np.pi) * psfsig)
        detsn = gaussian_filter(img / sig1, psfsig) / psfnorm
        # zero out the edges -- larger margin here?
        detsn[0 ,:] = 0
        detsn[:, 0] = 0
        detsn[-1,:] = 0
        detsn[:,-1] = 0
        return detsn

    def detect_sources(self, detsn, thresh):
        '''Detect peaks (sources) from the detection map.'''
        from scipy.ndimage.measurements import label, find_objects

        # HACK -- Just keep the brightest pixel in each blob!
        peaks = (detsn > thresh)
        blobs,nblobs = label(peaks)
        slices = find_objects(blobs)
        return slices
        
    def run(self, ps=None, momentsize=5, n_fwhm=100):

        fn = self.fn
        ext = self.ext
        pixsc = self.pixscale

        # Not sure what units the images are in...
        primhdr = fitsio.read_header(fn, ext=0)
        img, hdr = fitsio.read(fn, ext=ext, header=True)

        self.primhdr = primhdr
        self.hdr = hdr

        fullH, fullW = img.shape

        # Get the pre sky-subtracted percentile range of pixel values.
        mn, mx = np.percentile(img.ravel(), [25, 98])
        self.imgkwa = dict(vmin=mn, vmax=mx, cmap='gray')
        
        if self.debug and ps is not None:
            plt.clf()
            dimshow(img, **self.imgkwa)
            plt.title('Raw image')
            ps.savefig()
    
            M = 200
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(img[-M:, :M], ticks=False, **self.imgkwa)
            plt.subplot(2,2,2)
            dimshow(img[-M:, -M:], ticks=False, **self.imgkwa)
            plt.subplot(2,2,3)
            dimshow(img[:M, :M], ticks=False, **self.imgkwa)
            plt.subplot(2,2,4)
            dimshow(img[:M, -M:], ticks=False, **self.imgkwa)
            plt.suptitle('Raw image corners')
            ps.savefig()

        if self.debug and ps is not None:
            plt.clf()
            dimshow(img, **self.imgkwa)
            plt.title('Trimmed image')
            ps.savefig()
    
            M = 200
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(img[-M:, :M], ticks=False, **self.imgkwa)
            plt.subplot(2,2,2)
            dimshow(img[-M:, -M:], ticks=False, **self.imgkwa)
            plt.subplot(2,2,3)
            dimshow(img[:M, :M], ticks=False, **self.imgkwa)
            plt.subplot(2,2,4)
            dimshow(img[:M, -M:], ticks=False, **self.imgkwa)
            plt.suptitle('Trimmed corners')
            ps.savefig()
            
        band = self.get_band(primhdr)
        exptime = primhdr['EXPTIME']
        airmass = primhdr['AIRMASS']
        print('Band {}, Exptime {}, Airmass {}'.format(band, exptime, airmass))

        # Find the sky value and noise level
        sky, sig1 = self.get_sky_and_sigma(img)
        img -= sky

        # Compare this with what's in the header...
        sky1 = np.median(sky)

        #if zp0 is not None:
        #    skybr = -2.5 * np.log10(sky1/pixsc/pixsc/exptime) + zp0
        #    print('Sky brightness: %8.3f mag/arcsec^2' % skybr)
        #    print('Fiducial:       %8.3f mag/arcsec^2' % sky0)
        #else:
        #    skybr = None

        # Read WCS header and compute boresight
        wcs = self.get_wcs(hdr)
        ra_ccd, dec_ccd = wcs.pixelxy2radec((fullW + 1) / 2.0, (fullH + 1) / 2.0)

        # Should probably initialize the CCDs Table above and pack everything
        # into it, rather than generating a dictionary here.
        skybr = None
        meas = dict(band=band, airmass=airmass, exptime=exptime,
                    skybright=skybr, rawsky=sky1,
                    pixscale=pixsc, primhdr=primhdr,
                    hdr=hdr, wcs=wcs, ra_ccd=ra_ccd, dec_ccd=dec_ccd,
                    extension=ext, camera=self.camera)

        # Consider whether this is necessary...
        self.remove_sky_gradients(img)

        # Get the post-sky-subtracted percentile range of pixel values.
        mn, mx = np.percentile(img.ravel(), [25, 98])
        self.imgkwa = dict(vmin=mn, vmax=mx, cmap='gray')

        if ps is not None:
            plt.clf()
            dimshow(img, **self.imgkwa)
            plt.title('Sky-sub image: %s-%s' % (os.path.basename(fn).replace('.fits','').replace('.fz',''), ext))
            plt.colorbar()
            ps.savefig()

        # Detect stars
        psfsig = self.nominal_fwhm / 2.35
        detsn = self.detection_map(img, sig1, psfsig)

        det_thresh = self.det_thresh
        slices = self.detect_sources(detsn, det_thresh)
        if len(slices) < 20:
            det_thresh = self.det_thresh / 2.0
            slices = self.detect_sources(detsn, det_thresh)
        ndetected = len(slices)
        print('{} sources detected with detection threshold {}'.format(ndetected, det_thresh))

        #meas.update(ndetected=ndetected)
        if ndetected == 0:
            print('NO SOURCES DETECTED')
            pdb.set_trace()
            #return meas

        # TODO: throw out stars that are either masked or are in crowded positions!

        # Read in the PS1 catalog, and keep those within 0.25 deg of CCD center
        # and those with main sequence colors
        pscat = ps1cat(ccdwcs=wcs)
        stars = pscat.get_stars()
        #print('Got PS1 stars:', len(stars))
    
        # we add the color term later
        try:
            ps1band = ps1cat.ps1band[band]
            stars.mag = stars.median[:, ps1band]
        except KeyError:
            ps1band = None
            stars.mag = np.zeros(len(stars), np.float32)
        
        ok, px, py = wcs.radec2pixelxy(stars.ra, stars.dec)
        px -= 1
        py -= 1

        pdb.set_trace()
        
        if ps is not None:
            #kwa = dict(vmin=-3*sig1, vmax=50*sig1, cmap='gray')
            # Add to the 'detected sources' plot
            # mn,mx = np.percentile(img.ravel(), [50,99])
            # kwa = dict(vmin=mn, vmax=mx, cmap='gray')
            # plt.clf()
            # dimshow(img, **kwa)
            ax = plt.axis()
            #plt.plot(fx, fy, 'go', mec='g', mfc='none', ms=10)
            K = np.argsort(stars.mag)
            plt.plot(px[K[:10]]-trim_x0, py[K[:10]]-trim_y0, 'o', mec='m', mfc='none',ms=12,mew=2)
            plt.plot(px[K[10:]]-trim_x0, py[K[10:]]-trim_y0, 'o', mec='m', mfc='none', ms=8)
            plt.axis(ax)
            plt.title('PS1 stars')
            #plt.colorbar()
            ps.savefig()

        # Match PS1 to our detections, find offset
        radius = self.maxshift / pixsc

        I,J,dx,dy = self.match_ps1_stars(px, py, fullx, fully, radius, stars)
        print(len(I), 'spatial matches with large radius', self.maxshift,
              'arcsec,', radius, 'pix')

        pdb.set_trace()
        


        bins = 2*int(np.ceil(radius))
        #print('Histogramming with', bins, 'bins')
        histo,xe,ye = np.histogram2d(dx, dy, bins=bins,
                                     range=((-radius,radius),(-radius,radius)))
        # smooth histogram before finding peak -- fuzzy matching
        from scipy.ndimage.filters import gaussian_filter
        histo = gaussian_filter(histo, 1.)
        histo = histo.T
        mx = np.argmax(histo)
        my,mx = np.unravel_index(mx, histo.shape)
        shiftx = (xe[mx] + xe[mx+1])/2.
        shifty = (ye[my] + ye[my+1])/2.
        
        if ps is not None:
            plt.clf()
            plothist(dx, dy, range=((-radius,radius),(-radius,radius)))
            plt.xlabel('dx (pixels)')
            plt.ylabel('dy (pixels)')
            plt.title('Offsets to PS1 stars')
            ax = plt.axis()
            plt.axhline(0, color='b')
            plt.axvline(0, color='b')
            plt.plot(shiftx, shifty, 'o', mec='m', mfc='none', ms=15, mew=3)
            plt.axis(ax)
            ps.savefig()
    
        # Refine with smaller search radius
        radius2 = 3. / pixsc
        I,J,dx,dy = self.match_ps1_stars(px, py, fullx+shiftx, fully+shifty,
                                         radius2, stars)
        print(len(J), 'matches to PS1 with small radius', 3, 'arcsec')
        shiftx2 = np.median(dx)
        shifty2 = np.median(dy)
        #print('Stage-1 shift', shiftx, shifty)
        #print('Stage-2 shift', shiftx2, shifty2)
        sx = shiftx + shiftx2
        sy = shifty + shifty2
        print('Astrometric shift (%.0f, %.0f) pixels' % (sx,sy))

        #from astromalign.astrom_common import Alignment

        # Find affine transformation
        cx = fullW/2
        cy = fullH/2

        A = np.zeros((len(J), 3))
        A[:,0] = 1.
        A[:,1] = (fullx[J] + sx) - cx
        A[:,2] = (fully[J] + sy) - cy

        R = np.linalg.lstsq(A, px[I] - cx)
        resx = R[0]
        print('Affine transformation for X:', resx)
        R = np.linalg.lstsq(A, py[I] - cy)
        resy = R[0]
        print('Affine transformation for Y:', resy)

        meas.update(affine = [cx, cy] + list(resx) + list(resy))
        
        if True:
            r0,d0 = stars.ra[I[0]], stars.dec[I[0]]
            print('Pan-STARRS RA,Dec:', r0,d0)
            ok,px0,py0 = wcs.radec2pixelxy(r0, d0)
            print('Header WCS -> x,y:', px0-1, py0-1)
            mx0,my0 = fullx[J[0]], fully[J[0]]
            print('VS Matched star x,y:', mx0, my0)
        
        if self.debug and ps is not None:
            plt.clf()
            plothist(dx, dy, range=((-radius2,radius2),(-radius2,radius2)))
            plt.xlabel('dx (pixels)')
            plt.ylabel('dy (pixels)')
            plt.title('Offsets to PS1 stars')
            ax = plt.axis()
            plt.axhline(0, color='b')
            plt.axvline(0, color='b')
            plt.plot(shiftx2, shifty2, 'o', mec='m', mfc='none', ms=15, mew=3)
            plt.axis(ax)
            ps.savefig()

        if ps is not None:
            afx = (fullx[J] + sx) - cx
            afy = (fully[J] + sy) - cy
            affx = cx + (resx[0] + resx[1] * afx + resx[2] * afy)
            affy = cy + (resy[0] + resy[1] * afx + resy[2] * afy)
            affdx = px[I] - affx
            affdy = py[I] - affy
            plt.clf()
            plothist(affdx, affdy, range=((-radius2,radius2),(-radius2,radius2)))
            plt.xlabel('dx (pixels)')
            plt.ylabel('dy (pixels)')
            plt.title('Offsets to PS1 stars after Affine correction')
            ax = plt.axis()
            plt.axhline(0, color='b')
            plt.axvline(0, color='b')
            #plt.plot(shiftx, shifty, 'o', mec='m', mfc='none', ms=15, mew=3)
            plt.axis(ax)
            ps.savefig()
        
        if ps is not None:
            mn,mx = np.percentile(img.ravel(), [50,99])
            kwa2 = dict(vmin=mn, vmax=mx, cmap='gray')
            plt.clf()
            dimshow(img, **kwa2)
            ax = plt.axis()
            plt.plot(fx[J], fy[J], 'go', mec='g', mfc='none', ms=10, mew=2)
            plt.plot(px[I]-sx-trim_x0, py[I]-sy-trim_y0, 'm+', ms=10, mew=2)
            plt.axis(ax)
            plt.title('Matched PS1 stars')
            plt.colorbar()
            ps.savefig()

            plt.clf()
            dimshow(img, **kwa2)
            ax = plt.axis()
            plt.plot(fx[J], fy[J], 'go', mec='g', mfc='none', ms=10, mew=2)
            K = np.argsort(stars.mag)
            plt.plot(px[K[:10]]-sx-trim_x0, py[K[:10]]-sy-trim_y0, 'o', mec='m', mfc='none',ms=12,mew=2)
            plt.plot(px[K[10:]]-sx-trim_x0, py[K[10:]]-sy-trim_y0, 'o', mec='m', mfc='none', ms=8,mew=2)
            plt.axis(ax)
            plt.title('All PS1 stars')
            plt.colorbar()
            ps.savefig()

        # Perform aperture photometry on each star.
        apxy = np.vstack((fx, fy)).T
        ap = []
        aprad_pix = self.aprad / pixsc
        aper = photutils.CircularAperture(apxy, aprad_pix)
        p = photutils.aperture_photometry(img, aper)
        apflux = p.field('aperture_sum')
    
        # Manual aperture photometry to get clipped means in sky annulus
        sky_inner_r, sky_outer_r = [r / pixsc for r in self.skyrad]
        sky = []
        for xi,yi in zip(fx,fy):
            ix = int(np.round(xi))
            iy = int(np.round(yi))
            skyR = int(np.ceil(sky_outer_r))
            xlo = max(0, ix-skyR)
            xhi = min(W, ix+skyR+1)
            ylo = max(0, iy-skyR)
            yhi = min(H, iy+skyR+1)
            xx,yy = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
            r2 = (xx - xi)**2 + (yy - yi)**2
            inannulus = ((r2 >= sky_inner_r**2) * (r2 < sky_outer_r**2))
            skypix = img[ylo:yhi, xlo:xhi][inannulus]
            #print('ylo,yhi, xlo,xhi', ylo,yhi, xlo,xhi, 'img subshape', img[ylo:yhi, xlo:xhi].shape, 'inann shape', inannulus.shape)
            s,nil = sensible_sigmaclip(skypix)
            sky.append(s)
        sky = np.array(sky)

        apflux2 = apflux - sky * (np.pi * aprad_pix**2)
        good = (apflux2>0)*(apflux>0)
        apflux = apflux[good]
        apflux2 = apflux2[good]
        fx = fx[good]
        fy = fy[good]


            
        # Now cut to just *stars* with good colors
        stars.gicolor = stars.median[:,0] - stars.median[:,2]
        keep = (stars.gicolor > 0.4) * (stars.gicolor < 2.7)
        stars.cut(keep)
        if len(stars) == 0:
            print('No overlap or too few stars in PS1')
            return None
        px = px[keep]
        py = py[keep]
        # Re-match
        I,J,dx,dy = self.match_ps1_stars(px, py, fullx+sx, fully+sy,
                                         radius2, stars)
        print('Cut to', len(stars), 'PS1 stars with good colors; matched', len(I))

        nmatched = len(I)

        meas.update(dx=sx, dy=sy, nmatched=nmatched)

        # Compute photometric offset compared to PS1
        # as the PS1 minus observed mags
        colorterm = self.colorterm_ps1_to_observed(stars.median, band)
        stars.mag += colorterm
        ps1mag = stars.mag[I]
        
        if False and ps is not None:
            plt.clf()
            plt.semilogy(ps1mag, apflux2[J], 'b.')
            plt.xlabel('PS1 mag')
            plt.ylabel('DECam ap flux (with sky sub)')
            ps.savefig()
        
            plt.clf()
            plt.semilogy(ps1mag, apflux[J], 'b.')
            plt.xlabel('PS1 mag')
            plt.ylabel('DECam ap flux (no sky sub)')
            ps.savefig()

        # "apmag2" is the annular sky-subtracted flux
        apmag2 = -2.5 * np.log10(apflux2) + zp0 + 2.5 * np.log10(exptime)
        apmag  = -2.5 * np.log10(apflux ) + zp0 + 2.5 * np.log10(exptime)
    
        if ps is not None:
            plt.clf()
            plt.plot(ps1mag, apmag[J], 'b.', label='No sky sub')
            plt.plot(ps1mag, apmag2[J], 'r.', label='Sky sub')
            # ax = plt.axis()
            # mn = min(ax[0], ax[2])
            # mx = max(ax[1], ax[3])
            # plt.plot([mn,mx], [mn,mx], 'k-', alpha=0.1)
            # plt.axis(ax)
            plt.xlabel('PS1 mag')
            plt.ylabel('DECam ap mag')
            plt.legend(loc='upper left')
            plt.title('Zeropoint')
            ps.savefig()
    
        dm = ps1mag - apmag[J]
        dmag,dsig = sensible_sigmaclip(dm, nsigma=2.5)
        print('Mag offset: %8.3f' % dmag)
        print('Scatter:    %8.3f' % dsig)

        if not np.isfinite(dmag) or not np.isfinite(dsig):
            print('FAILED TO GET ZEROPOINT!')
            meas.update(zp=None)
            return meas

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

        if ps is not None:
            plt.clf()
            plt.plot(ps1mag, apmag[J] + dmag - ps1mag, 'b.', label='No sky sub')
            plt.plot(ps1mag, apmag2[J]+ dmag2- ps1mag, 'r.', label='Sky sub')
            plt.xlabel('PS1 mag')
            plt.ylabel('DECam ap mag - PS1 mag')
            plt.legend(loc='upper left')
            plt.ylim(-0.25, 0.25)
            plt.axhline(0, color='k', alpha=0.25)
            plt.title('Zeropoint')
            ps.savefig()

        zp_mean = zp0 + dmag

        zp_obs = zp0 + dmagmedsky
        transparency = 10.**(-0.4 * (zp0 - zp_obs - kx * (airmass - 1.)))
        meas.update(zp=zp_obs, transparency=transparency)

        print('Zeropoint %6.3f' % zp_obs)
        print('Fiducial  %6.3f' % zp0)
        print('Transparency: %.3f' % transparency)

        # Also return the zeropoints using the sky-subtracted mags,
        # and using median rather than clipped mean.
        meas.update(zp_mean = zp_mean,
                    zp_skysub = zp0 + dmag2,
                    zp_med = zp_med,
                    zp_med_skysub = zp0 + dmagmedsky)
        

        # print('Using sky-subtracted values:')
        # zp_sky = zp0 + dmag2
        # trans_sky = 10.**(-0.4 * (zp0 - zp_sky - kx * (airmass - 1.)))
        # print('Zeropoint %6.3f' % zp_sky)
        # print('Transparency: %.3f' % trans_sky)
        
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
                
            if doplot:
                mod1 = tr.getModelImage(0)
                chi1 = tr.getChiImage(0)
            
                plt.clf()
                plt.subplot(2,2,1)
                plt.title('Image')
                dimshow(pix, **self.imgkwa)
                plt.subplot(2,2,2)
                plt.title('Initial model')
                dimshow(mod0, **self.imgkwa)
                plt.subplot(2,2,3)
                plt.title('Final model')
                dimshow(mod1, **self.imgkwa)
                plt.subplot(2,2,4)
                plt.title('Final chi')
                dimshow(chi1, vmin=-10, vmax=10)
                plt.suptitle('PSF fit')
                ps.savefig()
    
        fwhms = np.array(fwhms)
        fwhm = np.median(fwhms)
        print('Median FWHM: %.3f' % np.median(fwhms))
        meas.update(seeing=fwhm)
    
        if False and ps is not None:
            lo,hi = np.percentile(fwhms, [5,95])
            lo -= 0.1
            hi += 0.1
            plt.clf()
            plt.hist(fwhms, 25, range=(lo,hi), histtype='step', color='b')
            plt.xlabel('FWHM (arcsec)')
            ps.savefig()
    
        if ps is not None:
            plt.clf()
            for i,(xi,yi) in enumerate(zip(fx[J],fy[J])[:50]):
                ix = int(np.round(xi))
                iy = int(np.round(yi))
                xlo = max(0, ix-psf_r)
                xhi = min(W, ix+psf_r+1)
                ylo = max(0, iy-psf_r)
                yhi = min(H, iy+psf_r+1)
                pix = img[ylo:yhi, xlo:xhi]
        
                slc = pix[iy-ylo, :].copy()
                slc /= np.sum(slc)
                p1 = plt.plot(slc, 'b-', alpha=0.2)
                slc = pix[:, ix-xlo].copy()
                slc /= np.sum(slc)
                p2 = plt.plot(slc, 'r-', alpha=0.2)
                ph,pw = pix.shape
                cx,cy = pw/2, ph/2
                if i == 0:
                    xx = np.linspace(0, pw, 300)
                    dx = xx[1]-xx[0]
                    sig = fwhm / pixsc / 2.35
                    yy = np.exp(-0.5 * (xx-cx)**2 / sig**2) # * np.sum(pix)
                    yy /= (np.sum(yy) * dx)
                    p3 = plt.plot(xx, yy, 'k-', zorder=20)
            #plt.ylim(-0.2, 1.0)
            plt.legend([p1[0], p2[0], p3[0]],
                       ['image slice (y)', 'image slice (x)', 'fit'])
            plt.title('PSF fit')
            ps.savefig()

        return meas


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

    def read_raw(self, F, ext):
        '''
        F: fitsio FITS object
        '''
        img = F[ext].read()
        hdr = F[ext].read_header()
        img = img.astype(np.float32)
        #print('img qts', np.percentile(img.ravel(), [0,25,50,75,100]))
        # Subtract median overscan and multiply by gains 
        dataA = parse_section(hdr['DATASEC'], slices=True)
        biasA = parse_section(hdr['BIASSEC'], slices=True)
        gainA = hdr['GAIN']
        b = np.median(img[biasA])
        #print('subtracting bias', b)
        img[dataA] = (img[dataA] - b) * gainA
    
        # Trim the image
        trimA = parse_section(hdr['TRIMSEC'], slices=True)
        # zero out all but the trim section
        trimg = img[trimA].copy()
        img[:,:] = 0
        img[trimA] = trimg
        return img,hdr

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
        return ps1_to_mosaic(ps1stars, band)

class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.'''

    def __init__(self, filename, ext, aprad, ps):
        super(NinetyPrimeMeasurer, self).__init__(filename, ext, aprad, ps)
        
        self.pixscale = 0.445 # Check this!
        self.camera = '90prime'

    def get_band(self, primhdr):
        band = super(NinetyPrimeMeasurer, self).get_band(primhdr)
        # Don't use band[0], otherwise we'll get "bokr-->b"
        return band

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

    def get_wcs(self, hdr):
        '''Figure out why the headers give CTYPE[1,2] as RA---TAN,DEC--TAN even though
        there are PV distortion coefficients.

        ''' 
        from astrometry.util.util import wcs_pv2sip_hdr
        wcs = wcs_pv2sip_hdr(hdr)
        return wcs

    def colorterm_ps1_to_observed(self, ps1stars, band):
        return ps1_to_mosaic(ps1stars, band)


def measure_raw_decam(fn, ext='N4', nom=None, ps=None, measargs={}, **kwargs):
    '''
    Reads the given file *fn*, extension *ext*', and measures a number of
    quantities that are useful for depth estimates:

    Returns:
    --------
    dict containing:

    band : string, eg 'g'

    airmass : float

    seeing : float, FWHM in arcsec.  From Gaussian fit to PS1 stars.

    zp : float.  From comparison to PS1, using aperture photometry.

    skybright : float, magnitudes/arcsec^2, based on a sky estimate and the
        canonical zeropoint.

    transparency : float, fraction.  Atmospheric transparency, based
        on computed zeropoint versus canonical zeropoint.


    Details:
    --------

    What this function does:
        
    - read raw image, apply bias and gain to the two amps & trim

    - estimate sky level via sigma-clip of central pixels
    (y in [1500,255), x in [500,1000); this is like decstat.pro)

    -> sky brightness, assuming canonical zeropoint

    - hack: remove sky gradients (or flat / gain / bias problems) by
    subtracting first a column-wise and then row-wise median from the
    image.

    - hack: remove a 100-pixel margin

    - assume a FWHM of 5 pixels, smooth by a gaussian and detect sources
    above 20 sigma.  Within each 'blob' of pixels above threshold, take
    the max pixel as the peak, and then compute the centroid in +- 5
    pixel box around the peak.  Keep sources where the centroid is
    within 1 pixel of the peak.

    - aperture-photometer peaks with a radius of 3.5"

    - I also tried computing a sky annulus, but it doesn't seem to work
    better, so it's not used

    - Read PS1 stars, apply color cut for stars, apply color
      correction to DECam

      - Match PS1 stars to our peaks

      - Dumbly find dx,dy offset (via histogram peak)

      - Sigma-clip to find mag offset PS1 to our apmags.

      -> zeropoint & transparency (vs canonical zeropoint)

      - For each matched star, cut out a +- 15 pixel box and use the
      Tractor to fit a single-Gaussian PSF

      - Take the median of the fit FWHMs -> FWHM estimate

    '''
    if nom is None:
        import decam
        nom = decam.DecamNominalCalibration()
    meas = DECamMeasurer(fn, ext, nom, **measargs)
    results = meas.run(ps, **kwargs)
    return results

def measure_mosaic3(fn, ext='im4', nom=None, ps=None, **kwargs):
    if nom is None:
        import mosaic
        nom = mosaic.MosaicNominalCalibration()
    meas = Mosaic3Measurer(fn, ext, nom, **measargs)
    results = meas.run(ps, **kwargs)
    return results

def measure_90prime(fn, ext='CCD1', aprad=default_aprad, ps=None):
    measure = NinetyPrimeMeasurer(filename=fn, ext=ext, aprad=default_aprad, ps=ps)
    return measure.run(ps)

def camera_name(primhdr):
    '''
    Returns 'mosaic3', 'decam', or '90prime'
    '''
    return primhdr.get('INSTRUME','').strip().lower()
    
def measure_image(fn, aprad=default_aprad, ps=None):

    primhdr = fitsio.read_header(fn)
    cam = camera_name(primhdr)

    if cam == 'decam':
        # CP-processed DECam image
        import decam
        nom = decam.DecamNominalCalibration()
        ext = kwargs.pop('ext')
        meas = DECamCPMeasurer(fn, ext, nom)
        results = meas.run(**kwargs)
    elif cam == 'mosaic3':
        results = measure_mosaic3(fn)
    elif cam == '90prime':
        results = []
        for ext in ('CCD1', 'CCD1', 'CCD1', 'CCD4'):
            results.append(measure_90prime(fn, ext, ps=ps))
        return merge_tables(results)

def sensible_sigmaclip(arr, nsigma = 4.):
    from scipy.stats import sigmaclip
    goodpix,lo,hi = sigmaclip(arr, low=nsigma, high=nsigma)
    # sigmaclip returns unclipped pixels, lo,hi, where lo,hi are
    # mean(goodpix) +- nsigma * sigma
    meanval = np.mean(goodpix)
    sigma = (meanval - lo) / nsigma
    return meanval, sigma

def parse_section(s, slices=False):
    '''
    parse '[57:2104,51:4146]' into integers; also subtract 1.
    '''
    s = s.replace('[','').replace(']','').replace(',',' ').replace(':',' ')
    #print('String', s)
    i = [int(si)-1 for si in s.split()]
    assert(len(i) == 4)
    if not slices:
        return i
    slc = slice(i[2], i[3]+1), slice(i[0], i[1]+1)
    #print('Slice', slc)
    return slc

def read_raw_decam(F, ext):
    '''
    F: fitsio FITS object
    '''
    img = F[ext].read()
    hdr = F[ext].read_header()
    #print('Image type', img.dtype, img.shape)

    img = img.astype(np.float32)
    #print('Converted image to', img.dtype, img.shape)

    if False:
        mn,mx = np.percentile(img.ravel(), [25,95])
        kwa = dict(vmin=mn, vmax=mx)

        plt.clf()
        dimshow(img, **kwa)
        plt.title('Raw image')
        ps.savefig()

        M = 200
        plt.clf()
        plt.subplot(2,2,1)
        dimshow(img[-M:, :M], ticks=False, **kwa)
        plt.subplot(2,2,2)
        dimshow(img[-M:, -M:], ticks=False, **kwa)
        plt.subplot(2,2,3)
        dimshow(img[:M, :M], ticks=False, **kwa)
        plt.subplot(2,2,4)
        dimshow(img[:M, -M:], ticks=False, **kwa)
        plt.suptitle('Raw corners')
        ps.savefig()
    
    if 'DESBIAS' in hdr:
        assert(False)

    # DECam RAW image

    # Raw image size 2160 x 4146

    # Subtract median overscan and multiply by gains 
    # print('DATASECA', hdr['DATASECA'])
    # print('BIASSECA', hdr['BIASSECA'])
    # print('DATASECB', hdr['DATASECB'])
    # print('BIASSECB', hdr['BIASSECB'])
    dataA = parse_section(hdr['DATASECA'], slices=True)
    biasA = parse_section(hdr['BIASSECA'], slices=True)
    dataB = parse_section(hdr['DATASECB'], slices=True)
    biasB = parse_section(hdr['BIASSECB'], slices=True)
    gainA = hdr['GAINA']
    gainB = hdr['GAINB']
    # print('DataA', dataA)
    # print('BiasA', biasA)
    # print('DataB', dataB)
    # print('BiasB', biasB)

    if False:
        plt.clf()
        plt.plot([np.median(img[i,:] )     for i in range(100)], 'b-')
        plt.plot([np.median(img[-(i+1),:]) for i in range(100)], 'c-')
        plt.plot([np.median(img[:,i] )     for i in range(100)], 'r-')
        plt.plot([np.median(img[:,-(i+1)]) for i in range(100)], 'm-')
        plt.title('Img')
        ps.savefig()
    
        plt.clf()
        plt.plot([np.median(img[dataA][i,:] )     for i in range(100)], 'b-')
        plt.plot([np.median(img[dataA][-(i+1),:]) for i in range(100)], 'c-')
        plt.plot([np.median(img[dataA][:,i] )     for i in range(100)], 'r-')
        plt.plot([np.median(img[dataA][:,-(i+1)]) for i in range(100)], 'm-')
        plt.title('Img DataA')
        ps.savefig()
    
        plt.clf()
        plt.plot([np.median(img[dataB][i,:] )     for i in range(100)], 'b-')
        plt.plot([np.median(img[dataB][-(i+1),:]) for i in range(100)], 'c-')
        plt.plot([np.median(img[dataB][:,i] )     for i in range(100)], 'r-')
        plt.plot([np.median(img[dataB][:,-(i+1)]) for i in range(100)], 'm-')
        plt.title('Img DataB')
        ps.savefig()

    
    img[dataA] = (img[dataA] - np.median(img[biasA])) * gainA
    img[dataB] = (img[dataB] - np.median(img[biasB])) * gainB
    
    # Trim the image -- could just take the min/max of TRIMSECA/TRIMSECB...
    trimA = parse_section(hdr['TRIMSECA'], slices=True)
    trimB = parse_section(hdr['TRIMSECB'], slices=True)
    # copy the TRIM A,B sections into a new image...
    trimg = np.zeros_like(img)
    trimg[trimA] = img[trimA]
    trimg[trimB] = img[trimB]
    # ... and then cut that new image
    trim = parse_section(hdr['TRIMSEC'], slices=True)
    img = trimg[trim]
    #print('Trimmed image:', img.dtype, img.shape)

    return img,hdr
    
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
