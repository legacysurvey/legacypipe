#!/usr/bin/env python

"""Check the PSFs

Get PS1 stars for this brick:
bb = mrdfits('~/dr1/decals-bricks.fits',1)
bb = bb[where(bb.brickname eq '2402p062')]
print, bb.ra1, bb.ra2, bb.dec1, bb.dec2
       240.08380       240.33520       6.1250000       6.3750000
on edison:
pp = read_ps1cat([240.08380D,240.33520D],[6.1250000D,6.3750000D])
mwrfits, pp, 'ps1cat-all-2402p062.fits', /create

jj = mrdfits('ps1cat-all-2402p062.fits',1)
tt = mrdfits('vanilla/tractor/240/tractor-2402p062.fits',1)
spherematch, jj.ra, jj.dec, tt.ra, tt.dec, 1D/3600, m1, m2
mwrfits, jj[m1], 'ps1cat-2402p062.fits', /create

djs_plot, tt.ra, tt.dec, psym=3, ysty=3
djs_oplot, jj.ra, jj.dec, psym=6, color='orange'
djs_oplot, jj[m1].ra, jj[m1].dec, psym=6, color='blue'

"""

from __future__ import division, print_function

import os
import sys
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from analysis.ps1 import ps1cat
from tractor.psfex import PsfEx
from tractor.basics import GaussianMixtureEllipsePSF, PixelizedPSF, RaDecPos
from astrometry.util.fits import fits_table
from legacypipe.common import Decals, DecamImage

#logging.basicConfig(format='%(message)s',level=logging.INFO,stream=sys.stdout)
logging.basicConfig(format='%(message)s', level=logging.DEBUG, stream=sys.stdout)
log = logging.getLogger('check-psfs')

def psf_residuals(expnum,ccdname)

    stampsize = 25
    
    decals = Decals()
    ccd = decals.find_ccds(expnum=expnum,ccdname=ccdname)[0]
    ccd.about()

    print('CCD center:', ccd.ra, ccd.dec)

    # For a given CCD:
    # - find PS1 stars
    # - for each PS1 star:
    #     - create tractor Image object in a teeny patch around this star
    #       (for only this one CCD)
    #     - create PointSource object with PS1 RA,Dec,flux
    #     - tractor.optimize with DR1-style PSF -> model1
    #     - tractor.optimize with PsfEx PSF     -> model2
    #     - plot image, model1, model2

    band = ccd.filter
    im = DecamImage(decals,ccd)
    wcs = im.get_wcs()
    print('Band: ', band)
    print('Reading: ', im.imgfn)

    iminfo = im.get_image_info()
    print('img: ', iminfo)
    H,W = iminfo['dims']

    # Get all the PS1 stars on this CCD
    ps1 = ps1cat(ccdwcs=wcs)
    cat = ps1.get_stars(rmagcut=[17,20])

    for star in cat[5]:
        ra, dec = (star.ra, star.dec)
        mag = star.median[1] # r-band

        ok, xpos, ypos = wcs.radec2pixelxy(ra, dec)
        ix,iy = int(xpos), int(ypos)

        # create little tractor Image object around the star
        slc = (slice(max(iy-stampsize, 0), min(iy+stampsize+1, H)),
               slice(max(ix-stampsize, 0), min(ix+stampsize+1, W)))
        tim = im.get_tractor_image(slc=slc, const2psf=True)
        # the PSF model 'const2Psf' is the one used in DR1 -- 2-component Gaussian fit
        # to PsfEx instantiated in the image center.

        # create tractor PointSource from PS1 measurements
        flux = NanoMaggies.magToNanomaggies(mag)
        star = PointSource(RaDecPos(ra, dec), NanoMaggies(**{band: flux}))

        # re-fit the source RA,Dec,flux.
        tractor = Tractor([tim], [star])
        # only fit the source
        tractor.freezeParam('images')

        #alphas = [0.1, 0.3, 1.0]
        #optargs = dict(priors=False, shared_params=False, alphas=alphas)
        optargs = {}

        print('PSF model:', tim.psf)

        print('Fitting params:')
        tractor.printThawedParams()

        for step in range(50):
            dlnp,X,alpha = tractor.optimize(**optargs)
            print('dlnp', dlnp)
            print('X,alpha', X, alpha)
            if dlnp < 0.1:

                m0 = tractor.getModelImage(0)
                chi0 = tractor.getChiImage(0)
                p0 = np.array(tractor.getParams())
                for i,step in enumerate([1e-3, 1e-2, 1e-1, 1.]):
                    tractor.setParams(p0 + step * np.array(X))
                    print('Trying update:', star)
                    m1 = tractor.getModelImage(0)
                    chi1 = tractor.getChiImage(0)
                    imchi = dict(interpolation='nearest', origin='lower', vmin=-10, vmax=10)
                    plt.clf()
                    plt.subplot(2,2,1)
                    plt.imshow(m0, **tim.ima)
                    plt.subplot(2,2,2)
                    plt.imshow(m1, **tim.ima)
                    plt.subplot(2,2,3)
                    plt.imshow(chi0, **imchi)
                    plt.subplot(2,2,4)
                    plt.imshow(chi1, **imchi)
                    plt.savefig('fail1-%i.png' % i)
                tractor.setParams(p0)
                break
        print('Fit:', star)

        mod1 = tractor.getModelImage(0)

        # Now change the PSF model to a pixelized PSF model from PsfEx instantiated
        # at this place in the image.
        psfimg = tim.psfex.instantiateAt(xpos, ypos, nativeScale=True)
        pixpsf = PixelizedPSF(psfimg)

        # use the new PSF model...
        tim.psf = pixpsf

        print()
        print('PSF model:', tim.psf)
        for step in range(50):
            dlnp,X,alpha = tractor.optimize(**optargs)
            print('dlnp', dlnp)
            print('X,alpha', X, alpha)
            if dlnp < 0.1:

                m0 = tractor.getModelImage(0)
                p0 = np.array(tractor.getParams())
                tractor.setParams(p0 + 0.001 * np.array(X))
                print('Trying update:', star)
                m1 = tractor.getModelImage(0)
                plt.clf()
                plt.subplot(1,2,1)
                plt.imshow(m0, **tim.ima)
                plt.subplot(1,2,2)
                plt.imshow(m1, **tim.ima)
                plt.savefig('fail2.png')
                tractor.setParams(p0)

            break
        print('Fit:', star)

        mod2 = tractor.getModelImage(0)

        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(tim.getImage(), **tim.ima)
        plt.title('Image')
        plt.subplot(1,3,2)
        plt.imshow(mod1, **tim.ima)
        plt.title('2-Gaussian model')
        plt.subplot(1,3,3)
        plt.imshow(mod1, **tim.ima)
        plt.title('PsfEx model')
        plt.savefig('psf.png')

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--expnum', type=long, default='396086', metavar='', 
                        help='exposure number')
    parser.add_argument('-c', '--ccdname', type=str, default='S31', metavar='', 
                        help='CCD name')
    args = parser.parse_args()

    psf_residuals(expnum=args.expnum,ccdname=args.ccdname)
    
if __name__ == "__main__":
    main()
