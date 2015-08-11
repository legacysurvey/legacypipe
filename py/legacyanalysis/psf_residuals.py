#!/usr/bin/env python

"""Construct a set of PSF diagnostic QAplots.

For a given CCD:
- find PS1 stars
- for each PS1 star:
    - create tractor Image object in a teeny patch around this star
      (for only this one CCD)
    - create PointSource object with PS1 RA,Dec,flux
    - tractor.optimize with DR1-style PSF -> model1
    - tractor.optimize with PsfEx PSF     -> model2
    - plot image, model1, model2

"""

from __future__ import division, print_function

import os
import sys
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tractor.psfex import PsfEx
from tractor import Tractor
from tractor.basics import (NanoMaggies, PointSource, GaussianMixtureEllipsePSF,
                            PixelizedPSF, RaDecPos)
from legacyanalysis.ps1cat import ps1cat

from astrometry.util.fits import fits_table
from legacypipe.common import Decals, DecamImage

def psf_residuals(expnum,ccdname,stampsize=25,
                  rmagcut=[17,20],verbose=0):

    # Set the debugging level.
    if verbose==0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)

    # Gather all the info we need about this CCD.
    decals = Decals()
    ccd = decals.find_ccds(expnum=expnum,ccdname=ccdname)[0]
    band = ccd.filter
    print('Band {}'.format(band))

    im = DecamImage(decals,ccd)
    iminfo = im.get_image_info()
    H,W = iminfo['dims']

    wcs = im.get_wcs()
    radec = wcs.radec_bounds()
    print(radec)

    # Get all the PS1 stars on this CCD.
    ps1 = ps1cat(ccdwcs=wcs)
    cat = ps1.get_stars(rmagcut=rmagcut)
    cat = cat[np.argsort(cat.median[:,1])] # sort by r-band magnitude

    qafile = 'qapsf-onccd.png'
    fig = plt.figure()
    ax = fig.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.scatter(cat.ra,cat.dec)
    ax.set_xlim([radec[1],radec[0]])#*[1.0002,0.9998])
    ax.set_ylim([radec[2],radec[3]])#*[0.985,1.015])
    ax.set_xlabel('$RA\ (deg)$',fontsize=18)
    ax.set_ylabel('$Dec\ (deg)$',fontsize=18)
    fig.savefig(qafile)

    # Initialize the QAplot
    ncols = 3
    nrows = 2
    cat = cat[:ncols*nrows]

    inchperstamp = 2.0
    fig = plt.figure(figsize=(inchperstamp*3*ncols,inchperstamp*nrows))
    gs = gridspec.GridSpec(nrows,3*ncols)
    irow = 0
    icol = 0
    
    for istar, ps1star in enumerate(cat):
        ra, dec = (ps1star.ra, ps1star.dec)
        mag = ps1star.median[1] # r-band

        ok, xpos, ypos = wcs.radec2pixelxy(ra, dec)
        ix,iy = int(xpos), int(ypos)

        # create a little tractor Image object around the star
        slc = (slice(max(iy-stampsize, 0), min(iy+stampsize+1, H)),
               slice(max(ix-stampsize, 0), min(ix+stampsize+1, W)))

        # The PSF model 'const2Psf' is the one used in DR1: a 2-component
        # Gaussian fit to PsfEx instantiated in the image center.
        tim = im.get_tractor_image(slc=slc, const2psf=True)
        stamp = tim.getImage()

        # Initialize a tractor PointSource from PS1 measurements
        flux = NanoMaggies.magToNanomaggies(mag)
        star = PointSource(RaDecPos(ra,dec), NanoMaggies(**{band: flux}))

        # Fit just the source RA,Dec,flux.
        tractor = Tractor([tim], [star])
        tractor.freezeParam('images')

        print('2-component MOG:', tim.psf)
        tractor.printThawedParams()

        for step in range(50):
            dlnp,X,alpha = tractor.optimize()
            if dlnp < 0.1:
                break
        print('Fit:', star)
        model_mog = tractor.getModelImage(0)
        chi2_mog = -2.0*tractor.getLogLikelihood()
        mag_mog = NanoMaggies.nanomaggiesToMag(star.brightness)[0]

        # Now change the PSF model to a pixelized PSF model from PsfEx instantiated
        # at this place in the image.
        psfimg = tim.psfex.instantiateAt(xpos, ypos, nativeScale=True)
        tim.psf = PixelizedPSF(psfimg)

        #print('PSF model:', tim.psf)
        #tractor.printThawedParams()
        for step in range(50):
            dlnp,X,alpha = tractor.optimize()
            if dlnp < 0.1:
                break

        print('Fit:', star)
        model_psfex = tractor.getModelImage(0)
        chi2_psfex = -2.0*tractor.getLogLikelihood()
        mag_psfex = NanoMaggies.nanomaggiesToMag(star.brightness)[0]

        # Generate a QAplot.
        if (istar>0) and (istar%(ncols)==0):
            irow = irow+1
        icol = 3*istar - 3*ncols*irow
        #print(istar, irow, icol, icol+1, icol+2)

        ax1 = plt.subplot2grid((nrows,3*ncols), (irow,icol), aspect='equal')
        ax1.axis('off')
        ax1.imshow(stamp, **tim.ima)
        gs.update(wspace=0.0,hspace=0.0,bottom=0.0,top=0.0,left=0.0,right=0.0) 

        ax2 = plt.subplot2grid((nrows,3*ncols), (irow,icol+1), aspect='equal')
        ax2.axis('off')
        ax2.imshow(stamp-model_mog, **tim.ima)

        #ax2.set_title('{:.3f}, {:.2f}'.format(mag_psfex,chi2_psfex),fontsize=14)
        #ax2.set_title('{:.3f}, $\chi^{2}$={:.2f}'.format(mag_psfex,chi2_psfex))

        ax3 = plt.subplot2grid((nrows,3*ncols), (irow,icol+2), aspect='equal')
        ax3.axis('off')
        ax3.imshow(stamp-model_psfex, **tim.ima)
        gs.update(wspace=0.1,hspace=0.0,bottom=0.0,top=0.0,left=0.0,right=0.0) 

    fig.savefig('qapsf.png')

def main():
    """
    Main routine.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--expnum', type=long, default='396086', metavar='', 
                        help='exposure number')
    parser.add_argument('-c', '--ccdname', type=str, default='S31', metavar='', 
                        help='CCD name')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0,
                        help='Toggle on verbose output')
    args = parser.parse_args()

    psf_residuals(expnum=args.expnum,ccdname=args.ccdname,verbose=args.verbose)
    
if __name__ == "__main__":
    main()
