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


from legacypipe.survey import Decals
decals = Decals()
all = decals.get_ccds()
plt.scatter(all.seeing,all.airmass) ; plt.xlim(0,3) ; plt.show(block=False)
cut1 = np.where(((all.airmass>0)*1)*((all.airmass<1.1)*1)*((all.seeing<1.0)*1))[0]
cut2 = np.unique(all.expnum[cut1],return_index=True)[1]
good = all[cut1[cut2]]
print(good.expnum)
[175034 175035 175040 175041 175042 197042 197043 197044 197045 197070
 197071 197072 197073 197074 197075 197076 197077 205060 205063 295339
 295340 295341 295342 432735 432736 432739 432740 432743 432744 432745
 432746 432747 432748 432749 432750 432759 432786 432787 432790 432791
 432792 432796 432797 432798 432799 432800 432801 432802 432803 432804
 432805 432806 432807 432808 432809 432810 432811 432812 432813 432814
 432815 432816 432817 432818 432819 432820 432821 432822 432823 432824
 432825 432826 432827 432828 432829 432830 432831 432832 432833 432834
 432835 432836 432837 432838 432839 432840 432841 432842 432843]

cut1 = np.where(((all.airmass>2.0)*1)*((all.seeing<1.0)*1))[0]
cut2 = np.unique(all[cut1].expnum,return_index=True)[1]
bad = all[cut1[cut2]]
print(bad.expnum)
[347427 347512 347513 347518 347519 347526 347763 392888 393579 393580
 393581 393584 393684 430521 430522 430525 430960]

# good seeing, high airmass, edge of FOV
psf_residuals -e 347427 -c S31 -m 14 17 -n 30

# good seeing, low airmass, edge of FOV
psf_residuals -e 175034 -c S31 -m 14 17 -n 30

"""

from __future__ import division, print_function

import os
import sys
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tractor.psfex import PsfEx, PixelizedPsfEx
from tractor import Tractor
from tractor.basics import (NanoMaggies, PointSource, GaussianMixtureEllipsePSF,
                            PixelizedPSF, RaDecPos)
from legacyanalysis.ps1cat import ps1cat

from astrometry.util.fits import fits_table
from legacypipe.survey import LegacySurveyData

def psf_residuals(expnum,ccdname,stampsize=35,nstar=30,
                  magrange=(13,17),verbose=0, splinesky=False):

    # Set the debugging level.
    if verbose==0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)

    pngprefix = 'qapsf-{}-{}'.format(expnum,ccdname)

    # Gather all the info we need about this CCD.
    survey = LegacySurveyData()
    ccd = survey.find_ccds(expnum=expnum,ccdname=ccdname)[0]
    band = ccd.filter
    ps1band = dict(g=0,r=1,i=2,z=3,Y=4)
    print('Band {}'.format(band))

    #scales = dict(g=0.0066, r=0.01, z=0.025)
    #vmin, vmax = np.arcsinh(-1), np.arcsinh(100)
    #print(scales[band])

    im = survey.get_image_object(ccd)
    iminfo = im.get_image_info()
    H,W = iminfo['dims']

    wcs = im.get_wcs()

    # Choose a uniformly selected subset of PS1 stars on this CCD.
    ps1 = ps1cat(ccdwcs=wcs)
    cat = ps1.get_stars(band=band,magrange=magrange)

    rand = np.random.RandomState(seed=expnum*ccd.ccdnum)
    these = rand.choice(len(cat)-1,nstar,replace=False)
    #these = rand.random_integers(0,len(cat)-1,nstar)
    cat = cat[these]
    cat = cat[np.argsort(cat.median[:,ps1band[band]])] # sort by magnitude
    #print(cat.nmag_ok)

    get_tim_kwargs = dict(pixPsf=True, splinesky=splinesky)

    # Make a QAplot of the positions of all the stars.
    tim = im.get_tractor_image(**get_tim_kwargs)
    img = tim.getImage()
    #img = tim.getImage()/scales[band]

    fig = plt.figure(figsize=(5,10))
    ax = fig.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    #ax.imshow(np.arcsinh(img),cmap='gray',interpolation='nearest',
    #          origin='lower',vmin=vmax,vmax=vmax)
    
    ax.imshow(img, **tim.ima)
    ax.axis('off')
    ax.set_title('{}: {}/{} AM={:.2f} Seeing={:.3f}"'.
                 format(band,expnum,ccdname,ccd.airmass,ccd.seeing))

    for istar, ps1star in enumerate(cat):
        ra, dec = (ps1star.ra, ps1star.dec)
        ok, xpos, ypos = wcs.radec2pixelxy(ra, dec)
        ax.text(xpos,ypos,'{:2d}'.format(istar+1),color='red',
                horizontalalignment='left')
        circ = plt.Circle((xpos,ypos),radius=30,color='g',fill=False,lw=1)
        ax.add_patch(circ)

    #radec = wcs.radec_bounds()
    #ax.scatter(cat.ra,cat.dec)
    #ax.set_xlim([radec[1],radec[0]])#*[1.0002,0.9998])
    #ax.set_ylim([radec[2],radec[3]])#*[0.985,1.015])
    #ax.set_xlabel('$RA\ (deg)$',fontsize=18)
    #ax.set_ylabel('$Dec\ (deg)$',fontsize=18)
    fig.savefig(pngprefix+'-ccd.png',bbox_inches='tight')

    # Initialize the many-stamp QAplot
    ncols = 3
    nrows = np.ceil(nstar/ncols).astype('int')

    inchperstamp = 2.0
    fig = plt.figure(figsize=(inchperstamp*3*ncols,inchperstamp*nrows))
    irow = 0
    icol = 0
    
    for istar, ps1star in enumerate(cat):
        ra, dec = (ps1star.ra, ps1star.dec)
        mag = ps1star.median[ps1band[band]] # r-band

        ok, xpos, ypos = wcs.radec2pixelxy(ra, dec)
        ix,iy = int(xpos), int(ypos)

        # create a little tractor Image object around the star
        slc = (slice(max(iy-stampsize, 0), min(iy+stampsize+1, H)),
               slice(max(ix-stampsize, 0), min(ix+stampsize+1, W)))

        # The PSF model 'const2Psf' is the one used in DR1: a 2-component
        # Gaussian fit to PsfEx instantiated in the image center.
        tim = im.get_tractor_image(slc=slc, **get_tim_kwargs)
        stamp = tim.getImage()
        ivarstamp = tim.getInvvar()

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
        psf = PixelizedPsfEx(im.psffn)
        tim.psf = psf.constantPsfAt(xpos, ypos)

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

        #mn, mx = np.percentile((stamp-model_psfex)[ivarstamp>0],[1,95])
        sig = np.std((stamp-model_psfex)[ivarstamp>0])
        mn, mx = [-2.0*sig,5*sig]

        # Generate a QAplot.
        if (istar>0) and (istar%(ncols)==0):
            irow = irow+1
        icol = 3*istar - 3*ncols*irow
        #print(istar, irow, icol, icol+1, icol+2)

        ax1 = plt.subplot2grid((nrows,3*ncols), (irow,icol), aspect='equal')
        ax1.axis('off')
        #ax1.imshow(stamp, **tim.ima)
        ax1.imshow(stamp,cmap='gray',interpolation='nearest',
                   origin='lower',vmin=mn,vmax=mx)
        ax1.text(0.1,0.9,'{:2d}'.format(istar+1),color='white',
                horizontalalignment='left',verticalalignment='top',
                transform=ax1.transAxes)

        ax2 = plt.subplot2grid((nrows,3*ncols), (irow,icol+1), aspect='equal')
        ax2.axis('off')
        #ax2.imshow(stamp-model_mog, **tim.ima)
        ax2.imshow(stamp-model_mog,cmap='gray',interpolation='nearest',
                   origin='lower',vmin=mn,vmax=mx)
        ax2.text(0.1,0.9,'MoG',color='white',
                horizontalalignment='left',verticalalignment='top',
                transform=ax2.transAxes)
        ax2.text(0.08,0.08,'{:.3f}'.format(mag_mog),color='white',
                 horizontalalignment='left',verticalalignment='bottom',
                 transform=ax2.transAxes)

        #ax2.set_title('{:.3f}, {:.2f}'.format(mag_psfex,chi2_psfex),fontsize=14)
        #ax2.set_title('{:.3f}, $\chi^{2}$={:.2f}'.format(mag_psfex,chi2_psfex))

        ax3 = plt.subplot2grid((nrows,3*ncols), (irow,icol+2), aspect='equal')
        ax3.axis('off')
        #ax3.imshow(stamp-model_psfex, **tim.ima)
        ax3.imshow(stamp-model_psfex,cmap='gray',interpolation='nearest',
                   origin='lower',vmin=mn,vmax=mx)
        ax3.text(0.1,0.9,'PSFEx',color='white',
                horizontalalignment='left',verticalalignment='top',
                transform=ax3.transAxes)
        ax3.text(0.08,0.08,'{:.3f}'.format(mag_psfex),color='white',
                 horizontalalignment='left',verticalalignment='bottom',
                 transform=ax3.transAxes)

        if istar==(nstar-1):
            break
    fig.savefig(pngprefix+'-stargrid.png',bbox_inches='tight')

def main():
    """
    Main routine.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--expnum', type=long, default='396086', metavar='', 
                        help='exposure number')
    parser.add_argument('-c', '--ccdname', type=str, default='S31', metavar='', 
                        help='CCD name')
    parser.add_argument('-n', '--nstar', type=long, default=30, metavar='', 
                        help='number of stars to display')
    parser.add_argument('-m', '--magrange', type=float, default=(13,17), nargs=2, metavar='', 
                        help='PS1 magnitude range')

    parser.add_argument('--splinesky', action='store_true', help='Use spline sky model?')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count', default=0,
                        help='Toggle on verbose output')
    args = parser.parse_args()

    psf_residuals(expnum=args.expnum,ccdname=args.ccdname,
                  nstar=args.nstar,magrange=args.magrange,
                  verbose=args.verbose,splinesky=args.splinesky)
    
if __name__ == "__main__":
    main()
