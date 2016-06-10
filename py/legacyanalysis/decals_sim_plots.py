#!/usr/bin/env python

"""Analyze the output of decals_simulations.

3216p000
"""

from __future__ import division, print_function

import os
import sys
import pdb
import logging
import argparse

import numpy as np
import matplotlib

from astropy.io import fits
from astropy.table import vstack, Table
from astrometry.libkd.spherematch import match_radec

# import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

matplotlib.use('Agg') # display backend

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='process this brick (required input)')
    parser.add_argument('-o', '--objtype', type=str, choices=['STAR', 'ELG', 'LRG', 'BGS'], default='STAR', metavar='', 
                        help='object type (STAR, ELG, LRG, BGS)') 
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='toggle on verbose output')

    args = parser.parse_args()
    if args.brick is None:
        parser.print_help()
        sys.exit(1)

    # Set the debugging level
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    brickname = args.brick
    objtype = args.objtype.upper()
    lobjtype = objtype.lower()
    log.info('Analyzing objtype {} on brick {}'.format(objtype, brickname))

    if 'DECALS_SIM_DIR' in os.environ:
        decals_sim_dir = os.getenv('DECALS_SIM_DIR')
    else:
        decals_sim_dir = '.'
    output_dir = os.path.join(decals_sim_dir, brickname)
    
    # Plotting preferences
    #sns.set(style='white',font_scale=1.6,palette='dark')#,font='fantasy')
    #col = sns.color_palette('dark')
    col = ['b', 'k', 'c', 'm', 'y', 0.8]
        
    # Read the metadata catalog.
    metafile = os.path.join(output_dir, 'metacat-{}-{}.fits'.format(brickname, lobjtype))
    log.info('Reading {}'.format(metafile))
    meta = fits.getdata(metafile, 1)

    # We need this for our histograms below
    magbinsz = 0.2
    rminmax = np.squeeze(meta['RMAG_RANGE'])
    nmagbin = long((rminmax[1]-rminmax[0])/magbinsz)
    
    # Work in chunks.
    allsimcat = []
    bigsimcat = []
    bigtractor = []
    nchunk = np.squeeze(meta['nchunk'])
    for ichunk in range(nchunk):
        print('ichunk= ', ichunk)
        #log.info('Working on chunk {:02d}/{:02d}'.format(ichunk+1, nchunk))
        chunksuffix = '{:02d}'.format(ichunk)
        
        # Read the simulated object catalog
        simcatfile = os.path.join(output_dir, 'simcat-{}-{}-{}.fits'.format(brickname, lobjtype, chunksuffix))
        log.info('Reading {}'.format(simcatfile))
        simcat = Table(fits.getdata(simcatfile, 1))

        # Read and match to the Tractor catalog
        tractorfile = os.path.join(output_dir, 'tractor-{}-{}-{}.fits'.format(
            brickname, lobjtype, chunksuffix))
        log.info('Reading {}'.format(tractorfile))
        tractor = Table(fits.getdata(tractorfile, 1))

        m1, m2, d12 = match_radec(tractor['ra'], tractor['dec'],
                                  simcat['RA'], simcat['DEC'], 1.0/3600.0)
        missing = np.delete(np.arange(len(simcat)), m2, axis=0)
        log.info('Missing {}/{} sources'.format(len(missing), len(simcat)))

        #good = np.where((np.abs(tractor['decam_flux'][m1,2]/simcat['rflux'][m2]-1)<0.3)*1)

        # Build matching catalogs for the plots below.
        if len(bigsimcat) == 0:
            bigsimcat = simcat[m2]
            bigtractor = tractor[m1]
        else:
            bigsimcat = vstack((bigsimcat, simcat[m2]))
            bigtractor = vstack((bigtractor, tractor[m1]))
        if len(allsimcat) == 0:
            allsimcat = simcat
        else:
            allsimcat = vstack((allsimcat, simcat))

        # Get cutouts of the missing sources in each chunk (if any)
        if len(missing) > 0:
            imfile = os.path.join(output_dir, 'qa-{}-{}-image-{}.jpg'.format(brickname, lobjtype, chunksuffix))
            hw = 30 # half-width [pixels]
            rad = 5
            ncols = 5
            nrows = 5
            nthumb = ncols*nrows
            dims = (ncols*hw*2,nrows*hw*2)
            mosaic = Image.new('RGB',dims)

            miss = missing[np.argsort(simcat['R'][missing])]
            xpos, ypos = np.meshgrid(np.arange(0, dims[0], hw*2, dtype='int'),
                                     np.arange(0, dims[1], hw*2, dtype='int'))
            im = Image.open(imfile)
            sz = im.size
            iobj = 0
            for ic in range(ncols):
                for ir in range(nrows):
                    xx = int(simcat['X'][miss[iobj]])
                    yy = int(sz[1]-simcat['Y'][miss[iobj]])
                    crop = (xx-hw, yy-hw, xx+hw, yy+hw)
                    box = (xpos[ir, ic], ypos[ir, ic])
                    thumb = im.crop(crop)
                    mosaic.paste(thumb, box)
                    iobj = iobj + 1

            # Add a border and circle the missing source.
            draw = ImageDraw.Draw(mosaic)
            sz = mosaic.size
            for ic in range(ncols):
                for ir in range(nrows):
                    draw.rectangle([(xpos[ir, ic], ypos[ir, ic]),
                                    (xpos[ir, ic]+hw*2, ypos[ir, ic]+hw*2)])
                    xx = xpos[ir, ic] + hw
                    yy = ypos[ir, ic] + hw
                    draw.ellipse((xx-rad, sz[1]-yy-rad, xx+rad, sz[1]-yy+rad), outline='yellow')

            qafile = os.path.join(output_dir, 'qa-{}-{}-missing-{}.png'.format(brickname, lobjtype, chunksuffix))
            log.info('Writing {}'.format(qafile))
            mosaic.save(qafile)

        # Annotate the coadd image and residual files so the simulated sources
        # are labeled.
        rad = 15
        for suffix in ('image', 'resid'):
            imfile = os.path.join(output_dir, 'qa-{}-{}-{}-{}.jpg'.format(brickname, lobjtype, suffix, chunksuffix))
            qafile = imfile.replace('.jpg', '-annot.png')

            im = Image.open(imfile)
            sz = im.size
            draw = ImageDraw.Draw(im)
            [draw.ellipse((cat['X']-rad, sz[1]-cat['Y']-rad, cat['X']+rad,
                           sz[1]-cat['Y']+rad), outline='yellow') for cat in simcat]
            log.info('Writing {}'.format(qafile))
            im.save(qafile)

    # Flux residuals vs r-band magnitude
    fig, ax = plt.subplots(3, sharex=True, figsize=(6,8))

    rmag = bigsimcat['R']
    for thisax, thiscolor, band, indx in zip(ax, col, ('G', 'R', 'Z'), (1, 2, 4)):
        simflux = bigsimcat[band+'FLUX']
        tractorflux = bigtractor['decam_flux'][:, indx]
        thisax.scatter(rmag, -2.5*np.log10(tractorflux/simflux),
                       color=thiscolor, s=10)
      
        thisax.set_ylim(-0.7,0.7)
        thisax.set_xlim(rminmax + [-0.1, 0.0])
        thisax.axhline(y=0.0,lw=2,ls='solid',color='gray')
        
    
        thisax.text(0.05,0.05, band.lower(), horizontalalignment='left',
                    verticalalignment='bottom',transform=thisax.transAxes,
                    fontsize=16)
        
    ax[1].set_ylabel(r'$\Delta$m (Tractor minus Input)')
    ax[2].set_xlabel('Input r magnitude (AB mag)')

    fig.subplots_adjust(left=0.18,hspace=0.1)
    qafile = os.path.join(output_dir, 'qa-{}-{}-flux.png'.format(brickname, lobjtype))
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)
    
    # Color residuals
    gr_tra = -2.5*np.log10(bigtractor['decam_flux'][:, 1]/bigtractor['decam_flux'][:, 2])
    rz_tra = -2.5*np.log10(bigtractor['decam_flux'][:, 2]/bigtractor['decam_flux'][:, 4])
    gr_sim = -2.5*np.log10(bigsimcat['GFLUX']/bigsimcat['RFLUX'])
    rz_sim = -2.5*np.log10(bigsimcat['RFLUX']/bigsimcat['ZFLUX'])

    fig, ax = plt.subplots(2,sharex=True,figsize=(6,8))
    
    ax[0].scatter(rmag, gr_tra-gr_sim, color=col[0], s=10)
    ax[1].scatter(rmag, rz_tra-rz_sim, color=col[1], s=10)
    
    [thisax.set_ylim(-0.7,0.7) for thisax in ax]
    [thisax.set_xlim(rminmax + [-0.1, 0.0]) for thisax in ax]
    [thisax.axhline(y=0.0, lw=2, ls='solid', color='gray') for thisax in ax]
    
    ax[0].set_ylabel('$\Delta$(g - r) (Tractor minus Input)')
    ax[1].set_ylabel('$\Delta$(r - z) (Tractor minus Input)')
    ax[1].set_xlabel('Input r magnitude (AB mag)')
    fig.subplots_adjust(left=0.18,hspace=0.1)

    qafile = os.path.join(output_dir, 'qa-{}-{}-color.png'.format(brickname, lobjtype))
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)

    # Fraction of matching sources
    rmaghist, magbins = np.histogram(allsimcat['R'], bins=nmagbin, range=rminmax)
    cmagbins = (magbins[:-1] + magbins[1:]) / 2.0
    ymatch, binsmatch = np.histogram(bigsimcat['R'], bins=nmagbin, range=rminmax)
    #ymatchgood, binsgood = np.histogram(bigsimcat['R'][good],bins=nmagbin,range=rminmax)

    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.step(cmagbins, 1.0*ymatch/rmaghist, lw=3, alpha=0.5, label='All objects')
    #ax.step(cmagbins, 1.0*ymatchgood/rmaghist, lw=3, ls='dashed', label='|$\Delta$m|<0.3')
    ax.axhline(y=1.0,lw=2,ls='dashed',color='gray')
    ax.set_xlabel('Input r magnitude (AB mag)')
    ax.set_ylabel('Fraction of Matching {}s'.format(objtype))
    ax.set_ylim([0.0, 1.1])
    ax.legend(loc='lower left')
    fig.subplots_adjust(bottom=0.15)
    qafile = os.path.join(output_dir, 'qa-{}-{}-frac.png'.format(brickname, lobjtype))
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)

    # Distribution of object types for matching sources.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    rmaghist, magbins = np.histogram(bigsimcat['R'], bins=nmagbin, range=rminmax)
    cmagbins = (magbins[:-1] + magbins[1:]) / 2.0
    tractortype = np.char.strip(bigtractor['type'].data)
    for otype in ['PSF', 'SIMP', 'EXP', 'DEV', 'COMP']:
        these = np.where(tractortype == otype)[0]
        if len(these)>0:
            yobj, binsobj = np.histogram(bigsimcat['R'][these], bins=nmagbin, range=rminmax)
            #plt.step(cmagbins,1.0*yobj,lw=3,alpha=0.5,label=otype)
            plt.step(cmagbins,1.0*yobj/rmaghist,lw=3,alpha=0.5,label=otype)
    plt.axhline(y=1.0,lw=2,ls='dashed',color='gray')
    plt.xlabel('Input r magnitude (AB mag)')
    #plt.ylabel('Number of Objects')
    plt.ylabel('Fraction of {}s classified'.format(objtype))
    plt.ylim([0.0,1.1])
    plt.legend(loc='center left', bbox_to_anchor=(0.08,0.5))
    fig.subplots_adjust(bottom=0.15)

    qafile = os.path.join(output_dir, 'qa-{}-{}-type.png'.format(brickname, lobjtype))
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)

    '''
    # Morphology plots
    if objtype=='ELG':
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,3,1)
        plt.plot(rmag,deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlim(rminmax)
        plt.xlabel('r (AB mag)')

        plt.subplot(1,3,2)
        plt.plot(bigsimcat['R50_1'],deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlabel('$r_{50}$ (arcsec)')

        plt.subplot(1,3,3)
        plt.plot(bigsimcat['BA_1'],deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlabel('b/a')
        plt.xlim([0.2,1.0])
        fig.subplots_adjust(bottom=0.18)
        qafile = os.path.join(output_dir,'qa-'+brickname+'-'+lobjtype+'-morph.png')
        log.info('Writing {}'.format(qafile))
        plt.savefig(qafile)
    '''
    
if __name__ == "__main__":
    main()
