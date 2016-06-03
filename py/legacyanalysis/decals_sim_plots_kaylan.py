#!/usr/bin/env python

"""
Only depends on anaconda python
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg') #display backend
import os
import sys
import argparse
import numpy as np
#import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

from astropy.io import fits

def bash(cmd):
    test= os.system('%s' % cmd)
    if test: 
        print('command failed: %s' % cmd) 
        raise ValueError

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-dir', help='dir to where diff brickname dirs are',required=True)
    parser.add_argument('-brick', help='name of brickname dir to process',required=True)
    parser.add_argument('-o', '--objtype', type=str, choices=['STAR', 'ELG', 'LRG', 'BGS'],default='ELG', metavar='', 
                        help='object type (STAR, ELG, LRG, BGS)') 
    parser.add_argument('-overwrite', help='set to anything to overwrite existing QA dir',required=False)
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='toggle on verbose output')

    args = parser.parse_args()

    outdir=os.path.join(args.dir,args.brick,'QA')
    if os.path.exists(outdir) and args.overwrite:
        bash('rm %s' % os.path.join(outdir,'*.png'))
    elif os.path.exists(outdir):
        print('%s exists, QUITTING, specify -overwrite 1 to proceed' % outdir)
        sys.exit()
    else: bash('mkdir %s' % outdir)

    objtype = args.objtype.upper()
    lobjtype = objtype.lower()

    # Plotting preferences
    #sns.set(style='white',font_scale=1.6,palette='dark')#,font='fantasy')
    #col = sns.color_palette('dark')
    col = ['b','k','c','m','y',0.8]
        
    # Read the meta-catalog.
    metafile = os.path.join(args.dir,args.brick,'metacat-'+args.brick+'-'+lobjtype+'.fits')
    meta = fits.getdata(metafile,1)

    # We need this for our histograms below
    magbinsz = 0.2
    if meta['rmag_range'].shape == (1,2): rmin,rmax = meta['rmag_range'][0][0],meta['rmag_range'][0][1]
    elif meta['rmag_range'].shape == (2,): rmin,rmax = meta['rmag_range'][0],meta['rmag_range'][1]
    rminmax= np.array([rmin,rmax])
    nmagbin = long((rminmax[1]-rminmax[0])/magbinsz)
    
    # Read the simulated object catalog
    simcatfile = os.path.join(args.dir,args.brick,'simcat-'+args.brick+'-'+lobjtype+'.fits')
    #log.info('Reading {}'.format(simcatfile))
    simcat = fits.getdata(simcatfile, 1)

    # Read and match to the Tractor catalog
    tractorfile = os.path.join(args.dir,args.brick,'tractor',args.brick[:3],'tractor-'+args.brick+'.fits')
    #log.info('Reading {}'.format(tractorfile))
    tractor = fits.getdata(tractorfile, 1)

    # Modify the coadd image and residual files so the simulated sources
    # are labeled.
    rad = 15
    imfile = os.path.join(args.dir,args.brick,'coadd',args.brick[:3],args.brick,'legacysurvey-'+args.brick+'-image.jpg') 
    imfile = [imfile,imfile.replace('-image','-resid')]
    for ifile in imfile:
        im=Image.open(ifile)
        im=np.asarray(im)
        kwargs=dict(origin='upper',interpolation='nearest') #upper so agrees with ds9
        fig,ax=plt.subplots() #,sharey=True,sharex=True)
        ax.imshow(im,**kwargs)
        for cat in simcat:
            p= patches.Circle((cat['x'],im.shape[1]-cat['y']), radius=rad,fc = 'none', ec = 'yellow')
            ax.add_patch(p)
        ax.tick_params(direction='out')
        plt.savefig(os.path.join(outdir,os.path.basename(ifile).replace('.jpg','-showingsrcs.png')), bbox_inches='tight',dpi=150)
    
if __name__ == "__main__":
    main()
