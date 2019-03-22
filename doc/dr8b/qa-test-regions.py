#!/usr/bin/env python

"""
Visualize the brick and CCD coverage of the test regions.

"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})

import os, pdb
import numpy as np
from astrometry.util.fits import fits_table
import matplotlib.pyplot as plt

def main():

    for region in ('dr8-test-overlap', 'dr8-test-s82', 'dr8-test-hsc-sgc', 'dr8-test-hsc-ngc',
                   'dr8-test-edr', 'dr8-test-hsc-north', 'dr8-test-deep2-egs'):
        
        if os.path.isfile('dr8b/ccds-{}-decam.fits'.format(region)):
            ccds_decam = fits_table('dr8b/ccds-{}-decam.fits'.format(region))
            bricks_decam = fits_table('dr8b/bricks-{}-decam.fits'.format(region))
        else:
            ccds_decam, bricks_decam = [], []
            
        if os.path.isfile('dr8b/ccds-{}-90prime-mosaic.fits'.format(region)):
            ccds_mosaic = fits_table('dr8b/ccds-{}-90prime-mosaic.fits'.format(region))
            bricks_mosaic = fits_table('dr8b/bricks-{}-90prime-mosaic.fits'.format(region))
        else:
            ccds_mosaic, bricks_mosaic = [], []

        # DR7    
        if os.path.isfile('dr7/ccds-{}.fits'.format(region)):
            ccds_dr7 = fits_table('dr7/ccds-{}.fits'.format(region))
            ccds_dr7 = ccds_dr7[ccds_dr7.ccd_cuts == 0]
        else:
            ccds_dr7 = []

        # DR6
        if os.path.isfile('dr6/ccds-{}.fits'.format(region)):
            ccds_dr6 = fits_table('dr6/ccds-{}.fits'.format(region))
            ccds_dr6 = ccds_dr6[ccds_dr6.ccd_cuts == 0]
        else:
            ccds_dr6 = []
            
        fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)    
        ax[0, 0].set_title('{} Bricks'.format(region.upper()))
        ax[0, 1].set_title('{} CCDs'.format(region.upper()))
        
        if len(bricks_decam):
            ax[0, 0].scatter(bricks_decam.ra, bricks_decam.dec, s=5, marker='s', 
                             label='DR8/DECam (N={})'.format(len(bricks_decam)))
            ax[0, 1].scatter(ccds_decam.ra, ccds_decam.dec, s=5, marker='s', 
                             label='DR8/DECam (N={})'.format(len(ccds_decam)))
            if len(ccds_dr7):
                ax[0, 1].scatter(ccds_dr7.ra, ccds_dr7.dec, s=2, alpha=0.5, 
                                 label='DR7/DECam (N={})'.format(len(ccds_dr7)))
            ax[0, 0].set_ylabel('Dec')
            
        if len(bricks_mosaic):
            ax[1, 0].scatter(bricks_mosaic.ra, bricks_mosaic.dec, s=5, marker='s', 
                             label='DR8/90Prime+Mosaic (N={})'.format(len(bricks_mosaic)))
            ax[1, 1].scatter(ccds_mosaic.ra, ccds_mosaic.dec, s=5, marker='s', 
                             label='DR8/90Prime+Mosaic (N={})'.format(len(ccds_mosaic)))
            if len(ccds_dr6):
                ax[0, 1].scatter(ccds_dr6.ra, ccds_dr6.dec, s=2, alpha=0.5,
                                 label='DR6/90Prime+Mosaic (N={})'.format(len(ccds_dr6)))
            ax[1, 0].set_ylabel('Dec')
            ax[1, 0].set_xlabel('RA')
            ax[1, 1].set_xlabel('RA')
            #ax[1, 0].set_title('{} Bricks'.format(region.upper()))
            #ax[1, 1].set_title('{} CCDs'.format(region.upper()))
        for xx in ax.flat:
            xx.legend(frameon=False, loc='upper right', fontsize=10, markerscale=3, handletextpad=0.1)
            xx.margins(x=0.1, y=0.5)

        plt.subplots_adjust(wspace=0.1, hspace=0.05)
        #plt.suptitle(region.upper())
        plt.savefig('{}.png'.format(region))
            
if __name__ == '__main__':
    main()
