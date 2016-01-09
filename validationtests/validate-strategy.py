#!/usr/bin/env python

"""Validate aspects of nightlystrategy.py by looking at the DR2 annotated CCDs
file.

J. Moustakas
Siena College
2016 Jan 8

"""

from __future__ import division, print_function

import os
import sys

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import seaborn as sns

def main():

    key = 'DECALS_DIR'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        raise EnvironmentError('Required ${} environment variable not set'.format(key))
        sys.exit(1)
    decalsdir = os.getenv(key)

    annfile = os.path.join(decalsdir,'decals-ccds-annotated.fits')
    if os.path.isfile(annfile):
        print('Reading {}'.format(annfile))
        ann = fits.getdata(annfile,1)
    else:
        print('Annotated CCDs file {} not found'.format(annfile))
        sys.exit(1)

    # Get just the DECaLS data.
    nall = len(ann)
    keep = ann['tilepass']>0
    ann = ann[keep]
    nkeep = len(ann)
    print('Keeping {}/{} CCDs from DECaLS itself'.format(nkeep,nall))

    # Convenience variables
    allpass = [1,2,3]
    allband = ['g','r','z']

    seeing = ann['seeing']
    mjd = ann['mjd_obs']

    phot = ann['photometric']
    pass1 = ann['tilepass']==1
    pass2 = ann['tilepass']==2
    pass3 = ann['tilepass']==3
    print('Total number of unphotometric CCDs {:g}'.format(np.sum((phot*1)==0)*1))

    sbins = 25
    srange = [0.6,2.7]
    sbinsz = (srange[1]-srange[0])/sbins

    sns.set(style='white',font_scale=1.3,palette='Set2')

    #-------------------
    # g-band seeing vs UT time in pass 1
    qafile = 'seeing_bytime.png'

    col = sns.color_palette()

    fig, ax = plt.subplots(3, 1, figsize=(6,8))
    for iband, band in enumerate(allband):
        indx = np.where((ann['filter']==band)*1)[0]
        indxarr = np.arange(len(indx))/1E4
        for ipass, thispass in enumerate(allpass):
            these = np.where(ann['tilepass'][indx]==thispass)[0]
            srt = np.argsort(mjd[indx][these])
            ax[iband].scatter(indxarr[these][srt],seeing[indx][these][srt],marker='o',
                              s=0.4,color=col[ipass],
                              label='Pass {:g}'.format(thispass))
        ax[iband].set_xlim([0,indxarr.max()])
        ax[iband].set_ylim(srange)
        plt.text(0.05,0.85,band,transform=ax[iband].transAxes,
                 horizontalalignment='left',fontsize=14)

        if band=='g':
            ax[iband].legend(prop={'size': 11}, labelspacing=0.25, markerscale=8)

        for thisax in ax:
            xlim = thisax.get_xlim()
            thisax.hlines(1.3,xlim[0],xlim[1]*0.99999,colors='k',linestyles='dashed')

    ax[1].set_ylabel('FWHM Seeing (arcsec)')
    ax[2].set_xlabel('Observation Number / 10000')

    fig.tight_layout()
    fig.subplots_adjust(left=0.3, bottom=0.3, wspace=0.1)
    print('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_inches='tight')

    sys.exit(1)
    
    #-------------------
    # Plot the distribution of seeing values for each pass and band.
    qafile = 'seeing_bypass.png'

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10,4))
    for iband, band in enumerate(allband):
        ninband = np.sum((ann['filter']==band)*1)
        for thispass in allpass:
            these = (ann['tilepass']==thispass)*(ann['filter']==band)
            nthese = np.sum(these)
            if nthese>0:
                weights = np.ones_like(seeing[these])/ninband # fraction of CCDs
                (n, bins, patches) = ax[iband].hist(seeing[these],sbins,range=srange,
                                                    alpha=1.0,weights=weights,
                                                    label='Pass {:g}'.format(thispass))
                                                    
                #if band=='g':
                #    print(np.sum(n))
        ax[iband].set_xlim(srange)
        plt.text(0.1,0.9,band,transform=ax[iband].transAxes,
                 horizontalalignment='left',fontsize=14)
        if band=='g':
            ax[iband].set_ylabel('Fraction of CCDs')
        if band=='r':
            ax[iband].set_xlabel('FWHM Seeing (arcsec)')
        if band=='z':
            ax[iband].legend(prop={'size': 11}, labelspacing=0.25)

    # Add a vertical line at 1.3 arcsec
    for thisax in ax:
        ylim = thisax.get_ylim()
        thisax.vlines(1.3,ylim[0],ylim[1]*0.99999,colors='k',linestyles='dashed')

    fig.tight_layout()
    fig.subplots_adjust(left=0.2, bottom=0.2, wspace=0.1)
    print('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_inches='tight')

    sys.exit(1)

    ##-------------------
    ## Plot the distribution of seeing values for photometric vs non-photometric
    ## CCDs as a function of pass number.
    #qafile = 'seeing_bypass_byphot.png'
    #
    #fig, ax = plt.subplots(3, 3, sharey=True, figsize=(10,4))
    #for ipass, thispass in enumerate(allpass):
    #    for iband, band in enumerate(allband):
    #        ninband = np.sum((ann['filter']==band)*(ann['tilepass']==thispass)*1)
    #        phot = (ann['tilepass']==thispass)*(ann['filter']==band)*(ann['photometric']==True)
    #        unphot = (ann['tilepass']==thispass)*(ann['filter']==band)*(ann['photometric']==False)
    #        print(np.sum(phot*1), np.sum(unphot*1))
    #        #weights = np.ones_like(seeing[these])/ninband # fraction of CCDs
    #        ax[ipass, iband].hist(seeing[phot],sbins,range=srange,color="#3498db")
    #        ax[ipass, iband].hist(seeing[unphot],sbins,range=srange,color="#34495e")
    #        print(thispass, band)
    #
    #fig.tight_layout()
    #fig.subplots_adjust(left=0.2, bottom=0.2, wspace=0.1)
    #print('Writing {}'.format(qafile))
    #plt.savefig(qafile,bbox_inches='tight')

if __name__ == "__main__":
    main()
