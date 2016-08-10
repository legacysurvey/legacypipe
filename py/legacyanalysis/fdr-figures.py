#!/usr/bin/env python

"""Generate the figures for Sec 3.3 (ELG target selection) of the FDR.

J. Moustakas
Siena College
2016 Feb 24

"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import os
import sys
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

def readfile(infile):
    if os.path.isfile(infile):
        print('Reading {}'.format(infile))
        data = fits.getdata(infile,1)
    else:
        print('File {} not found'.format(infile))
        sys.exit(1)
    return data

def build_sample(topdir='.', build_cfhtls=False, build_sdss=False):

    # Write out the matched CFHTLS-DECALS file
    if build_cfhtls:
        allcfhtls = readfile(os.path.join(topdir, 'cfhtls-d2-i.fits.gz'))
        alldecals = readfile(os.path.join(topdir, 'decals-dr2-cfhtls-d2-i.fits.gz'))

        keep = np.where((alldecals['BRICKID']!=-999)*
                        (alldecals['BRICK_PRIMARY']=='T')*
                        (np.sum((alldecals['DECAM_FLUX'][:,[1,2,4]]>0)*1,axis=1)==3)*
                        (np.sum((alldecals['DECAM_ANYMASK'][:,[1,2,4]]>0)*1,axis=1)==0)*
                        (alldecals['DECAM_FLUX'][:,2]<(10**(0.4*(22.5-17.5))))*
                        (alldecals['DECAM_FLUX'][:,2]>(10**(0.4*(22.5-21.5))))*
                        (allcfhtls['G_MAG_AUTO']<90)*
                        (allcfhtls['R_MAG_AUTO']<90)*
                        (allcfhtls['Z_MAG_AUTO']<90)*
                        (allcfhtls['G_MAG_AUTO']>0)*
                        (allcfhtls['R_MAG_AUTO']>0)*
                        (allcfhtls['Z_MAG_AUTO']>0)*
                        (allcfhtls['G_MAGERR_AUTO']>0)*
                        (allcfhtls['R_MAGERR_AUTO']>0)*
                        (allcfhtls['Z_MAGERR_AUTO']>0)*
                        (allcfhtls['I_DUBIOUS']==0)*
                        (allcfhtls['I_FLAGS']==0))[0]
        nobj = len(keep)
        print('Number of objects = {}'.format(nobj))

        decals = alldecals[keep]
        cfhtls = allcfhtls[keep]

        cols = [
            ('DECALS_G', 'f4'),
            ('DECALS_R', 'f4'),
            ('DECALS_Z', 'f4'),
            ('DECALS_GR', 'f4'),
            ('DECALS_RZ', 'f4'),
            ('STAR', 'B'),
            #('DECALS_TYPE', 'S10'),
            ('CFHTLS_U', 'f4'),
            ('CFHTLS_G', 'f4'),
            ('CFHTLS_R', 'f4'),
            ('CFHTLS_I', 'f4'),
            ('CFHTLS_Z', 'f4'),
            ('CFHTLS_UG','f4'),
            ('CFHTLS_GR','f4'),
            ('CFHTLS_RI','f4'),
            ('CFHTLS_IZ','f4'),
            ('CFHTLS_RAD','f4')]
        out = Table(np.zeros(nobj, dtype=cols))
        out['DECALS_G'] = -2.5*np.log10(decals['DECAM_FLUX'][:,1])+22.5
        out['DECALS_R'] = -2.5*np.log10(decals['DECAM_FLUX'][:,2])+22.5
        out['DECALS_Z'] = -2.5*np.log10(decals['DECAM_FLUX'][:,4])+22.5
        out['DECALS_GR'] = out['DECALS_G']-out['DECALS_R']
        out['DECALS_RZ'] = out['DECALS_R']-out['DECALS_Z']
        out['STAR'] = decals['TYPE']=='PSF'
        #out['DECALS_TYPE'] = decals['TYPE']

        out['CFHTLS_U'] = cfhtls['U_MAG_APER_6']
        out['CFHTLS_G'] = cfhtls['G_MAG_APER_6']
        out['CFHTLS_R'] = cfhtls['R_MAG_APER_6']
        out['CFHTLS_I'] = cfhtls['I_MAG_APER_6']
        out['CFHTLS_Z'] = cfhtls['Z_MAG_APER_6']
        out['CFHTLS_UG'] = out['CFHTLS_U']-out['CFHTLS_G']
        out['CFHTLS_GR'] = out['CFHTLS_G']-out['CFHTLS_R']
        out['CFHTLS_RI'] = out['CFHTLS_R']-out['CFHTLS_I']
        out['CFHTLS_IZ'] = out['CFHTLS_I']-out['CFHTLS_Z']
        out['CFHTLS_RAD'] = cfhtls['I_FLUX_RADIUS']*0.187 # arcsec

        outfile = 'cfhtls-decals.fits'
        print('Writing {}'.format(outfile))
        if os.path.isfile(outfile):
            os.remove(outfile)
        out.write(outfile)
 
    # Write out the matched SDSS-DECALS file
    if build_sdss:
        allsdss = readfile(os.path.join(topdir, 'stripe82-dr12-stars.fits.gz'))
        alldecals = readfile(os.path.join(topdir, 'decals-dr2-stripe82-dr12-stars.fits.gz'))

        keep = np.where((alldecals['BRICKID']!=-999)*
                        (alldecals['BRICK_PRIMARY']=='T')*
                        (np.sum((alldecals['DECAM_FLUX'][:,[1,2,4]]>0)*1,axis=1)==3)*
                        (np.sum((alldecals['DECAM_ANYMASK'][:,[1,2,4]]>0)*1,axis=1)==0)*
                        (alldecals['DECAM_FLUX'][:,2]<(10**(0.4*(22.5-18.0))))*
                        (alldecals['DECAM_FLUX'][:,2]>(10**(0.4*(22.5-19.0))))*
                        (allsdss['PSFMAG_G']<24.5)*
                        (allsdss['PSFMAG_R']<22.0)*
                        (allsdss['PSFMAG_Z']<22.5)*
                        (allsdss['PSFMAGERR_Z']<1.0))[0] # z
        nobj = len(keep)
        print('Number of objects = {}'.format(nobj))

        decals = alldecals[keep]
        sdss = allsdss[keep]

        cols = [
            ('DECALS_G', 'f4'),
            ('DECALS_R', 'f4'),
            ('DECALS_Z', 'f4'),
            ('DECALS_GR', 'f4'),
            ('DECALS_RZ', 'f4'),
            ('DECALS_TYPE', 'S10'),
            ('SDSS_U', 'f4'),
            ('SDSS_G', 'f4'),
            ('SDSS_R', 'f4'),
            ('SDSS_I', 'f4'),
            ('SDSS_Z', 'f4'),
            ('SDSS_UG','f4'),
            ('SDSS_GR','f4'),
            ('SDSS_RI','f4'),
            ('SDSS_IZ','f4')]
        out = Table(np.zeros(nobj, dtype=cols))
        out['DECALS_G'] = -2.5*np.log10(decals['DECAM_FLUX'][:,1])+22.5
        out['DECALS_R'] = -2.5*np.log10(decals['DECAM_FLUX'][:,2])+22.5
        out['DECALS_Z'] = -2.5*np.log10(decals['DECAM_FLUX'][:,4])+22.5
        out['DECALS_GR'] = out['DECALS_G']-out['DECALS_R']
        out['DECALS_RZ'] = out['DECALS_R']-out['DECALS_Z']
        out['DECALS_TYPE'] = decals['TYPE']

        out['SDSS_U'] = sdss['PSFMAG_U']
        out['SDSS_G'] = sdss['PSFMAG_G']
        out['SDSS_R'] = sdss['PSFMAG_R']
        out['SDSS_I'] = sdss['PSFMAG_I']
        out['SDSS_Z'] = sdss['PSFMAG_Z']
        out['SDSS_UG'] = out['SDSS_U']-out['SDSS_G']
        out['SDSS_GR'] = out['SDSS_G']-out['SDSS_R']
        out['SDSS_RI'] = out['SDSS_R']-out['SDSS_I']
        out['SDSS_IZ'] = out['SDSS_I']-out['SDSS_Z']

        outfile = 'sdss-decals.fits'
        print('Writing {}'.format(outfile))
        if os.path.isfile(outfile):
            os.remove(outfile)
        out.write(outfile)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--grz-plot', action='store_true', help='g-r vs r-z coded by [OII] strength')
    parser.add_argument('--dr2_plot', action='store_true', help='use matched dr2 cats')

    #if len(sys.argv)==1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()

    key = 'DESI_ROOT'
    if key in os.environ:
		desidir = os.getenv(key)
    else:
		desidir = '/project/projectdirs/desi'
    #targdir = os.path.join(desidir, 'target/analysis/deep2/v1.0')
    targdir = os.path.join(desidir, 'target/analysis/deep2/v2.0')
    dr2dir=os.path.join(desidir,'target/analysis/truth')
    #targdir = os.path.join(desidir, 'target/analysis/truth')
    outdir = 'figures' # output directory
    if not os.path.exists(outdir): os.mkdir(outdir)

    sns.set(style='white', font_scale=1.4, palette='Set2')
    col = sns.color_palette()

    area = 0.4342   # [deg^2]
    oiicut1 = 8E-17 # [erg/s/cm2]

    ## Build the samples.
    #if args.build_cfhtls:
    #    build_sample(topdir, build_cfhtls=True)
    #if args.build_sdss:
    #    build_sample(topdir, build_sdss=True)

    # Read the samples
    #phot = fits.getdata('deep2-phot.fits.gz', 1)
    zcat = fits.getdata(os.path.join(targdir, 'deep2-oii.fits.gz'), 1)
    stars = fits.getdata(os.path.join(targdir, 'deep2-stars.fits.gz'), 1)
    #zcat = fits.getdata(os.path.join(targdir, 'deep2egs-oii.fits.gz'), 1)
    #stars = fits.getdata(os.path.join(targdir, 'deep2egs-stars.fits.gz'), 1)

    # --------------------------------------------------
    # g-r vs r-z coded by [OII] strength
    if args.grz_plot:
        zmin = 0.6
        zmax = 1.6
        rfaint = 23.4
        grrange = (-0.3,2.0)
        rzrange = (-0.5,2.1)

        loz = np.where((zcat['FIELD']==1)*
                       (zcat['ZBEST']<zmin)*
                       (zcat['CFHTLS_R']<rfaint)*1)[0]
        oiifaint = np.where((zcat['FIELD']==1)*
                            (zcat['ZBEST']>zmin)*
                            (zcat['CFHTLS_R']<rfaint)*
                            (zcat['OII_3727_ERR']!=-2.0)*
                            (zcat['OII_3727']<oiicut1)*1)[0]
        oiibright_loz = np.where((zcat['FIELD']==1)*
                                 (zcat['ZBEST']>zmin)*
                                 (zcat['ZBEST']<1.0)*
                                 (zcat['CFHTLS_R']<rfaint)*
                                 (zcat['OII_3727_ERR']!=-2.0)*
                                 (zcat['OII_3727']>oiicut1)*1)[0]
        oiibright_hiz = np.where((zcat['FIELD']==1)*
                                 (zcat['ZBEST']>1.0)*
                                 (zcat['CFHTLS_R']<rfaint)*
                                 (zcat['OII_3727_ERR']!=-2.0)*
                                 (zcat['OII_3727']>oiicut1)*1)[0]
        print(len(loz), len(oiibright_loz), len(oiibright_hiz), len(oiifaint))

        def getgrz(zcat, index):
            gr = zcat['CFHTLS_G'][index] - zcat['CFHTLS_R'][index]
            rz = zcat['CFHTLS_R'][index] - zcat['CFHTLS_Z'][index]
            return gr, rz

        figfile = os.path.join(outdir, 'deep2-elg-grz-oii.png')
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        gr, rz = getgrz(zcat, loz)
        ax.scatter(rz, gr, marker='^', color=col[2], label=r'$z<0.6$')

        gr, rz = getgrz(zcat, oiifaint)
        ax.scatter(rz, gr, marker='s', color='tan', 
                   label=r'$z>0.6, [OII]<8\times10^{-17}$')

        gr, rz = getgrz(zcat, oiibright_loz)
        ax.scatter(rz, gr, marker='o', color='powderblue', 
                   label=r'$z>0.6, [OII]>8\times10^{-17}$')

        gr, rz = getgrz(zcat, oiibright_hiz)
        ax.scatter(rz, gr, marker='o', color='powderblue', edgecolor='black', 
                   label=r'$z>1.0, [OII]>8\times10^{-17}$')

        gr, rz = getgrz(stars, np.where((stars['FIELD']==1)*1)[0])
        sns.kdeplot(rz, gr, cmap='Greys_r', clip=(rzrange, grrange),
                    levels=(1-0.75, 1-0.5, 1-0.25, 1-0.1))
                    #levels=(0.5, 0.6, 0.75, 0.9, 0.99))
        
        ax.set_xlabel(r'$(r - z)$')
        ax.set_ylabel(r'$(g - r)$')
        ax.set_xlim(rzrange)
        ax.set_ylim(grrange)
        plt.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
                   markerscale=1.5)
        #fig.subplots_adjust(left=0.3, bottom=0.3, wspace=0.1)
        print('Writing {}'.format(figfile))
        plt.savefig(figfile, bbox_inches='tight')
    if args.dr2_plot:
        # Fix me!! combine fields 2+3
        field=2
        trac = fits.getdata(os.path.join(dr2dir, 'decals-dr2-deep2-field2.fits.gz'), 1)
        deep = fits.getdata(os.path.join(dr2dir, 'deep2-field2.fits.gz'), 1)
        mags= {}
        for iband,band in zip([1,2,4],['g','r','z']):
            #mags[band]= 22.5 -2.5*np.log10(trac['DECAM_FLUX'][:,iband]/trac['DECAM_MW_TRANSMISSION'][:,iband])
            mags[band]= deep['CFHTLS_%s' % band.upper()]
            figfile = os.path.join(outdir, 'cfhtls_deep2-elg-grz-oii.png')
            #figfile = os.path.join(outdir, 'decam_deep2-elg-grz-oii.png')
        # deep['ZHELIO'] = zcat['ZBEST'] 
        zmin = 0.6
        zmax = 1.6
        rfaint = 23.4
        grrange = (-0.3,2.0)
        rzrange = (-0.5,2.1)

        loz = np.where((deep['FIELD']==field)*
                       (deep['ZHELIO']<zmin)*
                       (mags['r']<rfaint)*1)[0]
        oiifaint = np.where((deep['FIELD']==field)*
                            (deep['ZHELIO']>zmin)*
                            (mags['r']<rfaint)*
                            (deep['OII_3727_ERR']!=-2.0)*
                            (deep['OII_3727']<oiicut1)*1)[0]
        oiibright_loz = np.where((deep['FIELD']==field)*
                                 (deep['ZHELIO']>zmin)*
                                 (deep['ZHELIO']<1.0)*
                                 (mags['r']<rfaint)*
                                 (deep['OII_3727_ERR']!=-2.0)*
                                 (deep['OII_3727']>oiicut1)*1)[0]
        oiibright_hiz = np.where((deep['FIELD']==field)*
                                 (deep['ZHELIO']>1.0)*
                                 (mags['r']<rfaint)*
                                 (deep['OII_3727_ERR']!=-2.0)*
                                 (deep['OII_3727']>oiicut1)*1)[0]
        print(len(loz), len(oiibright_loz), len(oiibright_hiz), len(oiifaint))

        def getgrz_mags(mags, index):
            gr = mags['g'][index] - mags['r'][index]
            rz = mags['r'][index] - mags['z'][index]
            return gr, rz

        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        gr, rz = getgrz_mags(mags, loz)
        ax.scatter(rz, gr, marker='^', color=col[2], label=r'$z<0.6$')

        gr, rz = getgrz_mags(mags, oiifaint)
        ax.scatter(rz, gr, marker='s', color='tan', 
                   label=r'$z>0.6, [OII]<8\times10^{-17}$')

        gr, rz = getgrz_mags(mags, oiibright_loz)
        ax.scatter(rz, gr, marker='o', color='powderblue', 
                   label=r'$z>0.6, [OII]>8\times10^{-17}$')

        gr, rz = getgrz_mags(mags, oiibright_hiz)
        ax.scatter(rz, gr, marker='o', color='powderblue', edgecolor='black', 
                   label=r'$z>1.0, [OII]>8\times10^{-17}$')

        # Star contours
        #gr, rz = getgrz(stars, np.where((stars['FIELD']==1)*1)[0])
        #sns.kdeplot(rz, gr, cmap='Greys_r', clip=(rzrange, grrange),
        #            levels=(1-0.75, 1-0.5, 1-0.25, 1-0.1))
                    #levels=(0.5, 0.6, 0.75, 0.9, 0.99))
        
        ax.set_xlabel(r'$(r - z)$')
        ax.set_ylabel(r'$(g - r)$')
        ax.set_xlim(rzrange)
        ax.set_ylim(grrange)
        plt.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
                   markerscale=1.5)
        #fig.subplots_adjust(left=0.3, bottom=0.3, wspace=0.1)
        print('Writing {}'.format(figfile))
        plt.savefig(figfile, bbox_inches='tight')
    
if __name__ == "__main__":
    main()
