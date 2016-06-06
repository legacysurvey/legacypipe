#!/usr/bin/env python

"""Redo the Tractor photometry of the "large" galaxies in Legacy Survey imaging.

J. Moustakas
Siena College
2016 June 6

"""
from __future__ import division, print_function

import os
import sys
import argparse
import pdb

import numpy as np
import logging as log

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
import seaborn as sns
import matplotlib as mpl

sns.set(style='ticks', font_scale=1.2, palette='Set1')
col = sns.color_palette()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sample', action='store_true', help='Build the sample.')

    args = parser.parse_args()

    # Top-level directory
    key = 'LEGACY_SURVEY_LARGE_GALAXIES'
    if key not in os.environ:
        log.fatal('Required ${} environment variable not set'.format(key))
        return 0
    largedir = os.getenv(key)

    # --------------------------------------------------
    # Build the sample of "big" galaxies and identify the corresponding CCDs and bricks. 

    if args.build_sample:
        catdir = os.getenv('CATALOGS_DIR')
        rc3 = fits.getdata(os.path.join(catdir, 'rc3', 'rc3_catalog.fits'), 1)

        d25_maj = 0.1*10.0**cat['logd_25'] # [arcmin]

        # For each object, create a simple WCS header and then find the unique
        # set of bricks containing the galaxies.  Need to think carefully about
        # galaxies spanning more than one brick.

        
        
        plt.hist(d25_maj, bins=50, range=(0.02,5))
        plt.show()

        

        pdb.set_trace()


    # --------------------------------------------------
    # Generate the brick and truth files.
    if args.bricks:
        brightoptions = [
            '--brickname', '{}'.format(brickname),
            '--nbrick', '{}'.format(nbrick),
            '--nspec', '{}'.format(nspec),
            '--outdir', '{}'.format(simsdir),
            '--brickdir', '{}'.format(brickdir),
            '--seed', '{}'.format(seed),
            '--objtype', 'BGS']

        brightargs = brightsims.parse(np.hstack((brightoptions, simoptions)))
        brightargs.verbose = True
        brightsims.main(brightargs)

    # --------------------------------------------------
    # Fit the redshifts
    if args.zfind:
        inputfile = os.path.join(simsdir, brickname+'-input.fits')
        log.info('Reading {}'.format(inputfile))
        cat = fits.getdata(inputfile, 1)

        log.info('Testing with just one brick!')
        #for ib in range(10, 11):
        for ib in range(nbrick):
            thisbrick = cat['BRICKNAME'][ib]
            brickfiles = [os.path.join(brickdir, 'brick-{}-{}.fits'.format(ch, thisbrick)) for ch in ['b', 'r', 'z']]
            redoptions = [
                '--brick', thisbrick,
                '--nproc', '{}'.format(args.nproc),
                '--specprod_dir', simsdir, 
                '--zrange-galaxy', '{}'.format(redshift[0]), '{}'.format(redshift[1]), 
                '--outfile', os.path.join(brickdir, thisbrick, 'zbest-{}.fits'.format(thisbrick)),
                '--objtype', 'ELG,LRG']
            redargs = zfind.parse(redoptions)
            zfind.main(redargs)

    # --------------------------------------------------
    # Parse and write out the simulation inputs, brick spectra, and redshifts
    if args.results:
        inputfile = os.path.join(simsdir, brickname+'-input.fits')
        log.info('Reading {}'.format(inputfile))
        cat = fits.getdata(inputfile, 1)

        # Build a results table.
        resultfile = makepath(os.path.join(simsdir, '{}-results.fits'.format(brickname)))
        resultcols = [
            ('EXPTIME', 'f4'),
            ('AIRMASS', 'f4'),
            ('MOONPHASE', 'f4'),
            ('MOONANGLE', 'f4'),
            ('MOONZENITH', 'f4'),
            ('SNR_B', 'f4'),
            ('SNR_R', 'f4'),
            ('SNR_Z', 'f4'),
            ('TARGETID', 'i8'),
            ('RMAG', 'f4'),
            ('D4000', 'f4'),
            ('EWHBETA', 'f4'), 
            ('ZTRUE', 'f4'), 
            ('Z', 'f4'), 
            ('ZERR', 'f4'), 
            ('ZWARNING', 'f4')]
        result = Table(np.zeros(nobj, dtype=resultcols))

        result['EXPTIME'].unit = 's'
        result['MOONANGLE'].unit = 'deg'
        result['MOONZENITH'].unit = 'deg'

        for ib in range(nbrick):
            # Copy over some data.
            thisbrick = cat['BRICKNAME'][ib]
            result['EXPTIME'][nspec*ib:nspec*(ib+1)] = cat['EXPTIME'][ib]
            result['AIRMASS'][nspec*ib:nspec*(ib+1)] = cat['AIRMASS'][ib]
            result['MOONPHASE'][nspec*ib:nspec*(ib+1)] = cat['MOONPHASE'][ib]
            result['MOONANGLE'][nspec*ib:nspec*(ib+1)] = cat['MOONANGLE'][ib]
            result['MOONZENITH'][nspec*ib:nspec*(ib+1)] = cat['MOONZENITH'][ib]

            # Read the truth file of the first channel to get the metadata.
            truthfile = os.path.join(brickdir, thisbrick, 'truth-brick-{}-{}.fits'.format('b', thisbrick))
            log.info('Reading {}'.format(truthfile))
            truth = io.Brick(truthfile).hdu_list[4].data

            result['TARGETID'][nspec*ib:nspec*(ib+1)] = truth['TARGETID']
            result['RMAG'][nspec*ib:nspec*(ib+1)] = 22.5-2.5*np.log10(truth['DECAM_FLUX'][:,2])
            result['D4000'][nspec*ib:nspec*(ib+1)] = truth['D4000']
            result['EWHBETA'][nspec*ib:nspec*(ib+1)] = truth['EWHBETA']
            result['ZTRUE'][nspec*ib:nspec*(ib+1)] = truth['TRUEZ']

            # Finally read the zbest file. 
            zbestfile = os.path.join(brickdir, thisbrick, 'zbest-{}.fits'.format(thisbrick))
            if os.path.isfile(zbestfile):
                log.info('Reading {}'.format(zbestfile))
                zbest = read_zbest(zbestfile)
                # There's gotta be a better way than looping here!
                for ii in range(nspec):
                    this = np.where(zbest.targetid[ii] == result['TARGETID'])[0]
                    result['Z'][this] = zbest.z[ii]
                    result['ZERR'][this] = zbest.zerr[ii]
                    result['ZWARNING'][this] = zbest.zwarn[ii]

            #pdb.set_trace()
                    
            # Finally, read the spectra and truth tables, one per channel.
            for channel in ('b','r','z'):
                brickfile = os.path.join(brickdir, thisbrick, 'brick-{}-{}.fits'.format(channel, thisbrick))

                log.info('Reading {}'.format(brickfile))
                brick = io.Brick(brickfile)
                wave = brick.get_wavelength_grid()

                for iobj in range(nspec):
                    flux = brick.hdu_list[0].data[iobj,:]
                    ivar = brick.hdu_list[1].data[iobj,:]
                    these = np.where((wave>np.mean(wave)-50)*(wave<np.mean(wave)+50)*(flux>0))[0]
                    result['SNR_'+channel.upper()][nspec*ib+iobj] = \
                      np.median(np.sqrt(flux[these]*ivar[these]))

        log.info('Writing {}'.format(resultfile))
        write_bintable(resultfile, result, extname='RESULTS', clobber=True)
        
    # --------------------------------------------------
    # Build QAplots
    if args.qaplots:
        phaserange = (-0.05, 1.05)
        snrrange = (0, 3)
        rmagrange = (17.5, 20)
        anglerange = (-5, 155)
        d4000range = (0.9, 2.2)
        snrinterval = 1.0

        cmap = mpl.colors.ListedColormap(sns.color_palette('muted'))
        
        resultfile = os.path.join(simsdir, '{}-results.fits'.format(brickname))
        log.info('Reading {}'.format(resultfile))
        res = fits.getdata(resultfile, 1)

        qafile = os.path.join(simsdir, 'qa-{}.pdf'.format(brickname))
        log.info('Writing {}'.format(qafile))

        # ------------------------------
        # Simulation 1
        if args.sim == 1:
            fig, ax0 = plt.subplots(1, 1, figsize=(6, 4.5))

            ax0.scatter(res['MOONANGLE']+rand.normal(0, 0.2, nobj), res['SNR_B'], label='b channel', c=col[1])
            ax0.scatter(res['MOONANGLE']+rand.normal(0, 0.2, nobj), res['SNR_R'], label='r channel', c=col[2])
            ax0.scatter(res['MOONANGLE']+rand.normal(0, 0.2, nobj), res['SNR_Z'], label='z channel', c=col[0])
            ax0.set_xlabel('Object-Moon Angle (deg)')
            ax0.set_ylabel(r'Signal-to-Noise Ratio (pixel$^{-1}$)')
            ax0.set_xlim(anglerange)
            ax0.set_ylim(snrrange)

            plt.legend(loc='upper left', labelspacing=0.25)
            plt.text(0.95, 0.7, 't = {:g} s\nr = {:g} mag\nRedshift = {:.2f}-{:.2f}'.format(exptime, rmag,
                                                                                            redshift[0], redshift[1])+\
                     '\nLunar Zenith Angle = {:g} deg\nLunar Phase = {:g}'.format(zenith, phase),
                     horizontalalignment='right', transform=ax0.transAxes,
                     fontsize=11)
    
            plt.subplots_adjust(bottom=0.2, right=0.95, left=0.15)
            plt.savefig(qafile)
            plt.close()

            #pdb.set_trace()

        # ------------------------------
        # Simulation 2
        if args.sim == 2:
            fig, ax0 = plt.subplots(1, 1, figsize=(6, 4.5))

            ax0.scatter(res['MOONPHASE']+rand.normal(0, 0.02, nobj), res['SNR_B'], label='b channel', c=col[1])
            ax0.scatter(res['MOONPHASE']+rand.normal(0, 0.02, nobj), res['SNR_R'], label='r channel', c=col[2])
            ax0.scatter(res['MOONPHASE']+rand.normal(0, 0.02, nobj), res['SNR_Z'], label='z channel', c=col[0])
            ax0.set_xlabel('Lunar Phase (0=Full, 1=New)')
            ax0.set_ylabel(r'Signal-to-Noise Ratio (pixel$^{-1}$)')
            ax0.set_xlim(phaserange)
            ax0.set_ylim(snrrange)

            plt.legend(loc='upper left', labelspacing=0.25)
            plt.text(0.95, 0.7, 't = {:g} s\nr = {:g} mag\nRedshift = {:.2f}-{:.2f}'.format(exptime, rmag,
                                                                                            redshift[0], redshift[1])+\
                     '\nLunar Zenith Angle = {:g} deg\nObject-Moon Angle = {:g} deg'.format(zenith, angle),
                     horizontalalignment='right', transform=ax0.transAxes,
                     fontsize=11)
    
            plt.subplots_adjust(bottom=0.2, right=0.95, left=0.15)
            plt.savefig(qafile)
            plt.close()

        # ------------------------------
        # Simulation 3
        if args.sim == 3:
            fig, ax0 = plt.subplots(1, 1, figsize=(6, 4))

            im = ax0.scatter(res['MOONANGLE']+rand.normal(0, 0.3, nobj), res['SNR_R'], c=res['MOONPHASE'], vmin=phaserange,
                             cmap=cmap)
            ax0.set_xlabel('Object-Moon Angle (deg)')
            ax0.set_ylabel(r'r channel Signal-to-Noise Ratio (pixel$^{-1}$)')
            ax0.set_xlim(anglerange)
            ax0.set_ylim(snrrange)
            ax0.yaxis.set_major_locator(mpl.ticker.MultipleLocator(snrinterval))

            plt.text(0.95, 0.7, 't = {:g} s\nr = {:g} mag\nRedshift = {:.2f}-{:.2f}'.format(exptime, rmag,
                                                                                            redshift[0], redshift[1])+\
                     '\nLunar Zenith Angle = {:g} deg'.format(zenith),
                     horizontalalignment='right', transform=ax0.transAxes,
                     fontsize=11)
    
            cbar = fig.colorbar(im)
            cbar.set_ticks([0, 0.5, 1])
            cbar.ax.set_yticklabels(['Full','Quarter','New'], rotation=90)
            cbar.ax.set_ylabel('Lunar Phase')

            plt.subplots_adjust(bottom=0.2, right=1.0)
            plt.savefig(qafile)
            plt.close()

        # ------------------------------
        # Simulation 4
        if args.sim == 4:
            bins = 30
            zgood = (np.abs(res['Z']-res['ZTRUE'])<5E-5)*(res['ZWARNING']==0)*1

            H, xedges, yedges = np.histogram2d(res['RMAG'], res['MOONPHASE'], bins=bins, weights=zgood)
            H2, _, _ = np.histogram2d(res['RMAG'], res['MOONPHASE'], bins=bins)
            extent = np.array((rmagrange, phaserange)).flatten()

            #pdb.set_trace()            

            fig, ax0 = plt.subplots(1, 1, figsize=(6, 4))
            im = ax0.imshow(H/H2, extent=extent, interpolation='nearest')#, cmap=cmap)
            ax0.set_xlabel('r (AB mag)')
            ax0.set_ylabel('Lunar Phase (0=Full, 1=New)')
            ax0.set_xlim(rmagrange)
            ax0.set_ylim(phaserange)

            #plt.text(0.95, 0.7, 't = {:g} s\nr = {:g} mag\nRedshift = {:.2f}-{:.2f}'.format(exptime, rmag,
            #                                                                                redshift[0], redshift[1])+\
            #         '\nLunar Zenith Angle = {:g} deg'.format(zenith),
            #         horizontalalignment='right', transform=ax0.transAxes,
            #         fontsize=11)
    
            cbar = fig.colorbar(im)
            #cbar.set_ticks([0, 0.5, 1])
            #cbar.ax.set_yticklabels(['Full','Quarter','New'], rotation=90)
            #cbar.ax.set_ylabel('Lunar Phase')

            plt.subplots_adjust(bottom=0.2, right=1.0)
            plt.savefig(qafile)
            plt.close()
        
        # ------------------------------
        # Simulation 10
        if args.sim == 10:
            zgood = (np.abs(res['Z']-res['ZTRUE'])<0.001)*(res['ZWARNING']==0)*1
            #pdb.set_trace()            
            
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6,6))

            # moon phase vs S/N(r)
            im = ax0.scatter(res['RMAG'], res['SNR_R'], c=res['MOONPHASE'], vmin=phaserange,
                             cmap=cmap)
            ax0.set_xlabel('r (AB mag)')
            ax0.set_ylabel('S/N (r channel)')
            ax0.set_xlim(rmagrange)
            ax0.set_ylim(snrrange)
            ax0.yaxis.set_major_locator(mpl.ticker.MultipleLocator(snrinterval))

            # object-moon angle vs S/N(r)
            im = ax1.scatter(res['MOONANGLE'], res['SNR_R'], c=res['MOONPHASE'], vmin=phaserange,
                             cmap=cmap)
            ax1.set_xlabel('Object-Moon Angle (deg)')
            ax1.set_ylabel('S/N (r channel)')
            ax1.set_xlim(anglerange)
            ax1.set_ylim(snrrange)
            ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(snrinterval))

            # D(4000) vs S/N(r)
            im = ax2.scatter(res['D4000'], res['SNR_R'], c=res['MOONPHASE'], vmin=phaserange,
                             cmap=cmap)
            ax2.set_xlabel('$D_{n}(4000)$')
            ax2.set_ylabel('S/N (r channel)')
            ax2.set_xlim(d4000range)
            ax2.set_ylim(snrrange)
            ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(snrinterval))

            # Shared colorbar
            cbarax = fig.add_axes([0.83, 0.15, 0.03, 0.8])
            cbar = fig.colorbar(im, cax=cbarax)
            ticks = ['Full','Quarter','New']
            cbar.set_ticks([0, 0.5, 1])
            cbar.ax.set_yticklabels(ticks, rotation=-45)
        
            plt.tight_layout(pad=0.5)#, h_pad=0.2, w_pad=0.3)
            plt.subplots_adjust(right=0.78)
            plt.savefig(qafile)
            plt.close()
        
            
if __name__ == "__main__":
    main()
