from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import PlotSequence

from legacyanalysis.ps1cat import ps1cat, ps1_to_decam
from legacypipe.common import *

if __name__ == '__main__':
    survey_dir = '/project/projectdirs/desiproc/dr3'
    survey = LegacySurveyData(survey_dir=survey_dir)

    ralo,rahi = 240,245
    declo,dechi = 5, 12

    ps = PlotSequence('comp')
    
    bricks = survey.get_bricks_readonly()
    bricks = bricks[(bricks.ra > ralo) * (bricks.ra < rahi) *
                    (bricks.dec > declo) * (bricks.dec < dechi)]
    print(len(bricks), 'bricks')

    I, = np.nonzero([os.path.exists(survey.find_file('tractor', brick=b.brickname))
                    for b in bricks])
    print(len(I), 'bricks with catalogs')
    bricks.cut(I)

    bands = 'grz'

    for band in bands:
        bricks.set('diff_%s' % band, np.zeros(len(bricks), np.float32))
    
    diffs = dict([(b,[]) for b in bands])
    for ibrick,b in enumerate(bricks):
        fn = survey.find_file('tractor', brick=b.brickname)
        T = fits_table(fn)
        print(len(T), 'sources in', b.brickname)

        brickwcs = wcs_for_brick(b)
        
        ps1 = ps1cat(ccdwcs=brickwcs)
        stars = ps1.get_stars()
        print(len(stars), 'PS1 sources')

        # Now cut to just *stars* with good colors
        stars.gicolor = stars.median[:,0] - stars.median[:,2]
        keep = (stars.gicolor > 0.4) * (stars.gicolor < 2.7)
        stars.cut(keep)
        print(len(stars), 'PS1 stars with good colors')
        
        I,J,d = match_radec(T.ra, T.dec, stars.ra, stars.dec, 1./3600.)
        print(len(I), 'matches')
        
        for band in bands:
            colorterm = ps1_to_decam(stars.median[J], band)

            psmag = stars.median[J, ps1.ps1band[band]]
            psmag += colorterm
            
            decflux = T.decam_flux[I, survey.index_of_band(band)]
            decmag = -2.5 * (np.log10(decflux) - 9)

            #K = np.flatnonzero((psmag > 14) * (psmag < 24))
            #print(len(K), 'with mag 14 to 24')
            K = np.flatnonzero((psmag > 14) * (psmag < 21))
            print(len(K), 'with mag 14 to 21')
            decmag = decmag[K]
            psmag  = psmag [K]
            K = np.flatnonzero(np.abs(decmag - psmag) < 1)
            print(len(K), 'with good mag matches (< 1 mag difference)')
            decmag = decmag[K]
            psmag  = psmag [K]

            if ibrick == 0:
                plt.clf()
                #plt.plot(psmag, decmag, 'b.')
                plt.plot(psmag, decmag - psmag, 'b.')
                plt.xlabel('PS1 mag')
                plt.xlabel('DECam - PS1 mag')
                plt.title('PS1 matches for %s band, brick %s' % (band, b.brickname))
                ps.savefig()

            mdiff = np.median(decmag - psmag)
            diffs[band].append(mdiff)
            print('Median difference:', mdiff)

            bricks.get('diff_%s' % band)[ibrick] = mdiff

            
    for band in bands:
        d = diffs[band]

        plt.clf()
        plt.hist(d, bins=20, range=(-0.02, 0.02), histtype='step')
        plt.xlabel('Median mag difference per brick')
        plt.title('DR3 EDR PS1 vs DECaLS: %s band' % band)
        ps.savefig()

        print('Median differences in', band, 'band:', np.median(d))
        
    plt.clf()
    plt.hist(diffs['g'], bins=20, range=(-0.02, 0.02), histtype='step', color='g')
    plt.hist(diffs['r'], bins=20, range=(-0.02, 0.02), histtype='step', color='r')
    plt.hist(diffs['z'], bins=20, range=(-0.02, 0.02), histtype='step', color='m')
    plt.xlabel('Median mag difference per brick')
    plt.title('DR3 EDR PS1 vs DECaLS')
    ps.savefig()

    rr,dd = np.meshgrid(np.linspace(ralo,rahi, 400), np.linspace(declo,dechi, 400))
    I,J,d = match_radec(rr.ravel(), dd.ravel(), bricks.ra, bricks.dec, 0.18, nearest=True)
    print(len(I), 'matches')

    for band in bands:
        plt.clf()
        dmag = np.zeros_like(rr) - 1.
        dmag.ravel()[I] = bricks.get('diff_%s' % band)[J]
        plt.imshow(dmag, interpolation='nearest', origin='lower',
                   vmin=-0.01, vmax=0.01, cmap='hot',
                   extent=(ralo,rahi,declo,dechi))
        plt.colorbar()
        plt.title('DR3 EDR PS1 vs DECaLS: %s band' % band)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.axis([ralo,rahi,declo,dechi])
        ps.savefig()

    for band in bands:
        plt.clf()
        plt.scatter(bricks.ra, bricks.dec, c=bricks.get('diff_%s' % band), vmin=-0.01, vmax=0.01,
                    edgecolors='face', s=200)
        plt.colorbar()
        plt.title('DR3 EDR PS1 vs DECaLS: %s band' % band)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.axis('scaled')
        plt.axis([ralo,rahi,declo,dechi])
        ps.savefig()
        
    
