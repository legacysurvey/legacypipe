from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os
import sys

from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import PlotSequence

from legacyanalysis.ps1cat import ps1cat, ps1_to_decam
from legacypipe.common import *

def main():
    survey_dir = '/project/projectdirs/desiproc/dr3'
    survey = LegacySurveyData(survey_dir=survey_dir)

    ralo,rahi = 240,245
    declo,dechi = 5, 12

    ps = PlotSequence('comp')
    
    bands = 'grz'


    ccdfn = 'ccds-forced.fits'
    #if not os.path.exists(ccdfn):
    if True:
        ccds = survey.get_annotated_ccds()
        ccds.cut((ccds.ra > ralo) * (ccds.ra < rahi) *
                 (ccds.dec > declo) * (ccds.dec < dechi))
        print(len(ccds), 'CCDs')
        
        ccds.path = np.array([os.path.join('dr3', 'forced', ('%08i' % e)[:5], '%08i' % e, 'decam-%08i-%s-forced.fits' % (e, n.strip()))
                              for e,n in zip(ccds.expnum, ccds.ccdname)])
        I, = np.nonzero([os.path.exists(fn) for fn in ccds.path])
        print(len(I), 'CCDs with forced photometry')
        ccds.cut(I)

        #ccds = ccds[:500]
        #e,I = np.unique(ccds.expnum, return_index=True)
        #print(len(I), 'unique exposures')
        #ccds.cut(I)
        
        FF = read_forcedphot_ccds(ccds, survey)
        FF.writeto('forced-all-matches.fits')
        
        ccds.writeto(ccdfn)

    ccds = fits_table(ccdfn)

    plt.clf()
    plt.hist(ccds.nforced, bins=100)
    plt.title('nforced')
    ps.savefig()
    
    plt.clf()
    plt.hist(ccds.nmatched, bins=100)
    plt.title('nmatched')
    ps.savefig()

    #ccds.cut(ccds.nmatched >= 150)
    ccds.cut(ccds.nmatched >= 50)
    
    neff = 1. / ccds.psfnorm_mean**2
    # Narcsec is in arcsec**2
    narcsec = neff * ccds.pixscale_mean**2
    # to arcsec
    narcsec = np.sqrt(narcsec)
    # Correction factor to get back to equivalent of Gaussian sigma
    narcsec /= (2. * np.sqrt(np.pi))
    # Conversion factor to FWHM (2.35)
    narcsec *= 2. * np.sqrt(2. * np.log(2.))
    ccds.psfsize = narcsec
    
    for band in bands:
        I = np.flatnonzero(ccds.filter == band)

        mxsee = 4.
        mlo,mhi = -0.01, 0.05
        
        plt.clf()
        plt.plot(np.clip(ccds.psfsize[I], 0, mxsee),
                 np.clip(ccds.mdiff[I], mlo,mhi), 'k.', alpha=0.1)

        # for p in [1,2,3]:
        #     J = np.flatnonzero(ccds.tilepass[I] == p)
        #     if len(J):
        #         plt.plot(np.clip(ccds.psfsize[I[J]], 0, mxsee),
        #                  np.clip(ccds.mdiff[I[J]], mlo,mhi), '.', color='rgb'[p-1], alpha=0.2)

        #plt.plot(ccds.seeing[I], ccds.mdiff[I], 'b.')
        plt.xlabel('PSF size (arcsec)')
        plt.ylabel('DECaLS PSF - PS1 (mag)')
        plt.axhline(0, color='k', alpha=0.2)
        plt.axis([0, mxsee, mlo,mhi])
        plt.title('DR3: EDR region, Forced phot: %s band' % band)
        ps.savefig()


    # Group by exposure

    for band in bands:
        I = np.flatnonzero(ccds.filter == band)

        E,J = np.unique(ccds.expnum[I], return_index=True)
        print(len(E), 'unique exposures in', band)
        exps = ccds[I[J]]
        print(len(exps), 'unique exposures in', band)
        assert(len(np.unique(exps.expnum)) == len(exps))
        exps.ddiff = np.zeros(len(exps))
        exps.dsize = np.zeros(len(exps))
        exps.nccds = np.zeros(len(exps), int)
        
        for iexp,exp in enumerate(exps):
            J = np.flatnonzero(ccds.expnum[I] == exp.expnum)
            J = I[J]
            print(len(J), 'CCDs in exposure', exp.expnum)
            exps.mdiff[iexp] = np.median(ccds.mdiff[J])
            exps.ddiff[iexp] = (np.percentile(ccds.mdiff[J], 84) - np.percentile(ccds.mdiff[J], 16))/2.
            exps.psfsize[iexp] = np.median(ccds.psfsize[J])
            exps.dsize[iexp] = (np.percentile(ccds.psfsize[J], 84) - np.percentile(ccds.psfsize[J], 16))/2.
            exps.nccds[iexp] = len(J)

        mxsee = 4.
        mlo,mhi = -0.01, 0.05

        exps.cut(exps.nccds >= 10)

        
        plt.clf()
        plt.errorbar(np.clip(exps.psfsize, 0, mxsee),
                     np.clip(exps.mdiff, mlo,mhi), yerr=exps.ddiff,
                     #xerr=exps.dsize,
                     fmt='.')
        #plt.plot(ccds.seeing[I], ccds.mdiff[I], 'b.')
        plt.xlabel('PSF size (arcsec)')
        plt.ylabel('DECaLS PSF - PS1 (mag)')
        plt.axhline(0, color='k', alpha=0.2)
        plt.axis([0, mxsee, mlo,mhi])
        plt.title('DR3: EDR region, Forced phot: %s band' % band)
        ps.savefig()



        
    sys.exit(0)

    
def read_forcedphot_ccds(ccds, survey):
    ccds.mdiff = np.zeros(len(ccds))
    ccds.mscatter = np.zeros(len(ccds))

    Nap = 8
    ccds.apdiff = np.zeros((len(ccds), Nap))
    ccds.apscatter = np.zeros((len(ccds), Nap))

    ccds.nforced = np.zeros(len(ccds), np.int16)
    ccds.nunmasked = np.zeros(len(ccds), np.int16)
    ccds.nmatched = np.zeros(len(ccds), np.int16)
    ccds.nps1 = np.zeros(len(ccds), np.int16)
    
    brickcache = {}

    FF = []
    
    for iccd,ccd in enumerate(ccds):
        print('CCD', iccd, 'of', len(ccds))
        F = fits_table(ccd.path)
        print(len(F), 'sources in', ccd.path)

        ccds.nforced[iccd] = len(F)
        
        # arr, have to match with brick sources to get RA,Dec.
        F.ra  = np.zeros(len(F))
        F.dec = np.zeros(len(F))
        F.masked = np.zeros(len(F), bool)

        maglo,maghi = 14.,21.
        maxdmag = 1.
        
        F.mag = -2.5 * (np.log10(F.flux) - 9)
        F.cut((F.flux > 0) * (F.mag > maglo-maxdmag) * (F.mag < maghi+maxdmag))
        print(len(F), 'sources between', (maglo-maxdmag), 'and', (maghi+maxdmag), 'mag')

        im = survey.get_image_object(ccd)
        print('Reading DQ image for', im)
        dq = im.read_dq()
        H,W = dq.shape
        ix = np.clip(np.round(F.x), 0, W-1).astype(int)
        iy = np.clip(np.round(F.y), 0, H-1).astype(int)
        F.mask = dq[iy,ix]
        print(np.sum(F.mask != 0), 'sources are masked')
        
        for brickname in np.unique(F.brickname):
            if not brickname in brickcache:
                brickcache[brickname] = fits_table(survey.find_file('tractor', brick=brickname))
            T = brickcache[brickname]
            idtoindex = np.zeros(T.objid.max()+1, int) - 1
            idtoindex[T.objid] = np.arange(len(T))

            I = np.flatnonzero(F.brickname == brickname)
            J = idtoindex[F.objid[I]]
            assert(np.all(J >= 0))
            F.ra [I] = T.ra [J]
            F.dec[I] = T.dec[J]

            F.masked[I] = (T.decam_anymask[J,:].max(axis=1) > 0)

        #F.cut(F.masked == False)
        #print(len(F), 'not masked')
        print(np.sum(F.masked), 'masked in ANYMASK')

        ccds.nunmasked[iccd] = len(F)
            
        wcs = Tan(*[float(x) for x in [ccd.crval1, ccd.crval2, ccd.crpix1, ccd.crpix2,
                                       ccd.cd1_1, ccd.cd1_2, ccd.cd2_1, ccd.cd2_2,
                                       ccd.width, ccd.height]])

        ps1 = ps1cat(ccdwcs=wcs)
        stars = ps1.get_stars()
        print(len(stars), 'PS1 sources')
        ccds.nps1[iccd] = len(stars)
        
        # Now cut to just *stars* with good colors
        stars.gicolor = stars.median[:,0] - stars.median[:,2]
        keep = (stars.gicolor > 0.4) * (stars.gicolor < 2.7)
        stars.cut(keep)
        print(len(stars), 'PS1 stars with good colors')

        stars.cut(np.minimum(stars.stdev[:,1], stars.stdev[:,2]) < 0.05)
        print(len(stars), 'PS1 stars with min stdev(r,i) < 0.05')
        
        I,J,d = match_radec(F.ra, F.dec, stars.ra, stars.dec, 1./3600.)
        print(len(I), 'matches')

        band = ccd.filter

        colorterm = ps1_to_decam(stars.median[J], band)

        F.cut(I)
        F.psmag = stars.median[J, ps1.ps1band[band]] + colorterm

        K = np.flatnonzero((F.psmag > maglo) * (F.psmag < maghi))
        print(len(K), 'with mag', maglo, 'to', maghi)
        F.cut(K)

        K = np.flatnonzero(np.abs(F.mag - F.psmag) < maxdmag)
        print(len(K), 'with good mag matches (<', maxdmag, 'mag difference)')
        ccds.nmatched[iccd] = len(K)
        if len(K) == 0:
            continue
        F.cut(K)
        
        ccds.mdiff[iccd] = np.median(F.mag - F.psmag)
        ccds.mscatter[iccd] = (np.percentile(F.mag - F.psmag, 84) -
                               np.percentile(F.mag - F.psmag, 16))/2.

        for i in range(Nap):
            apmag = -2.5 * (np.log10(F.apflux[:, i]) - 9)

            ccds.apdiff[iccd,i] = np.median(apmag - F.psmag)
            ccds.apscatter[iccd,i] = (np.percentile(apmag - F.psmag, 84) -
                                      np.percentile(apmag - F.psmag, 16))/2.

        #F.about()
        for c in ['apflux_ivar', 'brickid', 'flux_ivar',
                  'mjd', 'objid', 'fracflux', 'rchi2', 'x','y']:
            F.delete_column(c)

        F.expnum = np.zeros(len(F), np.int32) + ccd.expnum
        F.ccdname = np.array([ccd.ccdname] * len(F))
        F.iforced = np.zeros(len(F), np.int32) + iccd
        
        FF.append(F)

    FF = merge_tables(FF)
        
    return FF



    

    bricks = survey.get_bricks_readonly()
    bricks = bricks[(bricks.ra > ralo) * (bricks.ra < rahi) *
                    (bricks.dec > declo) * (bricks.dec < dechi)]
    print(len(bricks), 'bricks')

    I, = np.nonzero([os.path.exists(survey.find_file('tractor', brick=b.brickname))
                    for b in bricks])
    print(len(I), 'bricks with catalogs')
    bricks.cut(I)
    
    for band in bands:
        bricks.set('diff_%s' % band, np.zeros(len(bricks), np.float32))
        bricks.set('psfsize_%s' % band, np.zeros(len(bricks), np.float32))
        
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
            bricks.get('psfsize_%s' % band)[ibrick] = np.median(
                T.decam_psfsize[:, survey.index_of_band(band)])

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

            if False and ibrick == 0:
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
        
    if False:
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


        plt.clf()
        # reuse 'dmag' map...
        dmag = np.zeros_like(rr)
        dmag.ravel()[I] = bricks.get('psfsize_%s' % band)[J]
        plt.imshow(dmag, interpolation='nearest', origin='lower',
                   cmap='hot', extent=(ralo,rahi,declo,dechi))
        plt.colorbar()
        plt.title('DR3 EDR: DECaLS PSF size: %s band' % band)
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.axis([ralo,rahi,declo,dechi])
        ps.savefig()

    if False:
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


    plt.clf()
    plt.plot(bricks.psfsize_g, bricks.diff_g, 'g.')
    plt.plot(bricks.psfsize_r, bricks.diff_r, 'r.')
    plt.plot(bricks.psfsize_z, bricks.diff_z, 'm.')
    plt.xlabel('PSF size (arcsec)')
    plt.ylabel('DECaLS PSF - PS1 (mag)')
    plt.title('DR3 EDR')
    ps.savefig()



if __name__ == '__main__':
    main()
    
