'''
This is a little script for comparing DECaLS to Pan-STARRS magnitudes for
investigating zeropoint and other issues.
'''
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import os
import scipy.ndimage

from tractor.brightness import NanoMaggies
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.miscutils import *
from astrometry.util.plotutils import *

from legacyanalysis.ps1cat import *
from legacypipe.common import *
from astrometry.libkd.spherematch import *
import photutils

def compare_mags(TT, name, ps):
    for i,T in enumerate(TT):
        T.set('exp', np.zeros(len(T), np.uint8)+i)

    plt.clf()
    ap = 5    
    for i,T in enumerate(TT):
        cc = 'rgb'[i]
        plt.plot(T.flux, T.apflux[:, ap] / T.flux,
                 '.', color=cc, alpha=0.5)

        ff, frac = [],[]
        mags = np.arange(14, 24)
        for mlo,mhi in zip(mags, mags[1:]):
            flo = NanoMaggies.magToNanomaggies(mhi)
            fhi = NanoMaggies.magToNanomaggies(mlo)
            I = np.flatnonzero((T.flux > flo) * (T.flux <= fhi))
            ff.append(np.sqrt(flo * fhi))
            frac.append(np.median(T.apflux[I,ap] / T.flux[I]))
        plt.plot(ff, frac, 'o-', color=cc)
        
    plt.xscale('symlog')
    plt.xlim(1., 1e3)
    plt.ylim(0.9, 1.1)
    plt.xlabel('Forced-phot flux')
    plt.ylabel('Aperture / Forced-phot flux')
    plt.axhline(1, color='k', alpha=0.1)
    plt.title('%s region: Aperture %i fluxes' % (name, ap))
    ps.savefig()

    T = merge_tables(TT)

    T.bobjid = T.brickid.astype(int) * 10000 + T.objid
    bomap = {}
    for i,bo in enumerate(T.bobjid):
        try:
            bomap[bo].append(i)
        except KeyError:
            bomap[bo] = [i]

    II = []
    for bo,ii in bomap.items():
        if len(ii) != 3:
            continue
        II.append(ii)
    II = np.array(II)
    print 'II', II.shape

    exps = T.exp[II]
    print 'exposures:', exps
    assert(np.all(T.exp[II[:,0]] == 0))
    assert(np.all(T.exp[II[:,1]] == 1))
    assert(np.all(T.exp[II[:,2]] == 2))

    fluxes = T.flux[II]
    print 'fluxes', fluxes.shape
    meanflux = np.mean(fluxes, axis=1)
    print 'meanfluxes', meanflux.shape

    plt.clf()
    for i in range(3):
        plt.plot(meanflux, fluxes[:,i] / meanflux, '.',
                 color='rgb'[i], alpha=0.5)
    #plt.yscale('symlog')
    plt.xscale('symlog')
    plt.xlabel('Mean flux (nanomaggies)')
    plt.ylabel('Forced-phot flux / Mean')
    #plt.ylim(0, 2)
    plt.ylim(0.9, 1.1)
    plt.xlim(0, 1e3)
    plt.axhline(1, color='k', alpha=0.1)
    plt.title('%s region: Forced-phot fluxes' % name)
    ps.savefig()

    for ap in [4,5,6]:
        apfluxes = T.apflux[:,ap][II,]
        print 'ap fluxes', apfluxes.shape

        plt.clf()
        for i in range(3):
            plt.plot(meanflux, apfluxes[:,i] / meanflux, '.',
                     color='rgb'[i], alpha=0.5)
        plt.xscale('symlog')
        plt.xlabel('Mean flux (nanomaggies)')
        plt.ylabel('Aperture(%i) flux / Mean' % ap)
        plt.ylim(0.9, 1.1)
        plt.xlim(0, 1e3)
        plt.axhline(1, color='k', alpha=0.1)
        plt.title('%s region: Aperture %i fluxes' % (name, ap))
        ps.savefig()

        plt.clf()
        for i in range(3):
            plt.plot(fluxes[:,i], apfluxes[:,i] / fluxes[:,i], '.',
                     color='rgb'[i], alpha=0.5)
        plt.xscale('symlog')
        plt.xlim(0, 1e3)
        plt.ylim(0.9, 1.1)
        plt.xlabel('Forced-phot flux')
        plt.ylabel('Aperture / Forced-phot flux')
        plt.axhline(1, color='k', alpha=0.1)
        plt.title('%s region: Aperture %i fluxes' % (name, ap))
        ps.savefig()
    

def compare_to_ps1(ps, ccds):
    
    decals = Decals()

    allplots = []

    for expnum,ccdname in ccds:
        ccd = decals.find_ccds(expnum=expnum, ccdname=ccdname)
        assert(len(ccd) == 1)
        ccd = ccd[0]
        im = decals.get_image_object(ccd)
        print 'Reading', im

        wcs = im.get_wcs()

        magrange = (15,20)
        ps1 = ps1cat(ccdwcs=wcs)
        ps1 = ps1.get_stars(band=im.band, magrange=magrange)
        print 'Got', len(ps1), 'PS1 stars'
        # ps1.about()

        F = fits_table('forced-%i-%s.fits' % (expnum, ccdname))
        print 'Read', len(F), 'forced-phot results'
        
        F.ra,F.dec = wcs.pixelxy2radec(F.x+1, F.y+1)

        I,J,d = match_radec(F.ra, F.dec, ps1.ra, ps1.dec, 1./3600.)
        print 'Matched', len(I), 'stars to PS1'

        F.cut(I)
        ps1.cut(J)

        F.mag = NanoMaggies.nanomaggiesToMag(F.flux)
        F.apmag = NanoMaggies.nanomaggiesToMag(F.apflux[:,5])

        iband = ps1cat.ps1band[im.band]
        ps1mag = ps1.median[:,iband]
        mags = np.arange(magrange[0], 1+magrange[1])

        psf = im.read_psf_model(0, 0, pixPsf=True)
        pixscale = 0.262
        apertures = apertures_arcsec / pixscale
        h,w = ccd.height, ccd.width
        psfimg = psf.getPointSourcePatch(w/2., h/2.).patch
        ph,pw = psfimg.shape
        cx,cy = pw/2, ph/2
        apphot = []
        for rad in apertures:
            aper = photutils.CircularAperture((cx,cy), rad)
            p = photutils.aperture_photometry(psfimg, aper)
            apphot.append(p.field('aperture_sum'))
        apphot = np.hstack(apphot)
        print 'aperture photometry:', apphot
        skyest = apphot[6] - apphot[5]
        print 'Sky estimate:', skyest
        skyest /= np.pi * (apertures[6]**2 - apertures[5]**2)
        print 'Sky estimate per pixel:', skyest
        fraction = apphot[5] - skyest * np.pi * apertures[5]**2
        print 'Fraction of flux:', fraction
        zp = 2.5 * np.log10(fraction)
        print 'ZP adjustment:', zp
        
        plt.clf()

        for cc,mag,label in [('b', F.mag, 'Forced mag'), ('r', F.apmag, 'Aper mag')]:
            plt.plot(ps1mag, mag - ps1mag, '.', color=cc, label=label, alpha=0.6)

            mm,dd = [],[]
            for mlo,mhi in zip(mags, mags[1:]):
                I = np.flatnonzero((ps1mag > mlo) * (ps1mag <= mhi))
                mm.append((mlo+mhi)/2.)
                dd.append(np.median(mag[I] - ps1mag[I]))
            plt.plot(mm, dd, 'o-', color=cc)

            mm = np.array(mm)
            dd = np.array(dd)
            plt.plot(mm, dd - zp, 'o--', lw=3, alpha=0.5, color=cc)

            allplots.append((mm, dd, zp, cc, label))
            
        plt.xlabel('PS1 %s mag' % im.band)
        plt.ylabel('Mag - PS1 (mag)')
        plt.title('PS1 - Single-epoch mag: %i-%s' % (expnum, ccdname))
        plt.ylim(-0.2, 0.2)
        mlo,mhi = magrange
        plt.xlim(mhi, mlo)
        plt.axhline(0., color='k', alpha=0.1)
        plt.legend()
        ps.savefig()

    plt.clf()
    # for mm,dd,zp,cc,label in allplots:
    #     plt.plot(mm, dd, 'o-', color=cc, label=label)
    #     plt.plot(mm, dd - zp, 'o--', lw=3, alpha=0.5, color=cc)
    for sp,add in [(1,False),(2,True)]:
        plt.subplot(2,1,sp)
        for mm,dd,zp,cc,label in allplots:
            if add:
                plt.plot(mm, dd - zp, 'o--', lw=3, alpha=0.5, color=cc)
            else:
                plt.plot(mm, dd, 'o-', color=cc, label=label)
        plt.ylabel('Mag - PS1 (mag)')
        plt.ylim(-0.2, 0.05)
        mlo,mhi = magrange
        plt.xlim(mhi, mlo)
        plt.axhline(0., color='k', alpha=0.1)
        plt.axhline(-0.05, color='k', alpha=0.1)
        plt.axhline(-0.1, color='k', alpha=0.1)
    plt.xlabel('PS1 %s mag' % im.band)
    plt.suptitle('PS1 - Single-epoch mags')
    #plt.legend()
    ps.savefig()

    plt.clf()
    for mm,dd,zp,cc,label in allplots:
        plt.plot(mm, dd, 'o-', color=cc, label=label)
        plt.plot(mm, dd - zp, 'o--', lw=3, alpha=0.5, color=cc)
    plt.ylabel('Mag - PS1 (mag)')
    plt.ylim(-0.2, 0.05)
    mlo,mhi = magrange
    plt.xlim(mhi, mlo)
    plt.axhline(0., color='k', alpha=0.1)
    plt.xlabel('PS1 %s mag' % im.band)
    plt.suptitle('PS1 - Single-epoch mags')
    ps.savefig()





def compare_brick_to_ps1(brickname, ps, name='', basedir=''):
    decals = Decals()
    brick = decals.get_brick_by_name(brickname)
    wcs = wcs_for_brick(brick)

    magrange = (15,20)
    ps1 = ps1cat(ccdwcs=wcs)
    ps1 = ps1.get_stars(magrange=magrange)
    print 'Got', len(ps1), 'PS1 stars'

    T = fits_table(os.path.join(basedir, 'tractor', brickname[:3],
                                'tractor-%s.fits' % brickname))
    I,J,d = match_radec(T.ra, T.dec, ps1.ra, ps1.dec, 1./3600.)
    print 'Matched', len(I), 'stars to PS1'

    T.cut(I)
    ps1.cut(J)

    bands = 'z'
    ap = 5    

    allbands = 'ugrizY'
    mags = np.arange(magrange[0], 1+magrange[1])
    
    for band in bands:
        iband = allbands.index(band)
        piband = ps1cat.ps1band[band]
        T.flux = T.decam_flux[:,iband]
        T.mag = NanoMaggies.nanomaggiesToMag(T.flux)
        print 'apflux shape', T.decam_apflux.shape
        
        T.apflux = T.decam_apflux[:, iband, ap]
        T.apmag = NanoMaggies.nanomaggiesToMag(T.apflux)

        ps1mag = ps1.median[:,piband]

        plt.clf()

        for cc,mag,label in [('b', T.mag, 'Mag'), ('r', T.apmag, 'Aper mag')]:
            plt.plot(ps1mag, mag - ps1mag, '.', color=cc, label=label, alpha=0.6)

            mm,dd = [],[]
            for mlo,mhi in zip(mags, mags[1:]):
                I = np.flatnonzero((ps1mag > mlo) * (ps1mag <= mhi))
                mm.append((mlo+mhi)/2.)
                dd.append(np.median(mag[I] - ps1mag[I]))
            plt.plot(mm, dd, 'o-', color=cc)
            
        plt.xlabel('PS1 %s mag' % band)
        plt.ylabel('Mag - PS1 (mag)')
        plt.title('%sPS1 comparison: brick %s' % (name, brickname))
        plt.ylim(-0.2, 0.2)
        mlo,mhi = magrange
        plt.xlim(mhi, mlo)
        plt.axhline(0., color='k', alpha=0.1)
        plt.legend()
        ps.savefig()
        
        
if __name__ == '__main__':

    ps = PlotSequence('zp')

    compare_brick_to_ps1('2431p055', ps, name='New ZPs: ')
    compare_brick_to_ps1('2423p087', ps, name='New ZPs: ')

    compare_brick_to_ps1('2431p055', ps, basedir='dr2m', name='dr2m: ')
    compare_brick_to_ps1('2423p087', ps, basedir='dr2m', name='dr2m: ')
    
    import sys
    sys.exit(0)
    
    ccds = [
        (346642, 'S5'),
        (349159, 'S4'),
        (349182, 'S3'),
        (200650, 'S15'),
        (200663, 'N12'),
        (346638, 'S5'),
        ]

    compare_to_ps1(ps, ccds)

    
    TT1 = [fits_table('forced-%i-%s.fits' % (e,c))
           for e,c in ccds[:3]]

    TT2 = [fits_table('forced-%i-%s.fits' % (e,c))
           for e,c in ccds[3:]]

    for TT,name in [(TT1,'clean'),(TT2,'edge')]:
        compare_mags(TT, name, ps)
