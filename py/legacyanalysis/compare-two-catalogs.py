from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import pylab as plt
import numpy as np
from glob import glob
import os
import re
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import PlotSequence
from tractor.brightness import NanoMaggies
import scipy.stats

'''
This is a little script for comparing two directories full of tractor
catalogs.

'''

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name1', help='Name for first data set')
    parser.add_argument('--name2', help='Name for second data set')
    parser.add_argument('--plot-prefix', default='compare',
                        help='Prefix for plot filenames; default "%default"')
    parser.add_argument('--match', default=1.0, type=float,
                        help='Astrometric cross-match distance in arcsec')
    parser.add_argument('dir1', help='First directory to compare')
    parser.add_argument('dir2', help='Second directory to compare')

    opt = parser.parse_args()
    
    ps = PlotSequence(opt.plot_prefix)

    name1 = opt.name1
    if name1 is None:
        name1 = os.path.basename(opt.dir1)
        if not len(name1):
            name1 = os.path.basename(os.path.dirname(opt.dir1))
    name2 = opt.name2
    if name2 is None:
        name2 = os.path.basename(opt.dir2)
        if not len(name2):
            name2 = os.path.basename(os.path.dirname(opt.dir2))
    tt = 'Comparing %s to %s' % (name1, name2)

    # regex for tractor-*.fits catalog filename
    catre = re.compile('tractor-.*.fits')
        
    cat1,cat2 = [],[]
    for basedir,cat in [(opt.dir1, cat1), (opt.dir2, cat2)]:
        for dirpath,dirnames,filenames in os.walk(basedir, followlinks=True):
            for fn in filenames:
                if not catre.match(fn):
                    print('Skipping', fn, 'due to filename')
                    continue
                fn = os.path.join(dirpath, fn)
                t = fits_table(fn)
                print(len(t), 'from', fn)
                cat.append(t)
    cat1 = merge_tables(cat1, columns='fillzero')
    cat2 = merge_tables(cat2, columns='fillzero')
    print('Total of', len(cat1), 'from', name1)
    print('Total of', len(cat2), 'from', name2)
    cat1.cut(cat1.brick_primary)
    cat2.cut(cat2.brick_primary)
    print('Total of', len(cat1), 'BRICK_PRIMARY from', name1)
    print('Total of', len(cat2), 'BRICK_PRIMARY from', name2)

    # cat1.cut((cat1.decam_anymask[:,1] == 0) *
    #          (cat1.decam_anymask[:,2] == 0) *
    #          (cat1.decam_anymask[:,4] == 0))
    # cat2.cut((cat2.decam_anymask[:,1] == 0) *
    #          (cat2.decam_anymask[:,2] == 0) *
    #          (cat2.decam_anymask[:,4] == 0))
    # print('Total of', len(cat1), 'unmasked from', name1)
    # print('Total of', len(cat2), 'unmasked from', name2)
    
    I,J,d = match_radec(cat1.ra, cat1.dec, cat2.ra, cat2.dec, opt.match/3600.,
                        nearest=True)
    print(len(I), 'matched')

    plt.clf()
    plt.hist(d * 3600., 100)
    plt.xlabel('Match distance (arcsec)')
    plt.title(tt)
    ps.savefig()

    matched1 = cat1[I]
    matched2 = cat2[J]

    matched1.type = np.array([t.strip() for t in matched1.type])
    matched2.type = np.array([t.strip() for t in matched2.type])
    
    # Confusion matrix for source types
    types = ['PSF', 'SIMP', 'EXP', 'DEV', 'COMP']
    confusion = np.zeros((len(types), len(types)))
    labels = []
    assert(len(set(np.unique(matched1.type)) - set(types)) == 0)
    assert(len(set(np.unique(matched2.type)) - set(types)) == 0)
    for i,t1 in enumerate(types):
        I = np.flatnonzero(matched1.type == t1)
        if len(I) == 0:
            continue
        for j,t2 in enumerate(types):
            J = np.flatnonzero(matched2.type[I] == t2)
            if len(J) == 0:
                continue
            confusion[i, j] = float(len(J)) / float(len(I))
            labels.append((i, j, '%i/%i' % (len(J), len(I))))
    plt.clf()
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    for r,c,s in labels:
        plt.text(c, r, s, color='k', ha='center', fontsize=8)
    plt.xticks(range(len(types)), types)
    plt.yticks(range(len(types)), types)
    plt.ylabel(name1)
    plt.xlabel(name2)
    ps.savefig()

    plt.clf()
    I = np.flatnonzero((matched1.type == 'PSF') * (matched2.type == 'PSF'))
    print(len(I), 'PSF to PSF')
    plt.plot(matched1.dchisq[I,0] - matched1.dchisq[I,1],
             matched2.dchisq[I,0] - matched2.dchisq[I,1], 'k.', label='PSF to PSF')
    I = np.flatnonzero((matched1.type == 'PSF') * (matched2.type == 'SIMP'))
    print(len(I), 'PSF to SIMP')
    plt.plot(matched1.dchisq[I,0] - matched1.dchisq[I,1],
             matched2.dchisq[I,0] - matched2.dchisq[I,1], 'r.', label='PSF to SIMP')
    I = np.flatnonzero((matched1.type == 'SIMP') * (matched2.type == 'PSF'))
    print(len(I), 'SIMP to PSF')
    plt.plot(matched1.dchisq[I,0] - matched1.dchisq[I,1],
             matched2.dchisq[I,0] - matched2.dchisq[I,1], 'g.', label='SIMP to PSF')
    I = np.flatnonzero((matched1.type == 'SIMP') * (matched2.type == 'SIMP'))
    print(len(I), 'SIMP to SIMP')
    plt.plot(matched1.dchisq[I,0] - matched1.dchisq[I,1],
             matched2.dchisq[I,0] - matched2.dchisq[I,1], 'b.', label='SIMP to SIMP')
    plt.xlabel('%s dchisq: PSF - SIMP' % name1)
    plt.ylabel('%s dchisq: PSF - SIMP' % name2)
    plt.legend(loc='upper left')
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    plt.plot([-20,20],[-20,20], 'k-', alpha=0.5)
    plt.axhline(0, color='k', alpha=0.5)
    plt.axvline(0, color='k', alpha=0.5)
    plt.axis([-20,20,-20,20])
    ps.savefig()


    plt.clf()
    I = np.flatnonzero((matched1.type == 'EXP') * (matched2.type == 'EXP'))
    plt.plot(matched1.shapeexp_r[I], matched2.shapeexp_r[I], 'r.', label='exp')
    I = np.flatnonzero((matched1.type == 'DEV') * (matched2.type == 'DEV'))
    plt.plot(matched1.shapedev_r[I], matched2.shapedev_r[I], 'b.', label='dev')
    plt.xlabel('%s radius (arcsec)' % name1)
    plt.ylabel('%s radius (arcsec)' % name2)
    plt.axis([0,4,0,4])
    plt.legend()
    ps.savefig()

    for iband,band,cc in [(1,'g','g'),(2,'r','r'),(4,'z','m')]:
        K = np.flatnonzero((matched1.decam_flux_ivar[:,iband] > 0) *
                           (matched2.decam_flux_ivar[:,iband] > 0))
        
        print('Median mw_trans', band, 'is',
              np.median(matched1.decam_mw_transmission[:,iband]))
        
        plt.clf()
        plt.errorbar(matched1.decam_flux[K,iband],
                     matched2.decam_flux[K,iband],
                     fmt='.', color=cc,
                     xerr=1./np.sqrt(matched1.decam_flux_ivar[K,iband]),
                     yerr=1./np.sqrt(matched2.decam_flux_ivar[K,iband]),
                     alpha=0.1,
                     )
        plt.xlabel('%s flux: %s' % (name1, band))
        plt.ylabel('%s flux: %s' % (name2, band))
        plt.plot([-1e6, 1e6], [-1e6,1e6], 'k-', alpha=1.)
        plt.axis([-100, 1000, -100, 1000])
        plt.title(tt)
        ps.savefig()


    for iband,band,cc in [(1,'g','g'),(2,'r','r'),(4,'z','m')]:
        good = ((matched1.decam_flux_ivar[:,iband] > 0) *
                (matched2.decam_flux_ivar[:,iband] > 0))
        K = np.flatnonzero(good)
        psf1 = (matched1.type == 'PSF ')
        psf2 = (matched2.type == 'PSF ')
        P = np.flatnonzero(good * psf1 * psf2)

        mag1, magerr1 = NanoMaggies.fluxErrorsToMagErrors(
            matched1.decam_flux[:,iband], matched1.decam_flux_ivar[:,iband])
        
        iv1 = matched1.decam_flux_ivar[:, iband]
        iv2 = matched2.decam_flux_ivar[:, iband]
        std = np.sqrt(1./iv1 + 1./iv2)
        
        plt.clf()
        plt.plot(mag1[K],
                 (matched2.decam_flux[K,iband] - matched1.decam_flux[K,iband]) / std[K],
                 '.', alpha=0.1, color=cc)
        plt.plot(mag1[P],
                 (matched2.decam_flux[P,iband] - matched1.decam_flux[P,iband]) / std[P],
                 '.', alpha=0.1, color='k')
        plt.ylabel('(%s - %s) flux / flux errors (sigma): %s' % (name2, name1, band))
        plt.xlabel('%s mag: %s' % (name1, band))
        plt.axhline(0, color='k', alpha=0.5)
        plt.axis([24, 16, -10, 10])
        plt.title(tt)
        ps.savefig()

    plt.clf()
    lp,lt = [],[]
    for iband,band,cc in [(1,'g','g'),(2,'r','r'),(4,'z','m')]:
        good = ((matched1.decam_flux_ivar[:,iband] > 0) *
                (matched2.decam_flux_ivar[:,iband] > 0))
        #good = True
        psf1 = (matched1.type == 'PSF ')
        psf2 = (matched2.type == 'PSF ')
        mag1, magerr1 = NanoMaggies.fluxErrorsToMagErrors(
            matched1.decam_flux[:,iband], matched1.decam_flux_ivar[:,iband])
        iv1 = matched1.decam_flux_ivar[:, iband]
        iv2 = matched2.decam_flux_ivar[:, iband]
        std = np.sqrt(1./iv1 + 1./iv2)
        #std = np.hypot(std, 0.01)
        G = np.flatnonzero(good * psf1 * psf2 *
                           np.isfinite(mag1) *
                           (mag1 >= 20) * (mag1 < dict(g=24, r=23.5, z=22.5)[band]))
        
        n,b,p = plt.hist((matched2.decam_flux[G,iband] -
                          matched1.decam_flux[G,iband]) / std[G],
                 range=(-4, 4), bins=50, histtype='step', color=cc,
                 normed=True)

        sig = (matched2.decam_flux[G,iband] -
               matched1.decam_flux[G,iband]) / std[G]
        print('Raw mean and std of points:', np.mean(sig), np.std(sig))
        med = np.median(sig)
        rsigma = (np.percentile(sig, 84) - np.percentile(sig, 16)) / 2.
        print('Median and percentile-based sigma:', med, rsigma)
        lp.append(p[0])
        lt.append('%s: %.2f +- %.2f' % (band, med, rsigma))
        
    bins = []
    gaussint = []
    for blo,bhi in zip(b, b[1:]):
        c = scipy.stats.norm.cdf(bhi) - scipy.stats.norm.cdf(blo)
        c /= (bhi - blo)
        #bins.extend([blo,bhi])
        #gaussint.extend([c,c])
        bins.append((blo+bhi)/2.)
        gaussint.append(c)
    plt.plot(bins, gaussint, 'k-', lw=2, alpha=0.5)
    
    plt.title(tt)
    plt.xlabel('Flux difference / error (sigma)')
    plt.axvline(0, color='k', alpha=0.1)
    plt.ylim(0, 0.45)
    plt.legend(lp, lt, loc='upper right')
    ps.savefig()
        
        
    for iband,band,cc in [(1,'g','g'),(2,'r','r'),(4,'z','m')]:
        plt.clf()
        mag1, magerr1 = NanoMaggies.fluxErrorsToMagErrors(
            matched1.decam_flux[:,iband], matched1.decam_flux_ivar[:,iband])
        mag2, magerr2 = NanoMaggies.fluxErrorsToMagErrors(
            matched2.decam_flux[:,iband], matched2.decam_flux_ivar[:,iband])

        meanmag = NanoMaggies.nanomaggiesToMag((
            matched1.decam_flux[:,iband] + matched2.decam_flux[:,iband]) / 2.)

        psf1 = (matched1.type == 'PSF ')
        psf2 = (matched2.type == 'PSF ')
        good = ((matched1.decam_flux_ivar[:,iband] > 0) *
                (matched2.decam_flux_ivar[:,iband] > 0) *
                np.isfinite(mag1) * np.isfinite(mag2))
        K = np.flatnonzero(good)
        P = np.flatnonzero(good * psf1 * psf2)
        
        plt.errorbar(mag1[K], mag2[K], fmt='.', color=cc,
                     xerr=magerr1[K], yerr=magerr2[K], alpha=0.1)
        plt.plot(mag1[P], mag2[P], 'k.', alpha=0.5)
        plt.xlabel('%s %s (mag)' % (name1, band))
        plt.ylabel('%s %s (mag)' % (name2, band))
        plt.plot([-1e6, 1e6], [-1e6,1e6], 'k-', alpha=1.)
        plt.axis([24, 16, 24, 16])
        plt.title(tt)
        ps.savefig()

        plt.clf()
        plt.errorbar(mag1[K], mag2[K] - mag1[K], fmt='.', color=cc,
                     xerr=magerr1[K], yerr=magerr2[K], alpha=0.1)
        plt.plot(mag1[P], mag2[P] - mag1[P], 'k.', alpha=0.5)
        plt.xlabel('%s %s (mag)' % (name1, band))
        plt.ylabel('%s %s - %s %s (mag)' % (name2, band, name1, band))
        plt.axhline(0., color='k', alpha=1.)
        plt.axis([24, 16, -1, 1])
        plt.title(tt)
        ps.savefig()

        magbins = np.arange(16, 24.001, 0.5)
        
        plt.clf()
        plt.plot(mag1[K], (mag2[K]-mag1[K]) / np.hypot(magerr1[K], magerr2[K]),
                     '.', color=cc, alpha=0.1)
        plt.plot(mag1[P], (mag2[P]-mag1[P]) / np.hypot(magerr1[P], magerr2[P]),
                     'k.', alpha=0.5)

        plt.xlabel('%s %s (mag)' % (name1, band))
        plt.ylabel('(%s %s - %s %s) / errors (sigma)' %
                   (name2, band, name1, band))
        plt.axhline(0., color='k', alpha=1.)
        plt.axis([24, 16, -10, 10])
        plt.title(tt)
        ps.savefig()

        y = (mag2 - mag1) / np.hypot(magerr1, magerr2)
        
        plt.clf()
        plt.plot(meanmag[P], y[P], 'k.', alpha=0.1)

        midmag = []
        vals = np.zeros((len(magbins)-1, 5))
        median_err1 = []
        
        iqd_gauss = scipy.stats.norm.ppf(0.75) - scipy.stats.norm.ppf(0.25)

        # FIXME -- should we do some stats after taking off the mean difference?
        
        for bini,(mlo,mhi) in enumerate(zip(magbins, magbins[1:])):
            I = P[(meanmag[P] >= mlo) * (meanmag[P] < mhi)]
            midmag.append((mlo+mhi)/2.)
            median_err1.append(np.median(magerr1[I]))
            if len(I) == 0:
                continue
            # median and +- 1 sigma quantiles
            ybin = y[I]
            vals[bini,0] = np.percentile(ybin, 16)
            vals[bini,1] = np.median(ybin)
            vals[bini,2] = np.percentile(ybin, 84)
            # +- 2 sigma quantiles
            vals[bini,3] = np.percentile(ybin, 2.3)
            vals[bini,4] = np.percentile(ybin, 97.7)

            iqd = np.percentile(ybin, 75) - np.percentile(ybin, 25)
            
            print('Mag bin', midmag[-1], ': IQD is factor', iqd / iqd_gauss,
                  'vs expected for Gaussian;', len(ybin), 'points')

            # if iqd > iqd_gauss:
            #     # What error adding in quadrature would you need to make the IQD match?
            #     err = median_err1[-1]
            #     target_err = err * (iqd / iqd_gauss)
            #     sys_err = np.sqrt(target_err**2 - err**2)
            #     print('--> add systematic error', sys_err)

        # ~ Johan's cuts
        mlo = 21.
        mhi = dict(g=24., r=23.5, z=22.5)[band]
        I = P[(meanmag[P] >= mlo) * (meanmag[P] < mhi)]
        ybin = y[I]
        iqd = np.percentile(ybin, 75) - np.percentile(ybin, 25)
        print('Mag bin', mlo, mhi, 'band', band, ': IQD is factor',
              iqd / iqd_gauss, 'vs expected for Gaussian;', len(ybin), 'points')
        if iqd > iqd_gauss:
            # What error adding in quadrature would you need to make
            # the IQD match?
            err = np.median(np.hypot(magerr1[I], magerr2[I]))
            print('Median error (hypot):', err)
            target_err = err * (iqd / iqd_gauss)
            print('Target:', target_err)
            sys_err = np.sqrt((target_err**2 - err**2) / 2.)
            print('--> add systematic error', sys_err)

            # check...
            err_sys = np.hypot(np.hypot(magerr1, sys_err),
                               np.hypot(magerr2, sys_err))
            ysys = (mag2 - mag1) / err_sys
            ysys = ysys[I]
            print('Resulting median error:', np.median(err_sys[I]))
            iqd_sys = np.percentile(ysys, 75) - np.percentile(ysys, 25)
            print('--> IQD', iqd_sys / iqd_gauss, 'vs Gaussian')
            # Hmmm, this doesn't work... totally overshoots.
            
            
        plt.errorbar(midmag, vals[:,1], fmt='o', color='b',
                     yerr=(vals[:,1]-vals[:,0], vals[:,2]-vals[:,1]),
                     capthick=3, zorder=20)
        plt.errorbar(midmag, vals[:,1], fmt='o', color='b',
                     yerr=(vals[:,1]-vals[:,3], vals[:,4]-vals[:,1]),
                     capthick=2, zorder=20)
        plt.axhline( 1., color='b', alpha=0.2)
        plt.axhline(-1., color='b', alpha=0.2)
        plt.axhline( 2., color='b', alpha=0.2)
        plt.axhline(-2., color='b', alpha=0.2)

        for mag,err,y in zip(midmag, median_err1, vals[:,3]):
            if not np.isfinite(err):
                continue
            if y < -6:
                continue
            plt.text(mag, y-0.1, '%.3f' % err, va='top', ha='center', color='k',
                     fontsize=10)
        
        plt.xlabel('(%s + %s)/2 %s (mag), PSFs' % (name1, name2, band))
        plt.ylabel('(%s %s - %s %s) / errors (sigma)' %
                   (name2, band, name1, band))
        plt.axhline(0., color='k', alpha=1.)

        plt.axvline(21, color='k', alpha=0.3)
        plt.axvline(dict(g=24, r=23.5, z=22.5)[band], color='k', alpha=0.3)

        plt.axis([24.1, 16, -6, 6])
        plt.title(tt)
        ps.savefig()

        #magbins = np.append([16, 18], np.arange(20, 24.001, 0.5))
        if band == 'g':
            magbins = [20, 24]
        elif band == 'r':
            magbins = [20, 23.5]
        elif band == 'z':
            magbins = [20, 22.5]

        slo,shi = -5,5
        plt.clf()
        ha = dict(bins=25, range=(slo,shi), histtype='step', normed=True)
        y = (mag2 - mag1) / np.hypot(magerr1, magerr2)
        midmag = []
        nn = []
        rgbs = []
        lt,lp = [],[]
        for bini,(mlo,mhi) in enumerate(zip(magbins, magbins[1:])):
            I = P[(mag1[P] >= mlo) * (mag1[P] < mhi)]
            if len(I) == 0:
                continue
            ybin = y[I]
            rgb = [0.,0.,0.]
            rgb[0] = float(bini) / (len(magbins)-1)
            rgb[2] = 1. - rgb[0]
            n,b,p = plt.hist(ybin, color=rgb, **ha)
            lt.append('mag %g to %g' % (mlo,mhi))
            lp.append(p[0])
            midmag.append((mlo+mhi)/2.)
            nn.append(n)
            rgbs.append(rgb)
            
        bins = []
        gaussint = []
        for blo,bhi in zip(b, b[1:]):
            #midbin.append((blo+bhi)/2.)
            #gaussint.append(scipy.stats.norm.cdf(bhi) -
            #                scipy.stats.norm.cdf(blo))
            c = scipy.stats.norm.cdf(bhi) - scipy.stats.norm.cdf(blo)
            c /= (bhi - blo)
            bins.extend([blo,bhi])
            gaussint.extend([c,c])
        plt.plot(bins, gaussint, 'k-', lw=2, alpha=0.5)
            
        plt.legend(lp, lt)
        plt.title(tt)
        plt.xlim(slo,shi)
        ps.savefig()

        bincenters = b[:-1] + (b[1]-b[0])/2.
        plt.clf()
        lp = []
        for n,rgb,mlo,mhi in zip(nn, rgbs, magbins, magbins[1:]):
            p = plt.plot(bincenters, n, '-', color=rgb)
            lp.append(p[0])
        plt.plot(bincenters, gaussint[::2], 'k-', alpha=0.5, lw=2)
        plt.legend(lp, lt)
        plt.title(tt)
        plt.xlim(slo,shi)
        ps.savefig()
        
        
if __name__ == '__main__':
    main()

