from __future__ import print_function
import os
import fitsio
import numpy as np
from glob import glob
from collections import Counter
#import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt

from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
from legacypipe.utils import find_unique_pixels

from legacyanalysis.coverage import cmap_discretize

'''
This script produces a FITS table that summarizes per-brick
information about a data release.  It takes on the command-line the
list of "*-nexp-BAND.fits.gz" files, pulls the brick names out, and
reads the corresponding tractor files.  This is kind of an odd way to
do it, but I'm sure it made sense to me at the time.

This takes long enough to run on a full data release that you might
want to run multiple threads by hand, eg,

(for B in 30; do python -u legacyanalysis/brick-summary.py -o dr4-brick-summary-$B.fits /global/projecta/projectdirs/cosmo/work/dr4b/coadd/$B*/*/*-nexp-*.fits.gz > bs-$B.log 2>&1; done) &

for a set of B, and then

python legacyanalysis/brick-summary.py --merge -o brick-summary-dr4.fits dr4-brick-summary-*.fits

to merge them into one file, and

python legacyanalysis/brick-summary.py --plot brick-summary-dr4.fits

to make a couple of plots.

Or, run this to generate a list of command-lines that you can copy-n-paste:

for ((b=0; b<36; b++)); do B=$(printf %02i $b); echo "python -u legacyanalysis/brick-summary.py --dr5 -o dr5-brick-summary-$B.fits /project/projectdirs/cosmo/work/legacysurvey/dr5/DR5_out/coadd/$B*/*/*-nexp-*.fits.fz > bs-$B.log 2>&1 &"; done

python legacyanalysis/brick-summary.py --merge -o survey-brick-dr5.fits dr5-brick-summary-*.fits

'''


def colorbar_axes(parent, frac=0.12, pad=0.03, aspect=20):
    pb = parent.get_position(original=True).frozen()
    # new parent box, padding, child box
    (pbnew, padbox, cbox) = pb.splitx(1.0-(frac+pad), 1.0-frac)
    cbox = cbox.anchored('C', cbox)
    parent.set_position(pbnew)
    parent.set_anchor((1.0, 0.5))
    cax = parent.get_figure().add_axes(cbox)
    cax.set_aspect(aspect, anchor=((0.0, 0.5)), adjustable='box')
    parent.get_figure().sca(parent)
    return cax

def plots(opt):
    from astrometry.util.plotutils import antigray
    import tractor.sfd

    T = fits_table(opt.files[0])
    print('Read', len(T), 'bricks summarized in', opt.files[0])
    import pylab as plt
    import matplotlib

    B = fits_table('survey-bricks.fits.gz')
    print('Looking up brick bounds')
    ibrick = dict([(n,i) for i,n in enumerate(B.brickname)])
    bi = np.array([ibrick[n] for n in T.brickname])
    T.ra1 = B.ra1[bi]
    T.ra2 = B.ra2[bi]
    T.dec1 = B.dec1[bi]
    T.dec2 = B.dec2[bi]
    assert(np.all(T.ra2 > T.ra1))
    T.area = ((T.ra2 - T.ra1) * (T.dec2 - T.dec1) *
              np.cos(np.deg2rad((T.dec1 + T.dec2) / 2.)))
    del B
    del bi
    del ibrick

    print('Total sources:', sum(T.nobjs))
    print('Approx area:', len(T)/16., 'sq deg')
    print('Area:', np.sum(T.area))
    print('g,r,z coverage:', sum((T.nexp_g > 0) * (T.nexp_r > 0) * (T.nexp_z > 0)) / 16.)

    decam = True
    # vs MzLS+BASS
    #release = 'MzLS+BASS DR4'
    release = 'DECaLS DR7'

    if decam:
        # DECam
        #ax = [360, 0, -21, 36]
        ax = [300, -60, -21, 36]

        def map_ra(r):
                return r + (-360 * (r > 300))

    else:
        # MzLS+BASS
        ax = [310, 90, 30, 80]

        def map_ra(r):
                return r

    udec = np.unique(T.dec)
    print('Number of unique Dec values:', len(udec))
    print('Number of unique Dec values in range', ax[2],ax[3],':',
          np.sum((udec >= ax[2]) * (udec <= ax[3])))

    def radec_plot():
        plt.axis(ax)
        plt.xlabel('RA (deg)')
        if decam:
            # plt.xticks(np.arange(0, 361, 45))
            #tt = np.arange(0, 361, 60)
            #plt.xticks(tt, map_ra(tt))
            plt.xticks([-60,0,60,120,180,240,300], [300,0,60,120,180,240,300])
        else:
            plt.xticks(np.arange(90, 311, 30))

        plt.ylabel('Dec (deg)')

        def plot_broken(rr, dd, *args, **kwargs):
            dr = np.abs(np.diff(rr))
            I = np.flatnonzero(dr > 90)
            #print('breaks:', rr[I])
            #print('breaks:', rr[I+1])
            if len(I) == 0:
                plt.plot(rr, dd, *args, **kwargs)
                return
            for lo,hi in zip(np.append([0], I+1), np.append(I+1, -1)):
                #print('Cut:', lo, ':', hi, '->', rr[lo], rr[hi-1])
                plt.plot(rr[lo:hi], dd[lo:hi], *args, **kwargs)

        # Galactic plane lines
        gl = np.arange(361)
        gb = np.zeros_like(gl)
        from astrometry.util.starutil_numpy import lbtoradec
        rr,dd = lbtoradec(gl, gb)
        plot_broken(map_ra(rr), dd, 'k-', alpha=0.5, lw=1)
        rr,dd = lbtoradec(gl, gb+10)
        plot_broken(map_ra(rr), dd, 'k-', alpha=0.25, lw=1)
        rr,dd = lbtoradec(gl, gb-10)
        plot_broken(map_ra(rr), dd, 'k-', alpha=0.25, lw=1)

    plt.figure(1, figsize=(8,5))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.93)

    plt.figure(2, figsize=(8,4))
    #plt.subplots_adjust(left=0.06, right=0.98, top=0.98)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98)
    plt.figure(1)

    # Map of the tile centers we want to observe...
    if decam:
        O = fits_table('obstatus/decam-tiles_obstatus.fits')
    else:
        O = fits_table('mosaic-tiles_obstatus.fits')
    O.cut(O.in_desi == 1)
    rr,dd = np.meshgrid(np.linspace(ax[1],ax[0], 700),
                        np.linspace(ax[2],ax[3], 200))
    from astrometry.libkd.spherematch import match_radec
    I,J,d = match_radec(O.ra, O.dec, rr.ravel(), dd.ravel(), 1.)
    desimap = np.zeros(rr.shape, bool)
    desimap.flat[J] = True

    # Smoothed DESI boundary contours
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    C = plt.contour(gaussian_filter(
        binary_dilation(desimap).astype(np.float32), 2),
        [0.5], extent=[ax[1],ax[0],ax[2],ax[3]])
    plt.clf()
    desi_map_boundaries = C.collections[0]
    def desi_map_outline():
        segs = desi_map_boundaries.get_segments()
        for seg in segs:
            plt.plot(seg[:,0], seg[:,1], 'b-')

    def desi_map():
        # Show the DESI tile map in the background.
        plt.imshow(desimap, origin='lower', interpolation='nearest',
                   extent=[ax[1],ax[0],ax[2],ax[3]], aspect='auto',
                   cmap=antigray, vmax=8)

    base_cmap = 'viridis'

    # Dust map -- B&W version
    nr,nd = 610,350
    plt.figure(2)
    plt.clf()
    dmap = np.zeros((nd,nr))
    rr = np.linspace(ax[0], ax[1], nr)
    dd = np.linspace(ax[2], ax[3], nd)
    rr = rr[:-1] + 0.5*(rr[1]-rr[0])
    dd = dd[:-1] + 0.5*(dd[1]-dd[0])
    rr,dd = np.meshgrid(rr,dd)
    I,J,d = match_radec(rr.ravel(), dd.ravel(),
                        O.ra, O.dec, 1.0, nearest=True)
    iy,ix = np.unravel_index(I, rr.shape)
    #dmap[iy,ix] = O.ebv_med[J]
    sfd = tractor.sfd.SFDMap()
    ebv = sfd.ebv(rr[iy,ix], dd[iy,ix])
    dmap[iy,ix] = ebv
    mx = np.percentile(dmap[dmap > 0], 98)
    plt.imshow(dmap, extent=[ax[0],ax[1],ax[2],ax[3]], interpolation='nearest', origin='lower',
                   aspect='auto', cmap='Greys', vmin=0, vmax=mx)
    #desi_map_outline()
    radec_plot()
    cax = colorbar_axes(plt.gca(), frac=0.12)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Extinction E(B-V)')
    plt.savefig('ext-bw.pdf')
    plt.clf()
    dmap = sfd.ebv(rr.ravel(), dd.ravel()).reshape(rr.shape)
    plt.imshow(dmap, extent=[ax[0],ax[1],ax[2],ax[3]],
               interpolation='nearest', origin='lower',
               aspect='auto', cmap='Greys', vmin=0, vmax=0.25)
    desi_map_outline()
    radec_plot()
    cax = colorbar_axes(plt.gca(), frac=0.12)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Extinction E(B-V)')
    plt.savefig('ext-bw-2.pdf')
    plt.figure(1)

    #sys.exit(0)
    plt.clf()
    depthlo,depthhi = 21.5, 25.5
    for band in 'grz':
        depth = T.get('galdepth_%s' % band)
        ha = dict(histtype='step',  bins=50, range=(depthlo,depthhi))
        ccmap = dict(g='g', r='r', z='m')
        plt.hist(depth[depth>0], label='%s band' % band,
                 color=ccmap[band], **ha)
    plt.xlim(depthlo, depthhi)
    plt.xlabel('Galaxy depth (median per brick) (mag)')
    plt.ylabel('Number of Bricks')
    plt.title(release)
    plt.savefig('galdepths.png')

    for band in 'grz':
        depth = T.get('galdepth_%s' % band)
        nexp = T.get('nexp_%s' % band)
        #lo,hi = 22.0-0.05, 24.2+0.05
        lo,hi = depthlo-0.05, depthhi+0.05
        nbins = 1 + int((depthhi - depthlo) / 0.1)
        ha = dict(histtype='step',  bins=nbins, range=(lo,hi))
        ccmap = dict(g='g', r='r', z='m')
        area = 0.25**2
        plt.clf()
        I = np.flatnonzero((depth > 0) * (nexp == 1))
        plt.hist(depth[I], label='%s band, 1 exposure' % band,
                 color=ccmap[band], lw=1,
                 weights=area * np.ones_like(depth[I]),
                 **ha)
        I = np.flatnonzero((depth > 0) * (nexp == 2))
        plt.hist(depth[I], label='%s band, 2 exposures' % band,
                 color=ccmap[band], lw=2, alpha=0.5,
                 weights=area * np.ones_like(depth[I]),
                 **ha)
        I = np.flatnonzero((depth > 0) * (nexp >= 3))
        plt.hist(depth[I], label='%s band, 3+ exposures' % band,
                 color=ccmap[band], lw=3, alpha=0.3,
                 weights=area * np.ones_like(depth[I]),
                 **ha)
        plt.title('%s: galaxy depths, %s band' % (release, band))
        plt.xlabel('5-sigma galaxy depth (mag)')
        plt.ylabel('Square degrees')
        plt.xlim(lo, hi)
        plt.xticks(np.arange(depthlo, depthhi+0.01, 0.2))
        plt.legend(loc='upper right')
        plt.savefig('depth-hist-%s.png' % band)

    for band in 'grz':
        plt.clf()
        desi_map()
        N = T.get('nexp_%s' % band)
        I = np.flatnonzero(N > 0)
        #cm = matplotlib.cm.get_cmap('jet', 6)
        #cm = matplotlib.cm.get_cmap('winter', 5)

        mx = 10
        cm = cmap_discretize(base_cmap, mx)
        plt.scatter(map_ra(T.ra[I]), T.dec[I], c=N[I], s=3,
                    edgecolors='none',
                    vmin=0.5, vmax=mx + 0.5, cmap=cm)
        radec_plot()
        cax = colorbar_axes(plt.gca(), frac=0.08)
        plt.colorbar(cax=cax, ticks=range(mx+1))
        plt.title('%s: Number of exposures in %s' % (release, band))
        plt.savefig('nexp-%s.png' % band)

        #cmap = cmap_discretize(base_cmap, 15)
        cmap = cmap_discretize(base_cmap, 10)
        plt.clf()
        desi_map()
        psf = T.get('psfsize_%s' % band)
        I = np.flatnonzero(psf > 0)
        plt.scatter(map_ra(T.ra[I]), T.dec[I], c=psf[I], s=3,
                    edgecolors='none', cmap=cmap,
                    vmin=0.5, vmax=2.5)
        #vmin=0, vmax=3.)
        radec_plot()
        plt.colorbar()
        plt.title('%s: PSF size, band %s' % (release, band))
        plt.savefig('psfsize-%s.png' % band)

        plt.clf()
        desi_map()

        depth = T.get('galdepth_%s' % band) - T.get('ext_%s' % band)
        mn,mx = np.percentile(depth[depth > 0], [10,98])
        mn = np.floor(mn * 10) / 10.
        mx = np.ceil(mx * 10) / 10.
        cmap = cmap_discretize(base_cmap, 1+int((mx-mn+0.001)/0.1))
        I = (depth > 0)
        plt.scatter(map_ra(T.ra[I]), T.dec[I], c=depth[I], s=3,
                    edgecolors='none', vmin=mn-0.05, vmax=mx+0.05, cmap=cmap)
        radec_plot()
        plt.colorbar()
        plt.title('%s: galaxy depth, band %s, median per brick, extinction-corrected' % (release, band))
        plt.savefig('galdepth-%s.png' % band)

        # B&W version
        plt.figure(2)
        plt.clf()
        mn,mx = np.percentile(depth[depth > 0], [2,98])
        print('Raw mn,mx', mn,mx)
        mn = np.floor((mn+0.05) * 10) / 10. - 0.05
        mx = np.ceil( (mx-0.05) * 10) / 10. + 0.05
        print('rounded mn,mx', mn,mx)
        nsteps = int((mx-mn+0.001)/0.1)
        print('discretizing into', nsteps, 'colormap bins')
        #nsteps = 1+int((mx-mn+0.001)/0.1)
        cmap = cmap_discretize(antigray, nsteps)
        nr,nd = 610,228
        dmap = np.zeros((nd,nr))
        rr = np.linspace(ax[0], ax[1], nr)
        dd = np.linspace(ax[2], ax[3], nd)
        rr = rr[:-1] + 0.5*(rr[1]-rr[0])
        dd = dd[:-1] + 0.5*(dd[1]-dd[0])
        rr,dd = np.meshgrid(rr,dd)
        I,J,d = match_radec(rr.ravel(), dd.ravel(),
                            T.ra, T.dec, 0.2, nearest=True)
        iy,ix = np.unravel_index(I, rr.shape)
        dmap[iy,ix] = depth[J]
        plt.imshow(dmap, extent=[ax[0],ax[1],ax[2],ax[3]], interpolation='nearest', origin='lower',
                   aspect='auto', cmap=cmap, vmin=mn, vmax=mx)
        desi_map_outline()
        radec_plot()
        cax = colorbar_axes(plt.gca(), frac=0.12)
        cbar = plt.colorbar(cax=cax, ticks=np.arange(20, 26, 0.5)) #ticks=np.arange(np.floor(mn/5.)*5., 0.1+np.ceil(mx/5.)*5, 0.2))
        cbar.set_label('Depth (5-sigma, galaxy profile, AB mag)')
        plt.savefig('galdepth-bw-%s.pdf' % band)
        plt.figure(1)

        plt.clf()
        desi_map()
        ext = T.get('ext_%s' % band)
        mn = 0.
        mx = 0.5
        cmap = 'hot'
        cmap = cmap_discretize(cmap, 10)
        #cmap = cmap_discretize(base_cmap, 1+int((mx-mn+0.001)/0.1))
        plt.scatter(map_ra(T.ra), T.dec, c=ext, s=3,
                    edgecolors='none', vmin=mn, vmax=mx, cmap=cmap)
        radec_plot()
        plt.colorbar()
        plt.title('%s: extinction, band %s' % (release, band))
        plt.savefig('ext-%s.png' % band)


    T.ngal = T.nsimp + T.nrex + T.nexp + T.ndev + T.ncomp

    for col in ['nobjs', 'npsf', 'nsimp', 'nrex', 'nexp', 'ndev', 'ncomp', 'ngal']:
        if not col in T.get_columns():
            continue
        plt.clf()
        desi_map()
        N = T.get(col) / T.area
        mx = np.percentile(N, 99.5)
        plt.scatter(map_ra(T.ra), T.dec, c=N, s=3,
                    edgecolors='none', vmin=0, vmax=mx)
        radec_plot()
        cbar = plt.colorbar()
        cbar.set_label('Objects per square degree')
        tt = 'of type %s' % col[1:]
        if col == 'nobjs':
            tt = 'total'
        plt.title('%s: Number of objects %s' % (release, tt))
        plt.savefig('nobjs-%s.png' % col[1:])

        # B&W version
        plt.figure(2)
        plt.clf()
        # plt.scatter(map_ra(T.ra), T.dec, c=N, s=3,
        #             edgecolors='none', vmin=0, vmax=mx, cmap=antigray)
        # Approximate pixel size in PNG plot
        # This doesn't work correctly -- we've already binned to brick resolution, so get moire patterns
        # nobjs,xe,ye = np.histogram2d(map_ra(T.ra), T.dec, weights=T.get(col),
        #                              bins=(nr,nd), range=((ax[1],ax[0]),(ax[2],ax[3])))
        # nobjs = nobjs.T
        # area = np.diff(xe)[np.newaxis,:] * (np.diff(ye) * np.cos(np.deg2rad(ye[:-1])))[:,np.newaxis]
        # nobjs /= area
        # plt.imshow(nobjs, extent=[ax[1],ax[0],ax[2],ax[3]], interpolation='nearest', origin='lower',
        #           aspect='auto')
        #print('Computing neighbours for nobjs plot...')
        nr,nd = 610,228
        nobjs = np.zeros((nd,nr))
        rr = np.linspace(ax[0], ax[1], nr)
        dd = np.linspace(ax[2], ax[3], nd)
        rr = rr[:-1] + 0.5*(rr[1]-rr[0])
        dd = dd[:-1] + 0.5*(dd[1]-dd[0])
        rr,dd = np.meshgrid(rr,dd)
        I,J,d = match_radec(rr.ravel(), dd.ravel(),
                            T.ra, T.dec, 0.2, nearest=True)
        iy,ix = np.unravel_index(I, rr.shape)
        nobjs[iy,ix] = T.get(col)[J] / T.area[J]
        #print('done')

        #mx = 2. * np.median(nobjs[nobjs > 0])
        mx = np.percentile(N, 99)

        plt.imshow(nobjs, extent=[ax[0],ax[1],ax[2],ax[3]], interpolation='nearest', origin='lower',
                   aspect='auto', cmap='Greys', vmin=0, vmax=mx)
        desi_map_outline()
        radec_plot()
        #cax = colorbar_axes(plt.gca(), frac=0.08)
        cax = colorbar_axes(plt.gca(), frac=0.12)
        cbar = plt.colorbar(cax=cax,
                            format=matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        cbar.set_label('Objects per square degree')
        plt.savefig('nobjs-bw-%s.pdf' % col[1:])
        #plt.savefig('nobjs-bw-%s.png' % col[1:])
        plt.figure(1)

    Ntot = T.nobjs
    for col in ['npsf', 'nsimp', 'nrex', 'nexp', 'ndev', 'ncomp', 'ngal']:
        if not col in T.get_columns():
            continue
        plt.clf()
        desi_map()
        N = T.get(col) / (Ntot.astype(np.float32))
        N[Ntot == 0] = 0.
        print(col, 'max frac:', N.max())
        mx = np.percentile(N, 99.5)
        print('mx', mx)
        plt.scatter(map_ra(T.ra), T.dec, c=N, s=3,
                    edgecolors='none', vmin=0, vmax=mx)
        radec_plot()
        plt.colorbar()
        plt.title('%s: Fraction of objects of type %s' % (release, col[1:]))
        plt.savefig('fobjs-%s.png' % col[1:])

        # B&W version
        plt.figure(2)
        plt.clf()
        #plt.scatter(map_ra(T.ra), T.dec, c=N * 100., s=3,
        #            edgecolors='none', vmin=0, vmax=mx*100., cmap=antigray)

        fobjs = np.zeros((nd,nr))
        rr = np.linspace(ax[0], ax[1], nr)
        dd = np.linspace(ax[2], ax[3], nd)
        rr = rr[:-1] + 0.5*(rr[1]-rr[0])
        dd = dd[:-1] + 0.5*(dd[1]-dd[0])
        rr,dd = np.meshgrid(rr,dd)
        I,J,d = match_radec(rr.ravel(), dd.ravel(),
                            T.ra, T.dec, 0.2, nearest=True)
        iy,ix = np.unravel_index(I, rr.shape)
        fobjs[iy,ix] = N[J] * 100.

        #mx = 2. * np.median(fobjs[fobjs > 0])
        mx = np.percentile(N * 100., 99)

        plt.imshow(fobjs, extent=[ax[0],ax[1],ax[2],ax[3]], interpolation='nearest', origin='lower',
                   aspect='auto', cmap='Greys', vmin=0, vmax=mx)

        desi_map_outline()
        radec_plot()
        cax = colorbar_axes(plt.gca(), frac=0.12)
        cbar = plt.colorbar(cax=cax,
                            format=matplotlib.ticker.FuncFormatter(lambda x, p: '%.2g' % x))
        cbar.set_label('Percentage of objects of type %s' % col[1:].upper())
        plt.savefig('fobjs-bw-%s.pdf' % col[1:])
        #plt.savefig('fobjs-bw-%s.png' % col[1:])
        plt.figure(1)
    return 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', dest='outfn', help='Output filename',
                      default='TMP/nexp.fits')
    parser.add_argument('--merge', action='store_true', help='Merge sub-tables')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('files', metavar='nexp-file.fits.gz', nargs='+',
                        help='List of nexp files to process')

    opt = parser.parse_args()
    fns = opt.files

    if opt.merge:
        from astrometry.util.fits import merge_tables
        TT = []
        for fn in fns:
            T = fits_table(fn)
            print(fn, '->', len(T))
            TT.append(T)
        T = merge_tables(TT)
        T.writeto(opt.outfn)
        print('Wrote', opt.outfn)
        return

    if opt.plot:
        plots(opt)
        return

    fns.sort()
    print(len(fns), 'nexp files')
    if len(fns) == 1:
        if not os.path.exists(fns[0]):
            print('No such file.')
            return 0

    brickset = set()
    bricklist = []
    gn = []
    rn = []
    zn = []

    gnhist = []
    rnhist = []
    znhist = []

    nnhist = 6

    ibricks = []
    nsrcs = []
    npsf  = []
    nsimp = []
    nrex = []
    nexp  = []
    ndev  = []
    ncomp = []

    gpsfsize = []
    rpsfsize = []
    zpsfsize = []

    gpsfdepth = []
    rpsfdepth = []
    zpsfdepth = []
    ggaldepth = []
    rgaldepth = []
    zgaldepth = []

    wise_nobs = []
    wise_trans = []

    ebv = []
    gtrans = []
    rtrans = []
    ztrans = []

    bricks = fits_table('survey-bricks.fits.gz')

    #sfd = SFDMap()

    W = H = 3600
    # xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    unique = np.ones((H,W), bool)
    tlast = 0

    for ifn,fn in enumerate(fns):
        print('File', (ifn+1), 'of', len(fns), ':', fn)
        words = fn.split('/')
        dirprefix = '/'.join(words[:-4])
        #print('Directory prefix:', dirprefix)
        words = words[-4:]
        brick = words[2]
        #print('Brick', brick)
        if not brick in brickset:
            try:
                tfn = os.path.join(dirprefix, 'tractor', brick[:3], 'tractor-%s.fits'%brick)
                print('Tractor filename', tfn)
                T = fits_table(tfn, columns=['brick_primary', 'type',
                                             'psfsize_g', 'psfsize_r', 'psfsize_z',
                                             'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                                             'galdepth_g', 'galdepth_r', 'galdepth_z',
                                             'ebv',
                                             'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z',
                                             'nobs_w1', 'nobs_w2', 'nobs_w3', 'nobs_w4',
                                             'mw_transmission_w1', 'mw_transmission_w2', 'mw_transmission_w3', 'mw_transmission_w4'])
            except:
                print('Failed to read FITS table', tfn)
                import traceback
                traceback.print_exc()
                print('Carrying on.')
                continue

            brickset.add(brick)
            bricklist.append(brick)
            gn.append(0)
            rn.append(0)
            zn.append(0)

            gnhist.append([0 for i in range(nnhist)])
            rnhist.append([0 for i in range(nnhist)])
            znhist.append([0 for i in range(nnhist)])

            index = -1
            ibrick = np.nonzero(bricks.brickname == brick)[0][0]
            ibricks.append(ibrick)

            T.cut(T.brick_primary)
            nsrcs.append(len(T))
            types = Counter([t.strip() for t in T.type])
            npsf.append(types['PSF'])
            nsimp.append(types['SIMP'])
            nrex.append(types['REX'])
            nexp.append(types['EXP'])
            ndev.append(types['DEV'])
            ncomp.append(types['COMP'])
            print('N sources', nsrcs[-1])

            gpsfsize.append(np.median(T.psfsize_g))
            rpsfsize.append(np.median(T.psfsize_r))
            zpsfsize.append(np.median(T.psfsize_z))

            gpsfdepth.append(np.median(T.psfdepth_g))
            rpsfdepth.append(np.median(T.psfdepth_r))
            zpsfdepth.append(np.median(T.psfdepth_z))

            ggaldepth.append(np.median(T.galdepth_g))
            rgaldepth.append(np.median(T.galdepth_r))
            zgaldepth.append(np.median(T.galdepth_z))

            wise_nobs.append(np.median(
                np.vstack((T.nobs_w1, T.nobs_w2, T.nobs_w3, T.nobs_w4)).T,
                axis=0))
            wise_trans.append(np.median(
                np.vstack((T.mw_transmission_w1,
                           T.mw_transmission_w2,
                           T.mw_transmission_w3,
                           T.mw_transmission_w4)).T,
                           axis=0))

            gtrans.append(np.median(T.mw_transmission_g))
            rtrans.append(np.median(T.mw_transmission_r))
            ztrans.append(np.median(T.mw_transmission_z))

            ebv.append(np.median(T.ebv))

            br = bricks[ibrick]

            #print('Computing unique brick pixels...')
            pixscale = 0.262/3600.
            wcs = Tan(br.ra, br.dec, W/2.+0.5, H/2.+0.5,
                      -pixscale, 0., 0., pixscale,
                      float(W), float(H))
            unique[:,:] = True
            find_unique_pixels(wcs, W, H, unique,
                               br.ra1, br.ra2, br.dec1, br.dec2)
            U = np.flatnonzero(unique)
            #print(len(U), 'of', W*H, 'pixels are unique to this brick')

        else:
            index = bricklist.index(brick)
            assert(index == len(bricklist)-1)

        index = bricklist.index(brick)
        assert(index == len(bricklist)-1)

        filepart = words[-1]
        filepart = filepart.replace('.fits.gz', '')
        filepart = filepart.replace('.fits.fz', '')
        print('File:', filepart)
        band = filepart[-1]
        assert(band in 'grz')

        nlist,nhist = dict(g=(gn,gnhist), r=(rn,rnhist), z=(zn,znhist))[band]

        upix = fitsio.read(fn).flat[U]
        med = np.median(upix)
        print('Band', band, ': Median', med)
        nlist[index] = med

        hist = nhist[index]
        for i in range(nnhist):
            if i < nnhist-1:
                hist[i] = np.sum(upix == i)
            else:
                hist[i] = np.sum(upix >= i)
        assert(sum(hist) == len(upix))
        print('Number of exposures histogram:', hist)

    ibricks = np.array(ibricks)

    T = fits_table()
    T.brickname = np.array(bricklist)
    T.ra  = bricks.ra [ibricks]
    T.dec = bricks.dec[ibricks]
    T.nexp_g = np.array(gn).astype(np.int16)
    T.nexp_r = np.array(rn).astype(np.int16)
    T.nexp_z = np.array(zn).astype(np.int16)
    T.nexphist_g = np.array(gnhist).astype(np.int32)
    T.nexphist_r = np.array(rnhist).astype(np.int32)
    T.nexphist_z = np.array(znhist).astype(np.int32)
    T.nobjs  = np.array(nsrcs).astype(np.int16)
    T.npsf   = np.array(npsf ).astype(np.int16)
    T.nsimp  = np.array(nsimp).astype(np.int16)
    T.nrex   = np.array(nrex ).astype(np.int16)
    T.nexp   = np.array(nexp ).astype(np.int16)
    T.ndev   = np.array(ndev ).astype(np.int16)
    T.ncomp  = np.array(ncomp).astype(np.int16)
    T.psfsize_g = np.array(gpsfsize).astype(np.float32)
    T.psfsize_r = np.array(rpsfsize).astype(np.float32)
    T.psfsize_z = np.array(zpsfsize).astype(np.float32)
    with np.errstate(divide='ignore'):
        T.psfdepth_g = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(gpsfdepth))))).astype(np.float32)
        T.psfdepth_r = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(rpsfdepth))))).astype(np.float32)
        T.psfdepth_z = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(zpsfdepth))))).astype(np.float32)
        T.galdepth_g = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(ggaldepth))))).astype(np.float32)
        T.galdepth_r = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(rgaldepth))))).astype(np.float32)
        T.galdepth_z = (-2.5*(-9.+np.log10(5.*np.sqrt(1. / np.array(zgaldepth))))).astype(np.float32)
    for k in ['psfdepth_g', 'psfdepth_r', 'psfdepth_z', 'galdepth_g', 'galdepth_r', 'galdepth_z']:
        v = T.get(k)
        v[np.logical_not(np.isfinite(v))] = 0.
    T.ebv = np.array(ebv).astype(np.float32)
    T.trans_g = np.array(gtrans).astype(np.float32)
    T.trans_r = np.array(rtrans).astype(np.float32)
    T.trans_z = np.array(ztrans).astype(np.float32)
    T.ext_g = -2.5 * np.log10(T.trans_g)
    T.ext_r = -2.5 * np.log10(T.trans_r)
    T.ext_z = -2.5 * np.log10(T.trans_z)
    T.wise_nobs = np.array(wise_nobs).astype(np.int16)
    T.trans_wise = np.array(wise_trans).astype(np.float32)
    T.ext_w1 = -2.5 * np.log10(T.trans_wise[:,0])
    T.ext_w2 = -2.5 * np.log10(T.trans_wise[:,1])
    T.ext_w3 = -2.5 * np.log10(T.trans_wise[:,2])
    T.ext_w4 = -2.5 * np.log10(T.trans_wise[:,3])

    T.writeto(opt.outfn)

if __name__ == '__main__':
    main()
