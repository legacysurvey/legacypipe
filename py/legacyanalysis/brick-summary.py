from __future__ import print_function
import os
import fitsio
import numpy as np
from glob import glob
from collections import Counter
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

for ((b=0; b<36; b++)); do B=$(printf %02i $b); echo "python -u legacyanalysis/brick-summary.py --dr5 -o dr5-brick-summary-$B.fits /project/projectdirs/cosmo/work/legacysurvey/dr5/coadd/$B*/*/*-nexp-*.fits.fz > bs-$B.log 2>&1 &"; done



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
    T = fits_table(opt.files[0])
    import pylab as plt
    import matplotlib

    print('Total sources:', sum(T.nobjs))
    print('Area:', len(T)/16., 'sq deg')
    print('g,r,z coverage:', sum((T.nexp_g > 0) * (T.nexp_r > 0) * (T.nexp_z > 0)) / 16.)

    decam = True
    # vs MzLS+BASS
    #release = 'MzLS+BASS DR4'
    release = 'DECaLS DR5'

    
    if decam:
        # DECam
        ax = [360, 0, -21, 36]
    else:
        # MzLS+BASS
        ax = [310, 90, 30, 80]

    def radec_plot():
        plt.axis(ax)
        plt.xlabel('RA (deg)')
        if decam:
            plt.xticks(np.arange(0, 361, 45))
        else:
            plt.xticks(np.arange(90, 311, 30))

        plt.ylabel('Dec (deg)')

        gl = np.arange(361)
        gb = np.zeros_like(gl)
        from astrometry.util.starutil_numpy import lbtoradec
        rr,dd = lbtoradec(gl, gb)
        plt.plot(rr, dd, 'k-', alpha=0.5, lw=1)
        rr,dd = lbtoradec(gl, gb+10)
        plt.plot(rr, dd, 'k-', alpha=0.25, lw=1)
        rr,dd = lbtoradec(gl, gb-10)
        plt.plot(rr, dd, 'k-', alpha=0.25, lw=1)
        
    plt.figure(figsize=(8,5))
    plt.subplots_adjust(left=0.1, right=0.98, top=0.93)
    
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

    def desi_map():
        # Show the DESI tile map in the background.
        from astrometry.util.plotutils import antigray
        plt.imshow(desimap, origin='lower', interpolation='nearest',
                   extent=[ax[1],ax[0],ax[2],ax[3]], aspect='auto',
                   cmap=antigray, vmax=8)

    base_cmap = 'viridis'

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
        plt.scatter(T.ra[I], T.dec[I], c=N[I], s=3,
                    edgecolors='none',
                    vmin=0.5, vmax=mx + 0.5, cmap=cm)
        radec_plot()
        cax = colorbar_axes(plt.gca(), frac=0.06)
        plt.colorbar(cax=cax, ticks=range(mx+1))
        plt.title('%s: Number of exposures in %s' % (release, band))
        plt.savefig('nexp-%s.png' % band)

        #cmap = cmap_discretize(base_cmap, 15)
        cmap = cmap_discretize(base_cmap, 10)
        plt.clf()
        desi_map()
        psf = T.get('psfsize_%s' % band)
        I = np.flatnonzero(psf > 0)
        plt.scatter(T.ra[I], T.dec[I], c=psf[I], s=3,
                    edgecolors='none', cmap=cmap,
                    vmin=0.5, vmax=2.5)
        #vmin=0, vmax=3.)
        radec_plot()
        plt.colorbar()
        plt.title('%s: PSF size, band %s' % (release, band))
        plt.savefig('psfsize-%s.png' % band)

        plt.clf()
        desi_map()

        depth = T.get('galdepth_%s' % band)
        mn,mx = np.percentile(depth[depth > 0], [10,98])
        mn = np.floor(mn * 10) / 10.
        mx = np.ceil(mx * 10) / 10.
        cmap = cmap_discretize(base_cmap, 1+int((mx-mn+0.001)/0.1))
        I = (depth > 0)
        plt.scatter(T.ra[I], T.dec[I], c=depth[I], s=3,
                    edgecolors='none', vmin=mn-0.05, vmax=mx+0.05, cmap=cmap)
        radec_plot()
        plt.colorbar()
        plt.title('%s: galaxy depth (median per brick), band %s' % (release, band))
        plt.savefig('galdepth-%s.png' % band)

        plt.clf()
        desi_map()
        ext = T.get('ext_%s' % band)
        mn = 0.
        mx = 0.5
        cmap = 'hot'
        cmap = cmap_discretize(cmap, 10)
        #cmap = cmap_discretize(base_cmap, 1+int((mx-mn+0.001)/0.1))
        plt.scatter(T.ra, T.dec, c=ext, s=3,
                    edgecolors='none', vmin=mn, vmax=mx, cmap=cmap)
        radec_plot()
        plt.colorbar()
        plt.title('%s: extinction, band %s' % (release, band))
        plt.savefig('ext-%s.png' % band)

        
    for col in ['nobjs', 'npsf', 'nsimp', 'nrex', 'nexp', 'ndev', 'ncomp']:
        if not col in T.get_columns():
            continue
        plt.clf()
        desi_map()
        N = T.get(col)
        mx = np.percentile(N, 99.5)
        plt.scatter(T.ra, T.dec, c=N, s=3,
                    edgecolors='none', vmin=0, vmax=mx)
        radec_plot()
        plt.colorbar()
        tt = 'of type %s' % col[1:]
        if col == 'nobjs':
            tt = 'total'
        plt.title('%s: Number of objects %s' % (release, tt))
        plt.savefig('nobjs-%s.png' % col[1:])

    Ntot = T.nobjs
    for col in ['npsf', 'nsimp', 'nrex', 'nexp', 'ndev', 'ncomp']:
        if not col in T.get_columns():
            continue
        plt.clf()
        desi_map()
        N = T.get(col) / (Ntot.astype(np.float32))
        N[Ntot == 0] = 0.
        print(col, 'max frac:', N.max())
        mx = np.percentile(N, 99.5)
        print('mx', mx)
        plt.scatter(T.ra, T.dec, c=N, s=3,
                    edgecolors='none', vmin=0, vmax=mx)
        radec_plot()
        plt.colorbar()
        plt.title('%s: Fraction of objects of type %s' % (release, col[1:]))
        plt.savefig('fobjs-%s.png' % col[1:])
        
    return 0
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', dest='outfn', help='Output filename',
                      default='TMP/nexp.fits')
    parser.add_argument('--merge', action='store_true', help='Merge sub-tables')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--dr5', action='store_true', help='DR5 format?')
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
    
    brickset = set()
    bricklist = []
    gn = []
    rn = []
    zn = []
    
    gnhist = []
    rnhist = []
    znhist = []
    
    nnhist = 6
    
    gdepth = []
    rdepth = []
    zdepth = []
    
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
    # H=3600
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
                if opt.dr5:
                    T = fits_table(tfn, columns=['brick_primary', 'type',
                                                 'psfsize_g', 'psfsize_r', 'psfsize_z',
                                                 'psfdepth_g', 'psfdepth_r', 'psfdepth_z',
                                                 'galdepth_g', 'galdepth_r', 'galdepth_z',
                                                 'ebv',
                                                 'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z',
                                                 'nobs_w1', 'nobs_w2', 'nobs_w3', 'nobs_w4',
                                                 'mw_transmission_w1', 'mw_transmission_w2', 'mw_transmission_w3', 'mw_transmission_w4'])
                else:
                    T = fits_table(tfn, columns=['brick_primary', 'type', 'decam_psfsize',
                                             'decam_depth', 'decam_galdepth',
                                             'ebv', 'decam_mw_transmission',
                                             'wise_nobs', 'wise_mw_transmission'])
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

            if opt.dr5:
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
                
            else:
                gpsfsize.append(np.median(T.decam_psfsize[:,1]))
                rpsfsize.append(np.median(T.decam_psfsize[:,2]))
                zpsfsize.append(np.median(T.decam_psfsize[:,4]))

                gpsfdepth.append(np.median(T.decam_depth[:,1]))
                rpsfdepth.append(np.median(T.decam_depth[:,2]))
                zpsfdepth.append(np.median(T.decam_depth[:,4]))

                ggaldepth.append(np.median(T.decam_galdepth[:,1]))
                rgaldepth.append(np.median(T.decam_galdepth[:,2]))
                zgaldepth.append(np.median(T.decam_galdepth[:,4]))
    
                wise_nobs.append(np.median(T.wise_nobs, axis=0))
                wise_trans.append(np.median(T.wise_mw_transmission, axis=0))

                gtrans.append(np.median(T.decam_mw_transmission[:,1]))
                rtrans.append(np.median(T.decam_mw_transmission[:,2]))
                ztrans.append(np.median(T.decam_mw_transmission[:,4]))
                
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
    
    #print('Maximum number of sources:', max(nsrcs))
    
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
