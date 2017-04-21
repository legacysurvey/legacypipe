from __future__ import print_function
import os
import fitsio
import numpy as np
from glob import glob
from collections import Counter
from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
from legacypipe.utils import find_unique_pixels

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
    
    # DECam
    #ax = [360, 0, -21, 36]

    ax = [310, 90, 30, 80]

    def radec_plot():
        plt.axis(ax)
        plt.xlabel('RA (deg)')
        # DECam
        #plt.xticks(np.arange(0, 361, 45))
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
    #O = fits_table('obstatus/decam-tiles_obstatus.fits')
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

    release = 'MzLS+BASS DR4'

    for band in 'grz':
        plt.clf()
        desi_map()
        N = T.get('nexp_%s' % band)
        I = np.flatnonzero(N > 0)
        #cm = matplotlib.cm.get_cmap('jet', 6)
        #cm = matplotlib.cm.get_cmap('winter', 5)
        cm = matplotlib.cm.viridis
        cm = matplotlib.cm.get_cmap(cm, 5)
        plt.scatter(T.ra[I], T.dec[I], c=N[I], s=3,
                    edgecolors='none',
                    vmin=0.5, vmax=5.5, cmap=cm)
        radec_plot()
        cax = colorbar_axes(plt.gca(), frac=0.06)
        plt.colorbar(cax=cax, ticks=range(6))
        #plt.colorbar(ticks=range(6))
        plt.title('%s: Number of exposures in %s' % (release, band))
        plt.savefig('nexp-%s.png' % band)

        plt.clf()
        desi_map()
        plt.scatter(T.ra, T.dec, c=T.get('nexp_%s' % band), s=3,
                    edgecolors='none', vmin=0, vmax=2.)
        radec_plot()
        plt.colorbar()
        plt.title('%s: PSF size, band %s' % (release, band))
        plt.savefig('psfsize-%s.png' % band)

        plt.clf()
        desi_map()

        depth = T.get('galdepth_%s' % band)
        mn,mx = np.percentile(depth, [25,95])
        plt.scatter(T.ra, T.dec, c=depth, s=3,
                    edgecolors='none', vmin=mn, vmax=mx)
        radec_plot()
        plt.colorbar()
        plt.title('%s: galaxy depth, band %s' % (release, band))
        plt.savefig('galdepth-%s.png' % band)


    for col in ['nobjs', 'npsf', 'nsimp', 'nexp', 'ndev', 'ncomp']:
        plt.clf()
        desi_map()
        N = T.get(col)
        mx = np.percentile(N, 99.5)
        plt.scatter(T.ra, T.dec, c=N, s=3,
                    edgecolors='none', vmin=0, vmax=mx)
        radec_plot()
        plt.colorbar()
        plt.title('%s: Number of objects of type %s' % (release, col[1:]))
        plt.savefig('nobjs-%s.png' % col[1:])

    Ntot = T.nobjs
    for col in ['npsf', 'nsimp', 'nexp', 'ndev', 'ncomp']:
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
            tfn = os.path.join(dirprefix, 'tractor', brick[:3], 'tractor-%s.fits'%brick)
            print('Tractor filename', tfn)
            T = fits_table(tfn, columns=['brick_primary', 'type', 'decam_psfsize',
                                         'decam_depth', 'decam_galdepth',
                                         'ebv', 'decam_mw_transmission',
                                         'wise_nobs', 'wise_mw_transmission'])
            T.cut(T.brick_primary)
            nsrcs.append(len(T))
            types = Counter([t.strip() for t in T.type])
            npsf.append(types['PSF'])
            nsimp.append(types['SIMP'])
            nexp.append(types['EXP'])
            ndev.append(types['DEV'])
            ncomp.append(types['COMP'])
            print('N sources', nsrcs[-1])
    
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

            ebv.append(np.median(T.ebv))
            gtrans.append(np.median(T.decam_mw_transmission[:,1]))
            rtrans.append(np.median(T.decam_mw_transmission[:,2]))
            ztrans.append(np.median(T.decam_mw_transmission[:,4]))
    
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
