from __future__ import print_function
from glob import glob
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence

def brickname_from_filename(fn):
    fn = os.path.basename(fn)
    words = fn.split('-')
    assert(len(words) == 3)
    return words[1]

def summarize_depths(basedir, outfn, summaryfn, allfn):
    pat = os.path.join(basedir, 'coadd', '*', '*', '*-depth.fits')
    #pat = os.path.join(basedir, 'coadd', '000', '*', '*-depth.fits')
    print('Pattern', pat)
    fns = glob(pat)
    fns.sort()
    print(len(fns), 'depth files')
    
    fn = fns.pop(0)
    print('Reading', fn)

    bands = 'grz'
    # We'll keep all files for merging...
    TT = []

    T = fits_table(fn)
    # Create / upgrade the count columns to int64.
    for band in bands:
        for pro in ['ptsrc', 'gal']:
            col = 'counts_%s_%s' % (pro, band)
            if not col in T.columns():
                v = np.zeros(len(T), np.int64)
            else:
                v = T.get(col).astype(np.int64)
            T.set(col, v)
    T.brickname = np.array([brickname_from_filename(fn)] * len(T))
    TT.append(T.copy())
    
    for ifn,fn in enumerate(fns):
        print('Reading', ifn, 'of', len(fns), ':', fn)
        t = fits_table(fn)
        assert(np.all(t.depthlo == T.depthlo))
        assert(np.all(t.depthhi == T.depthhi))
        cols = t.get_columns()
        t.brickname = np.array([brickname_from_filename(fn)] * len(t))
        for band in bands:
            col = 'counts_ptsrc_%s' % band
            if not col in cols:
                continue
            C = T.get(col)
            C += t.get(col)
            col = 'counts_gal_%s' % band
            C = T.get(col)
            C += t.get(col)
        TT.append(t)

    T.delete_column('brickname')
    T.writeto(summaryfn)
    print('Wrote', summaryfn)
    
    T = merge_tables(TT, columns='fillzero')
    T.writeto(outfn)
    print('Wrote', outfn)

    alldepths = fits_table()
    t = TT[0]
    nd = len(t.depthlo)
    for band in bands:
        psfcol = 'counts_ptsrc_%s' % band
        psf = np.zeros((len(TT), nd), np.int32)
        alldepths.set(psfcol, psf)
        galcol = 'counts_gal_%s' % band
        gal = np.zeros((len(TT), nd), np.int32)
        alldepths.set(galcol, gal)
        for i,T in enumerate(TT):
            cols = T.get_columns()
            if not psfcol in cols:
                continue
            #print('File', i, T.get(psfcol))
            psf[i, :] = T.get(psfcol)
            gal[i, :] = T.get(galcol)
    alldepths.brickname = np.array([t.brickname[0] for t in TT])
    alldepths.writeto(allfn)
    depths = fits_table()
    depths.depthlo = t.depthlo
    depths.depthhi = t.depthhi
    depths.writeto(allfn, append=True)
    print('Wrote', allfn)

def summary_plots(summaryfn, ps):
    T = fits_table(summaryfn)
    dlo = T.depthlo.copy()
    dd = dlo[2] - dlo[1]
    dlo[0] = dlo[1] - dd

    for band in 'grz':

        I = np.flatnonzero((dlo >= 21.5) * (dlo < 24.8))
        #I = np.flatnonzero((dlo >= 21.5))

        pixarea = (0.262 / 3600.)**2
        
        plt.clf()
        plt.bar(dlo[I], T.get('counts_ptsrc_%s' % band)[I] * pixarea, width=dd)
        plt.xlabel('Depth: %s band' % band)
        #plt.ylabel('Number of pixels')
        plt.ylabel('Area (sq.deg)')
        plt.title('DECaLS DR3 Depth: Point Sources, %s' % band)
        plt.xlim(21.5, 24.8)
        #plt.xlim(21.5, 25.)
        ps.savefig()

        plt.clf()
        plt.bar(dlo[I], T.get('counts_gal_%s' % band)[I] * pixarea, width=dd)
        plt.xlabel('Depth: %s band' % band)
        #plt.ylabel('Number of pixels')
        plt.ylabel('Area (sq.deg)')
        plt.title('DECaLS DR3 Depth: Canonical Galaxy, %s' % band)
        plt.xlim(21.5, 24.8)
        #plt.xlim(21.5, 25.)
        ps.savefig()

    for band in 'grz':
        c = list(reversed(np.cumsum(list(reversed(T.get('counts_gal_%s' % band))))))
        #N = np.sum(T.get('counts_gal_%s' % band))
        # Skip bin with no observations?
        N = np.sum(T.get('counts_gal_%s' % band)[1:])

        plt.clf()
        plt.bar(dlo, c, width=dd)

        target = dict(g=24.0, r=23.4, z=22.5)[band]
        plt.axvline(target)
        plt.axvline(target - 0.3)
        plt.axvline(target - 0.6)
        plt.axhline(N * 0.90)
        plt.axhline(N * 0.95)
        plt.axhline(N * 0.98)
        
        plt.xlabel('Depth: %s band' % band)
        plt.ylabel('Number of pixels')
        plt.title('Depth: SIMP Galaxy, %s' % band)

        ps.savefig()

if __name__ == '__main__':
    # outfn = 'dr3-depth.fits'
    # summaryfn = 'dr3-depth-summary.fits'
    # basedir = '/project/projectdirs/cosmo/data/legacysurvey/dr3'
    # summarize_depths(basedir, outfn, summaryfn)

    # outfn = 'dr4-depth.fits'
    # summaryfn = 'dr4-depth-summary.fits'
    # basedir = '/global/cscratch1/sd/dstn/galdepths-dr4'
    # summarize_depths(basedir, outfn, summaryfn)

    outfn = 'dr5-depth-concat.fits'
    summaryfn = 'dr5-depth-summary.fits'
    allfn = 'dr5-depth.fits'
    basedir = '/project/projectdirs/cosmo/work/legacysurvey/dr5/DR5_out'
    summarize_depths(basedir, outfn, summaryfn, allfn)

    ps = PlotSequence('depth')
    summary_plots(summaryfn, ps)
    import sys
    sys.exit(0)
