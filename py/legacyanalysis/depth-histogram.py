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

if __name__ == '__main__':
    outfn = 'dr3-depth.fits'
    summaryfn = 'dr3-depth-summary.fits'

    fns = glob('coadd/*/*/*-depth.fits')
    fns.sort()
    print(len(fns), 'depth files')
    
    fn = fns.pop(0)
    print('Reading', fn)
    # We'll keep all files for merging...
    TT = []

    T = fits_table(fn)
    # Create / upgrade the count columns to int64.
    for band in 'grz':
        for pro in ['ptsrc', 'gal']:
            col = 'counts_%s_%s' % (pro, band)
            if not col in T.columns():
                v = np.array(len(T), np.int64)
            else:
                v = T.get(col).astype(np.int64)
            T.set(col, v)
    T.brickname = np.array([brickname_from_filename(fn)] * len(T))
    TT.append(T)
    
    for ifn,fn in enumerate(fns):
        print('Reading', ifn, 'of', len(fns), ':', fn)
        t = fits_table(fn)
        assert(np.all(t.depthlo == T.depthlo))
        assert(np.all(t.depthhi == T.depthhi))
        cols = t.get_columns()
        t.brickname = np.array([brickname_from_filename(fn)] * len(t))
        for band in 'grz':
            col = 'counts_ptsrc_%s' % band
            if not col in cols:
                continue
            C = T.get(col)
            C += t.get(col)
            col = 'counts_gal_%s' % band
            C = T.get(col)
            C += t.get(col)
        TT.append(t)
            
    T.writeto(summaryfn)
    print('Wrote', summaryfn)
    
    T = merge_tables(TT)
    T.writeto(outfn)
    print('Wrote', outfn)


    T = fits_table(summaryfn)
    dlo = T.depthlo.copy()
    dd = dlo[2] - dlo[1]
    dlo[0] = dlo[1] - dd

    ps = PlotSequence('depth')
    
    for band in 'grz':
        plt.clf()
        plt.bar(dlo, T.get('counts_ptsrc_%s' % band), width=dd)
        plt.xlabel('Depth: %s band' % band)
        plt.ylabel('Number of pixels')
        plt.title('DECaLS DR3 Depth: Point Sources, %s' % band)
        ps.savefig()

        plt.clf()
        plt.bar(dlo, T.get('counts_gal_%s' % band), width=dd)
        plt.xlabel('Depth: %s band' % band)
        plt.ylabel('Number of pixels')
        plt.title('DECaLS DR3 Depth: Canonical Galaxy, %s' % band)
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
        

    
