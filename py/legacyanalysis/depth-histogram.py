from astrometry.util.fits import *
from glob import glob
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.plotutils import *

fn = 'dr2-depth.fits'
if os.path.exists(fn):
    T = fits_table(fn)
    dlo = T.depthlo.copy()
    dd = dlo[2] - dlo[1]
    dlo[0] = dlo[1] - dd

    ps = PlotSequence('depth')
    
    for band in 'grz':
        plt.clf()
        plt.bar(dlo, T.get('counts_ptsrc_%s' % band), width=dd)
        plt.xlabel('Depth: %s band' % band)
        plt.ylabel('Number of pixels')
        plt.title('DECaLS DR2 Depth: Point Sources, %s' % band)
        ps.savefig()

        plt.clf()
        plt.bar(dlo, T.get('counts_gal_%s' % band), width=dd)
        plt.xlabel('Depth: %s band' % band)
        plt.ylabel('Number of pixels')
        plt.title('DECaLS DR2 Depth: Canonical Galaxy, %s' % band)
        ps.savefig()

    import sys
    sys.exit(0)
        
        
if __name__ == '__main__':
    fns = glob('dr2p/coadd/*/*/*-depth.fits')
    fns.sort()
    print len(fns), 'depth files'

    fn = fns.pop(0)
    print 'Reading', fn
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
                
    for ifn,fn in enumerate(fns):
        print 'Reading', ifn, 'of', len(fns), ':', fn
        t = fits_table(fn)
        assert(np.all(t.depthlo == T.depthlo))
        assert(np.all(t.depthhi == T.depthhi))
        cols = t.get_columns()
        for band in 'grz':
            col = 'counts_ptsrc_%s' % band
            if not col in cols:
                continue
            C = T.get(col)
            C += t.get(col)
            col = 'counts_gal_%s' % band
            C = T.get(col)
            C += t.get(col)

    T.writeto('dr2-depth.fits')
