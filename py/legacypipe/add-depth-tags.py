import fitsio
from legacypipe.common import *

from astrometry.util.fits import fits_table

if __name__ == '__main__':
    outdir = 'tractor2'
    decals = Decals()
    bricks = decals.get_bricks()
    for b in bricks:
        fn = decals.find_file('tractor', brick=b.brickname)
        if not os.path.exists(fn):
            print 'Does not exist:', fn
            continue
        T = fits_table(fn)
        print 'Read', len(T), 'from', fn
        T.decam_depth    = np.zeros((len(T), len(decals.allbands)), np.float32)
        T.decam_galdepth = np.zeros((len(T), len(decals.allbands)), np.float32)
        bands = 'grz'
        ibands = [decals.index_of_band(b) for b in bands]
        ix = np.clip(np.round(T.bx).astype(int), 0, 3599)
        iy = np.clip(np.round(T.by).astype(int), 0, 3599)
        for iband,band in zip(ibands, bands):
            fn = decals.find_file('depth', brick=b.brickname, band=band)
            if os.path.exists(fn):
                print 'Reading', fn
                img = fitsio.read(fn)
                T.decam_depth[:,iband] = img[iy, ix]

            fn = decals.find_file('galdepth', brick=b.brickname, band=band)
            if os.path.exists(fn):
                print 'Reading', fn
                img = fitsio.read(fn)
                T.decam_galdepth[:,iband] = img[iy, ix]
        outfn = os.path.join(outdir, 'tractor', b.brickname[:3], 'tractor-%s.fits' % b.brickname)
        T.writeto(outfn)
        print 'Wrote', outfn
