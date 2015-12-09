import fitsio
from legacypipe.common import *

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs

def add_depth_tag(decals, brick, outdir):
    outfn = os.path.join(outdir, 'tractor', brick[:3], 'tractor-%s.fits' % brick)
    if os.path.exists(outfn):
        print 'Exists:', outfn
        return
    fn = decals.find_file('tractor', brick=brick)
    if not os.path.exists(fn):
        print 'Does not exist:', fn
        return
    T = fits_table(fn, lower=False)
    primhdr = fitsio.read_header(fn)
    hdr = fitsio.read_header(fn, ext=1)
    print 'Read', len(T), 'from', fn
    T.decam_depth    = np.zeros((len(T), len(decals.allbands)), np.float32)
    T.decam_galdepth = np.zeros((len(T), len(decals.allbands)), np.float32)
    bands = 'grz'
    ibands = [decals.index_of_band(b) for b in bands]
    ix = np.clip(np.round(T.bx).astype(int), 0, 3599)
    iy = np.clip(np.round(T.by).astype(int), 0, 3599)
    for iband,band in zip(ibands, bands):
        fn = decals.find_file('depth', brick=brick, band=band)
        if os.path.exists(fn):
            print 'Reading', fn
            img = fitsio.read(fn)
            T.decam_depth[:,iband] = img[iy, ix]

        fn = decals.find_file('galdepth', brick=brick, band=band)
        if os.path.exists(fn):
            print 'Reading', fn
            img = fitsio.read(fn)
            T.decam_galdepth[:,iband] = img[iy, ix]
    outfn = os.path.join(outdir, 'tractor', brick[:3], 'tractor-%s.fits' % brick)
    trymakedirs(outfn, dir=True)

    for s in [
        'Data product of the DECam Legacy Survey (DECaLS)',
        'Full documentation at http://legacysurvey.org',
        ]:
        primhdr.add_record(dict(name='COMMENT', value=s, comment=s))

    # print 'Header:', hdr
    # T.writeto(outfn, header=hdr, primheader=primhdr)

    # Yuck, all this to get the units right
    tmpfn = outfn + '.tmp'
    fits = fitsio.FITS(tmpfn, 'rw', clobber=True)
    fits.write(None, header=primhdr)
    cols = T.get_columns()
    units = []
    for i in range(1, len(cols)+1):
        u = hdr.get('TUNIT%i' % i, '')
        units.append(u)
    # decam_depth units
    fluxiv = '1/nanomaggy^2'
    units[-2] = fluxiv
    units[-1] = fluxiv
    fits.write([T.get(c) for c in cols], names=cols, header=hdr, units=units)
    fits.close()
    os.rename(tmpfn, outfn)
    print 'Wrote', outfn

def bounce_add_depth_tag(X):
    return add_depth_tag(*X)

if __name__ == '__main__':
    outdir = 'tractor2'
    decals = Decals()
    bricks = decals.get_bricks()
    bricks.cut(bricks.dec > -15)

    if True:
        for brick in bricks.brickname:
            add_depth_tag(decals, brick, outdir)
    else:
        # totally I/O-bound; this doesn't help.
        from astrometry.util.multiproc import *
        mp = multiproc(24)
        mp.map(bounce_add_depth_tag,
               [(decals, brick, outdir) for brick in bricks.brickname])


