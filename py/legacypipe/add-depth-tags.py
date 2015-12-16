import fitsio
from legacypipe.common import *

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs

def add_depth_tag(decals, brick, outdir, overwrite=False):
    outfn = os.path.join(outdir, 'tractor', brick[:3], 'tractor-%s.fits' % brick)
    if os.path.exists(outfn) and not overwrite:
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
    import sys
    outdir = 'tractor2'
    decals = Decals()
    bricks = decals.get_bricks()
    bricks.cut(bricks.dec > -15)

    # Which bricks are missing the depth tags?
    if True:
        for brick in bricks.brickname:
            #fn = 'tractor2/tractor/%s/tractor-%s.fits' % (brick[:3], brick)
            fn = '/project/projectdirs/desiproc/dr2/tractor+depth/%s/tractor-%s.fits' % (brick[:3], brick)
            print 'reading', fn
            if not os.path.exists(fn):
                continue
            T = fits_table(fn)
            print 'Brick', brick, ':', len(T)
            if len(T) == 0:
                continue
            bad = np.flatnonzero((T.decam_depth[:,1] == 0) * (T.decam_nobs[:,1] > 0))
            pg = (100*len(bad)/len(T))
            bad = np.flatnonzero((T.decam_depth[:,2] == 0) * (T.decam_nobs[:,2] > 0))
            pr = (100*len(bad)/len(T))
            bad = np.flatnonzero((T.decam_depth[:,4] == 0) * (T.decam_nobs[:,4] > 0))
            pz = (100*len(bad)/len(T))

            bad = np.flatnonzero((T.decam_galdepth[:,1] == 0) * (T.decam_nobs[:,1] > 0))
            gg = (100*len(bad)/len(T))
            bad = np.flatnonzero((T.decam_galdepth[:,2] == 0) * (T.decam_nobs[:,2] > 0))
            gr = (100*len(bad)/len(T))
            bad = np.flatnonzero((T.decam_galdepth[:,4] == 0) * (T.decam_nobs[:,4] > 0))
            gz = (100*len(bad)/len(T))
            if max(pg, pr, pz, gg, gr, gz) < 10:
                continue
            print '%3i' % pg, '% bad g ptsrc'
            print '%3i' % pr, '% bad r ptsrc'
            print '%3i' % pz, '% bad z ptsrc'
            print '%3i' % gg, '% bad g gal'
            print '%3i' % gr, '% bad r gal'
            print '%3i' % gz, '% bad z gal'

        # -> p202 through p242
    sys.exit(0)

    bricks.cut((bricks.dec > 20.1) * (bricks.dec < 24.3))
    print len(bricks), 'bricks to re-run'
    for brick in bricks.brickname:
        add_depth_tag(decals, brick, outdir, overwrite=True)


    if True:
        for brick in bricks.brickname:
            add_depth_tag(decals, brick, outdir)
    else:
        # totally I/O-bound; this doesn't help.
        from astrometry.util.multiproc import *
        mp = multiproc(24)
        mp.map(bounce_add_depth_tag,
               [(decals, brick, outdir) for brick in bricks.brickname])


