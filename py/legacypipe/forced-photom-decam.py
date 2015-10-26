"""This script does something.
"""
from __future__ import print_function
import os
import sys

import numpy as np
import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.ttime import Time, MemMeas

from tractor import Tractor

from legacypipe.common import Decals, bricks_touching_wcs, exposure_metadata, get_version_header, apertures_arcsec
from desi_common import read_fits_catalog
import tractor

# python projects/desi/forced-photom-decam.py decals/images/decam/CP20140810_g_v2/c4d_140816_032035_ooi_g_v2.fits.fz 43 DR1 f.fits

def main():
    """Main program
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_argument('--no-ceres', action='store_false', dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')
    parser.add_argument('--catalog-path', default=None,
                      help='Path to DECaLS DR1/DR2 catalogs; default "dr1" or "dr2", eg, /project/projectdirs/cosmo/data/legacysurvey/dr1')
    parser.add_argument('--plots', default=None, help='Create plots; specify a base filename for the plots')
    parser.add_argument('--write-cat', help='Write out the catalog subset on which forced phot was done')
    parser.add_argument('--apphot', action='store_true',
                      help='Do aperture photometry?')
    parser.add_argument('--no-forced', dest='forced', action='store_false',
                      help='Do NOT do regular forced photometry?  Implies --apphot')
    parser.add_argument('filename',help='Filename OR exposure number.')
    parser.add_argument('hdu',help='decam-HDU OR CCD name.')
    parser.add_argument('catfn',help='catalog filename OR "DR1/DR2".')
    parser.add_argument('outfn',help='Output catalog filename.')
    opt = parser.parse_args()

    Time.add_measurement(MemMeas)
    t0 = Time()

    filename, hdu, catfn, outfn = opt.filename, opt.hdu, opt.catfn, opt.outfn

    if os.path.exists(outfn):
        print('Ouput file exists:', outfn)
        sys.exit(0)

    if not opt.forced:
        opt.apphot = True

    zoomslice = None
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        zoomslice = (slice(y0,y1), slice(x0,x1))

    ps = None
    if opt.plots is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plots)

    # Try parsing filename as exposure number.
    try:
        expnum = int(filename)
        filename = None
    except:
        # make this 'None' for decals.find_ccds()
        expnum = None

    # Try parsing HDU number
    try:
        hdu = int(hdu)
        ccdname = None
    except:
        ccdname = hdu
        hdu = -1

    decals = Decals()

    if filename is not None and hdu >= 0:
        # Read metadata from file
        T = exposure_metadata([filename], hdus=[hdu])
        print('Metadata:')
        T.about()
    else:
        # Read metadata from decals-ccds.fits table
        T = decals.find_ccds(expnum=expnum, ccdname=ccdname)
        print(len(T), 'with expnum', expnum, 'and CCDname', ccdname)
        if hdu >= 0:
            T.cut(T.image_hdu == hdu)
            print(len(T), 'with HDU', hdu)
        if filename is not None:
            T.cut(np.array([f.strip() == filename for f in T.image_filename]))
            print(len(T), 'with filename', filename)
        assert(len(T) == 1)

    im = decals.get_image_object(T[0])
    tim = im.get_tractor_image(slc=zoomslice, pixPsf=True, splinesky=True)
    print('Got tim:', tim)

    if catfn in ['DR1', 'DR2']:
        if opt.catalog_path is None:
            opt.catalog_path = catfn.lower()

        margin = 20
        TT = []
        chipwcs = tim.subwcs
        bricks = bricks_touching_wcs(chipwcs, decals=decals)
        for b in bricks:
            # there is some overlap with this brick... read the catalog.
            fn = os.path.join(opt.catalog_path, 'tractor', b.brickname[:3],
                              'tractor-%s.fits' % b.brickname)
            if not os.path.exists(fn):
                print('WARNING: catalog', fn, 'does not exist.  Skipping!')
                continue
            print('Reading', fn)
            T = fits_table(fn)
            ok,xx,yy = chipwcs.radec2pixelxy(T.ra, T.dec)
            W,H = chipwcs.get_width(), chipwcs.get_height()
            I = np.flatnonzero((xx >= -margin) * (xx <= (W+margin)) *
                               (yy >= -margin) * (yy <= (H+margin)))
            T.cut(I)
            print('Cut to', len(T), 'sources within image + margin')
            # print('Brick_primary:', np.unique(T.brick_primary))
            T.cut(T.brick_primary)
            print('Cut to', len(T), 'on brick_primary')
            T.cut((T.out_of_bounds == False) * (T.left_blob == False))
            print('Cut to', len(T), 'on out_of_bounds and left_blob')
            TT.append(T)
        T = merge_tables(TT)
        T._header = TT[0]._header
        del TT

        # Fix up various failure modes:
        # FixedCompositeGalaxy(pos=RaDecPos[240.51147402832561, 10.385488075518923], brightness=NanoMaggies: g=(flux -2.87), r=(flux -5.26), z=(flux -7.65), fracDev=FracDev(0.60177207), shapeExp=re=3.78351e-44, e1=9.30367e-13, e2=1.24392e-16, shapeDev=re=inf, e1=-0, e2=-0)
        # -> convert to EXP
        I = np.flatnonzero(np.array([((t.type == 'COMP') and
                                      (not np.isfinite(t.shapedev_r)))
                                     for t in T]))
        if len(I):
            print('Converting', len(I), 'bogus COMP galaxies to EXP')
            for i in I:
                T.type[i] = 'EXP'

        # Same thing with the exp component.
        # -> convert to DEV
        I = np.flatnonzero(np.array([((t.type == 'COMP') and
                                      (not np.isfinite(t.shapeexp_r)))
                                     for t in T]))
        if len(I):
            print('Converting', len(I), 'bogus COMP galaxies to DEV')
            for i in I:
                T.type[i] = 'DEV'

        if opt.write_cat:
            T.writeto(opt.write_cat)
            print('Wrote catalog to', opt.write_cat)

    else:
        T = fits_table(catfn)

    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T

    cat = read_fits_catalog(T, ellipseClass=tractor.ellipses.EllipseE)
    # print('Got cat:', cat)

    print('Forced photom...')
    opti = None
    if opt.ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        B = 8
        opti = CeresOptimizer(BW=B, BH=B)

    tr = Tractor([tim], cat, optimizer=opti)
    tr.freezeParam('images')
    for src in cat:
        src.freezeAllBut('brightness')
        src.getBrightness().freezeAllBut(tim.band)

    F = fits_table()
    F.brickid   = T.brickid
    F.brickname = T.brickname
    F.objid     = T.objid

    F.filter  = np.array([tim.band]               * len(T))
    F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(T))
    F.exptime = np.array([tim.primhdr['EXPTIME']] * len(T))

    ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
    F.x = (x-1).astype(np.float32)
    F.y = (y-1).astype(np.float32)

    if opt.apphot:
        import photutils

        img = tim.getImage()
        ie = tim.getInvError()
        with np.errstate(divide='ignore'):
            imsigma = 1. / ie
        imsigma[ie == 0] = 0.

        apimg = []
        apimgerr = []

        # Aperture photometry locations
        xxyy = np.vstack([tim.wcs.positionToPixel(src.getPosition()) for src in cat]).T
        apxy = xxyy - 1.

        apertures = apertures_arcsec / tim.wcs.pixel_scale()
        print('Apertures:', apertures, 'pixels')

        for rad in apertures:
            aper = photutils.CircularAperture(apxy, rad)
            p = photutils.aperture_photometry(img, aper, error=imsigma)
            apimg.append(p.field('aperture_sum'))
            apimgerr.append(p.field('aperture_sum_err'))
        ap = np.vstack(apimg).T
        ap[np.logical_not(np.isfinite(ap))] = 0.
        F.apflux = ap
        ap = 1./(np.vstack(apimgerr).T)**2
        ap[np.logical_not(np.isfinite(ap))] = 0.
        F.apflux_ivar = ap

    if opt.forced:
        kwa = {}
        if opt.plots is None:
            kwa.update(wantims=False)

        R = tr.optimize_forced_photometry(variance=True, fitstats=True,
                                          shared_params=False, **kwa)

        if opt.plots:
            (data,mod,ie,chi,roi) = R.ims1[0]

            ima = tim.ima
            imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)
            plt.clf()
            plt.imshow(data, **ima)
            plt.title('Data: %s' % tim.name)
            ps.savefig()

            plt.clf()
            plt.imshow(mod, **ima)
            plt.title('Model: %s' % tim.name)
            ps.savefig()

            plt.clf()
            plt.imshow(chi, **imchi)
            plt.title('Chi: %s' % tim.name)
            ps.savefig()

        F.flux = np.array([src.getBrightness().getFlux(tim.band)
                           for src in cat]).astype(np.float32)
        F.flux_ivar = R.IV.astype(np.float32)

        F.fracflux = R.fitstats.profracflux.astype(np.float32)
        F.rchi2    = R.fitstats.prochi2    .astype(np.float32)

    program_name = sys.argv[0]
    version_hdr = get_version_header(program_name, decals.decals_dir)
    # HACK -- print only two directory names + filename of CPFILE.
    fname = os.path.basename(im.imgfn)
    d = os.path.dirname(im.imgfn)
    d1 = os.path.basename(d)
    d = os.path.dirname(d)
    d2 = os.path.basename(d)
    fname = os.path.join(d2, d1, fname)
    print('Trimmed filename to', fname)
    #version_hdr.add_record(dict(name='CPFILE', value=im.imgfn, comment='DECam comm.pipeline file'))
    version_hdr.add_record(dict(name='CPFILE', value=fname, comment='DECam comm.pipeline file'))
    version_hdr.add_record(dict(name='CPHDU', value=im.hdu, comment='DECam comm.pipeline ext'))
    version_hdr.add_record(dict(name='CAMERA', value='DECam', comment='Dark Energy Camera'))
    version_hdr.add_record(dict(name='EXPNUM', value=im.expnum, comment='DECam exposure num'))
    version_hdr.add_record(dict(name='CCDNAME', value=im.ccdname, comment='DECam CCD name'))
    version_hdr.add_record(dict(name='FILTER', value=tim.band, comment='Bandpass of this image'))
    version_hdr.add_record(dict(name='EXPOSURE', value='decam-%s-%s' % (im.expnum, im.ccdname), comment='Name of this image'))

    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME']
    for key in keys:
        if key in tim.primhdr:
            version_hdr.add_record(dict(name=key, value=tim.primhdr[key]))

    hdr = fitsio.FITSHDR()

    units = {'mjd':'sec', 'exptime':'sec', 'flux':'nanomaggy',
             'flux_ivar':'1/nanomaggy^2'}
    columns = F.get_columns()
    for i,col in enumerate(columns):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    outdir = os.path.dirname(outfn)
    if len(outdir):
        trymakedirs(outdir)
    fitsio.write(outfn, None, header=version_hdr, clobber=True)
    F.writeto(outfn, header=hdr, append=True)
    print('Wrote', outfn)

    print('Finished forced phot:', Time()-t0)
    return 0

if __name__ == '__main__':
    sys.exit(main())
