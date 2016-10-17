'''
This script performs forced photometry of individual DECam images
given a DECaLS catalog.
'''

from __future__ import print_function
import os
import sys

import numpy as np
import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.ttime import Time, MemMeas

from tractor import Tractor
from tractor.galaxy import disable_galaxy_cache
from tractor.ellipses import EllipseE

from legacypipe.survey import LegacySurveyData, bricks_touching_wcs, exposure_metadata, get_version_header, apertures_arcsec
from catalog import read_fits_catalog

# python projects/desi/forced-photom-decam.py decals/images/decam/CP20140810_g_v2/c4d_140816_032035_ooi_g_v2.fits.fz 43 DR1 f.fits

def get_parser():
    '''
    Returns the option parser for forced photometry of DECam images
    '''
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_argument('--no-ceres', action='store_false', dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')
    parser.add_argument('--plots', default=None, help='Create plots; specify a base filename for the plots')
    parser.add_argument('--write-cat', help='Write out the catalog subset on which forced phot was done')
    parser.add_argument('--apphot', action='store_true',
                      help='Do aperture photometry?')
    parser.add_argument('--no-forced', dest='forced', action='store_false',
                      help='Do NOT do regular forced photometry?  Implies --apphot')

    parser.add_argument('--constant-invvar', action='store_true',
                        help='Set inverse-variance to a constant across the image?')
    
    parser.add_argument('--save-model',
                        help='Compute and save model image?')
    parser.add_argument('--save-data',
                        help='Compute and save model image?')
    
    parser.add_argument('filename',help='Filename OR exposure number.')
    parser.add_argument('hdu',help='decam-HDU OR CCD name.')
    parser.add_argument('catfn',help='catalog filename OR "DR1/DR2/DR3".')
    parser.add_argument('outfn',help='Output catalog filename.')

    return parser
    
def main(survey=None, opt=None):
    '''Driver function for forced photometry of individual DECam images.
    '''
    if opt is None:
        parser = get_parser()
        opt = parser.parse_args()

    Time.add_measurement(MemMeas)
    t0 = Time()

    if os.path.exists(opt.outfn):
        print('Ouput file exists:', opt.outfn)
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
        expnum = int(opt.filename)
        opt.filename = None
    except:
        # make this 'None' for survey.find_ccds()
        expnum = None

    # Try parsing HDU number
    try:
        opt.hdu = int(opt.hdu)
        ccdname = None
    except:
        ccdname = opt.hdu
        opt.hdu = -1

    if survey is None:
        survey = LegacySurveyData()

    if opt.filename is not None and opt.hdu >= 0:
        # Read metadata from file
        T = exposure_metadata([opt.filename], hdus=[opt.hdu])
        print('Metadata:')
        T.about()
    else:
        # Read metadata from survey-ccds.fits table
        T = survey.find_ccds(expnum=expnum, ccdname=ccdname)
        print(len(T), 'with expnum', expnum, 'and CCDname', ccdname)
        if opt.hdu >= 0:
            T.cut(T.image_hdu == opt.hdu)
            print(len(T), 'with HDU', opt.hdu)
        if opt.filename is not None:
            T.cut(np.array([f.strip() == opt.filename for f in T.image_filename]))
            print(len(T), 'with filename', opt.filename)
        assert(len(T) == 1)

    ccd = T[0]
    im = survey.get_image_object(ccd)
    tim = im.get_tractor_image(slc=zoomslice, pixPsf=True, splinesky=True,
                               constant_invvar=opt.constant_invvar)
    print('Got tim:', tim)

    print('Read image:', Time()-t0)

    if opt.catfn in ['DR1', 'DR2', 'DR3']:

        margin = 20
        TT = []
        chipwcs = tim.subwcs
        bricks = bricks_touching_wcs(chipwcs, survey=survey)
        for b in bricks:
            # there is some overlap with this brick... read the catalog.
            fn = survey.find_file('tractor', brick=b.brickname)
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
            if len(T):
                TT.append(T)
        if len(TT) == 0:
            print('No sources to photometer.')
            return 0
        T = merge_tables(TT, columns='fillzero')
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
        T = fits_table(opt.catfn)

    surveydir = survey.get_survey_dir()
    del survey
        
    cat = read_fits_catalog(T)
    # print('Got cat:', cat)

    print('Read catalog:', Time()-t0)

    print('Forced photom...')
    opti = None
    forced_kwargs = {}
    if opt.ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        B = 8
        opti = CeresOptimizer(BW=B, BH=B)
        #forced_kwargs.update(verbose=True)

    for src in cat:
        # Limit sizes of huge models
        from tractor.galaxy import ProfileGalaxy
        if isinstance(src, ProfileGalaxy):
            px,py = tim.wcs.positionToPixel(src.getPosition())
            h = src._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
            MAXHALF = 128
            if h > MAXHALF:
                print('halfsize', h,'for',src,'-> setting to',MAXHALF)
                src.halfsize = MAXHALF
        
    tr = Tractor([tim], cat, optimizer=opti)
    tr.freezeParam('images')
    for src in cat:
        src.freezeAllBut('brightness')
        src.getBrightness().freezeAllBut(tim.band)
    disable_galaxy_cache()
        
    F = fits_table()
    F.brickid   = T.brickid
    F.brickname = T.brickname
    F.objid     = T.objid

    F.filter  = np.array([tim.band]               * len(T))
    F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(T))
    F.exptime = np.array([tim.primhdr['EXPTIME']] * len(T)).astype(np.float32)

    ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
    F.x = (x-1).astype(np.float32)
    F.y = (y-1).astype(np.float32)

    if opt.forced:
        if opt.plots is None:
            forced_kwargs.update(wantims=False)

        R = tr.optimize_forced_photometry(variance=True, fitstats=True,
                                          shared_params=False, priors=False, **forced_kwargs)

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

        print('Forced photom:', Time()-t0)

        
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
        F.apflux = ap.astype(np.float32)
        ap = 1./(np.vstack(apimgerr).T)**2
        ap[np.logical_not(np.isfinite(ap))] = 0.
        F.apflux_ivar = ap.astype(np.float32)
        print('Aperture photom:', Time()-t0)

    program_name = sys.argv[0]
    version_hdr = get_version_header(program_name, surveydir)
    filename = getattr(ccd, 'image_filename')
    if filename is None:
        # HACK -- print only two directory names + filename of CPFILE.
        fname = os.path.basename(im.imgfn)
        d = os.path.dirname(im.imgfn)
        d1 = os.path.basename(d)
        d = os.path.dirname(d)
        d2 = os.path.basename(d)
        filename = os.path.join(d2, d1, fname)
        print('Trimmed filename to', filename)
    version_hdr.add_record(dict(name='CPFILE', value=filename, comment='CP file'))
    version_hdr.add_record(dict(name='CPHDU', value=im.hdu, comment='CP ext'))
    version_hdr.add_record(dict(name='CAMERA', value=ccd.camera, comment='Camera'))
    version_hdr.add_record(dict(name='EXPNUM', value=im.expnum, comment='Exposure num'))
    version_hdr.add_record(dict(name='CCDNAME', value=im.ccdname, comment='CCD name'))
    version_hdr.add_record(dict(name='FILTER', value=tim.band, comment='Bandpass of this image'))
    version_hdr.add_record(dict(name='EXPOSURE',
                                value='%s-%s-%s' % (ccd.camera, im.expnum, im.ccdname),
                                comment='Name of this image'))

    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME']
    for key in keys:
        if key in tim.primhdr:
            version_hdr.add_record(dict(name=key, value=tim.primhdr[key]))

    hdr = fitsio.FITSHDR()
    units = {'exptime':'sec', 'flux':'nanomaggy', 'flux_ivar':'1/nanomaggy^2'}
    columns = F.get_columns()
    for i,col in enumerate(columns):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    outdir = os.path.dirname(opt.outfn)
    if len(outdir):
        trymakedirs(outdir)
    fitsio.write(opt.outfn, None, header=version_hdr, clobber=True)
    F.writeto(opt.outfn, header=hdr, append=True)
    print('Wrote', opt.outfn)
    
    if opt.save_model or opt.save_data:
        hdr = fitsio.FITSHDR()
        tim.getWcs().wcs.add_to_header(hdr)
    if opt.save_model:
        print('Getting model image...')
        mod = tr.getModelImage(tim)
        fitsio.write(opt.save_model, mod, header=hdr, clobber=True)
        print('Wrote', opt.save_model)
    if opt.save_data:
        fitsio.write(opt.save_data, tim.getImage(), header=hdr, clobber=True)
        print('Wrote', opt.save_data)
    
    print('Finished forced phot:', Time()-t0)
    return 0

if __name__ == '__main__':
    sys.exit(main())
