'''
This script performs forced photometry of individual Legacy Survey
images given a data release catalog.
'''

from __future__ import print_function
import os
import sys

import numpy as np
import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.ttime import Time, MemMeas

from tractor import Tractor, Catalog, NanoMaggies
from tractor.galaxy import disable_galaxy_cache
from tractor.ellipses import EllipseE

from legacypipe.survey import LegacySurveyData, bricks_touching_wcs, get_version_header, apertures_arcsec
from catalog import read_fits_catalog

def get_parser():
    '''
    Returns the option parser for forced photometry of Legacy Survey images
    '''
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')

    parser.add_argument('--catalog', help='Use the given FITS catalog file, rather than reading from a data release directory')

    parser.add_argument('--skip-calibs', dest='do_calib', default=True, action='store_false',
                        help='Do not try to run calibrations')
    parser.add_argument('--skip', dest='skip', default=False, action='store_true',
                        help='Exit if the output file already exists')

    parser.add_argument('--zoom', type=int, nargs=4, help='Set target image extent (eg "0 100 0 200")')
    parser.add_argument('--no-ceres', action='store_false', dest='ceres', help='Do not use Ceres optimization engine (use scipy)')
    parser.add_argument('--plots', default=None, help='Create plots; specify a base filename for the plots')
    parser.add_argument('--write-cat', help='Write out the catalog subset on which forced phot was done')
    parser.add_argument('--apphot', action='store_true',
                      help='Do aperture photometry?')
    parser.add_argument('--no-forced', dest='forced', action='store_false',
                      help='Do NOT do regular forced photometry?  Implies --apphot')

    parser.add_argument('--derivs', action='store_true',
                        help='Include RA,Dec derivatives in forced photometry?')

    parser.add_argument('--agn', action='store_true',
                        help='Add a point source to the center of each DEV/EXP/COMP galaxy?')

    parser.add_argument('--constant-invvar', action='store_true',
                        help='Set inverse-variance to a constant across the image?')

    parser.add_argument('--no-hybrid-psf', dest='hybrid_psf', action='store_false',
                        default=True,
                        help='Do not use hybrid pixelized-MoG PSF model?')
    parser.add_argument('--no-normalize-psf', dest='normalize_psf', action='store_false',
                        default=True,
                        help='Do not normalize PSF?')

    parser.add_argument('--no-move-gaia', dest='move_gaia', action='store_false',
                        default=True, help='Do not move Gaia stars to image epoch?')

    parser.add_argument('--save-model',
                        help='Compute and save model image?')
    parser.add_argument('--save-data',
                        help='Compute and save model image?')

    parser.add_argument('--camera', help='Cut to only CCD with given camera name?')

    parser.add_argument('expnum', help='Exposure number')
    parser.add_argument('ccdname', help='Image HDU OR CCD name.')
    parser.add_argument('outfn', help='Output catalog filename.')

    return parser

def main(survey=None, opt=None):

    import sys
    print(' '.join(sys.argv))

    '''Driver function for forced photometry of individual Legacy
    Survey images.
    '''
    if opt is None:
        parser = get_parser()
        opt = parser.parse_args()

    Time.add_measurement(MemMeas)
    t0 = tlast = Time()

    if opt.skip and os.path.exists(opt.outfn):
        print('Ouput file exists:', opt.outfn)
        sys.exit(0)

    if opt.derivs and opt.agn:
        print('Sorry, can\'t do --derivs AND --agn')
        sys.exit(0)

    if not opt.forced:
        opt.apphot = True

    zoomslice = None
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        zoomslice = (slice(y0,y1), slice(x0,x1))

    ps = None
    if opt.plots is not None:
        import pylab as plt
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plots)

    # Try parsing filename as exposure number.
    try:
        expnum = int(opt.expnum)
        filename = None
    except:
        # make this 'None' for survey.find_ccds()
        expnum = None
        filename = opt.expnum

    # Try parsing HDU number
    try:
        hdu = int(opt.ccdname)
        ccdname = None
    except:
        hdu = -1
        ccdname = opt.ccdname

    if survey is None:
        survey = LegacySurveyData()

    catsurvey = survey
    if opt.catalog_dir is not None:
        catsurvey = LegacySurveyData(survey_dir = opt.catalog_dir)

    if filename is not None and hdu >= 0:
        # FIXME -- try looking up in CCDs file?
        # Read metadata from file
        print('Warning: faking metadata from file contents')
        T = exposure_metadata([filename], hdus=[hdu])
        print('Metadata:')
        T.about()

        if not 'ccdzpt' in T.columns():
            phdr = fitsio.read_header(filename)
            T.ccdzpt = np.array([phdr['MAGZERO']])
            print('WARNING: using header MAGZERO')
            T.ccdraoff = np.array([0.])
            T.ccddecoff = np.array([0.])
            print('WARNING: setting CCDRAOFF, CCDDECOFF to zero.')

    else:
        # Read metadata from survey-ccds.fits table
        T = survey.find_ccds(expnum=expnum, ccdname=ccdname)
        print(len(T), 'with expnum', expnum, 'and CCDname', ccdname)
        if hdu >= 0:
            T.cut(T.image_hdu == hdu)
            print(len(T), 'with HDU', hdu)
        if filename is not None:
            T.cut(np.array([f.strip() == filename for f in T.image_filename]))
            print(len(T), 'with filename', filename)
        if opt.camera is not None:
            T.cut(T.camera == opt.camera)
            print(len(T), 'with camera', opt.camera)
        assert(len(T) == 1)

    ccd = T[0]
    im = survey.get_image_object(ccd)

    if opt.do_calib:
        #from legacypipe.survey import run_calibs
        #kwa = dict(splinesky=True)
        #run_calibs((im, kwa))
        im.run_calibs(splinesky=True)

    tim = im.get_tractor_image(slc=zoomslice, pixPsf=True, splinesky=True,
                               constant_invvar=opt.constant_invvar,
                               hybridPsf=opt.hybrid_psf,
                               normalizePsf=opt.normalize_psf)
    print('Got tim:', tim)

    tnow = Time()
    print('Read image:', tnow-tlast)
    tlast = tnow

    if opt.catalog:
        T = fits_table(opt.catalog)
    else:
        margin = 20
        TT = []
        chipwcs = tim.subwcs
        bricks = bricks_touching_wcs(chipwcs, survey=catsurvey)
        for b in bricks:
            # there is some overlap with this brick... read the catalog.
            fn = catsurvey.find_file('tractor', brick=b.brickname)
            if not os.path.exists(fn):
                print('WARNING: catalog', fn, 'does not exist.  Skipping!')
                continue
            print('Reading', fn)
            T = fits_table(fn, columns=[
                'ra', 'dec', 'brick_primary', 'type', 'release', 'brickid', 'brickname', 'objid',
                'fracdev', 'flux_r',
                'shapedev_r', 'shapedev_e1', 'shapedev_e2',
                'shapeexp_r', 'shapeexp_e1', 'shapeexp_e2',
                'ref_epoch', 'pmra', 'pmdec', 'parallax'
            ])
            ok,xx,yy = chipwcs.radec2pixelxy(T.ra, T.dec)
            W,H = chipwcs.get_width(), chipwcs.get_height()
            I = np.flatnonzero((xx >= -margin) * (xx <= (W+margin)) *
                               (yy >= -margin) * (yy <= (H+margin)))
            T.cut(I)
            print('Cut to', len(T), 'sources within image + margin')
            T.cut(T.brick_primary)
            print('Cut to', len(T), 'on brick_primary')
            for col in ['out_of_bounds', 'left_blob']:
                if col in T.get_columns():
                    T.cut(T.get(col) == False)
                    print('Cut to', len(T), 'on', col)
            if len(T):
                TT.append(T)
        if len(TT) == 0:
            print('No sources to photometer.')
            return 0
        T = merge_tables(TT, columns='fillzero')
        T._header = TT[0]._header
        del TT
        print('Total of', len(T), 'catalog sources')

        # Fix up various failure modes:
        # FixedCompositeGalaxy(pos=RaDecPos[240.51147402832561, 10.385488075518923], brightness=NanoMaggies: g=(flux -2.87), r=(flux -5.26), z=(flux -7.65), fracDev=FracDev(0.60177207), shapeExp=re=3.78351e-44, e1=9.30367e-13, e2=1.24392e-16, shapeDev=re=inf, e1=-0, e2=-0)
        # -> convert to EXP
        I, = np.nonzero([t == 'COMP' and not np.isfinite(r) for t,r in zip(T.type, T.shapedev_r)])
        if len(I):
            print('Converting', len(I), 'bogus COMP galaxies to EXP')
            for i in I:
                T.type[i] = 'EXP'

        # Same thing with the exp component.
        # -> convert to DEV
        I, = np.nonzero([t == 'COMP' and not np.isfinite(r) for t,r in zip(T.type, T.shapeexp_r)])
        if len(I):
            print('Converting', len(I), 'bogus COMP galaxies to DEV')
            for i in I:
                T.type[i] = 'DEV'

        if opt.write_cat:
            T.writeto(opt.write_cat)
            print('Wrote catalog to', opt.write_cat)

    surveydir = survey.get_survey_dir()
    del survey

    if opt.move_gaia:
        # Gaia stars: move RA,Dec to the epoch of this image.
        I = np.flatnonzero(T.ref_epoch > 0)
        if len(I):
            from legacypipe.survey import radec_at_mjd
            orig_ra = T.ra.copy()
            orig_dec = T.dec.copy()

            print('Moving', len(I), 'Gaia stars to MJD', tim.time.toMjd())
            ra,dec = radec_at_mjd(T.ra[I], T.dec[I], T.ref_epoch[I].astype(float),
                                  T.pmra[I], T.pmdec[I], T.parallax[I],
                                  tim.time.toMjd())
            T.ra [I] = ra
            T.dec[I] = dec

    kwargs = {}
    cols = T.get_columns()
    if 'flux_r' in cols and not 'decam_flux_r' in cols:
        kwargs.update(fluxPrefix='')

    tnow = Time()
    print('Read catalog:', tnow-tlast)
    tlast = tnow

    cat = read_fits_catalog(T, bands='r', **kwargs)
    # Replace the brightness (which will be a NanoMaggies with g,r,z)
    # with a NanoMaggies with this image's band only.
    for src in cat:
        src.brightness = NanoMaggies(**{tim.band: 1.})

    tnow = Time()
    print('Parse catalog:', tnow-tlast)
    tlast = tnow

    print('Forced photom...')
    F = run_forced_phot(cat, tim,
                        ceres=opt.ceres,
                        derivs=opt.derivs,
                        fixed_also=True,
                        agn=opt.agn,
                        do_forced=opt.forced,
                        do_apphot=opt.apphot,
                        get_model=opt.save_model,
                        ps=ps)

    if opt.save_model:
        # unpack results
        F,model_img = F

    F.release   = T.release
    F.brickid   = T.brickid
    F.brickname = T.brickname
    F.objid     = T.objid

    F.camera  = np.array([ccd.camera] * len(F))
    F.expnum  = np.array([im.expnum]  * len(F)).astype(np.int32)
    F.ccdname = np.array([im.ccdname] * len(F))

    # "Denormalizing"
    F.filter  = np.array([tim.band]               * len(F))
    F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(F))
    F.exptime = np.array([tim.primhdr['EXPTIME']] * len(F)).astype(np.float32)
    F.psfsize = np.array([tim.psf_fwhm * tim.imobj.pixscale] * len(F)).astype(np.float32)
    #### FIXME -- units?
    ### --> also add units to the dict below so the FITS headers have units
    F.sky     = np.array([tim.midsky / tim.zpscale / tim.imobj.pixscale**2] * len(F)).astype(np.float32)
    # in the same units as the depth maps -- flux inverse-variance.
    F.psfdepth = np.array([(1. / (tim.sig1 / tim.psfnorm)**2)] * len(F)).astype(np.float32)
    F.galdepth = np.array([(1. / (tim.sig1 / tim.galnorm)**2)] * len(F)).astype(np.float32)
    # F.psfdepth = np.array([-2.5 * (np.log10(5. * tim.sig1 / tim.psfnorm) - 9)] * len(F)).astype(np.float32)
    # F.galdepth = np.array([-2.5 * (np.log10(5. * tim.sig1 / tim.galnorm) - 9)] * len(F)).astype(np.float32)

    # super units questions here
    if opt.derivs:
        cosdec = np.cos(np.deg2rad(T.dec))
        F.dra  = (F.flux_dra  / F.flux) * 3600. / cosdec
        F.ddec = (F.flux_ddec / F.flux) * 3600.
        F.dra_ivar  = F.flux_dra_ivar  * (F.flux / 3600. * cosdec)**2
        F.ddec_ivar = F.flux_ddec_ivar * (F.flux / 3600.)**2
        F.delete_column('flux_dra')
        F.delete_column('flux_ddec')
        F.delete_column('flux_dra_ivar')
        F.delete_column('flux_ddec_ivar')
        F.flux = F.flux_fixed
        F.flux_ivar = F.flux_fixed_ivar
        F.delete_column('flux_fixed')
        F.delete_column('flux_fixed_ivar')

        for c in ['dra', 'ddec', 'dra_ivar', 'ddec_ivar', 'flux', 'flux_ivar']:
            F.set(c, F.get(c).astype(np.float32))

    F.ra  = T.ra
    F.dec = T.dec

    ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
    F.x = (x-1).astype(np.float32)
    F.y = (y-1).astype(np.float32)

    ## FIXME -- read outlier_masks?

    h,w = tim.shape
    F.mask = tim.dq[np.clip(np.round(F.y).astype(int), 0, h-1),
                    np.clip(np.round(F.x).astype(int), 0, w-1)]

    program_name = sys.argv[0]
    ## FIXME -- from catalog?
    release = 0
    version_hdr = get_version_header(program_name, surveydir, release)
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
    version_hdr.add_record(dict(name='PLVER', value=ccd.plver, comment='CP pipeline version'))
    version_hdr.add_record(dict(name='PROCDATE', value=ccd.procdate, comment='CP image DATE'))

    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME']
    for key in keys:
        if key in tim.primhdr:
            version_hdr.add_record(dict(name=key, value=tim.primhdr[key]))

    hdr = fitsio.FITSHDR()
    units = {'exptime':'sec',
             'flux':'nanomaggy', 'flux_ivar':'1/nanomaggy^2',
             'apflux':'nanomaggy', 'apflux_ivar':'1/nanomaggy^2',
             'psfdepth':'1/nanomaggy^2', 'galdepth':'1/nanomaggy^2',
             'sky':'nanomaggy/arcsec^2', 'psfsize':'arcsec' }
    if opt.derivs:
        units.update({'dra':'arcsec', 'ddec':'arcsec',
                      'dra_ivar':'1/arcsec^2', 'ddec_ivar':'1/arcsec^2'})
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
        fitsio.write(opt.save_model, model_img, header=hdr, clobber=True)
        print('Wrote', opt.save_model)
    if opt.save_data:
        fitsio.write(opt.save_data, tim.getImage(), header=hdr, clobber=True)
        print('Wrote', opt.save_data)

    tnow = Time()
    print('Forced phot:', tnow-tlast)
    print('Total:', tnow-t0)
    return 0

def run_forced_phot(cat, tim, ceres=True, derivs=False, agn=False,
                    do_forced=True, do_apphot=True, get_model=False, ps=None,
                    timing=False,
                    fixed_also=False):
    '''
    fixed_also: if derivs=True, also run without derivatives and report
    that flux too?
    '''
    if timing:
        t0 = Time()
    if ps is not None:
        import pylab as plt
    opti = None
    forced_kwargs = {}
    if ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        B = 8
        opti = CeresOptimizer(BW=B, BH=B)
        #forced_kwargs.update(verbose=True)

    nsize = 0
    for src in cat:

        # Limit sizes of huge models
        # from tractor.galaxy import ProfileGalaxy
        # if isinstance(src, ProfileGalaxy):
        #     px,py = tim.wcs.positionToPixel(src.getPosition())
        #     h = src._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
        #     MAXHALF = 128
        #     if h > MAXHALF:
        #         #print('halfsize', h,'for',src,'-> setting to',MAXHALF)
        #         nsize += 1
        #         src.halfsize = MAXHALF

        src.freezeAllBut('brightness')
        src.getBrightness().freezeAllBut(tim.band)
    #print('Limited the size of', nsize, 'large galaxy models')

    if derivs:
        realsrcs = []
        derivsrcs = []
        for src in cat:
            realsrcs.append(src)

            brightness_dra  = src.getBrightness().copy()
            brightness_ddec = src.getBrightness().copy()
            brightness_dra .setParams(np.zeros(brightness_dra .numberOfParams()))
            brightness_ddec.setParams(np.zeros(brightness_ddec.numberOfParams()))
            brightness_dra .freezeAllBut(tim.band)
            brightness_ddec.freezeAllBut(tim.band)

            dsrc = SourceDerivatives(src, [brightness_dra, brightness_ddec],
                                     tim, ps)
            derivsrcs.append(dsrc)

        # For convenience, put all the real sources at the front of
        # the list, so we can pull the IVs off the front of the list.
        cat = realsrcs + derivsrcs

    if agn:
        from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy
        from tractor import PointSource
        from legacypipe.survey import SimpleGalaxy, RexGalaxy

        realsrcs = []
        agnsrcs = []
        iagn = []
        for i,src in enumerate(cat):
            realsrcs.append(src)
            ## ??
            if isinstance(src, (SimpleGalaxy, RexGalaxy)):
                #print('Skipping SIMP or REX:', src)
                continue
            if isinstance(src, (ExpGalaxy, DevGalaxy, FixedCompositeGalaxy)):
                iagn.append(i)
                bright = src.getBrightness().copy()
                bright.setParams(np.zeros(bright.numberOfParams()))
                bright.freezeAllBut(tim.band)
                agn = PointSource(src.pos, bright)
                agn.freezeAllBut('brightness')
                #print('Adding "agn"', agn, 'to', src)
                #print('agn params:', agn.getParamNames())
                agnsrcs.append(src)
        iagn = np.array(iagn)
        cat = realsrcs + agnsrcs
        print('Added AGN to', len(iagn), 'galaxies')

    tr = Tractor([tim], cat, optimizer=opti)
    tr.freezeParam('images')
    disable_galaxy_cache()

    F = fits_table()

    if do_forced:

        if ps is None and not get_model:
            forced_kwargs.update(wantims=False)

        R = tr.optimize_forced_photometry(variance=True, fitstats=True,
                                          shared_params=False, priors=False,
                                          **forced_kwargs)

        if ps is not None or get_model:
            (data,mod,ie,chi,roi) = R.ims1[0]

        if ps is not None:
            ima = dict(vmin=-2.*tim.sig1, vmax=5.*tim.sig1,
                       interpolation='nearest', origin='lower',
                       cmap='gray')
            imchi = dict(interpolation='nearest', origin='lower',
                         vmin=-5, vmax=5, cmap='RdBu')
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

            if derivs:
                trx = Tractor([tim], realsrcs)
                trx.freezeParam('images')

                modx = trx.getModelImage(0)
                chix = (data - modx) * tim.getInvError()

                plt.clf()
                plt.imshow(modx, **ima)
                plt.title('Model without derivatives: %s' % tim.name)
                ps.savefig()

                plt.clf()
                plt.imshow(chix, **imchi)
                plt.title('Chi without derivatives: %s' % tim.name)
                ps.savefig()

        if derivs:
            cat = realsrcs
        if agn:
            cat = realsrcs

        F.flux = np.array([src.getBrightness().getFlux(tim.band)
                           for src in cat]).astype(np.float32)
        N = len(cat)
        F.flux_ivar = R.IV[:N].astype(np.float32)

        F.fracflux = R.fitstats.profracflux[:N].astype(np.float32)
        F.rchi2    = R.fitstats.prochi2    [:N].astype(np.float32)
        # 1 - 
        #F.fracmasked = R.fitstats.pronpix  [:N].astype(np.float32)
        try:
            F.fracmasked = R.fitstats.promasked[:N].astype(np.float32)
        except:
            print('No "fracmasked" available (only in recent Tractor versions)')

        if derivs:
            F.flux_dra  = np.array([src.getParams()[0] for src in derivsrcs]).astype(np.float32)
            F.flux_ddec = np.array([src.getParams()[1] for src in derivsrcs]).astype(np.float32)
            F.flux_dra_ivar  = R.IV[N  ::2].astype(np.float32)
            F.flux_ddec_ivar = R.IV[N+1::2].astype(np.float32)

        if agn:
            F.flux_agn = np.zeros(len(F), np.float32)
            F.flux_agn_ivar = np.zeros(len(F), np.float32)
            F.flux_agn[iagn] = np.array([src.getParams()[0] for src in agnsrcs])
            F.flux_agn_ivar[iagn] = R.IV[N:].astype(np.float32)

        if timing:
            print('Forced photom:', Time()-t0)

        if derivs and fixed_also:
            cat = realsrcs
            tr.setCatalog(Catalog(*cat))
            R = tr.optimize_forced_photometry(variance=True, fitstats=False,
                                              shared_params=False, priors=False,
                                              **forced_kwargs)
            F.flux_fixed = np.array([src.getBrightness().getFlux(tim.band)
                                     for src in cat]).astype(np.float32)
            N = len(cat)
            F.flux_fixed_ivar = R.IV[:N].astype(np.float32)

    if do_apphot:
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
        #print('Apertures:', apertures, 'pixels')

        #print('apxy shape', apxy.shape)  # --> (2,N)

        # The aperture photometry routine doesn't like pixel positions outside the image
        H,W = img.shape
        Iap = np.flatnonzero((apxy[0,:] >= 0)   * (apxy[1,:] >= 0) *
                             (apxy[0,:] <= W-1) * (apxy[1,:] <= H-1))
        print('Aperture photometry for', len(Iap), 'of', len(apxy[0,:]), 'sources within image bounds')

        for rad in apertures:
            aper = photutils.CircularAperture(apxy[:,Iap], rad)
            p = photutils.aperture_photometry(img, aper, error=imsigma)
            apimg.append(p.field('aperture_sum'))
            apimgerr.append(p.field('aperture_sum_err'))
        ap = np.vstack(apimg).T
        ap[np.logical_not(np.isfinite(ap))] = 0.
        F.apflux = np.zeros((len(F), len(apertures)), np.float32)
        F.apflux[Iap,:] = ap.astype(np.float32)

        apimgerr = np.vstack(apimgerr).T
        apiv = np.zeros(apimgerr.shape, np.float32)
        apiv[apimgerr != 0] = 1./apimgerr[apimgerr != 0]**2

        F.apflux_ivar = np.zeros((len(F), len(apertures)), np.float32)
        F.apflux_ivar[Iap,:] = apiv
        if timing:
            print('Aperture photom:', Time()-t0)

    if get_model:
        return F,mod
    return F

from tractor import MultiParams, BasicSource
class SourceDerivatives(MultiParams, BasicSource):
    def __init__(self, real, brights, tim, ps):
        from tractor.patch import Patch
        '''
        *real*: The real source whose derivatives are my profiles.
        '''
        # This a subclass of MultiParams and we pass the brightnesses
        # as our params.
        super(SourceDerivatives,self).__init__(*brights)
        self.real = real
        self.brights = brights
        self.umods = None

        # Get the current source profile and take pixel-space
        # derivatives by hand, for speed and to get symmetric
        # derivatives.
        realmod = real.getUnitFluxModelPatch(tim)
        p = realmod.patch
        dx = np.zeros_like(p)
        dy = np.zeros_like(p)
        # Omit a boundary of 1 pixel on all sides in both derivatives
        dx[1:-1, 1:-1] = (p[1:-1, :-2] - p[1:-1, 2:]) / 2.
        dy[1:-1, 1:-1] = (p[:-2, 1:-1] - p[2:, 1:-1]) / 2.
        # Convert from pixel-space to RA,Dec derivatives via CD matrix
        px, py = tim.getWcs().positionToPixel(real.pos, real)
        cdi = tim.getWcs().cdInverseAtPixel(px, py)
        dra  = dx * cdi[0, 0] + dy * cdi[1, 0]
        ddec = dx * cdi[0, 1] + dy * cdi[1, 1]

        if ps is not None:
            import pylab as plt
            mx = p.max()
            plt.clf()
            plt.subplot(2,3,1)
            plt.imshow(p, interpolation='nearest', origin='lower',
                       vmin=0, vmax=mx)
            mx *= 0.25
            plt.subplot(2,3,2)
            plt.imshow(dx, interpolation='nearest', origin='lower',
                       vmin=-mx, vmax=mx)
            plt.title('dx')
            plt.subplot(2,3,3)
            plt.imshow(dy, interpolation='nearest', origin='lower',
                       vmin=-mx, vmax=mx)
            plt.title('dy')
            plt.subplot(2,3,5)
            sc = 3600/0.262
            plt.imshow(dra/sc, interpolation='nearest', origin='lower',
                       vmin=-mx, vmax=mx)
            plt.title('dra')
            plt.subplot(2,3,6)
            plt.imshow(ddec/sc, interpolation='nearest', origin='lower',
                       vmin=-mx, vmax=mx)
            plt.title('ddec')
            ps.savefig()
        # These are our "unit-flux" models
        self.umods = [Patch(realmod.x0, realmod.y0, dra),
                      Patch(realmod.x0, realmod.y0, ddec)]

    @staticmethod
    def getNamedParams():
        return dict(dra=0, ddec=1)

    # forced photom calls getUnitFluxModelPatches
    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        return self.umods

    def getModelPatch(self, img, minsb=0., modelMask=None):
        from tractor.patch import add_patches
        pc = img.getPhotoCal()
        p1 = self.umods[0] * pc.brightnessToCounts(self.brights[0])
        p2 = self.umods[1] * pc.brightnessToCounts(self.brights[1])
        return add_patches(p1, p2)

if __name__ == '__main__':
    sys.exit(main())
