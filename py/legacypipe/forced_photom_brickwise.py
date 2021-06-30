import os
import sys
import numpy as np
import fitsio

def main():
    from astrometry.util.fits import fits_table, merge_tables
    from legacypipe.survey import LegacySurveyData, wcs_for_brick

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brick', help='Brick name to run', required=True)
    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 3600 0 3600")')
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)

    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')
    parser.add_argument('--survey-dir', help='Override LEGACY_SURVEY_DIR for reading images')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')

    #parser.add_argument('--apphot', action='store_true',
    #                  help='Do aperture photometry?')
    #parser.add_argument('--no-forced', dest='forced', action='store_false',
    #                  help='Do NOT do regular forced photometry?  Implies --apphot')
    #parser.add_argument('--derivs', action='store_true',
    #                    help='Include RA,Dec derivatives in forced photometry?')

    parser.add_argument('--no-ceres', action='store_false', dest='ceres',
                        help='Do not use Ceres optimization engine (use scipy)')
    parser.add_argument('--ceres-threads', type=int, default=1,
                        help='Set number of threads used by Ceres')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    opt = parser.parse_args()

    import logging
    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    #if not opt.forced:
    #    opt.apphot = True

    survey = LegacySurveyData(survey_dir=opt.survey_dir,
                              output_dir=opt.output_dir)
    if opt.catalog_dir is None:
        catsurvey = survey
    else:
        catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir)
    tfn = catsurvey.find_file('tractor', brick=opt.brick)
    print('Reading catalog from', tfn)
    T = fits_table(tfn)
    tprimhdr = fitsio.read_header(tfn)

    brick = catsurvey.get_brick_by_name(opt.brick)
    #ra1,ra2,dec1,dec2 = brick.ra1, brick.ra2, brick.dec1, brick.dec2
    #radecpoly = np.array([[ra2,dec1], [ra1,dec1], [ra1,dec2], [ra2,dec2], [ra2,dec1]])

    targetwcs = wcs_for_brick(brick)
    H,W = targetwcs.shape
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    radecpoly = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                          [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        print(len(ccds), 'CCDs survive cuts')
    # # Cut on bands to be used
    # ccds.cut(np.array([b in bands for b in ccds.filter]))
    # print('Cut to', len(ccds), 'CCDs in bands', ','.join(bands))

    # args for forced_photom_one_ccd:
    opt.apphot = True
    opt.forced = True
    opt.derivs = False
    opt.agn = False
    opt.do_calib = False
    opt.constant_invvar = False
    opt.hybrid_psf = True
    opt.normalize_psf = True
    opt.outlier_mask = None
    opt.catalog = False
    opt.write_cat = False
    opt.move_gaia = True
    opt.save_model = False
    opt.save_data = False

    args = []
    for ccd in ccds:
        args.append((survey, catsurvey, None, None, ccd, opt, None, radecpoly, None))

    from legacypipe.forced_photom import bounce_one_ccd
    if opt.threads:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)
        FF = mp.map(bounce_one_ccd, args)
        del mp
    else:
        FF = map(bounce_one_ccd, args)

    FF = [F for F in FF if F is not None]
    if len(FF) == 0:
        print('No photometry results to write.')
        return 0
    # Keep only the first header
    _,version_hdr,_,_ = FF[0]
    version_hdr.delete('CPHDU')
    version_hdr.delete('CCDNAME')

    # unpack results
    outlier_masks = [m for _,_,m,_ in FF]
    outlier_hdrs  = [h for _,_,_,h in FF]
    FF            = [F for F,_,_,_ in FF]
    F = merge_tables(FF)

    # with survey.write_output('forced-brick', brick=opt.brick) as out:
    #     F.writeto(None, fits_object=out.fits, primheader=version_hdr,
    #               units=units, columns=columns)
    #     print('Wrote', out.real_fn)
    from legacypipe.units import get_units_for_columns
    columns = F.get_columns()
    units = get_units_for_columns(columns)
    outfn = os.path.join(survey.output_dir, 'forced-brickwise-%s.fits' % opt.brick)
    F.writeto(outfn, primheader=version_hdr, units=units, columns=columns)
    print('Wrote', outfn)

    # Also average the flux measurements by band for each source and
    # add them to a new tractor file!
    bands = list(set(F.filter))
    bands.sort()

    _,Nap = F.apflux.shape

    print('Forced-phot bands:', bands)
    for b in bands:
        T.set('forced_flux_%s' % b, np.zeros(len(T), np.float32))
        T.set('forced_flux_ivar_%s' % b, np.zeros(len(T), np.float32))
        T.set('forced_apflux_%s' % b, np.zeros((len(T), Nap), np.float32))
        T.set('forced_apflux_ivar_%s' % b, np.zeros((len(T), Nap), np.float32))
        T.set('forced_psfdepth_%s' % b, np.zeros(len(T), np.float32))
        T.set('forced_galdepth_%s' % b, np.zeros(len(T), np.float32))
        T.set('forced_nexp_%s' % b, np.zeros(len(T), np.int32))

    objidmap = dict([(o,i) for i,o in enumerate(T.objid)])
    brickid = brick.brickid

    for bid,objid,band,flux,fluxiv,psfdepth,galdepth,apflux,apfluxiv in zip(
            F.brickid, F.objid, F.filter, F.flux, F.flux_ivar, F.psfdepth, F.galdepth,
            F.apflux, F.apflux_ivar):
        if bid != brickid:
            continue
        try:
            i = objidmap[objid]
        except KeyError:
            continue
        band = band.strip()
        T.get('forced_flux_%s' % band)[i] += flux * fluxiv
        T.get('forced_flux_ivar_%s' % band)[i] += fluxiv
        T.get('forced_apflux_%s' % band)[i] += apflux * apfluxiv
        T.get('forced_apflux_ivar_%s' % band)[i] += apfluxiv
        T.get('forced_psfdepth_%s' % band)[i] += psfdepth
        T.get('forced_galdepth_%s' % band)[i] += galdepth
        T.get('forced_nexp_%s' % band)[i] += 1

    eunits = {}
    for b in bands:
        iv = T.get('forced_flux_ivar_%s' % b)
        f = T.get('forced_flux_%s' % b)
        T.get('forced_flux_%s' % b)[iv > 0] = f[iv > 0] / iv[iv > 0]
        T.get('forced_flux_%s' % b)[iv == 0] = 0.
        iv = T.get('forced_apflux_ivar_%s' % b)
        f = T.get('forced_apflux_%s' % b)
        T.get('forced_apflux_%s' % b)[iv > 0] = f[iv > 0] / iv[iv > 0]
        T.get('forced_apflux_%s' % b)[iv == 0] = 0.

        flux = 'nanomaggy'
        fluxiv = '1/nanomaggy^2'
        eunits.update({'forced_flux_%s'%b: flux,
                       'forced_flux_ivar_%s'%b: fluxiv,
                       'forced_apflux_%s'%b: flux,
                       'forced_apflux_ivar_%s'%b: fluxiv,
                       'forced_psfdepth_%s'%b: fluxiv,
                       'forced_galdepth_%s'%b: fluxiv,
                       })

    outfn = os.path.join(survey.output_dir, 'tractor-forced-%s.fits' % opt.brick)
    columns = T.get_columns()
    tbands = []
    for col in columns:
        words = col.split('_')
        if len(words) != 2 or words[0] != 'flux':
            continue
        tbands.append(words[1])
    T.writeto(outfn, units=get_units_for_columns(columns, bands=tbands, extras=eunits),
              primhdr=tprimhdr)
    print('Wrote', outfn)
        
if __name__ == '__main__':
    main()
