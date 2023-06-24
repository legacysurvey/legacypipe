import os
import sys
import numpy as np
import fitsio
from collections import Counter

from legacypipe.forced_photom import forced_photom_one_ccd

def main():
    from astrometry.util.fits import fits_table, merge_tables
    from legacypipe.survey import LegacySurveyData, wcs_for_brick

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brick', help='Brick name to run') #, required=True)
    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 3600 0 3600")')

    parser.add_argument('--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')

    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)

    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')
    parser.add_argument('--survey-dir', help='Override LEGACY_SURVEY_DIR for reading images')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')

    #parser.add_argument('--apphot', action='store_true',
    #                  help='Do aperture photometry?')
    #parser.add_argument('--no-forced', dest='forced', action='store_false',
    #                  help='Do NOT do regular forced photometry?  Implies --apphot')
    parser.add_argument('--derivs', action='store_true',
                        help='Include RA,Dec derivatives in forced photometry?')
    parser.add_argument('--do-calib', default=False, action='store_true',
                        help='Run calibs if necessary?')

    parser.add_argument('--bands', default=None,
                        help='Comma-separated list of bands to forced-photometer.')

    parser.add_argument('--no-ceres', action='store_false', dest='ceres',
                        help='Do not use Ceres optimization engine (use scipy)')
    parser.add_argument('--ceres-threads', type=int, default=1,
                        help='Set number of threads used by Ceres')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument('--plots', default=None, help='Create plots; specify a base filename for the plots')
    opt = parser.parse_args()

    import logging
    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    ps = None
    if opt.plots is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plots)

    #if not opt.forced:
    #    opt.apphot = True

    survey = LegacySurveyData(survey_dir=opt.survey_dir,
                              output_dir=opt.output_dir)

    if opt.catalog_dir is None:
        catsurvey = survey
    else:
        catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir)

    custom_brick = (opt.radec is not None)
    if custom_brick:
        # Custom brick...
        from legacypipe.survey import BrickDuck
        # Custom brick; create a fake 'brick' object
        ra,dec = opt.radec
        ra,dec = float(ra), float(dec)
        print('RA,Dec', ra, dec)
        rdstring = '%06i%s%05i' % (int(1000*ra), 'm' if dec < 0 else 'p',
                                   int(1000*np.abs(dec)))
        brickname = 'custom-%s' % rdstring
        brick = BrickDuck(ra, dec, brickname)
        outfn = os.path.join(survey.output_dir, 'tractor-forced-%s.fits' % rdstring)
    else:
        brick = catsurvey.get_brick_by_name(opt.brick)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)
        outfn = os.path.join(survey.output_dir, 'tractor-forced-%s.fits' % opt.brick)
    #ra1,ra2,dec1,dec2 = brick.ra1, brick.ra2, brick.dec1, brick.dec2
    #radecpoly = np.array([[ra2,dec1], [ra1,dec1], [ra1,dec2], [ra2,dec2], [ra2,dec1]])

    print('Brick', brick)
    if os.path.exists(outfn):
        print('Output file exists:', outfn)
        return 0

    targetwcs = wcs_for_brick(brick, W=opt.width, H=opt.height)
    H,W = targetwcs.shape
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    radecpoly = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                          [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    # custom brick -- set RA,Dec bounds
    if custom_brick:
        brick.ra1,_  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,_  = targetwcs.pixelxy2radec(1, H/2)
        _, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        _, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)

    #tfn = catsurvey.find_file('tractor', brick=opt.brick)
    tfn = catsurvey.find_file('tractor', brick=brick.brickname)
    print('Reading catalog from', tfn)
    T = fits_table(tfn)
    tprimhdr = fitsio.read_header(tfn)

    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        from legacypipe.utils import NothingToDoError
        raise NothingToDoError('No CCDs touching brick')
    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        print(len(ccds), 'CCDs survive cuts.')
        print('CCD filters:', Counter(ccds.filter).most_common())
    if opt.bands:
        # Cut on bands to be used
        bands = opt.bands.split(',')
        ccds.cut(np.array([b in bands for b in ccds.filter]))
        print('Cut to', len(ccds), 'CCDs in bands', ','.join(bands))

    print('Forced-photometering CCDs:')
    for ccd in ccds:
        print('  ', ccd.image_filename)

    # args for forced_photom_one_ccd:
    opt.apphot = True
    opt.forced = True
    opt.agn = False
    opt.constant_invvar = False
    opt.hybrid_psf = True
    opt.normalize_psf = True
    opt.outlier_mask = None
    if custom_brick:
        opt.catalog = tfn
    else:
        opt.catalog = False
    opt.write_cat = False
    opt.move_gaia = True
    opt.save_model = False
    opt.save_data = False
    opt.plot_wcs = None
    if ps is not None:
        opt.plot_wcs = targetwcs

    if opt.threads:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)

    N = len(ccds)
    if opt.do_calib:
        args = []
        for i,ccd in enumerate(ccds):
            args.append((i, N, survey, ccd))
        if opt.threads:
            mp.map(calib_one_ccd, args)
        else:
            map(calib_one_ccd, args)
    opt.do_calib = False

    args = []
    for i,ccd in enumerate(ccds):
        args.append((i, N, survey, catsurvey, None, None, ccd, opt, None, radecpoly,
                     [brick], ps))

    if opt.threads:
        FF = mp.map(photom_one_ccd, args)
        del mp
    else:
        FF = map(photom_one_ccd, args)

    FF = [F for F in FF if F is not None]
    if len(FF) == 0:
        print('No photometry results to write.')
        return 0
    # Keep only the first header
    _,version_hdr,_,_ = FF[0]
    version_hdr.delete('CPHDU')
    version_hdr.delete('CCDNAME')

    # unpack results
    #outlier_masks = [m for _,_,m,_ in FF]
    #outlier_hdrs  = [h for _,_,_,h in FF]
    FF            = [F for F,_,_,_ in FF]
    F = merge_tables(FF)

    # with survey.write_output('forced-brick', brick=opt.brick) as out:
    #     F.writeto(None, fits_object=out.fits, primheader=version_hdr,
    #               units=units, columns=columns)
    #     print('Wrote', out.real_fn)
    from legacypipe.units import get_units_for_columns
    from astrometry.util.file import trymakedirs
    columns = F.get_columns()
    units = get_units_for_columns(columns)
    #outfn = os.path.join(survey.output_dir, 'forced-brickwise-%s.fits' % opt.brick)
    boutfn = outfn.replace('tractor-forced-', 'forced-brickwise-')
    dirnm = os.path.dirname(boutfn)
    trymakedirs(dirnm)
    F.writeto(boutfn, primheader=version_hdr, units=units, columns=columns)
    print('Wrote', boutfn)

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

    columns = T.get_columns()
    tbands = []
    for col in columns:
        words = col.split('_')
        if len(words) != 2 or words[0] != 'flux':
            continue
        tbands.append(words[1])
    #outfn = os.path.join(survey.output_dir, 'tractor-forced-%s.fits' % opt.brick)
    T.writeto(outfn, units=get_units_for_columns(columns, bands=tbands, extras=eunits),
              primhdr=tprimhdr)
    print('Wrote', outfn)

def calib_one_ccd(X):
    (i, N, survey,ccd) = X
    im = survey.get_image_object(ccd)
    print('Checking calibs for CCD', (i+1), 'of', N, ':', im)
    if survey.cache_dir is not None:
        im.check_for_cached_files(survey)
    im.run_calibs(splinesky=True, survey=survey,
                  halos=True, subtract_largegalaxies=True)

def photom_one_ccd(X):
    i = X[0]
    N = X[1]
    print('Forced photometry for CCD', (i+1), 'of', N)
    X = X[2:]
    return forced_photom_one_ccd(*X)

if __name__ == '__main__':
    main()
