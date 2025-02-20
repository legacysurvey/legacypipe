import os
import sys
import numpy as np
import fitsio
from collections import Counter

from legacypipe.forced_photom import forced_photom_one_ccd, find_missing_sga

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
    parser.add_argument('-r', '--run', default=None, help='Set the run type to execute')
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
    parser.add_argument('--plots', default=None, help='Create plots; specify a base filename for the plots')
    parser.add_argument('--too-many-sources', type=int, default=0,
                        help='Fail if more than this number of catalog entries')
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

    ps = None
    if opt.plots is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plots)

    #if not opt.forced:
    #    opt.apphot = True

    from legacypipe.runs import get_survey
    survey = get_survey(opt.run,
                        survey_dir=opt.survey_dir,
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
        outfn = survey.find_file('tractor-forced', brick=opt.brick, output=True)
        print('Output filename:', outfn)
        #outfn = os.path.join(survey.output_dir, opt.brick[:3], 'tractor-forced-%s.fits' % opt.brick)

    print('Brick', brick.brickname)
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

    opt.catalog = False
    tfn = catsurvey.find_file('tractor', brick=brick.brickname)
    if os.path.exists(tfn):
        print('Reading catalog from', tfn)
        T = fits_table(tfn)
        tprimhdr = fitsio.read_header(tfn)
        if custom_brick:
            opt.catalog = tfn
        print('Checking for nearby SGA galaxies...')
        # from get_catalog_in_wcs...
        surveys = [(catsurvey, None)]
        SGA = find_missing_sga(T, targetwcs, survey, surveys, None)#columns)
        if SGA is not None:
            #print('SGA columns:', SGA.get_columns())
            #print('T columns:', T.get_columns())
            # They have the same set of columns
            ## Add 'em in!
            T = merge_tables([T, SGA], columns='fillzero')
            #print('merged T columns:', T.get_columns())
    else:
        from legacypipe.forced_photom import get_catalog_in_wcs
        T = get_catalog_in_wcs(targetwcs, survey, catsurvey)
        tprimhdr = None
        print('Got catalog in WCS:')
        T.about()

    if opt.too_many_sources and len(T) > opt.too_many_sources:
        print('Number of catalog entries: %i exceeds --too-many-sources=%i.  Failing!' %
              (len(T), opt.too_many_sources))
        return -1

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
        print('  ', ccd.image_filename, 'expnum', ccd.expnum, 'ccdname', ccd.ccdname)

    # args for forced_photom_one_ccd:
    opt.apphot = True
    opt.forced = True
    opt.agn = False
    opt.constant_invvar = False
    opt.hybrid_psf = True
    opt.normalize_psf = True
    opt.outlier_mask = None
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
    Iphot = np.flatnonzero(np.logical_or(T.brick_primary, T.ref_cat == 'L3') *
                           (T.type != 'DUP') * (T.type != 'NUN'))
    Tphot = T[Iphot]
    Torig = Tphot
    for i,ccd in enumerate(ccds):
        if not opt.threads:
            Tphot = Torig.copy()
        args.append((i, N, survey, catsurvey, None, None, ccd, Tphot, opt, None, radecpoly,
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
    _,hdr,_,_ = FF[0]
    # Drop header cards that are about an individual image
    for key in ['CPHDU', 'CCDNAME', 'CPFILE', 'CAMERA', 'EXPNUM', 'FILTER',
                'PLVER', 'PLPROCID', 'PROCDATE', 'TELESCOP', 'OBSERVAT',
                'OBS-LAT', 'OBS-LONG', 'OBS-ELEV', 'INSTRUME']:
        hdr.delete(key)

    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions)
    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring
    gitver = get_git_version()
    hdr.add_record(dict(name='FORCEDV', value=gitver,
                        comment='forced-photom legacypipe git version'))
    deps = get_dependency_versions(None, None, None, None, mpl=False)
    for name,value,comment in deps:
        hdr.add_record(dict(name=name, value=value, comment=comment))
    command_line=' '.join(sys.argv)
    hdr.add_record(dict(name='CMDLINE', value=command_line,
                                   comment='forced-phot command-line'))
    hdr.add_record(dict(name='BRICK', value=brick.brickname,
                                comment='LegacySurveys brick RRRr[pm]DDd'))
    hdr.add_record(dict(name='BRICKID' , value=brick.brickid,
                                comment='LegacySurveys brick id'))
    hdr.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min (deg)'))
    hdr.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max (deg)'))
    hdr.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min (deg)'))
    hdr.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max (deg)'))
    # Add NOAO-requested headers
    hdr.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'), comment='Brick center RA (hms)'))
    hdr.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'), comment='Brick center DEC (dms)'))
    hdr.add_record(dict(
        name='CENTRA', value=brick.ra, comment='Brick center RA (deg)'))
    hdr.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='Brick center Dec (deg)'))
    for i,(r,d) in enumerate(radecpoly[:4]):
        hdr.add_record(dict(
            name='CORN%iRA' %(i+1), value=r, comment='Brick corner RA (deg)'))
        hdr.add_record(dict(
            name='CORN%iDEC'%(i+1), value=d, comment='Brick corner Dec (deg)'))

    print('Merging photometry results...')
    FF = [F for F,_,_,_ in FF]
    F = merge_tables(FF)

    flux_unit = 'nanomaggies'
    fluxiv_unit = 'nanomaggies^(-2)'
    from legacypipe.units import get_units_for_columns
    columns = F.get_columns()
    eunits = {'full_fit_dra': 'arcsec',
              'full_fit_ddec': 'arcsec',
              'full_fit_flux': flux_unit,
              'full_fit_flux_ivar': fluxiv_unit,
              'full_fit_dra_ivar': 'arcsec^(-2)',
              'full_fit_ddec_ivar': 'arcsec^(-2)',
              'win_dra': 'arcsec',
              'win_ddec': 'arcsec',
              'winpsf_dra': 'arcsec',
              'winpsf_ddec': 'arcsec',
              'flux_motion': flux_unit,
              'flux_motion_ivar': fluxiv_unit,
              }
    units = get_units_for_columns(columns, extras=eunits)
    with survey.write_output('forced-brick', brick=opt.brick) as out:
        F.writeto(None, fits_object=out.fits, primheader=hdr,
                  units=units, columns=columns)

    # Also average the flux measurements by band for each source and
    # add them to a new tractor file!
    bands = list(set(F.filter))
    bands.sort()

    _,Nap = F.apflux.shape

    #print('Forced-phot bands:', bands)
    for b in bands:
        from legacypipe.format_catalog import clean_column_name
        b = clean_column_name(b)
        T.set('forced_flux_%s'        % b, np.zeros(len(T), np.float32))
        T.set('forced_flux_ivar_%s'   % b, np.zeros(len(T), np.float32))
        T.set('forced_apflux_%s'      % b, np.zeros((len(T), Nap), np.float32))
        T.set('forced_apflux_ivar_%s' % b, np.zeros((len(T), Nap), np.float32))
        T.set('forced_psfdepth_%s'    % b, np.zeros(len(T), np.float32))
        T.set('forced_galdepth_%s'    % b, np.zeros(len(T), np.float32))
        T.set('forced_nexp_%s'        % b, np.zeros(len(T), np.int32))

    objidmap = dict([((b,o),i) for i,(b,o) in enumerate(zip(T.brickid, T.objid))])

    for bid,objid,band,flux,fluxiv,psfdepth,galdepth,apflux,apfluxiv in zip(
            F.brickid, F.objid, F.filter, F.flux, F.flux_ivar, F.psfdepth, F.galdepth,
            F.apflux, F.apflux_ivar):
        key = (bid, objid)
        try:
            i = objidmap[key]
        except KeyError:
            continue
        band = band.strip()
        band = clean_column_name(band)
        T.get('forced_flux_%s'        % band)[i] += flux * fluxiv
        T.get('forced_flux_ivar_%s'   % band)[i] += fluxiv
        T.get('forced_apflux_%s'      % band)[i] += apflux * apfluxiv
        T.get('forced_apflux_ivar_%s' % band)[i] += apfluxiv
        T.get('forced_psfdepth_%s'    % band)[i] += psfdepth
        T.get('forced_galdepth_%s'    % band)[i] += galdepth
        T.get('forced_nexp_%s'        % band)[i] += 1

    eunits = {}
    for b in bands:
        b = clean_column_name(b)
        iv = T.get('forced_flux_ivar_%s' % b)
        f = T.get('forced_flux_%s' % b)
        T.get('forced_flux_%s' % b)[iv > 0] = f[iv > 0] / iv[iv > 0]
        T.get('forced_flux_%s' % b)[iv == 0] = 0.
        iv = T.get('forced_apflux_ivar_%s' % b)
        f = T.get('forced_apflux_%s' % b)
        T.get('forced_apflux_%s' % b)[iv > 0] = f[iv > 0] / iv[iv > 0]
        T.get('forced_apflux_%s' % b)[iv == 0] = 0.

        eunits.update({'forced_flux_%s'%b: flux_unit,
                       'forced_flux_ivar_%s'%b: fluxiv_unit,
                       'forced_apflux_%s'%b: flux_unit,
                       'forced_apflux_ivar_%s'%b: fluxiv_unit,
                       'forced_psfdepth_%s'%b: fluxiv_unit,
                       'forced_galdepth_%s'%b: fluxiv_unit,
                       })

    columns = T.get_columns()
    tbands = []
    for col in columns:
        words = col.split('_')
        if len(words) != 2 or words[0] != 'flux':
            continue
        tbands.append(words[1])

    # Delete then add so that the new cards appear at the end?
    for r in hdr.records():
        key = r['name']
        if key == 'COMMENT':
            continue
        tprimhdr.delete(key)
    tprimhdr.add_record(dict(name='COMMENT', value=None,
                             comment='Headers below here are from the forced photometry'))
    for r in hdr.records():
        tprimhdr.add_record(r)

    with survey.write_output('tractor-forced', brick=opt.brick) as out:
        units=get_units_for_columns(columns, bands=tbands, extras=eunits)
        T.writeto(None, fits_object=out.fits, primheader=tprimhdr, units=units)

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
    sys.exit(main())
