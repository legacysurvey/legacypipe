'''
After running forced_photom.py on a set of CCDs, this script merges
the results back into a catalog.
'''
import sys
import logging
import numpy as np
from glob import glob
from collections import Counter

from astrometry.util.fits import fits_table, merge_tables

from legacypipe.survey import LegacySurveyData, get_version_header, apertures_arcsec, wcs_for_brick

def merge_forced(survey, brickname, cat, bands='grz'):
    # Get list of CCDs -- from pipeline run results, or straight from CCDs table?
    # ccdfn = survey.find_file('ccds-table', brick=brickname)
    # CCDs = fits_table(ccdfn)
    # print('Read', len(CCDs), 'CCDs')
    brick = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brick)
    CCDs = survey.ccds_touching_wcs(brickwcs)
    print('Read', len(CCDs), 'CCDs touching brick', brickname)

    # objects in the catalog: (release,brickid,objid)
    catobjs = set([(r,b,o) for r,b,o in
                   zip(cat.release, cat.brickid, cat.objid)])

    # (release, brickid, objid, band) -> [ index in forced-phot table ]
    phot_index = {}

    camexp = set()
    FF = []
    for ccd in CCDs:
        cam = ccd.camera.strip()
        key = (cam, ccd.expnum)
        if key in camexp:
            # already read this camera-expnum
            continue
        camexp.add(key)
        ffn = survey.find_file('forced', camera=cam, expnum=ccd.expnum)
        print('Forced phot filename:', ffn)
        F = fits_table(ffn)
        print('Read', len(F), 'forced-phot entries for CCD')
        ikeep = []
        for i,(r,b,o) in enumerate(zip(F.release, F.brickid, F.objid)):
            if (r,b,o) in catobjs:
                ikeep.append(i)
        if len(ikeep) == 0:
            print('No catalog objects found in this forced-phot table.')
            continue
        F.cut(np.array(ikeep))
        print('Cut to', len(F), 'phot entries matching catalog')
        FF.append(F)
    F = merge_tables(FF)
    F._header = FF[0]._header
    del FF

    I = np.lexsort((F.expnum, F.camera, F.filter, F.objid, F.brickid, F.release))
    F.cut(I)

    for i,(r,brick,o,band) in enumerate(
            zip(F.release, F.brickid, F.objid, F.filter)):
        key = (r,brick,o,band)
        if not key in phot_index:
            phot_index[key] = []
        phot_index[key].append(i)

    # find largest number of photometry measurements!
    Nmax = max([len(inds) for inds in phot_index.values()])
    print('Maximum number of photometry entries:', Nmax)

    for band in bands:
        nobs = np.zeros(len(cat), np.int32)
        indx = np.empty(len(cat), np.int32)
        indx[:] = -1
        cat.set('nobs_%s' % band, nobs)
        cat.set('index_%s' % band, indx)

        for i,(r,brick,o) in enumerate(zip(cat.release, cat.brickid, cat.objid)):
            key = (r,brick,o,band)
            try:
                inds = phot_index[key]
            except KeyError:
                continue
            nobs[i] = len(inds)
            # Indices are all contiguous (so we only need store the first one)
            assert(np.all(np.array(inds) == (np.arange(len(inds)) + inds[0])))
            indx[i] = inds[0]
    return cat, F

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')
    parser.add_argument('--out', help='Output filename -- if not set, defaults to path within --outdir.')
    parser.add_argument('-r', '--run', default=None,
                        help='Set the run type to execute (for images)')

    parser.add_argument('--catalog', help='Use the given FITS catalog file, rather than reading from a data release directory')
    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')
    parser.add_argument('--catalog-dir-north', help='Set LEGACY_SURVEY_DIR to use to read Northern catalogs')
    parser.add_argument('--catalog-dir-south', help='Set LEGACY_SURVEY_DIR to use to read Southern catalogs')
    parser.add_argument('--catalog-resolve-dec-ngc', type=float, help='Dec at which to switch from Northern to Southern catalogs (NGC only)', default=32.375)
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    opt = parser.parse_args()
    if opt.brick is None:
        parser.print_help()
        return -1
    verbose = opt.verbose
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    from legacypipe.runs import get_survey
    survey = get_survey(opt.run,
                        survey_dir=opt.survey_dir,
                        output_dir=opt.output_dir)

    columns = ['release', 'brickid', 'objid',]

    cat = None
    if opt.catalog is not None:
        cat = fits_table(opt.catalog, columns=columns)
        print('Read', len(cat), 'sources from', opt.catalog)
    elif opt.catalog_dir is not None:
        catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir)
        fn = catsurvey.find_file('tractor', brick=opt.brick)
        cat = fits_table(fn, columns=columns)
        print('Read', len(cat), 'sources from', fn)
    else:
        from astrometry.util.starutil_numpy import radectolb
        # The "north" and "south" directories often don't have
        # 'survey-bricks" files of their own -- use the 'survey' one
        # instead.
        brick = None
        for s in [survey, catsurvey]:
            try:
                brick = s.get_brick_by_name(opt.brick)
                break
            except:
                import traceback
                traceback.print_exc()
                pass

        l,b = radectolb(brick.ra, brick.dec)
        # NGC and above resolve line? -> north
        if b > 0 and brick.dec >= opt.catalog_resolve_dec_ngc:
            if opt.catalog_dir_north:
                catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir_north)
        else:
            if opt.catalog_dir_south:
                catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir_south)

        fn = catsurvey.find_file('tractor', brick=opt.brick)
        cat = fits_table(fn, columns=columns)
        print('Read', len(cat), 'sources from', fn)

    program_name = sys.argv[0]
    ## FIXME -- from catalog?
    release = 9999
    version_hdr = get_version_header(program_name, opt.survey_dir, release)

    from legacypipe.utils import add_bits
    from legacypipe.bits import DQ_BITS
    add_bits(version_hdr, DQ_BITS, 'DQMASK', 'DQ', 'D')
    from legacyzpts.psfzpt_cuts import CCD_CUT_BITS
    add_bits(version_hdr, CCD_CUT_BITS, 'CCD_CUTS', 'CC', 'C')
    for i,ap in enumerate(apertures_arcsec):
        version_hdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                    comment='(optical) Aperture radius, in arcsec'))

    cat,forced = merge_forced(survey, opt.brick, cat)
    units = []
    for i,col in enumerate(forced.get_columns()):
        units.append(forced._header.get('TUNIT%i' % (i+1), ''))
    cols = forced.get_columns()

    if opt.out:
        cat.writeto(opt.out, primheader=version_hdr)
        forced.writeto(opt.out, append=True, units=units, columns=cols)
    else:
        with survey.write_output('forced-brick', brick=opt.brick) as out:
            cat.writeto(None, fits_object=out.fits, primheader=version_hdr)
            forced.writeto(None, fits_object=out.fits, append=True, units=units, columns=cols)

if __name__ == '__main__':
    main()
