import os
import sys
import numpy as np
import fitsio

from astrometry.util.file import trymakedirs
from legacypipe.survey import LegacySurveyData, bricks_touching_wcs, get_version_header
from legacypipe.outliers import read_outlier_mask_file

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)
    parser.add_argument('--survey-dir', help='Override LEGACY_SURVEY_DIR')

    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')
    parser.add_argument('--catalog-dir-north', help='Set LEGACY_SURVEY_DIR to use to read Northern catalogs')
    parser.add_argument('--catalog-dir-south', help='Set LEGACY_SURVEY_DIR to use to read Southern catalogs')
    parser.add_argument('--catalog-resolve-dec-ngc', type=float, help='Dec at which to switch from Northern to Southern catalogs (NGC only)')
    parser.add_argument('--camera', help='Cut to only CCD with given camera name?')

    parser.add_argument('--expnum', type=int, help='Exposure number')
    parser.add_argument('--ccdname', default=None, help='CCD name to cut to (default: all)')
    parser.add_argument('--out-dir', help='Output base directory', default='.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')
    return parser

def main(survey=None, opt=None, args=None):
    if args is None:
        args = sys.argv[1:]
    if opt is None:
        parser = get_parser()
        opt = parser.parse_args(args)
    import logging
    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if survey is None:
        survey = LegacySurveyData(survey_dir=opt.survey_dir,
                                  output_dir=opt.out_dir)
    surveydir = survey.get_survey_dir()
    program_name = sys.argv[0]
    release = 9999
    version_hdr = get_version_header(program_name, surveydir, release)

    # Read metadata from survey-ccds.fits table
    ccds = survey.find_ccds(camera=opt.camera, expnum=opt.expnum,
                         ccdname=opt.ccdname)
    print(len(ccds), 'with camera', opt.camera, 'and expnum', opt.expnum, 'and ccdname', opt.ccdname)
    # sort CCDs
    ccds.cut(np.lexsort((ccds.ccdname, ccds.expnum, ccds.camera)))

    # If there is only one catalog survey_dir, we pass it to get_catalog_in_wcs
    # as the northern survey.
    catsurvey_north = survey
    catsurvey_south = None

    if opt.catalog_dir_north is not None:
        assert(opt.catalog_dir_south is not None)
        assert(opt.catalog_resolve_dec_ngc is not None)
        catsurvey_north = LegacySurveyData(survey_dir = opt.catalog_dir_north)
        catsurvey_south = LegacySurveyData(survey_dir = opt.catalog_dir_south)
    elif opt.catalog_dir is not None:
        catsurvey_north = LegacySurveyData(survey_dir = opt.catalog_dir)

    args = []
    for ccd in ccds:
        args.append((survey,
                     catsurvey_north, catsurvey_south, opt.catalog_resolve_dec_ngc, ccd))

    if opt.threads:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)
        FF = mp.map(bounce_one_ccd, args)
        del mp
    else:
        FF = list(map(bounce_one_ccd, args))

    outlier_masks = [m for m,_ in FF]
    outlier_hdrs  = [h for _,h in FF]

    # Add outlier bit meanings to the primary header
    version_hdr.add_record(dict(name='COMMENT', value='Outlier mask bit meanings'))
    version_hdr.add_record(dict(name='OUTL_POS', value=1,
                                comment='Outlier mask bit for Positive outlier'))
    version_hdr.add_record(dict(name='OUTL_NEG', value=2,
                                comment='Outlier mask bit for Negative outlier'))

    outdir = os.path.join(opt.out_dir, 'outlier-masks')
    camexp = set(zip(ccds.camera, ccds.expnum))
    for c,e in camexp:
        I = np.flatnonzero((ccds.camera == c) * (ccds.expnum == e))
        ccd = ccds[I[0]]
        imfn = ccd.image_filename.strip()
        outfn = os.path.join(outdir, imfn.replace('.fits', '-outlier.fits'))
        trymakedirs(outfn, dir=True)
        tempfn = outfn.replace('.fits', '-tmp.fits')
        with fitsio.FITS(tempfn, 'rw', clobber=True) as fits:
            fits.write(None, header=version_hdr)
            for i in I:
                mask = outlier_masks[i]
                _,_,_,meth,tile = survey.get_compression_args('outliers_mask', shape=mask.shape)
                fits.write(mask, header=outlier_hdrs[i], extname=ccds.ccdname[i],
                           compress=meth, tile_dims=tile)
        os.rename(tempfn, outfn)
        print('Wrote', outfn)

def bounce_one_ccd(X):
    # for multiprocessing
    return run_one_ccd(*X)

def run_one_ccd(survey, catsurvey_north, catsurvey_south, resolve_dec,
                ccd):
    # Outliers masks are computed within a survey (north/south for dr9), and are stored
    # in a brick-oriented way, in the results directories.

    # Grab original image headers (including WCS)
    im = survey.get_image_object(ccd)
    imhdr = im.read_image_header()

    chipwcs = im.get_wcs(hdr=imhdr)

    outlier_header = None
    outlier_mask = None
    posneg_mask = np.zeros((im.height,im.width), np.uint8)

    class faketim(object):
        pass
    tim = faketim()
    tim.imobj = im
    tim.shape = (im.height,im.width)
    tim.x0 = 0
    tim.y0 = 0

    north_ccd = (ccd.camera.strip() != 'decam')
    catsurvey = catsurvey_north
    if not north_ccd and catsurvey_south is not None:
        catsurvey = catsurvey_south
    bricks = bricks_touching_wcs(chipwcs, survey=catsurvey)
    for b in bricks:
        print('Reading outlier mask for brick', b.brickname,
              ':', catsurvey.find_file('outliers_mask', brick=b.brickname, output=False))
        ok = read_outlier_mask_file(catsurvey, [tim], b.brickname, pos_neg_mask=posneg_mask,
                                    subimage=False, output=False, apply_masks=False)
        if not ok:
            print('WARNING: failed to read outliers mask file for brick', b.brickname)

    outlier_mask = np.zeros((ccd.height, ccd.width), np.uint8)
    H,W = ccd.height,ccd.width
    outlier_mask[tim.y0:tim.y0+H, tim.x0:tim.x0+W] = posneg_mask
    del posneg_mask
    imhdr['CAMERA'] = ccd.camera
    imhdr['EXPNUM'] = ccd.expnum
    imhdr['CCDNAME'] = ccd.ccdname
    imhdr['IMGFILE'] = ccd.image_filename.strip()
    outlier_header = imhdr
            
    return outlier_mask, outlier_header

if __name__ == '__main__':
    main()
