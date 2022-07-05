import sys

import logging
logger = logging.getLogger('legacypipe.runbrick')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)
import warnings
def formatwarning(message, category, filename, lineno, line=None):
    #return 'Warning: %s (%s:%i)' % (message, filename, lineno)
    return 'Warning: %s' % (message)
warnings.formatwarning = formatwarning

import numpy as np

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    
    #parser.add_argument('--survey-dir', type=str, default=None,
    #                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--catalog', help='Use the given FITS catalog file, rather than reading from a data release directory')
    
    parser.add_argument('--radec', nargs=2,
                        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')
    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')
    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')
    #parser.add_argument(
    #    '--unwise-tr-dir', default=None,
    #    help='Base directory for unWISE time-resolved coadds; may be a colon-separated list')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')
    return parser


def main():
    parser = get_parser()
    opt = parser.parse_args()

    if opt.radec is None:
        print('Need --radec')
        parser.print_help()
        return -1

    if opt.catalog is None:
        print('Need --catalog')
        parser.print_help()
        return -1

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)
    # silence "findfont: score(<Font 'DejaVu Sans Mono' ...)" messages
    logging.getLogger('matplotlib.font_manager').disabled = True
    # route warnings through the logging system
    logging.captureWarnings(True)

    ra,dec = opt.radec
    try:
        ra = float(ra)
    except:
        from astrometry.util.starutil_numpy import hmsstring2ra
        ra = hmsstring2ra(ra)
    try:
        dec = float(dec)
    except:
        from astrometry.util.starutil_numpy import dmsstring2dec
        dec = dmsstring2dec(dec)
    info('Parsed RA,Dec', ra,dec)

    from astrometry.util.fits import fits_table
    T = fits_table(opt.catalog)
    T.regular = (T.type != 'DUP')
    from collections import Counter
    print('T types:', Counter(T.type))

    bands = ['g','r','z']
    
    from legacypipe.catalog import read_fits_catalog
    # Add in a fake flux_{BAND} column, with flux 1.0 nanomaggies
    # for band in [1,2,3,4]:
    #     T.set('flux_w%i' % band, np.ones(len(T), np.float32))
    cat = read_fits_catalog(T, bands=bands)

    from legacypipe.survey import BrickDuck, wcs_for_brick
    brickname = 'custom_%.4f_%.4f' % (ra,dec)
    brick = BrickDuck(ra, dec, brickname)
    W,H = opt.width, opt.height
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=opt.pixscale)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    from legacypipe.survey import LegacySurveyData
    survey = LegacySurveyData('.', output_dir=opt.output_dir)
    
    from legacypipe.runbrick import stage_wise_forced
    import fitsio
    version_header = fitsio.read_header(opt.catalog)
    #version_header = fitsio.FITSHDR()

    from astrometry.util.multiproc import multiproc
    mp = multiproc()

    T.ibx = np.round(T.bx).astype(np.int32)
    T.iby = np.round(T.by).astype(np.int32)
    T.in_bounds = ((T.ibx >= 0) * (T.iby >= 0) * (T.ibx < W) * (T.iby < H))

    X = stage_wise_forced(survey=survey, cat=cat, T=T, targetwcs=targetwcs, targetrd=targetrd,
                          W=W, H=H, pixscale=opt.pixscale, brickname=brickname,
                          version_header=version_header, mp=mp,
                          unwise_dir=opt.unwise_dir,
                          unwise_tr_dir=None,
                          unwise_modelsky_dir=None,
    )

    print('X:', X.keys())

    WISE = X['WISE']
    wise_mask_maps = X['wise_mask_maps']
    wise_apertures_arcsec = X['wise_apertures_arcsec']

    import os
    
    from legacypipe.runbrick import copy_wise_into_catalog
    copy_wise_into_catalog(T, WISE, None, version_header)

    for col in ['regular', 'ibx', 'iby', 'in_bounds', 'wise_mask']:
        T.delete_column(col)

    from legacypipe.units import get_units_for_columns
    from astrometry.util.file import trymakedirs
    columns = T.get_columns()
    wbands = ['w1','w2','w3','w4']
    gbands = ['nuv','fuv']
    units = get_units_for_columns(columns, bands=bands + wbands + gbands)

    T.writeto(os.path.join(opt.output_dir, 'tractor.fits'), primheader=version_header,
              units=units, columns=columns)

if __name__ == '__main__':
    sys.exit(main())
