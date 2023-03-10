import sys
import os

import numpy as np

#from astrometry.util.ttime import Time

import logging
logger = logging.getLogger('legacypipe.maskbits-light')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def write_maskbits_light(survey, brick, brickname, version_header,
                         targetwcs, W, H, refstars,
                         only_if_bad_mags=False):
    from legacypipe.utils import find_unique_pixels, copy_header_with_wcs
    from legacypipe.bits import MASKBITS, IN_BLOB
    from legacypipe.reference import get_reference_map

    I = np.flatnonzero((refstars.donotfit==False) * (refstars.ref_id >= 0))
    # Drop Gaia stars with G = BP = RP = 0
    bad = ((refstars.ref_cat[I] == 'GE') * (refstars.phot_g_mean_mag[I] == 0) *
           (refstars.phot_bp_mean_mag[I] == 0) * (refstars.phot_rp_mean_mag[I] == 0))
    if only_if_bad_mags:
        nbad = sum(bad)
        if nbad == 0:
            info('Quitting because brick does not contain any stars with bad Gaia mags')
            return 0
    info('Dropping', sum(bad), 'Gaia-EDR3 stars with G=BP=RP=0')
    refstars = refstars[I[~bad]]

    # Construct a mask bits map
    maskbits = np.zeros((H,W), np.int32)
    # !PRIMARY
    U = find_unique_pixels(targetwcs, W, H, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    maskbits |= MASKBITS['NPRIMARY'] * np.logical_not(U).astype(np.int32)
    del U

    refmap = get_reference_map(targetwcs, refstars)

    # BRIGHT
    if refmap is not None:
        maskbits |= MASKBITS['BRIGHT']  * ((refmap & IN_BLOB['BRIGHT'] ) > 0)
        maskbits |= MASKBITS['MEDIUM']  * ((refmap & IN_BLOB['MEDIUM'] ) > 0)
        maskbits |= MASKBITS['GALAXY']  * ((refmap & IN_BLOB['GALAXY'] ) > 0)
        maskbits |= MASKBITS['CLUSTER'] * ((refmap & IN_BLOB['CLUSTER']) > 0)
    del refmap

    hdr = copy_header_with_wcs(version_header, targetwcs)
    with survey.write_output('maskbits-light', brick=brickname, shape=maskbits.shape) as out:
        out.fits.write(maskbits, header=hdr, extname='MASKBITS-LIGHT')


def main():
    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring
    from astrometry.util.fits import fits_table

    from legacypipe.runs import get_survey
    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions,
        wcs_for_brick)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brick',
                        help='Brick name to run; required')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--fix-bad-mags', default=False, action='store_true',
                        help='Only run on bricks that contain a ref star with Gaia EDR3 G=BP=RP=0; otherwise, skip')
    opt = parser.parse_args()

    if opt.brick is None:
        print('Must specify --brick')
        return -1

    kw = {}
    if opt.fix_bad_mags:
        kw.update(only_if_bag_mags=True)

    run=None
    survey = get_survey(run,
                        survey_dir=opt.survey_dir,
                        output_dir=opt.output_dir)

    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    brick = survey.get_brick_by_name(opt.brick)
    brickname = brick.brickname

    refs = survey.find_file('ref-sources', brick=brickname)
    refstars = fits_table(refs)

    # Get WCS object describing brick
    W = H = 3600
    pixscale = 0.262
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    program_name = 'maskbits-light.py'
    release = None
    command_line = ' '.join(sys.argv)

    # Create FITS header with version strings
    gitver = get_git_version()
    version_header = get_version_header(program_name, survey.survey_dir, release,
                                        git_version=gitver)
    deps = get_dependency_versions(None, None, None, None, mpl=False)
    for name,value,comment in deps:
        version_header.add_record(dict(name=name, value=value, comment=comment))
    if command_line is not None:
        version_header.add_record(dict(name='CMDLINE', value=command_line,
                                       comment='runbrick command-line'))
    version_header.add_record(dict(name='BRICK', value=brickname,
                                comment='LegacySurveys brick RRRr[pm]DDd'))
    version_header.add_record(dict(name='BRICKID' , value=brick.brickid,
                                comment='LegacySurveys brick id'))
    version_header.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min (deg)'))
    version_header.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max (deg)'))
    version_header.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min (deg)'))
    version_header.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max (deg)'))
    # Add NOAO-requested headers
    version_header.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'), comment='Brick center RA (hms)'))
    version_header.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'), comment='Brick center DEC (dms)'))
    version_header.add_record(dict(
        name='CENTRA', value=brick.ra, comment='Brick center RA (deg)'))
    version_header.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='Brick center Dec (deg)'))
    for i,(r,d) in enumerate(targetrd[:4]):
        version_header.add_record(dict(
            name='CORN%iRA' %(i+1), value=r, comment='Brick corner RA (deg)'))
        version_header.add_record(dict(
            name='CORN%iDEC'%(i+1), value=d, comment='Brick corner Dec (deg)'))

    write_maskbits_light(survey, brick, brickname, version_header, targetwcs,
                         W, H, refstars, **kw)
    return 0

if __name__ == '__main__':
    #from astrometry.util.ttime import MemMeas
    #Time.add_measurement(MemMeas)
    sys.exit(main())
