from __future__ import print_function
import os
from legacypipe.survey import LegacySurveyData, wcs_for_brick

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dr', '--drdir', dest='drdir',
                        default='/project/projectdirs/cosmo/data/legacysurvey/dr5',
                        help='Directory containing data release w/ tar-gzipped calibs')
    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')
    parser.add_argument(
        '--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')
    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')
    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')
    parser.add_argument('--no-psf', dest='do_psf', default=True, action='store_false',
                        help='Do not extract PsfEx files')
    parser.add_argument('--no-sky', dest='do_sky', default=True, action='store_false',
                        help='Do not extract SplineSky files')

    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    optdict = vars(opt)

    drdir = opt.drdir
    W = opt.width
    H = opt.height
    pixscale = opt.pixscale
    target_extent = opt.zoom

    do_psf = opt.do_psf
    do_sky = opt.do_sky

    #brickname = '1501p020'

    custom = (opt.radec is not None)
    #ra,dec = 216.03, 34.86
    if custom:
        ra,dec = opt.radec #27.30, -10.43
        ra  = float(ra)
        dec = float(dec)
        #W,H = 1000,1000
        #W,H = 1500,1500
        brickname = 'custom_%.3f_%.3f' % (ra,dec)
        #do_sky = False

    survey = LegacySurveyData()

    if custom:
        from legacypipe.survey import BrickDuck
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck(ra, dec, brickname)
    else:
        brickname = opt.brick
        brick = survey.get_brick_by_name(brickname)

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)

    # Find CCDs
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    print(len(ccds), 'CCDs touching target WCS')

    for ccd in ccds:
        im = survey.get_image_object(ccd)
        print('CCD', im)

        expnum = '%08i' % im.expnum

        if do_psf:
            if os.path.exists(im.psffn) or os.path.exists(im.merged_psffn):
                print('PSF file exists')
            else:
                print('Need PSF', im.psffn, im.merged_psffn)
    
                tarfn = os.path.join(drdir, 'calib', im.camera, 'psfex-merged',
                                     'legacysurvey_dr5_calib_decam_psfex-merged_%s.tar.gz' % expnum[:5])
                print(tarfn)
                if os.path.exists(tarfn):
                    outfn = '%s/%s-%s.fits' % (expnum[:5], im.camera, expnum)
                    cmd = 'cd %s/%s/psfex-merged && tar xvzf %s %s' % (survey.get_calib_dir(), im.camera, tarfn, outfn)
                    print(cmd)
                    os.system(cmd)

        if do_sky:
            if os.path.exists(im.splineskyfn) or os.path.exists(im.merged_splineskyfn):
                print('Sky file exists')
            else:
                print('Need sky', im.splineskyfn, im.merged_splineskyfn)
    
                tarfn = os.path.join(drdir, 'calib', im.camera, 'splinesky-merged',
                                     'legacysurvey_dr5_calib_decam_splinesky-merged_%s.tar.gz' % expnum[:5])
                print(tarfn)
                if os.path.exists(tarfn):
                    outfn = '%s/%s-%s.fits' % (expnum[:5], im.camera, expnum)
                    cmd = 'cd %s/%s/splinesky-merged && tar xvzf %s %s' % (survey.get_calib_dir(), im.camera, tarfn, outfn)
                    print(cmd)
                    os.system(cmd)

if __name__ == '__main__':
    main()

