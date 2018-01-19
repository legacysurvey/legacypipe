from __future__ import print_function
import os
from legacypipe.survey import LegacySurveyData, wcs_for_brick

def main():

    drdir = '/project/projectdirs/cosmo/data/legacysurvey/dr5'

    brickname = '1501p020'

    W=3600
    H=3600
    pixscale=0.262
    target_extent = None

    survey = LegacySurveyData()

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

