from __future__ import print_function

import os
import numpy as np
from astrometry.libkd.spherematch import match_radec
from legacypipe.survey import LegacySurveyData, get_git_version, wcs_for_brick
from legacypipe.runbrick import make_depth_cut

if __name__ == '__main__':
    survey = LegacySurveyData()
    ccds = survey.get_ccds()
    bricks = survey.get_bricks()
    print(len(ccds), 'CCDs')
    print(len(bricks), 'bricks')

    bricks.cut(bricks.dec >= -30)
    print(len(bricks), 'above Dec of -30')

    I = survey.photometric_ccds(ccds)
    ccds.cut(I)
    print(len(ccds), 'pass photometric cut')

    I,J,d = match_radec(bricks.ra, bricks.dec, ccds.ra, ccds.dec, 0.5, nearest=True)
    print(len(I), 'bricks with CCDs nearby')
    bricks.cut(I)

    for i,brick in enumerate(bricks):
        print()
        print()
        print('Brick', (i+1), 'of', len(bricks), ':', brick.brickname)

        dirnm = os.path.join('depthcuts', brick.brickname[:3])
        outfn = os.path.join(dirnm, 'ccds-%s.fits' % brick.brickname)
        if os.path.exists(outfn):
            print('Exists:', outfn)
            continue

        H,W = 3600,3600
        pixscale = 0.262
        bands = ['g','r','z']

        # Get WCS object describing brick
        targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        gitver = get_git_version()

        bccds = survey.ccds_touching_wcs(targetwcs)
        if bccds is None:
            print('No CCDs actually touching brick')
            continue
        print(len(bccds), 'CCDs actually touching brick')

        bccds.cut(np.in1d(bccds.filter, bands))
        print('Cut on filter:', len(bccds), 'CCDs remain.')

        I = survey.photometric_ccds(bccds)
        if I is None:
            print('None cut')
        else:
            print(len(I), 'of', len(bccds), 'CCDs are photometric')
            bccds.cut(I)
        if len(I) == 0:
            print('No CCDs left')
            continue

        plots = False
        ps = None
        splinesky = True
        gaussPsf = False
        pixPsf = True
        do_calibs = False

        try:
            I = make_depth_cut(survey, bccds, bands, targetrd, brick, W, H, pixscale,
                               plots, ps, splinesky, gaussPsf, pixPsf, do_calibs,
                               gitver, targetwcs)
        except:
            print('Failed to make_depth_cut():')
            import traceback
            traceback.print_exc()
            continue

        I = np.array(I)
        print(len(I), 'CCDs passed depth cut')
        keep = np.zeros(len(bccds), bool)
        if len(I):
            keep[I] = True
        bccds.passed_depth_cut = keep

        if not os.path.exists(dirnm):
            os.makedirs(dirnm)
        bccds.writeto(outfn)
        print('Wrote', outfn)


