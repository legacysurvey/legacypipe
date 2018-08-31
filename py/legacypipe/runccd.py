'''

This script runs the legacypipe code on a single CCD.

'''
from __future__ import print_function
import numpy as np
from legacypipe.survey import LegacySurveyData
from legacypipe.runbrick import get_parser, get_runbrick_kwargs, run_brick

def main():
    parser = get_parser()
    parser.set_defaults(wise=False)

    #hybridPsf=True, normalizePsf=True, rex=True, splinesky=True,
    #gaia_stars=True, wise=False, ceres=False,
    
    parser.add_argument('expnum', type=int, help='Exposure number')
    parser.add_argument('ccdname', help='CCD name (eg: "N4")')

    opt = parser.parse_args()
    optdict = vars(opt)
    verbose = optdict.pop('verbose')

    expnum = optdict.pop('expnum')
    ccdname = optdict.pop('ccdname')
    
    #print('optdict:', optdict)
    
    survey = LegacySurveyData(survey_dir=opt.survey_dir,
                              output_dir=opt.output_dir,
                              cache_dir=opt.cache_dir)

    ccds = survey.find_ccds(expnum=expnum, ccdname=ccdname)
    if len(ccds) == 0:
        print('Did not find EXPNUM', expnum, 'CCDNAME', ccdname)
        return -1
    ccd = ccds[0]
    print('Found CCD', ccd)

    awcs = survey.get_approx_wcs(ccd)
    ra,dec = awcs.radec_center()
    h,w = awcs.shape
    rr,dd = awcs.pixelxy2radec([1,1,w,w], [1,h,h,1])
    # Rotate RAs to be around RA=180 to avoid wrap-around
    rotra = np.fmod((rr - ra + 180) + 360, 360.)
    
    # assume default pixscale
    pixscale = 0.262 / 3600

    W = int(np.ceil((rotra.max() - rotra.min()) * np.cos(np.deg2rad(dec))
                    / pixscale))
    H = int(np.ceil((dd.max() - dd.min()) / pixscale))
    print('W, H', W, H)

    optdict.update(survey=survey)
    
    survey, kwargs = get_runbrick_kwargs(**optdict)

    kwargs.update(radec=(ra,dec), width=W, height=H, bands=[ccd.filter])
    
    #if opt.brick is None and opt.radec is None:

    run_brick(None, survey, **kwargs)

    #hybridPsf=True, normalizePsf=True, rex=True, splinesky=True,
    #gaia_stars=True, wise=False, ceres=False,


if __name__ == '__main__':
    main()
