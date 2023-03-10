'''

This script runs the legacypipe code on a single CCD.

'''
import numpy as np
from legacypipe.survey import LegacySurveyData
from legacypipe.runbrick import get_parser, get_runbrick_kwargs, run_brick

class FakeLegacySurveyData(LegacySurveyData):
    def filter_ccd_kd_files(self, fns):
        if self.no_kd:
            return []
        return fns

def main():
    from astrometry.util.ttime import Time

    t0 = Time()

    parser = get_parser()
    parser.set_defaults(wise=False)

    parser.add_argument('expnum', type=int, help='Exposure number')
    parser.add_argument('ccdname', help='CCD name (eg: "N4")')

    opt = parser.parse_args()
    optdict = vars(opt)
    verbose = optdict.pop('verbose')

    import logging
    import sys
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    expnum = optdict.pop('expnum')
    ccdname = optdict.pop('ccdname')
    survey = FakeLegacySurveyData(survey_dir=opt.survey_dir,
                                  output_dir=opt.output_dir,
                                  cache_dir=opt.cache_dir)
    survey.no_kd = False

    ccds = survey.find_ccds(expnum=expnum, ccdname=ccdname)
    if len(ccds) == 0:
        print('Did not find EXPNUM', expnum, 'CCDNAME', ccdname)
        return -1

    # Force the CCDs
    survey.ccds = ccds
    survey.no_kd = True
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

    # Only set W,H if they were not specified (to other than default values) on the command-line
    if opt.width == 3600 and opt.height == 3600:
        kwargs.update(width=W, height=H)
    if opt.radec is None and opt.brick is None:
        kwargs.update(radec=(ra,dec))
    kwargs.update(bands=[ccd.filter])
    print('kwargs:', kwargs)
    run_brick(None, survey, **kwargs)
    print('Finished:', Time()-t0)

if __name__ == '__main__':
    main()
