from __future__ import print_function
import os
import numpy as np

from legacypipe.decam import DecamImage
from legacypipe.survey import *

class SurveySubset(LegacySurveyData):
    def __init__(self, mjd_period, mjd_step, **kwargs):
        super(SurveySubset, self).__init__(**kwargs)
        self.mjd_step = mjd_step
        self.mjd_period = mjd_period
        
    def get_ccds(self):
        CCDs = super(SurveySubset, self).get_ccds()
        print(len(CCDs), 'CCDs total')
        mjd0 = CCDs.mjd_obs.min()
        mjd1 = CCDs.mjd_obs.max()
        print('MJD range', mjd0, 'to', mjd1)
        nsteps = int(np.ceil((mjd1 - mjd0) / self.mjd_period))
        print('N periods:', nsteps)
        k = 0
        for i in range(nsteps):
            mjdstart = mjd0 +  i    * self.mjd_period
            mjdstop  = mjd0 + (i+1) * self.mjd_period
            I = np.flatnonzero((CCDs.mjd_obs >= mjdstart) * (CCDs.mjd_obs <  mjdstop))
            print(len(I), 'in MJD range', mjdstart, 'to', mjdstop)
            if len(I) == 0:
                continue
            if k == self.mjd_step:
                CCDs.cut(I)
                break
            k += 1
        print('Returning', len(CCDs), 'CCDs')
        return CCDs
    
def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs
    
    parser = get_parser()
    # subset number
    parser.add_argument('--mjd-period', type=float, help='How long to make periods (in days), default 30', default=30)
    parser.add_argument('--subset', type=int, help='Which period of MJDs to keep?  Default 0', default=0)
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    survey, kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1,0]:
        return kwargs

    survey = SurveySubset(opt.mjd_period, opt.subset, survey_dir=opt.survey_dir, output_dir=opt.outdir)

    run_brick(opt.brick, survey, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
