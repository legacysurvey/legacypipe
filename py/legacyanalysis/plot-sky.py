from __future__ import print_function

import sys

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.plotutils import *

from legacypipe.survey import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--expnum', type=int, help='Cut to a single exposure')
    parser.add_argument('--ccdname', help='Cut to a single extension/CCD name')
    parser.add_argument('--run-calibs', action='store_true', help='Run calib if necessary?')
    opt = parser.parse_args()

    if opt.expnum is None or opt.ccdname is None:
        print('Need --expnum and --ccdname')
        sys.exit(-1)

    survey = LegacySurveyData(survey_dir=opt.survey_dir)
    ccd = survey.find_ccds(expnum=opt.expnum, ccdname=opt.ccdname)
    print('Found', len(ccd), 'CCD')
    assert(len(ccd) == 1)
    ccd = ccd[0]

    im = survey.get_image_object(ccd)

    if opt.run_calibs:
        # Run calibrations
        kwa = dict(splinesky=True)
        run_calibs((im, kwa))

    tim = im.get_tractor_image(gaussPsf=True, nanomaggies=False, subsky=False,
                               dq=False, invvar=False, splinesky=True)

    mn,mx = np.percentile(tim.getImage(), [50, 95])
    ima = dict(vmin=mn, vmax=mx)

    skymod = np.zeros_like(tim.getImage())
    sky = tim.getSky()
    sky.addTo(skymod)
    
    plt.clf()
    plt.subplot(1,3,1)
    dimshow(tim.getImage(), **ima)
    plt.subplot(1,3,2)
    dimshow(skymod, **ima)
    plt.subplot(1,3,3)
    dimshow(tim.getImage() - skymod, vmin=-3.*tim.sig1, vmax=3.*tim.sig1)
    plt.savefig('sky.png')
       
    
