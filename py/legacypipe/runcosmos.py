from __future__ import print_function
import os
import numpy as np

from legacypipe.decam import DecamImage
from legacypipe.survey import *

'''
Testing code for adding noise to deeper images to simulate DECaLS depth data.
'''

class DecamImagePlusNoise(DecamImage):
    '''
    A DecamImage subclass to add noise to DECam images upon read.
    '''
    def __init__(self, survey, t):
        t.camera = 'decam'
        super(DecamImagePlusNoise, self).__init__(survey, t)
        self.addnoise = t.addnoise

    def get_tractor_image(self, **kwargs):
        assert(kwargs.get('nanomaggies', True))
        tim = super(DecamImagePlusNoise, self).get_tractor_image(**kwargs)
        if tim is None:
            return None
        with np.errstate(divide='ignore'):
            ie = 1. / (np.hypot(1. / tim.inverr, self.addnoise))
        ie[tim.inverr == 0] = 0.
        tim.inverr = ie
        tim.data += np.random.normal(size=tim.shape) * self.addnoise
        print('Adding noise: sig1 was', tim.sig1)
        print('Adding', self.addnoise)
        sig1 = 1. / np.median(ie[ie > 0])
        print('New sig1 is', sig1)
        tim.sig1 = sig1
        #tim.zr = [-3. * sig1, 10. * sig1]
        #tim.ima.update(vmin=tim.zr[0], vmax=tim.zr[1])
        return tim

class CosmosSurvey(LegacySurveyData):
    def __init__(self, subset=0, **kwargs):
        super(CosmosSurvey, self).__init__(**kwargs)
        self.subset = subset
        self.image_typemap.update({'decam+noise' : DecamImagePlusNoise})
        
    def get_ccds(self):
        CCDs = super(CosmosSurvey, self).get_ccds()
        CCDs.cut(CCDs.subset == self.subset)
        return CCDs
    
def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs
    
    parser = get_parser()
    # subset number
    parser.add_argument('--subset', type=int, help='COSMOS subset number [0 to 4, 10 to 12]', default=0)
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    survey, kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1,0]:
        return kwargs

    survey = CosmosSurvey(survey_dir=opt.survey_dir, subset=opt.subset,
                          output_dir=opt.outdir)

    run_brick(opt.brick, survey, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
