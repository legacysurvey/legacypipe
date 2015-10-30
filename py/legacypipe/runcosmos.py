from __future__ import print_function
import os
import numpy as np

from legacypipe.decam import DecamImage
from legacypipe.common import *

'''
Testing code for adding noise to deeper images to simulate DECaLS depth data.
'''

class DecamImagePlusNoise(DecamImage):
    '''
    A DecamImage subclass to add noise to DECam images upon read.
    '''
    def __init__(self, decals, t):
        super(DecamImagePlusNoise, self).__init__(decals, t)
        self.addnoise = t.addnoise

    def get_tractor_image(self, **kwargs):
        assert(kwargs.get('nanomaggies', True))
        tim = super(DecamImagePlusNoise, self).get_tractor_image(**kwargs)
        if tim is None:
            return None
        ie = 1. / (np.hypot(1. / tim.inverr, self.addnoise))
        ie[tim.inverr == 0] = 0.
        tim.inverr = ie
        tim.data += np.random.normal(size=im.shape) * self.addnoise
        print('Adding noise: sig1 was', tim.sig1)
        print('Adding', self.addnoise)
        sig1 = 1. / np.median(ie[ie > 0])
        print('New sig1 is', sig1)
        tim.sig1 = sig1
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.ima.update(vmin=tim.zr[0], vmax=tim.zr[1])
        return tim

def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs
    
    parser = get_parser()
    # subset number?
    #parser.add_argument('--cosmos', type=int, help='COSMOS subset 
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    kwargs = get_runbrick_kwargs(opt)
    if kwargs == -1:
        return -1

    decals = Decals(opt.decals_dir)
    decals.image_typemap.update({'decam+noise' : DecamImagePlusNoise})
    kwargs['decals'] = decals
    
    # runbrick...
    run_brick(opt.brick, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
