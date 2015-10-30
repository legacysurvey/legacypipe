from __future__ import print_function
import os
import numpy as np
from .image import LegacySurveyImage
from .decam import DecamImage
from .common import *

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

    def read_invvar(self, header=False, **kwargs):
        X = super(DecamImagePlusNoise, self).read_invvar(header=header, **kwargs)
        if header:
            iv,hdr = X
        else:
            iv = X

        newiv = 1. / ((1. / iv) + self.addnoise**2)
        newiv[iv == 0] = 0.
        if header:
            return newiv,hdr
        else:
            return newiv
        
    def read_image(self, header=False, **kwargs):
        X = super(DecamImagePlusNoise, self).read_image(header=header,**kwargs)
        if header:
            im,hdr = X
        else:
            im = X

        im += np.random.normal(size=im.shape) * self.addnoise
        if header:
            return im,hdr
        else:
            return im




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
