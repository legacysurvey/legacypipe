from __future__ import print_function

import os
import fitsio

from legacypipe.image import LegacySurveyImage
from legacypipe.common import Decals
from legacypipe.runbrick import run_brick

class MosaicImage(LegacySurveyImage):
    def __init__(self, decals, t):
        super(MosaicImage, self).__init__(decals, t)


class MosaicDecals(Decals):
    def __init__(self, **kwargs):
        super(MosaicDecals, self).__init__(**kwargs)
        self.image_typemap.update(mosaic=MosaicImage)




if __name__ == '__main__':
    decals = MosaicDecals()
    
    run_brick('3521p000', width=400, height=400)
    
    
