from __future__ import print_function

import os
import fitsio

from image import LegacySurveyImage

class MosaicImage(LegacySurveyImage):
    def __init__(self, decals, t):
        super(MosaicImage, self).__init__(decals, t)



