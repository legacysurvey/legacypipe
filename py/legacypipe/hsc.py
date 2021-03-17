from __future__ import print_function
import numpy as np
import os
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS

class HscImage(LegacySurveyImage):
    def compute_filenames(self):
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn
