from __future__ import print_function
import numpy as np

from legacypipe.image import LegacySurveyImage

'''
Code specific to images from the 90prime camera on the Bok telescope.
'''
class BokImage(LegacySurveyImage):
    '''
    Class for handling images from the 90prime camera processed by the
    NOAO Community Pipeline.
    '''
    def __init__(self, survey, t):
        super(BokImage, self).__init__(survey, t)
        self.dq_saturation_bits = 0 #not used so set to 0

    def read_invvar(self, **kwargs):
        return self.read_invvar_clipped(**kwargs)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

