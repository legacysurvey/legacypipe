from __future__ import print_function
import os
import fitsio
import numpy as np

from legacypipe.cpimage import CPImage
from legacypipe.survey import LegacySurveyData

'''
Code specific to images from the 90prime camera on the Bok telescope.
'''
class BokImage(CPImage):
    '''
    Class for handling images from the 90prime camera processed by the
    NOAO Community Pipeline.
    '''

    # this is defined here for testing purposes (to handle small images)
    splinesky_boxsize = 256

    def __init__(self, survey, t):
        super(BokImage, self).__init__(survey, t)
        self.dq_saturation_bits = 0 #not used so set to 0
        self.name = self.imgfn

    def __str__(self):
        return 'Bok ' + self.name

    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq

    def read_invvar(self, clip=True, clipThresh=0.2, **kwargs):
        print('Reading the 90Prime oow weight map as Inverse Variance')
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            # Clamp near-zero (incl negative!) invvars to zero.
            # These arise due to fpack.
            if clipThresh > 0.:
                med = np.median(invvar[invvar > 0])
                thresh = clipThresh * med
            else:
                thresh = 0.
            invvar[invvar < thresh] = 0
        return invvar

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

