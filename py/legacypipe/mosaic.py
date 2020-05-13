from __future__ import print_function
import os
from legacypipe.image import LegacySurveyImage

class MosaicImage(LegacySurveyImage):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOAO Community Pipeline.
    '''
    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue

    def apply_amp_correction(self, img, invvar, x0, y0):
        self.apply_amp_correction_northern(img, invvar, x0, y0)

    def get_fwhm(self, primhdr, imghdr):
        # exposure 88865 has SEEINGP1 in the primary header, nothing anywhere else,
        # so FWHM in the CCDs file is NaN.
        import numpy as np
        print('mosaic get_fwhm: self.fwhm =', self.fwhm)
        if not np.isfinite(self.fwhm):
            self.fwhm = primhdr.get('SEEINGP1', 0.0)
        return self.fwhm

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        from legacypipe.image import remap_dq_cp_codes
        dq = remap_dq_cp_codes(dq, ignore_codes=[7]) # 8 also?
        return dq
