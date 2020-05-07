from __future__ import print_function

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

    def apply_amp_correction(self, img, invvar, x0, y0):
        self.apply_amp_correction_northern(img, invvar, x0, y0)

    def get_fwhm(self, primhdr, imghdr):
        # exposure BOK_CP/CP20160405/ksb_160406_104543_ooi_r_v1.fits.f
        # has SEEINGP1 in the primary header, nothing anywhere else,
        # so FWHM in the CCDs file is NaN.
        import numpy as np
        print('90prime get_fwhm: self.fwhm =', self.fwhm)
        if not np.isfinite(self.fwhm):
            self.fwhm = primhdr.get('SEEINGP1', 0.0)
        return self.fwhm

    def read_dq(self, slice=None, header=False, **kwargs):
        # Add supplemental static mask.
        import os
        import fitsio
        from pkg_resources import resource_filename
        dq = super(BokImage, self).read_dq(slice=slice, header=header, **kwargs)
        if header:
            # unpack tuple
            dq,hdr = dq
        dirname = resource_filename('legacypipe', 'config')
        fn = os.path.join(dirname, 'ksb_staticmask_ood_v1.fits.fz')
        F = fitsio.FITS(fn)[self.hdu]
        if slice is not None:
            mask = F[slice]
        else:
            mask = F.read()

        # Pixels where the mask==1 that are not already masked get marked
        # with code 1 ("bad").
        if mask.shape == dq.shape:
            dq[(mask == 1) * (dq == 0)] = 1
        else:
            print('WARNING: static mask shape', mask.shape, 'does not equal DQ shape', dq.shape, '-- not applying static mask!')

        if header:
            return dq,hdr
        return dq

    def remap_invvar
        (self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def remap_dq(self, dq, header):
        from legacypipe.image import remap_dq_cp_codes
        dq = remap_dq_cp_codes(dq, ignore_codes=[7]) # 8 also?
        return dq
