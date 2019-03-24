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

    def read_dq(self, slice=None, header=False, **kwargs):
        # Add supplemental static mask.
        import os
        import fitsio
        from pkg_resources import resource_filename
        dq = super(BokImage, self).read_dq(slice=slice, header=header, **kwargs)
        if header:
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
        dq[(mask == 1) * (dq == 0)] = 1
        if header:
            return dq,hdr
        return dq

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

