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
    def __init__(self, survey, t, image_fn=None, image_hdu=0):
        super(BokImage, self).__init__(survey, t, image_fn=image_fn, image_hdu=image_hdu)
        # Nominal zeropoints, sky brightness, and extinction values (taken from
        # rapala.ninetyprime.boketc.py)
        # /global/homes/a/arjundey/idl/pro/observing/bokstat.pro
        self.zp0 =  dict(g = 26.93,r = 27.01,z = 26.552) # ADU/sec
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.454

    def apply_amp_correction(self, img, invvar, x0, y0):
        self.apply_amp_correction_northern(img, invvar, x0, y0)

    def get_site(self):
        # FIXME -- this is for the Mayall, not the Bok!
        from astropy.coordinates import EarthLocation
        from astropy.units import m
        from astropy.utils import iers
        iers.conf.auto_download = False
        return EarthLocation(-1994503. * m, -5037539. * m, 3358105. * m)

    def get_fwhm(self, primhdr, imghdr):
        # exposure BOK_CP/CP20160405/ksb_160406_104543_ooi_r_v1.fits.f
        # has SEEINGP1 in the primary header, nothing anywhere else,
        # so FWHM in the CCDs file is NaN.
        import numpy as np
        fwhm = super().get_fwhm(primhdr, imghdr)
        if not np.isfinite(fwhm):
            fwhm = imghdr.get('SEEINGP1', 0.0)
        return fwhm

    def get_expnum(self, primhdr):
        """converts 90prime header key DTACQNAM into the unique exposure number"""
        # /descache/bass/20160710/d7580.0144.fits --> 75800144
        import re
        import os
        base= (os.path.basename(primhdr['DTACQNAM'])
               .replace('.fits','')
               .replace('.fz',''))
        return int( re.sub(r'([a-z]+|\.+)','',base) )

    def get_gain(self, primhdr, hdr):
        return 1.4

    def get_band(self, primhdr):
        band = primhdr['FILTER']
        band = band.split()[0]
        return band.replace('bokr', 'r')

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

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

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def remap_dq(self, dq, header):
        from legacypipe.image import remap_dq_cp_codes
        # code 8: see https://github.com/legacysurvey/legacypipe/issues/645
        dq = remap_dq_cp_codes(dq, ignore_codes=[7, 8])
        return dq

    # These are only used during zeropointing.
    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime
    def scale_weight(self, img):
        return img / (self.exptime**2)
