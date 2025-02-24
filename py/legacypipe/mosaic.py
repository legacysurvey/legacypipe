from legacypipe.cpimage import CPImage

class MosaicImage(CPImage):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOIRLab Community Pipeline.
    '''
    def __init__(self, survey, t, image_fn=None, image_hdu=0, **kwargs):
        super(MosaicImage, self).__init__(survey, t, image_fn=image_fn, image_hdu=image_hdu,
                                          **kwargs)

        self.zp0 = dict(z = 26.552,
                        D51 = 24.351, # from obsbot
        )
        self.k_ext = dict(z = 0.06,
                          D51 = 0.211, # from obsbot
        )

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.262

    def colorterm_sdss_to_observed(self, sdssstars, band):
        from legacypipe.ps1cat import sdss_to_decam
        print('Warning: using DECam color term for SDSS to Mosaic transformation')
        return sdss_to_decam(sdssstars, band)
    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacypipe.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

    def apply_amp_correction(self, img, invvar, x0, y0):
        self.apply_amp_correction_northern(img, invvar, x0, y0)

    # always recompute airmass with boresight RA,Dec
    def get_airmass(self, primhdr, imghdr, ra, dec):
        return self.recompute_airmass(primhdr, ra, dec)

    def get_site(self):
        from astropy.coordinates import EarthLocation
        from astropy.units import m
        from astropy.utils import iers
        iers.conf.auto_download = False
        return EarthLocation(-1994503. * m, -5037539. * m, 3358105. * m)

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        if cam == 'mosaic3':
            cam = 'mosaic'
        return cam

    def get_fwhm(self, primhdr, imghdr):
        # exposure 88865 has SEEINGP1 in the primary header, nothing anywhere else,
        # so FWHM in the CCDs file is NaN.
        import numpy as np
        fwhm = super().get_fwhm(primhdr, imghdr)
        if not np.isfinite(fwhm):
            fwhm = imghdr.get('SEEINGP1', 0.0)
        return fwhm

    def get_expnum(self, primhdr):
        if 'EXPNUM' in primhdr and primhdr['EXPNUM'] is not None:
            return primhdr['EXPNUM']
        # At the beginning of the survey, eg 2016-01-24, the EXPNUM
        # cards are blank.  Fake up an expnum like 160125082555
        # (yymmddhhmmss), same as the CP filename.
        # OBSID   = 'kp4m.20160125T082555' / Observation ID
        obsid = primhdr['OBSID']
        obsid = obsid.strip().split('.')[1]
        obsid = obsid.replace('T', '')
        obsid = int(obsid[2:], 10)
        print('Faked up EXPNUM', obsid)
        return obsid

    def get_band(self, primhdr):
        band = primhdr['FILTER']
        band = band.split()[0]
        band = {'zd':'z'}.get(band, band) # zd --> z
        return band

    def get_gain(self, primhdr, hdr):
        return hdr['GAIN']

    # Used during zeropointing only.
    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime
    def scale_weight(self, img):
        return img / (self.exptime**2)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        from legacypipe.cpimage import remap_dq_cp_codes
        # code 8: https://github.com/legacysurvey/legacypipe/issues/644
        dq = remap_dq_cp_codes(dq, ignore_codes=[7, 8])
        return dq
