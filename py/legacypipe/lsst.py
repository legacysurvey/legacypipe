import numpy as np
from legacypipe.hsc import HscImage
from legacypipe.image import LegacySurveyImage

class LsstImage(HscImage):
    def get_band(self, primhdr):
        band = primhdr['FILTBAND']
        band = band.split()[0]
        return band
    def get_propid(self, primhdr):
        return primhdr.get('PROGRAM', '')
    def get_ccdname(self, primhdr, hdr):
        return primhdr['DETNAME'].strip().upper()
    def get_fwhm(self, primhdr, imghdr):
        print("HACK - no FWHM readily available")
        return 5.0
    def get_radec_bore(self, primhdr):
        # miracle of miracles, they just put in decimal degrees
        return primhdr['RA'], primhdr['DEC']
    # There is a CCDGAIN ("rough guess") = 1.0,
    # or HIERARCH LSST ISR GAIN C10 = 1.64786902096928
    # for C0..C15 (oh but with some missing..?)
    def get_gain(self, primhdr, hdr):
        #print('Primhdr:', primhdr)
        vals = []
        for i in range(16):
            key = 'LSST ISR GAIN C%02i'%i
            if key in primhdr:
                vals.append(primhdr[key])
        if len(vals) == 0:
            return primhdr['CCDGAIN']
        return np.median(vals)

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        #sup = super(LegacySurveyImage, self)
        #print('sup:', sup)
        #return sup.estimate_sky(img, invvar, dq, primhdr, imghdr)
        return LegacySurveyImage.estimate_sky(self, img, invvar, dq, primhdr, imghdr)

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       normalizePsf=False, old_calibs_ok=False,
                       psf_sigma=1., w=0, h=0):
        assert(gaussPsf or pixPsf or hybridPsf)
