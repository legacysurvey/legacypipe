from legacypipe.image import LegacySurveyImage

import logging
logger = logging.getLogger('legacypipe.suprime')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class SuprimeImage(LegacySurveyImage):

    def read_image_primary_header(self, **kwargs):
        # SuprimeCam images have an empty primary header, with a bunch of duplicated cards
        # in the image HDUs, so we'll hack that here!
        self._primary_header = self.read_image_fits()[1].read_header()
        return self._primary_header

    def get_band(self, primhdr):
        # FILTER01, but not in the primary header!
        # read from first extension!
        #self.hdu = 1
        #imghdr = self.read_image_header()
        band = primhdr['FILTER01']
        band = band.split()[0]
        return band

    def get_propid(self, primhdr):
        return primhdr.get('PROP-ID', '')

    def get_expnum(self, primhdr):
        s = primhdr['EXP-ID']
        s = s.replace('SUPE','')
        return int(s, 10)

    def get_mjd(self, primhdr):
        return primhdr.get('MJD-STR')

    def get_ccdname(self, primhdr, hdr):
        # DET-ID  =                    0 / ID of the detector used for this data
        return str(hdr['DET-ID'])

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        self.dqfn = self.imgfn.replace('p.fits.fz', 'p.weight.fits.fz')
        assert(self.dqfn != self.imgfn)
        self.wtfn = None
