import numpy as np
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

    zp0 = {
        'I-A-L464': 25,
    }

    def get_ps1_band(self):
        from legacypipe.ps1cat import ps1cat
        # Returns the integer index of the band in Pan-STARRS1 to use for an image in filter
        # self.band.
        # eg, g=0, r=1, i=2, z=3, Y=4
        # FIXME
        return ps1cat.ps1band['g']
    
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

    def read_invvar(self, **kwargs):
        img = self.read_image(**kwargs)
        if self.sig1 is None or self.sig1 == 0.:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            self.sig1 = sig1
            print('Computed sig1 by Blanton method:', self.sig1)
        else:
            from tractor import NanoMaggies
            print('Suprime read_invvar: sig1 from CCDs file:', self.sig1)
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            print('scaled to image units:', sig1)
        iv = np.empty_like(img)
        iv[:,:] = 1./sig1**2
        return iv

    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1cat
        g_index = ps1cat.ps1band['g']
        r_index = ps1cat.ps1band['r']
        #i_index = ps1cat.ps1band['i']
        gmag = cat[:,g_index]
        rmag = cat[:,r_index]
        #imag = cat[:,i_index]
        colorterm = np.zeros(len(gmag))
        return colorterm
