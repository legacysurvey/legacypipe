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

    k_ext = {
        'I-A-L464': 0.173,   # made up!
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

    def read_dq(self, header=None, **kwargs):
        dq = super().read_dq(header=header, **kwargs)
        if header:
            dq,hdr = dq
        # .weight.fits.fz files: 1 = good
        dq = 1 - dq
        if header:
            dq = dq,hdr
        return dq

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

    def get_extension_list(self, debug=False):
        if debug:
            return [1]
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def override_ccd_table_types(self):
        return {'camera':'S10',
                'filter': 'S8',}

    # Flip the weight map (1=good) to a flag map (1=bad)
    def run_se(self, imgfn, maskfn):
        import fitsio
        import os
        from collections import Counter
        from legacypipe.survey import create_temp
        tmpmaskfn  = create_temp(suffix='.fits')
        print('run_se: maskfn', maskfn)
        #mask,hdr = self.read_dq(maskfn, header=True)
        mask,hdr = self.read_dq(header=True)
        print('Mask values:', Counter(mask.ravel()))
        fitsio.write(tmpmaskfn, mask, header=hdr)
        R = super().run_se(imgfn, tmpmaskfn)
        os.unlink(tmpmaskfn)
        return R

    def get_fwhm(self, primhdr, imghdr):
        # If PsfEx file exists, read FWHM from there
        if not hasattr(self, 'merged_psffn'):
            return super().get_fwhm(primhdr, imghdr)
        psf = self.read_psf_model(0, 0, pixPsf=True)
        fwhm = psf.fwhm
        return fwhm

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.2

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)
