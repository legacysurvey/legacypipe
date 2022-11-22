import os
from datetime import datetime
import numpy as np

from legacypipe.image import LegacySurveyImage

import logging
logger = logging.getLogger('legacypipe.wiro')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class WiroImage(LegacySurveyImage):

    zp0 = dict(
        g = 25.0,
        NB_A = 25.0,
        NB_B = 25.0,
        NB_C = 25.0,
        NB_D = 25.0,
        NB_E = 25.0,
        NB_F = 25.0,
        )

    k_ext = dict(
        g = 0.173,
        NB_A = 0.173,
        NB_B = 0.173,
        NB_C = 0.173,
        NB_D = 0.173,
        NB_E = 0.173,
        NB_F = 0.173,
    )

    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        self.dq_hdu = 1
        self.wt_hdu = 2

    def get_band(self, primhdr):
        f = primhdr['FILTER']
        filtmap = {
            'Filter 1: g 1736'  : 'g',
            'Filter 2: C 14859' : 'NB_C',
            'Filter 3: D 27981' : 'NB_D',
            'Filter 4: E 41102' : 'NB_E',
            'Filter 5: A 54195' : 'NB_A',
        }
        # ASSUME that the filter is one of the above!
        return filtmap[f]

    def get_radec_bore(self, primhdr):
        # Some TELDEC header cards (eg 20221030/a276) have a bug:
        # TELDEC  = '-4:-50:-23.-9'
        try:
            return super.get_radec_bore(primhdr)
        except:
            return None,None

    def get_expnum(self, primhdr):
        d = self.get_date(primhdr)
        expnum = d.second + 100*(d.minute + 100*(d.hour + 100*(d.day + 100*(d.month + 100*d.year))))
        return expnum

    def get_mjd(self, primhdr):
        from astrometry.util.starutil_numpy import datetomjd
        d = self.get_date(primhdr)
        return datetomjd(d)

    def get_date(self, primhdr):
        date = primhdr['DATE-OBS']
        # DATE-OBS= '2022-10-04T05:20:19.335'
        return datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        cam = {'wiroprime':'wiro'}.get(cam, cam)
        return cam

    def get_ccdname(self, primhdr, hdr):
        # return 'CCD'
        return ''

    def get_pixscale(self, primhdr, hdr):
        return 0.58

    def get_fwhm(self, primhdr, imghdr):
        # If PsfEx file exists, read FWHM from there
        if not hasattr(self, 'merged_psffn'):
            return super().get_fwhm(primhdr, imghdr)
        psf = self.read_psf_model(0, 0, pixPsf=True)
        fwhm = psf.fwhm
        return fwhm

    def get_gain(self, primhdr, hdr):
        # from https://iopscience.iop.org/article/10.1088/1538-3873/128/969/115003/ampdf
        return 2.6

    def get_object(self, primhdr):
        return primhdr.get('OBJNAME', '')

    def compute_filenames(self):
        # Masks and weight-maps are in HDUs following the image
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    def get_extension_list(self, debug=False):
        return [0]

    # def read_invvar(self, **kwargs):
    #     ie = super().read_invvar(**kwargs)
    #     return ie**2

    def read_invvar(self, **kwargs):
        # The reduced WIRO images have an Uncertainty HDU, but this only counts dark current
        # and readout noise only.
        img = self.read_image(**kwargs)
        if self.sig1 is None or self.sig1 == 0.:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            self.sig1 = sig1
            print('Computed sig1 by Blanton method:', self.sig1)
        # else:
        #     from tractor import NanoMaggies
        #     print('sig1 from CCDs file:', self.sig1)
        #     # sig1 in the CCDs file is in nanomaggy units --
        #     # but here we need to return in image units.
        #     zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
        #     sig1 = self.sig1 * zpscale
        #     print('scaled to image units:', sig1)
        iv = np.empty_like(img)
        iv[:,:] = 1./self.sig1**2
        return iv

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        # SATURATE header keyword is ~65536, but actual saturation in the images is
        # 32760.
        I,J = np.nonzero(img > 32700)
        from legacypipe.bits import DQ_BITS
        if len(I):
            dq[I,J] |= DQ_BITS['satur']
            invvar[I,J] = 0

    def get_wcs(self, hdr=None):
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        fn = os.path.join(calibdir, 'wcs', imgdir, self.name + '.wcs')
        print('WCS filename:', fn)
        from astrometry.util.util import Sip
        return Sip(fn)

    def get_crpixcrval(self, primhdr, hdr):
        wcs = self.get_wcs()
        p1,p2 = wcs.get_crpix()
        v1,v2 = wcs.get_crval()
        return p1,p2,v1,v2

    def get_cd_matrix(self, primhdr, hdr):
        wcs = self.get_wcs()
        return wcs.get_cd()

    def get_ps1_band(self):
        # Returns the integer index of the band in Pan-STARRS1 to use for an image in filter
        # self.band.
        # eg, g=0, r=1, i=2, z=3, Y=4
        # A known filter?
        from legacypipe.ps1cat import ps1cat
        if self.band in ps1cat.ps1band:
            return ps1cat.ps1band[self.band]
        # Narrow-band filters -- calibrate to PS1 g band.
        return dict(
            NB_A = 0,
            NB_B = 0,
            NB_C = 0,
            NB_D = 0,
            NB_E = 0,
            NB_F = 0,
            )[self.band]

    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1cat
        # See, eg, ps1cat.py's ps1_to_decam.
        # "cat" is a table of PS1 stars;
        # Grab the g-i color:
        g_index = ps1cat.ps1band['g']
        i_index = ps1cat.ps1band['i']
        gmag = cat[:,g_index]
        imag = cat[:,i_index]
        gi = gmag - imag

        coeffs = dict(
            g = [ 0. ],
            NB_A = [ 0. ],
            NB_B = [ 0. ],
            NB_C = [ 0. ],
            NB_D = [ 0. ],
            NB_E = [ 0. ],
            NB_F = [ 0. ],
            )[band]
        colorterm = np.zeros(len(gi))
        for power,coeff in enumerate(coeffs):
            colorterm += coeff * gi**power
        return colorterm
