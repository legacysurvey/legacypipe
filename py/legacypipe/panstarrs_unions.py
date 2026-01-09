import os
import numpy as np
import fitsio
from astrometry.util.util import Tan
from legacypipe.image import LegacySurveyImage, info, debug
from legacypipe.bits import DQ_BITS
from legacypipe.survey import create_temp

class PanStarrsImage(LegacySurveyImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)

        # Nominal zeropoints
        # These are used only for "ccdskybr", so are not critical.
        self.zp0 = dict(i = 16.87)
        self.k_ext = dict(i = 0.08)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.186

    def get_base_name(self):
        # Returns the base name to use for this Image object.  This is
        # used for calib paths, and is joined with the CCD name to
        # form the name of this Image object and for calib filenames.
        basename = os.path.basename(self.image_filename)
        # eg "PSS.DR4.219.312.i.fits"
        basename = basename.replace('.fits', '')
        return basename

    def set_calib_filenames(self):
        super().set_calib_filenames()
        # One image per file -- no separate merged / single PsfEx files
        self.psffn = self.merged_psffn
        # Sky has already been calibrated out so no external calib
        # files for them!
        self.skyfn = None
        self.merged_skyfn = None
        self.old_merged_skyfns = []
        self.old_single_skyfn = None

    def get_expnum(self, primhdr):
        # These are coadds, but the expnum is widely used as an identifier, so fake one up using
        # the tile name!
        # eg "PSS.DR4.219.312.i"
        base = self.get_base_name()
        words = base.split('.')
        tile1 = words[-3]
        tile2 = words[-2]
        tile1 = int(tile1, 10)
        tile2 = int(tile2, 10)
        return tile1 * 1000 + tile2

    def get_exptime(self, primhdr):
        # The coadds are AVERAGES of the input images; exptime isn't well defined,
        # and they're used to scale CCDZPT values.
        # Let's fake it so that this scaling is a NO-OP
        return 1.

    def get_camera(self, primhdr):
        # nothing in the headers...
        return 'panstarrs'

    def get_ccdname(self, primhdr, hdr):
        return 'coadd'

    def get_radec_bore(self, primhdr):
        wcs = self.get_wcs(hdr=primhdr)
        #print('Got WCS:', wcs)
        r,d = wcs.radec_center()
        return r,d

    def get_wcs(self, hdr=None):
        if hdr is not None:
            tan = Tan(hdr)
        else:
            print('Reading TAN header from filename "%s", hdu "%s"' % (self.imgfn, self.hdu))
            tan = Tan(str(self.imgfn), int(self.hdu))
        return tan

    def get_airmass(self, primhdr, imghdr, ra, dec):
        return 1.2
        #return None

    def get_fwhm(self, primhdr, imghdr):
        # ???
        if hasattr(self, 'merged_psffn'):
            try:
                psf = self.read_psf_model(0, 0, pixPsf=True)
                print('get_fwhm: read PSF model', psf)
                return psf.fwhm
            except:
                print('Failed to read PSF model to get_fwhm:')
                import traceback
                traceback.print_exc()
                pass
        return np.nan

    def read_dq(self, header=True, **kwargs):
        img = self.read_image(header=header, **kwargs)
        if header:
            img,hdr = img
        dq = np.zeros(img.shape, np.int16)
        if header:
            return dq,hdr
        return dq

    def has_astrometric_calibration(self, ccd):
        return True

    def compute_filenames(self):
        self.dqfn = None
        self.wtfn = self.imgfn.replace('.fits', '.weight.fits')

    def read_sky_model(self, **kwargs):
        from tractor import ConstantSky
        sky = ConstantSky(0.)
        return sky

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        from legacypipe.image import estimate_sky_from_pixels
        skymed, skyrms = estimate_sky_from_pixels(img)
        return 0., skymed, skyrms

    def check_image_header(self, imghdr):
        pass
            
    def run_se(self, imgfn, maskfn):
        tmpmaskfn = None
        if maskfn is None:
            # Create an all-zeros fake flags.fits file.
            phdr = self.read_image_primary_header()
            # Are these the right way around?
            H = phdr['NAXIS1']
            W = phdr['NAXIS2']
            tmpmaskfn = create_temp(suffix='.fits')
            debug('Writing fake mask file', tmpmaskfn)
            fitsio.write(tmpmaskfn, np.zeros((H,W), np.uint8), clobber=True)
            maskfn = tmpmaskfn
        R = super().run_se(imgfn, maskfn)
        if tmpmaskfn is not None:
            os.remove(tmpmaskfn)
        return R

    def gaia_to_observed(self, gaia, band):
        from functools import reduce
        G  = gaia.phot_g_mean_mag .astype(np.float32)
        BP = gaia.phot_bp_mean_mag.astype(np.float32)
        RP = gaia.phot_rp_mean_mag.astype(np.float32)
        color = BP - RP

        # From Rongpu, 2020-04-12
        # no BP-RP color: use average color
        average_color=1.4
        badcolor = reduce(np.logical_or,
                          [np.logical_not(np.isfinite(color)),
                           BP == 0, RP == 0,])
        color[badcolor] = average_color
        # clip colors
        lo,hi = (0.5, 3.0)
        color = np.clip(color, lo, hi)

        coeffs = {
            # Untitled458.ipynb
            'i': ('RP', [ 0.00900635, -0.04966931,  0.03259645]),
        }

        base_mag, coeff = coeffs[band]
        base_mags = dict(G=G, BP=BP, RP=RP)
        mag = base_mags[base_mag].copy()
        nomags = (mag == 0.)
        for order,c in enumerate(coeff):
            mag += c * color**order
        missing_mag = 30.
        mag[nomags] = missing_mag
        return mag

    #def get_photometric_calibrator_cuts(self, name, cat):
    #            color = cat.phot_bp_mean_mag - cat.phot_rp_mean_mag
    # def get_photocal_mag_limits(self):
    #     MAGLIM=dict(
    #         i=[16, 19.5],
    #     )
    #     return MAGLIM.get(self.band, (16.,20.))
