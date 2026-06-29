import os
import numpy as np
from legacypipe.hsc import HscImage
from legacypipe.image import LegacySurveyImage

class LsstImage(HscImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0,
                 camera_setup=False, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        if camera_setup:
            return
        self.set_calib_filenames()
        # Try grabbing fwhm from PSFEx file, if it exists.
        if hasattr(self, 'fwhm') and not np.isfinite(self.fwhm):
            try:
                # PSF model file may not have been created yet...
                self.fwhm = self.get_fwhm(None, None)
            except:
                pass

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_lsst_comcam
        return ps1_to_lsst_comcam(ps1stars, band)

    def set_calib_filenames(self):
        # Calib filenames
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        self.name = basename
        self.sefn         = os.path.join(calibdir, 'se',           imgdir, basename + '-se.fits')
        self.psffn        = os.path.join(calibdir, 'psfex-single', imgdir, basename + '-psfex.fits')
        self.skyfn        = None
        self.merged_psffn = None
        self.merged_skyfn = None
        self.old_merged_skyfns = []
        self.old_merged_psffns = []
        # not used by this code -- here for the sake of legacyzpts/merge_calibs.py
        self.old_single_psffn = None
        self.old_single_skyfn = None

    def get_band(self, primhdr):
        band = primhdr['FILTBAND']
        band = band.split()[0]
        return band

    def get_propid(self, primhdr):
        return primhdr.get('PROGRAM', '')

    def get_ccdname(self, primhdr, hdr):
        return primhdr['DETNAME'].strip().upper()

    def get_fwhm(self, primhdr, imghdr):
        psf = None
        try:
            # PSF model file may not have been created yet...
            psf = self.read_psf_model(0., 0., pixPsf=True)
        except:
            #import traceback
            #traceback.print_exc()
            pass
        if psf is None:
            print("HACK - no FWHM readily available")
            return np.nan
        fwhm = psf.fwhm
        return fwhm

    def get_radec_bore(self, primhdr):
        # miracle of miracles, they just put in decimal degrees
        return primhdr['RA'], primhdr['DEC']

    # There is a CCDGAIN ("rough guess") = 1.0,
    # or HIERARCH LSST ISR GAIN C10 = 1.64786902096928
    # for C0..C15 (oh but with some missing..?)
    def get_gain(self, primhdr, hdr):
        vals = []
        for i in range(16):
            key = 'LSST ISR GAIN C%02i'%i
            if key in primhdr:
                vals.append(primhdr[key])
        if len(vals) == 0:
            return primhdr['CCDGAIN']
        return np.median(vals)

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        return LegacySurveyImage.estimate_sky(self, img, invvar, dq, primhdr, imghdr)

    # DP1 ComCam images: PiffPSF is stored as a pickle in a variable-length binary table.
    # Appears to be Piff version 1.5.0
    # but that doesn't build at NERSC, so just fall back to PsfEx.
    def read_psf_model(self, *args, **kwargs):
        return LegacySurveyImage.read_psf_model(self, *args, **kwargs)
