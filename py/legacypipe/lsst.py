import os
import numpy as np
from legacypipe.hsc import HscImage
from legacypipe.image import LegacySurveyImage

class LsstImage(HscImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0,
                 camera_setup=False, **kwargs):
        print('LsstImage.__init__')
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)

        # if image_fn is not None:
        #     # Read metadata from image header.
        #     self.image_filename = image_fn
        # else:
        #     imgfn = ccd.image_filename.strip()
        #     self.image_filename = imgfn

        if camera_setup:
            return

        self.set_calib_filenames()
        # calibdir = self.survey.get_calib_dir()
        # imgdir = os.path.dirname(self.image_filename)
        # basename = self.get_base_name()
        # calname = self.name
        # #self.sefn         = os.path.join(calibdir, 'se',           imgdir, basename, calname + '-se.fits')
        # #self.psffn = os.path.join(calibdir, 'psfex-single', imgdir, basename, calname + '-psfex.fits')
        # self.sefn         = os.path.join(calibdir, 'se', imgdir, calname + '-se.fits')
        # self.psffn = os.path.join(calibdir, 'psfex-single', imgdir, calname + '-psfex.fits')
        # print('LSST: setting SE filename:', self.sefn)
        # print('LSST: setting PsfEx filename:', self.psffn)

    def set_calib_filenames(self):
        # Calib filenames
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        if len(self.ccdname):
            calname = basename + '-' + self.ccdname
        else:
            calname = basename
        self.name = calname
        #self.sefn         = os.path.join(calibdir, 'se',           imgdir, basename, calname + '-se.fits')
        #self.psffn        = os.path.join(calibdir, 'psfex-single', imgdir, basename, calname + '-psfex.fits')
        #self.skyfn        = os.path.join(calibdir, 'sky-single',   imgdir, basename, calname + '-splinesky.fits')
        self.sefn         = os.path.join(calibdir, 'se',           imgdir, calname + '-se.fits')
        self.psffn        = os.path.join(calibdir, 'psfex-single', imgdir, calname + '-psfex.fits')
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

    # def read_psf_model(self, x0, y0,
    #                    gaussPsf=False, pixPsf=False, hybridPsf=False,
    #                    normalizePsf=False, old_calibs_ok=False,
    #                    psf_sigma=1., w=0, h=0):
    #     # DP1 ComCam images: PiffPSF is stored as a pickle in a variable-length binary table.
    #     # Appears to be Piff version 1.5.0
    def read_psf_model(self, *args, **kwargs):
        return LegacySurveyImage.read_psf_model(self, *args, **kwargs)
