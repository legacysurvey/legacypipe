from __future__ import print_function
import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS
from legacypipe.survey import create_temp

class HscImage(LegacySurveyImage):
    def __init__(self, survey, ccd):
        ccd.plver = 'xxx'
        ccd.procdate = 'xxx'
        ccd.plprocid = 'xxx'
        super().__init__(survey, ccd)
        self.dq_hdu = 2
        self.wt_hdu = 3
        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)

        # FIXME -- these are just from DECam
        self.zp0 = dict(
            g = 26.610,
            r = 26.818,
            i = 26.758,
            z = 26.484,
        )
        self.k_ext = dict(g = 0.17,
                          r = 0.10,
                          i = 0.08,
                          z = 0.06,
                          )

    def get_extension_list(self, fn, debug=False):
        return [self.image_hdu,]

    def calibration_good(self, primhdr):
        return True

    '''
    def get_psfex_unmerged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-psfex.fits'
        fn = os.path.join(self.calibdir, 'psfex-single', basedir, base, fn)
        return fn
    def get_splinesky_unmerged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-splinesky.fits'
        fn = os.path.join(self.calibdir, 'sky-single', basedir, base, fn)
        return fn
    '''
    def compute_filenames(self):
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    def get_expnum(self, primhdr):
        return primhdr['EXPID']

    def get_fwhm(self, hdr, hdu):
        return self.primhdr['SEEING']

    def get_wcs(self, hdr=None):
        from astrometry.util.util import Sip
        if hdr is None:
            hdr = self.read_image_header()
        wcs = Sip(hdr)
        # Correction: ccd ra,dec offsets from zeropoints/CCDs file
        dra,ddec = self.dradec
        # debug('Applying astrometric zeropoint:', (dra,ddec))
        r,d = wcs.get_crval()
        wcs.set_crval((r + dra / np.cos(np.deg2rad(d)), d + ddec))
        wcs.version = ''
        wcs.plver = ''
        #phdr = self.read_image_primary_header()
        #wcs.plver = phdr.get('PLVER', '').strip()
        return wcs

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_hsc
        return ps1_to_hsc(ps1stars, band)

    def read_image(self, header=False, **kwargs):
        img = super().read_image(header=header, **kwargs)
        if header:
            img,hdr = img
        img[np.logical_not(np.isfinite(img))] = 0.
        if header:
            img = img,hdr
        return img
    
    def remap_dq(self, dq, header):
        return remap_hsc_bitmask(dq, header)

    def read_invvar(self, dq=None, **kwargs):
        # HSC has a VARIANCE map (not a weight map)
        v = self._read_fits(self.wtfn, self.wt_hdu, **kwargs)
        iv = 1./v
        iv[v==0] = 0.
        iv[np.logical_not(np.isfinite(iv))] = 0.
        #! this can happen
        iv[np.logical_not(np.isfinite(np.sqrt(iv)))] = 0.
        return iv

    def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
        # Before passing files to SourceExtractor / PsfEx, filter our mask image
        # because it marks DETECTED pixels with a mask bit.
        tmpimgfn,tmpmaskfn = super().funpack_files(imgfn, maskfn, imghdu, maskhdu, todelete)
        #print('Dropping mask bit 5 before running SE')
        m = fitsio.read(tmpmaskfn)
        m &= ~(1 << 5)
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpmaskfn)
        fitsio.write(tmpmaskfn, m, clobber=True)
        return tmpimgfn, tmpmaskfn

    def validate_version(self, *args, **kwargs):
        return True
    def check_image_header(self, imghdr):
        pass

def remap_hsc_bitmask(dq, header):
    new_dq = np.zeros(dq.shape, np.int16)
    # MP_BAD  =                    0
    # MP_SUSPECT =        7
    # MP_NO_DATA =        8
    # MP_CROSSTALK =      9
    # MP_UNMASKEDNAN =   11
    new_dq |= (DQ_BITS['badpix'] * ((dq & ((1<<0) | (1<<7) | (1<<8) | (1<<9) | (1<<11))) != 0))
    # MP_SAT  =                    1
    new_dq |= (DQ_BITS['satur' ] * ((dq & (1<<1)) != 0))
    # MP_INTRP=                    2
    new_dq |= (DQ_BITS['interp'] * ((dq & (1<<2)) != 0))
    # MP_CR   =                    3
    new_dq |= (DQ_BITS['cr'] * ((dq & (1<<3)) != 0))
    #MP_EDGE =                    4
    new_dq |= (DQ_BITS['edge'] * ((dq & (1<<4)) != 0))
    '''
    MP_DETECTED =       5
    MP_DETECTED_NEGATIVE = 6
    MP_NOT_DEBLENDED = 10
    '''
    return new_dq
