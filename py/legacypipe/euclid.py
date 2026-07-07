import os
from glob import glob
import numpy as np
import fitsio

from tractor import PixelizedPsfEx, PixelizedPSF, NormalizedPsf, NormalizedPixelizedPsf
from tractor.ducks import Sky

from legacypipe.image import LegacySurveyImage, NormalizedPixelizedPsfEx

import logging
logger = logging.getLogger('legacypipe.euclid')
def error(*args):
    from legacypipe.utils import log_error
    log_error(logger, args)
    import traceback
    traceback.print_exc()
def warning(*args):
    from legacypipe.utils import log_warning
    log_warning(logger, args)
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class EuclidImage(LegacySurveyImage):
    def get_expnum(self, primhdr):
        # ??
        #return primhdr['PTGID']

        # So bizarre, this seems like a SUPER basic thing, and yet
        obs = primhdr['OBS_ID']
        dith = primhdr['DITHOBS']
        exp = primhdr['EXPNUM']
        filt = primhdr.get('FILTER', 'vis')
        fnum = dict(Y=1, J=2, H=3, vis=0)[filt]
        assert(fnum < 10)
        assert(dith < 1000)
        return obs*100000 + dith*100 + exp*10 + fnum

    # plver...
    #ORIGIN  = 'OU-NIR  '           / Euclid SGS origin
    #SOFTVERS= '4.2     '

    def get_radec_bore(self, primhdr):
        # Good ol' decimal degrees
        return primhdr['RA'], primhdr['DEC']

    def get_airmass(self, primhdr, imghdr, ra, dec):
        return 0.0

    def get_gain(self, primhdr, hdr):
        return hdr['GAIN']

    def validate_version(self, typ, fn, *args, **kwargs):
        # Skip sky model -- super expects a FITS table
        #if '_W-CAL-IMAGE-BKG_' in fn:
        #    return True
        return os.path.exists(fn)

    def get_extension_list(self, debug=False):
        F = self.read_image_fits()
        exts = []
        for hdu in range(1, len(F)):
            f = F[hdu]
            extname = f.get_extname()
            #print('EXTNAME', extname)
            extname = extname.strip()
            if extname.endswith('.SCI'):
                exts.append(extname)
        return exts

    def read_invvar(self, **kwargs):
        rms = super().read_invvar(**kwargs)
        # RMS to IV
        with np.errstate(divide='ignore'):
            iv = 1./(rms**2)
        iv[~np.isfinite(rms)] = 0.
        iv[rms == 0.] = 0.
        print('NISP read_invvar: min %g, max %g, # finite: %i, # inf: %i, # zero: %i' %
              (iv.min(), iv.max(), np.sum(np.isfinite(iv)), np.sum(np.logical_not(np.isfinite(iv))),
               np.sum(iv == 0)))
        return iv

    def read_sky_model(self, slc=None, old_calibs_ok=False,
                       template_meta=None, **kwargs):
        print('Reading sky model:', self.merged_skyfn, self.skyextname)
        if slc is not None:
            img = fitsio.FITS(self.merged_skyfn)[self.skyextname][slc]
        else:
            img = fitsio.read(self.merged_skyfn, ext=self.skyextname)
        print('Sky: median', np.median(img))
        return PixelizedSky(img)

class PixelizedSky(Sky):
    def __init__(self, img):
        self.img = img
    def addTo(self, mod, scale=1.):
        assert(self.img.shape == mod.shape)
        mod += scale * self.img

class VisImage(EuclidImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0,
                 camera_setup=False, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        # Nominal zeropoints
        self.zp0 = dict(
            V = 24.566,
        )
        if camera_setup:
            return
        self.set_calib_filenames()
        # Try grabbing fwhm from PSF file, if it exists.
        # if hasattr(self, 'fwhm') and not np.isfinite(self.fwhm):
        #     print('grab FWHM from PSF model...')
        #     try:
        #         # PSF model file may not have been created yet...
        #         self.fwhm = self.get_fwhm(None, None)
        #     except:
        #         pass
        self.skyextname = self.ccdname.replace('.SCI','')
        print('CCDname', self.ccdname, 'sky ext', self.skyextname)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.3

    def get_fwhm(self, primhdr, imghdr):
        # shrug
        return 2.5

    # absolutely wild
    def get_band(self, primhdr):
        # they don't have anything in the header!!
        return 'V'

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn
        #self.dq_hdu =
        print('Setting DQ HDU: ccdname is', self.ccdname)
        self.dq_hdu = self.ccdname.replace('.SCI', '.FLG')
        self.wt_hdu = self.ccdname.replace('.SCI', '.RMS')

    def get_zeropoint(self, primhdr, hdr):
        s = hdr['MAGZEROP']
        # wtf
        # MAGZEROP= '24.56605'           / zero-point
        return float(s)

    def set_calib_filenames(self):
        '''
        EUC_VIS_GRD-PSF-000-000000-0000000__20240322T192915.424564Z.fits
        EUC_VIS_SWL-BKG-002681-00-1-0000000__20241017T042839.727792Z.fits
        EUC_VIS_SWL-DET-002681-00-1-0000000__20241017T042839.727728Z.fits
        EUC_VIS_SWL-WGT-002681-00-1-0000000__20241017T042839.727798Z.fits
        '''
        self.psffn = None
        fn = self.imgfn
        dirnm = os.path.dirname(fn)
        base = os.path.basename(fn)

        parts = base.split('_')
        prod = parts[2].split('-')
        assert(prod[0] == 'SWL')
        assert(prod[1] == 'DET')
        expid = '-'.join(prod[2:])

        self.name = '_'.join([parts[0], parts[1], expid])

        # PSF
        pat = os.path.join(dirnm, '_'.join(parts[:2] + ['GRD-PSF-000-000000-0000000', '*.fits']))
        fns = glob(pat)
        if len(fns) == 1:
            self.merged_psffn = fns[0]
            print('PSF filename:', self.merged_psffn)
        else:
            print('PSF model not found:', pat)

        # Background/sky
        pat = os.path.join(dirnm, '_'.join(parts[:2] + ['SWL-BKG-' + expid, '*.fits']))
        fns = glob(pat)
        if len(fns) == 1:
            self.merged_skyfn = fns[0]
            print('Background/sky model:', self.merged_skyfn)
        else:
            print('Sky/background model not found:', pat)

    def run_calibs(self, **kwargs):
        pass

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       normalizePsf=False, old_calibs_ok=False,
                       psf_sigma=1., w=0, h=0):
        print('Reading PSF model:', self.merged_psffn, 'CCDNAME', self.ccdname)
        ext = self.ccdname.replace('.SCI','')
        img = fitsio.read(self.merged_psffn, ext)

        '''
        FIXME -- this is not at all correct -- the PSF model images are 189x189 but it's actually something like a 9x9 grid of 21x21-pixel images???
        '''

        print('Got PSF image', img.shape)
        if normalizePsf:
            debug('Normalizing PSF')
            psf = NormalizedPixelizedPsf(img)
        else:
            psf = PixelizedPsf(img)
        psf.fwhm = self.fwhm
        if hybridPsf:
            from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
            psf = HybridPixelizedPSF(psf, cx=w/2., cy=h/2.,
                                     gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
        debug('Using PSF model', psf)
        return psf

    def remap_dq(self, dq, header, slc):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        ignore_mask = 0x1040000
        print('Ignoring DQ mask: 0x%x' % ignore_mask)
        ignore_mask = np.uint32(ignore_mask)
        dq = dq & ~ignore_mask
        return dq

# NISP - NIR images
# DpdNirCalibratedFrame data product
# https://euclid.esac.esa.int/dr/q1/dpdd/nirdpd/dpcards/nir_calibratedframe.html
# 

class NispImage(EuclidImage):

    def __init__(self, survey, ccd, image_fn=None, image_hdu=0,
                 camera_setup=False, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        # Nominal zeropoints
        self.zp0 = dict(
            Y = 29.77,
            J = 30.05,
            H = 29.95,
        )
        if camera_setup:
            return
        self.set_calib_filenames()
        # Try grabbing fwhm from PSF file, if it exists.
        if hasattr(self, 'fwhm') and not np.isfinite(self.fwhm):
            print('grab FWHM from PSF model...')
            try:
                # PSF model file may not have been created yet...
                self.fwhm = self.get_fwhm(None, None)
            except:
                pass
        self.skyextname = self.ccdname

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.3

    def get_fwhm(self, primhdr, imghdr):
        if hasattr(self, 'merged_psffn'):
            psf = self.read_psf_model(0., 0., pixPsf=True)
            fwhm = psf.fwhm
            return fwhm
        return np.nan

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn
        #self.dq_hdu =
        print('Setting DQ HDU: ccdname is', self.ccdname)
        self.dq_hdu = self.ccdname.replace('.SCI', '.DQ')
        self.wt_hdu = self.ccdname.replace('.SCI', '.RMS')

    def get_zeropoint(self, primhdr, hdr):
        return hdr['ZPAB']

    def set_calib_filenames(self):
        self.psffn = None

        #fn = self.image_filename
        fn = self.imgfn
        # NIR/2681/EUC_NIR_W-CAL-IMAGE-BKG_Y-2681-1_20240930T183543.377193Z.fits
        # NIR/2681/EUC_NIR_W-CAL-IMAGE_Y-2681-1_20240930T183522.647051Z.fits
        # NIR/2681/EUC_NIR_W-CAL-PSF-I_Y-2681-1_20240930T183602.216265Z.fits
        # NIR/2681/EUC_NIR_W-CAL-PSF-M_Y-2681-1_20240930T183602.217973Z.psf

        dirnm = os.path.dirname(fn)
        base = os.path.basename(fn)

        parts = base.split('_')
        assert(parts[2] == 'W-CAL-IMAGE')

        self.name = '_'.join([parts[0], parts[1], parts[3]])
        
        pat = os.path.join(dirnm, '_'.join(parts[:2] + ['W-CAL-PSF-M'] + [parts[3]] + ['*.psf']))
        fns = glob(pat)
        if len(fns) == 1:
            self.merged_psffn = fns[0]
        else:
            print('PSF model not found:', pat)

        pat = os.path.join(dirnm, '_'.join(parts[:2] + ['W-CAL-IMAGE-BKG'] + [parts[3]] + ['*.fits']))
        fns = glob(pat)
        if len(fns) == 1:
            self.merged_skyfn = fns[0]
        else:
            print('Sky/background model not found:', pat)

    def run_calibs(self, **kwargs):
        pass

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       normalizePsf=False, old_calibs_ok=False,
                       psf_sigma=1., w=0, h=0):
        # ccdname = 'det11.sci'
        # DET_ID  = '11      '
        print('Reading PSF model:', self.merged_psffn)
        F = fitsio.FITS(self.merged_psffn)
        T = None
        for hdu in range(1, len(F)):
            hdr = F[hdu].read_header()
            det = hdr['DET_ID']
            det = det.strip()
            ccdname = 'det'+det+'.sci'
            #print('PSF: ccd name "%s" vs "%s"' % (ccdname, self.ccdname))
            if ccdname == self.ccdname.lower():
                # found it!
                from astrometry.util.fits import fits_table
                T = fits_table(self.merged_psffn, hdu=hdu)
                #T.about()
                break
        assert(T is not None)
        from tractor import PsfExModel
        assert(len(T) == 1)
        # What's this???
        #  psf_coeffs (<class 'numpy.ndarray'>) shape (1, 1, 21) dtype >f4
        #  psf_mask (<class 'numpy.ndarray'>) shape (1, 1, 61, 61) dtype >f4
        row = T[0]
        #print('PSF coeffs:', row.psf_coeffs)
        n_comp,h,w = row.psf_mask.shape
        assert(n_comp == 1)
        # If degree 0, set polname* to avoid assertion error in tractor
        if True:
            row.polname1 = 'X_IMAGE'
            row.polname2 = 'Y_IMAGE'
            row.polgrp1 = 1
            row.polgrp2 = 1
            row.polngrp = 1
            row.poldeg1 = 0
            row.polzero1 = 0
            row.polzero2 = 0
            row.polscal1 = 1
            row.polscal2 = 1
            row.psfaxis3 = 1
        row.psf_samp = hdr['PSF_SAMP']
        row.psf_fwhm = hdr['PSF_FWHM']
        psfex = PsfExModel(Ti=row)
        if normalizePsf:
            debug('Normalizing PSF')
            psf = NormalizedPixelizedPsfEx(None, psfex=psfex)
        else:
            psf = PixelizedPsfEx(None, psfex=psfex)
        psf.version = ''
        psf.plver = ''
        psf.procdate = ''
        psf.plprocid = ''
        psf.datasum  = ''
        psf.fwhm = row.psf_fwhm
        psf.header = hdr

        psf.shift(x0, y0)
        if hybridPsf:
            from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
            psf = HybridPixelizedPSF(psf, cx=w/2., cy=h/2.,
                                     gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
        debug('Using PSF model', psf)
        return psf

    def remap_dq(self, dq, header, slc):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        # example file EUC_NIR_W-CAL-IMAGE_Y-2681-3_20240930T183522.757602Z.fits
        # has the majority of pixels with 0x800 or 0x8800.  Ignore these.
        ignore_names = ['NLINEAR', 'DARKNODET']
        ignore_mask = 0
        for name in ignore_names:
            key = 'MSK_FLAG_' + name
            bit = header[key]
            ignore_mask |= (1 << bit)
        print('Ignoring DQ mask: 0x%x' % ignore_mask)
        #print('DQ type:', dq.dtype)
        ignore_mask = np.uint32(ignore_mask)
        dq = dq & ~ignore_mask
        return dq
    # DQ bits:
    # https://euclid.esac.esa.int/msp/dpdd/live/nirdpd/dpcards/nir_calibratedframe.html#data-quality-layer
    #    MSK_FLAG_INVALID =  0
    #    MSK_FLAG_OBMASK =   1
    #    MSK_FLAG_DISCONNECTED = 2
    #    MSK_FLAG_ZEROQE =   3
    #    
    #    MSK_FLAG_BADBASE =  4
    #    MSK_FLAG_LOWQE =    5
    #    MSK_FLAG_SUPERQE =  6
    #    MSK_FLAG_HOT =      7
    #    
    #    MSK_FLAG_RTN =      8
    #    MSK_FLAG_SNOWBALL = 9
    #    MSK_FLAG_SATUR =   10
    #    MSK_FLAG_NLINEAR = 11
    #    
    #    MSK_FLAG_NLMODFAIL = 12
    #    MSK_FLAG_PERSIST = 13
    #    MSK_FLAG_PERMODFAIL = 14
    #    MSK_FLAG_DARKNODET = 15
    #    
    #    MSK_FLAG_COSMIC =  16
    #    MSK_FLAG_FLATLH =  17
    #    MSK_FLAG_GHOST =   18
    #    MSK_FLAG_SCATTER = 19
    #    
    #    MSK_FLAG_MOVING =  20
    #    MSK_FLAG_TRANS =   21
    #    MSK_FLAG_CROSSTALK = 22
    #    MSK_FLAG_FLOWER =  23
    #    
    #    MSK_FLAG_VIGNET =  24
