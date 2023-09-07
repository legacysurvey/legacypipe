import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS
from legacypipe.survey import create_temp


'''
This class handles Hyper-SuprimeCam CALEXP calibrated image files
produced by the LSST software stack.

These are one file per CCD, with variance maps, flags, WCS, and PsfEx
models included in BINTABLE HDUs.

The sky background is also estimated and subtracted, so no external
calib files required.
'''
class HscImage(LegacySurveyImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        if ccd is not None:
            ccd.plver = 'xxx'
            ccd.procdate = 'xxx'
            ccd.plprocid = 'xxx'
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        self.dq_hdu = 2
        self.wt_hdu = 3

        # Nominal zeropoints
        # These are used only for "ccdskybr", so are not critical.
        # These are just from DECam!!
        self.zp0 = dict(
            g = 25.001,
            r = 25.209,
            r2 = 25.209,
            # i,Y from DESY1_Stripe82 95th percentiles
            i = 25.149,
            i2 = 25.149,
            z = 24.875,
            y = 23.712,
        )
        self.k_ext = dict(g = 0.17,
                          r = 0.10,
                          r2 = 0.10,
                          i = 0.08,
                          i2 = 0.08,
                          z = 0.06,
                          y = 0.058,
                          )

        # Sky has already been calibrated out, and Psf is included in the CALEXP file,
        # so no external calib files!
        self.sefn = None
        self.psffn = None
        self.skyfn = None
        self.merged_psffn = None
        self.merged_skyfn = None
        self.old_merged_skyfns = []
        self.old_merged_psffns = []
        self.old_single_psffn = None
        self.old_single_skyfn = None

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.168

    def get_extension_list(self, debug=False):
        return [1,]

    def has_astrometric_calibration(self, ccd):
        return True

    def compute_filenames(self):
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    def get_expnum(self, primhdr):
        return primhdr['EXPID']

    def get_ccdname(self, primhdr, hdr):
        return primhdr['DETSER'].strip().upper()

    def get_airmass(self, primhdr, imghdr, ra, dec):
        return primhdr['BORE-AIRMASS']

    def get_gain(self, primhdr, hdr):
        return np.mean([primhdr['T_GAIN%i' % i] for i in [1,2,3,4]])

    def get_fwhm(self, primhdr, imghdr):
        fwhm = primhdr['SEEING']
        if fwhm == 0.0:
            psf = self.read_psf_model(0., 0., pixPsf=True)
            fwhm = psf.fwhm
            return fwhm
        # convert from arcsec to pixels (hard-coded pixscale here)
        fwhm /= HscImage.get_nominal_pixscale()
        return fwhm

    def get_propid(self, primhdr):
        pid = primhdr.get('PROP-ID')
        if pid is not None:
            return pid
        # CORR files
        return primhdr['OB-ID']

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        if cam == 'hyper suprime-cam':
            cam = 'hsc'
        return cam

    def get_mjd(self, primhdr):
        '''HSC CALEXP images have a MJD-OBS header that is incorrectly set to
        the constant value 51543.99925713.  They also have MJD and
        MJD-END that are correct.  MJD is for the start of the exposure.

        MJD     =      56743.311664789 / [d] Mod. Julian Date at typical time
        MJD-END =     56743.3151609958 / [d] Mod.Julian Date at the end of exposure
        MJD-OBS =     51544.4992571308 / Modified Julian Date of observation

        On the other hand, HSC CORR images have MJD-STR for the start MJD:

        MJD-STR =       58850.23208995 / [d] Mod.Julian Date at the start of exposure
        MJD-END =        58850.2326893 / [d] Mod.Julian Date at the end of exposure
        MJD-OBS =     51544.4992571308 / Modified Julian Date of observation
        '''
        mjd = primhdr.get('MJD')
        if mjd is not None:
            return mjd
        return primhdr.get('MJD-STR')

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

    def read_sky_model(self, **kwargs):
        from tractor import ConstantSky
        sky = ConstantSky(0.)
        return sky

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       normalizePsf=False, old_calibs_ok=False,
                       psf_sigma=1., w=0, h=0):
        assert(gaussPsf or pixPsf or hybridPsf)
        if gaussPsf:
            from tractor import GaussianMixturePSF
            v = psf_sigma**2
            psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
            psf.version = '0'
            psf.plver = ''
            return psf

        # spatially varying pixelized PsfEx
        from tractor import PsfExModel
        from astrometry.util.fits import fits_table
        from tractor import PixelizedPsfEx
        from legacypipe.image import NormalizedPixelizedPsfEx

        fn = self.imgfn
        # PsfEx model information is spread across two BINTABLE hdus,
        # each with AR_NAME='PsfexPsf' and no other easily recognized
        # headers.
        F = self.read_image_fits()
        TT = []
        for i in range(1, len(F)):
            hdr = F[i].read_header()
            if hdr.get('AR_NAME') == 'PsfexPsf':
                T = fits_table(fn, hdu=i)
                assert(len(T) == 1)
                TT.append(T)
        assert(len(TT) == 2)
        T1,T2 = TT
        T1.rename('_pixstep', 'pixstep')
        T2.rename('_comp', 'comp')
        T2.rename('_size', 'size')
        T2.rename('_context_first', 'context_first')
        T2.rename('_context_second', 'context_second')

        t1 = T1[0]
        t2 = T2[0]

        psfex = PsfExModel()
        psfex.sampling = t1.pixstep
        degree = psfex.degree = t2.degree
        # PSF distortion bases are polynomials of x,y
        psfex.x0, psfex.y0 = t2.context_first
        psfex.xscale, psfex.yscale = t2.context_second
        # number of terms in polynomial
        ne = (degree + 1) * (degree + 2) / 2
        size = t2.size
        assert(size[2] == ne)

        ims = t2.comp.reshape(list(reversed(size)))
        ims = ims.astype(np.float32)
        assert(len(ims.shape) == 3)
        assert(ims.shape[0] == ne)

        # Extra polynomial coefficients that (apparently) aren't folded into the ims (?)
        #print('Coefficients:', t2.coeff)
        assert(len(t2.coeff) == ne)
        for i,c in enumerate(t2.coeff):
            ims[:,:,i] *= c

        psfex.psfbases = ims

        bh, bw = psfex.psfbases[0].shape
        psfex.radius = (bh + 1) / 2.

        # We don't have a FWHM measurement, so hack up a measurement on the first
        # PSF basis image.
        import photutils
        from scipy.interpolate import interp1d
        psf0 = psfex.psfbases[0,:,:]
        cx = bw//2
        cy = bh//2
        sb = []
        rads = np.arange(0, 20.1, 0.5)
        for rad1,rad2 in zip(rads, rads[1:]):
            aper = photutils.CircularAnnulus((cx, cy), max(1e-3, rad1), rad2)
            p = photutils.aperture_photometry(psf0, aper)
            f = p.field('aperture_sum')[0]
            f /= (np.pi * (rad2**2 - rad1**2))
            sb.append(f)
        f = interp1d(sb, rads[:-1])
        mx = psf0.max()
        hwhm = f(psf0.max() / 2.)
        fwhm = hwhm * 2. * psfex.sampling
        psfex.fwhm = fwhm
        print('Measured PsfEx FWHM', fwhm)

        if normalizePsf:
            psf = NormalizedPixelizedPsfEx(None, psfex=psfex)
        else:
            psf = PixelizedPsfEx(None, psfex=psfex)

        psf.version = ''
        psf.plver = ''
        psf.procdate = ''
        psf.plprocid = ''
        psf.datasum  = ''
        psf.fwhm = fwhm
        psf.header = None

        psf.shift(x0, y0)
        if hybridPsf:
            from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
            psf = HybridPixelizedPSF(psf, cx=w/2., cy=h/2.,
                                     gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
        return psf

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

    def get_zeropoint(self, primhdr, hdr):
        # CALEXP data products have FLUXMAG0.
        flux = primhdr.get('FLUXMAG0')
        if flux is not None:
            zpt = 2.5 * np.log10(flux / self.exptime)
            return zpt
        # CORR data products don't... depending on versions, there
        # will be an HDU with AR_NAME = 'PhotoCalib', scaling the image to nanoJanskies.
        F = self.read_image_fits()
        photocal = None
        for i in range(1, len(F)):
            from astrometry.util.fits import fits_table
            hdr = F[i].read_header()
            if hdr.get('AR_NAME') != 'PhotoCalib':
                continue
            #print('Found PhotoCalib AR_NAME')
            #dat = F[i].read(lower=True)
            #print('data:', dat)
            #print('dtype:', dat.dtype)
            photocal = fits_table(F[i].read(lower=True))
            #print('Found PhotoCalib table:', photocal)
            #photocal.about()
            break
        if photocal is not None:
            # flux [nJy] = "counts" * calibration factor.
            # mag 0 = 3.631 e12 nJy.
            scale = photocal.calibrationmean[0]
            flux = 3.631e12 / scale
            zpt = 2.5 * np.log10(flux / self.exptime)
            print('Returning zeropoint:', zpt)
            return zpt
        # Have to measure from photometering reference catalog!
        return None

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        return 0., primhdr['SKYLEVEL'], primhdr['SKYSIGMA']

    def read_invvar(self, dq=None, **kwargs):
        # HSC has a VARIANCE map (not a weight map)
        v = self._read_fits(self.wtfn, self.wt_hdu, **kwargs)
        iv = np.zeros(v.shape, np.float32)
        ok = np.isfinite(iv) * (v > 0)
        iv[ok] = 1./v[ok]
        #! this can happen
        iv[np.logical_not(np.isfinite(np.sqrt(iv)))] = 0.
        if dq is not None:
            iv[dq != 0] = 0.
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

    def check_image_header(self, imghdr):
        pass

    # def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
    #     ## Mask very negative pixels
    #     from tractor.brightness import NanoMaggies
    #     zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
    #     sig = zpscale * self.sig1
    #     print('Sig1 =', self.sig1, 'nanomaggies ->', sig, 'image counts')
    #     Ineg,Jneg = np.nonzero(img < -5. * sig)
    #     print('Found', len(Ineg), 'pixels < -5 sigma -- masking')
    #     img[Ineg,Jneg] = 0.
    #     dq[Ineg,Jneg] |= DQ_BITS['badpix']
    #     invvar[Ineg,Jneg] = 0.

def remap_hsc_bitmask(dq, header):
    new_dq = np.zeros(dq.shape, np.int16)
    #### The bits don't seem to be constant / the same between CORR and CALEXP.
    #### Use the header names!!
    masks = {}
    for k in header:
        if not k.startswith('MP_'):
            continue
        name = k[3:]
        masks[name] = 1 << int(header[k])
    def val(name):
        return masks.get(name, 0)
    # MP_BAD
    # MP_SAT
    # MP_INTRP
    # MP_CR
    #MP_EDGE
    # MP_DETECTED
    # MP_DETECTED_NEGATIVE
    # MP_SUSPECT
    # MP_NO_DATA
    # MP_CROSSTALK
    # MP_NOT_DEBLENDED
    # MP_UNMASKEDNAN
    bits = (val('BAD') |
            val('SUSPECT') |
            val('NO_DATA') |
            val('CROSSTALK') |
            val('UNMASKEDNAN'))
    new_dq |= DQ_BITS['badpix'] * ((dq & bits) != 0)
    bits = val('SAT')
    new_dq |= DQ_BITS['satur' ] * ((dq & bits) != 0)
    bits = val('INTRP')
    new_dq |= DQ_BITS['interp'] * ((dq & bits) != 0)
    bits = val('CR')
    new_dq |= DQ_BITS['cr'] * ((dq & bits) != 0)
    bits = val('EDGE')
    new_dq |= DQ_BITS['edge'] * ((dq & bits) != 0)
    return new_dq
