import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS
from legacypipe.survey import create_temp

'''
This class handles Pan-STARRS STACKed image products (DR1/DR2).

https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images

These are sky-subtracted, warped to a common astrometric frame and
scaled to a common photometric zeropoint, and coadded.

Zeropoints is 25+2.5*log10(EXPTIME)
(header FPA.ZP = 25.0)

The image pixel values have an asinh scaling on them....

BZERO   =   3.632529139519E+00 / Scaling: TRUE = BZERO + BSCALE * DISK
BSCALE  =   2.115535619667E-04 / Scaling: TRUE = BZERO + BSCALE * DISK
BSOFTEN =   2.218863810555E+02 / Scaling: LINEAR = 2 * BSOFTEN * sinh(TRUE/a)
BOFFSET =   7.172080516815E+00 / Scaling: UNCOMP = BOFFSET + LINEAR

.wt maps are variances.

WCS header are weird,
CTYPE1  = 'RA---TAN'
CTYPE2  = 'DEC--TAN'
CRVAL1  =     351.887878417988
CRVAL2  =     34.4999999999988
CRPIX1  =                5000.
CRPIX2  =                5000.
CDELT1  =  5.1602346501999E-05
CDELT2  =  5.1602346501999E-05
PC001001=                  -1.
PC001002=                   0.
PC002001=                   0.
PC002002=                   1.

id:
STK_ID  = '5413501 '           / type of stack
SKYCELL = 'skycell.580.249'    / type of stack
TESS_ID = 'CFIS.V0 '           / type of stack

PSCAMERA= 'GPC1    '           / Camera name
PSFORMAT= 'SKYCELL '           / Camera format
IMAGEID =              5413501 / Image identifier

HIERARCH FPA.TELESCOPE = 'PS1     ' / Telescope of origin
HIERARCH FPA.INSTRUMENT = 'GPC1    ' / Instrument name (according to the instrum
HIERARCH FPA.FILTER = 'i.00000 ' / Filter used (instrument name)
MJD-OBS =     57515.0391846985 / Time of exposure

The "STARCORE" bit masks quite aggressively.

'''

class PanStarrsImage(LegacySurveyImage):
    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        if ccd is not None:
            ccd.plver = 'xxx'
            ccd.procdate = 'xxx'
            ccd.plprocid = 'xxx'
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)

        # Nominal zeropoints
        # These are used only for "ccdskybr", so are not critical.
        self.zp0 = dict(
            g = 25.0,
            r = 25.0,
            i = 25.0,
            z = 25.0,)
        self.k_ext = dict(g = 0.17,
                          r = 0.10,
                          i = 0.08,
                          z = 0.06,)

        # Sky has already been calibrated out so no external calib
        # files for them!
        self.skyfn = None
        self.merged_skyfn = None
        self.old_merged_skyfns = []
        self.old_single_skyfn = None

        # One image per file -- no separate merged / single PsfEx files
        self.psffn = self.merged_psffn
        
    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.186

    def get_base_name(self):
        import os
        basename = os.path.basename(self.image_filename)
        basename = basename.replace('.fits','')
        return basename

    # def get_extension_list(self, debug=False):
    #     #return [1,]
    #     return [0,]

    def has_astrometric_calibration(self, ccd):
        return True

    def compute_filenames(self):
        self.dqfn = self.imgfn.replace('.fits', '.mask.fits')
        self.wtfn = self.imgfn.replace('.fits', '.wt.fits')

    def get_cd_matrix(self, primhdr, hdr):
        #### Really we should read PC001001, PC001002, PC002001, PC002002...
        return -hdr['CDELT1'], 0., 0., hdr['CDELT2']

    def get_radec_bore(self, primhdr):
        hdr = self.read_image_header()
        return hdr['CRVAL1'], hdr['CRVAL2']

    def get_band(self, primhdr):
        key = 'FPA.FILTER'
        if key in primhdr:
            band = primhdr[key]
            band = band.split('.')[0]
            return band
        #self.hdu = 1        
        hdr = self.read_image_header()
        #print('Primary header:')
        #print(primhdr)
        band = hdr['FPA.FILTER']
        band = band.split('.')[0]
        return band

    def get_expnum(self, primhdr):
        key = 'IMAGEID'
        if key in primhdr:
            return primhdr[key]
        hdr = self.read_image_header()
        if key in hdr:
            return hdr[key]
        # WTF some images don't have this??!  eg rings.v3.skycell.0960.017.stk.r.unconv.fits lacks it
        # Fake it from:
        # TESS_ID = 'RINGS.V3'           / type of stack
        # SKYCELL = 'skycell.2594.011'   / type of stack
        tessid = None
        skycell = None
        for h in [primhdr, hdr]:
            if 'TESS_ID' in h:
                tessid = h['TESS_ID']
            if 'SKYCELL' in h:
                skycell = h['SKYCELL']
        if tessid is None or skycell is None:
            raise RuntimeError('No IMAGEID or TESS_ID & SKYCELL in header for Pan-STARRS image %s' % self.image_filename)
        words = tessid.strip().split('.')
        if len(words) != 2 or words[0] != 'RINGS' or words[1][0] != 'V':
            raise RuntimeError('Failed to parse TESS_ID for Pan-STARRS image %s' % self.image_filename)
        tid = int(words[1][1:])
        words = skycell.strip().split('.')
        if len(words) != 3 or words[0] != 'skycell':
            raise RuntimeError('Failed to parse SKYCELL for Pan-STARRS image %s' % self.image_filename)
        sk1 = int(words[1], 10)
        sk2 = int(words[2], 10)
        fakeid = tid * 10_000_000 + sk1 * 1_000 + sk2
        return fakeid

    def get_exptime(self, primhdr):
        key = 'EXPTIME'
        if key in primhdr:
            return primhdr[key]
        hdr = self.read_image_header()
        return hdr.get(key)

    def get_pixscale(self, primhdr, hdr):
        return 3600. * np.sqrt(np.abs(hdr['CDELT1'] * hdr['CDELT2']))

    def get_ccdname(self, primhdr, hdr):
        return ''

    def get_gain(self, primhdr, hdr):
        hdr = self.read_image_header()
        return hdr['CELL.GAIN']

    def get_fwhm(self, primhdr, imghdr):
        hdr = self.read_image_header()
        fwhm = hdr['CHIP.SEEING']
        fwhm = float(fwhm)
        if not np.isfinite(fwhm):
            # There's a get_fwhm() call early in the constructor; don't try this at that point.
            if hasattr(self, 'merged_psffn'):
                psf = self.read_psf_model(0., 0., pixPsf=True)
                fwhm = psf.fwhm
                return fwhm
        # convert from arcsec to pixels (hard-coded pixscale here)
        fwhm /= PanStarrsImage.get_nominal_pixscale()
        return fwhm

    def get_propid(self, primhdr):
        return ''

    def get_camera(self, primhdr):
        key = 'PSCAMERA'
        if key in primhdr:
            cam = primhdr[key]
        else:
            hdr = self.read_image_header()
            # PSCAMERA= 'GPC1    '           / Camera name
            cam = hdr[key]
        if cam == 'GPC1':
            return 'panstarrs'
        return cam

    def get_wcs(self, hdr=None):
        from astrometry.util.util import Tan
        if hdr is None:
            hdr = self.read_image_header()
        copyhdr = fitsio.FITSHDR()
        for r in hdr.records():
            copyhdr.add_record(r)
        if not 'CD1_1' in copyhdr:
            #### Really we should read PC001001, PC001002, PC002001, PC002002...
            copyhdr['CD1_1'] = -hdr['CDELT1']
            copyhdr['CD1_2'] = 0.
            copyhdr['CD2_1'] = 0.
            copyhdr['CD2_2'] = hdr['CDELT2']
        wcs = Tan(copyhdr)
        wcs.version = ''
        wcs.plver = ''
        return wcs

    def read_sky_model(self, **kwargs):
        from tractor import ConstantSky
        sky = ConstantSky(0.)
        return sky

    def read_image(self, header=False, **kwargs):
        img,hdr = super().read_image(header=True, **kwargs)

        # Arcsinh scaled pixel values!
        alpha = 2.5 * np.log10(np.e)
        boff  = hdr['BOFFSET']
        bsoft = hdr['BSOFTEN']
        img = boff + bsoft * 2. * np.sinh(img / alpha)

        if header:
            img = img,hdr
        return img

    def read_invvar(self, dq=None, header=False, **kwargs):
        # VARIANCE map (not a weight map)
        v,hdr = self._read_fits(self.wtfn, self.wt_hdu, header=True, **kwargs)

        # Arcsinh scaled values!
        alpha = 2.5 * np.log10(np.e)
        boff  = hdr['BOFFSET']
        bsoft = hdr['BSOFTEN']
        v = boff + bsoft * 2. * np.sinh(v / alpha)

        iv = 1./v
        iv[v<=0] = 0.
        iv[np.logical_not(np.isfinite(iv))] = 0.
        #! this can happen
        #iv[np.logical_not(np.isfinite(np.sqrt(iv)))] = 0.
        if dq is not None:
            iv[dq != 0] = 0.
        if header:
            iv = iv,hdr
        return iv

    def get_mask_names(self, hdr):
        maskvals = dict()
        # number of bits
        nmasks = hdr['MSKNUM']
        for i in range(nmasks):
            name = hdr['MSKNAM%02i' % i].strip()
            val  = hdr['MSKVAL%02i' % i]
            maskvals[name] = val
        return maskvals

    def remap_dq(self, dq, hdr):
        new_dq = np.zeros(dq.shape, np.int16)
        maskvals = self.get_mask_names(hdr)
        # Ignore STARCORE
        new_dq |= DQ_BITS['badpix'] * ((dq & (maskvals['DETECTOR'] |
                                              maskvals['FLAT'] |
                                              maskvals['DARK'] |
                                              maskvals['BLANK'] |
                                              maskvals['CTE'] |
                                              maskvals['SUSPECT'] |
                                              maskvals['BURNTOOL'] |
                                              maskvals['SPIKE'] |
                                              maskvals['GHOST'] |
                                              maskvals['STREAK'] |
                                              maskvals['CONV.BAD'] |
                                              maskvals['CONV.POOR']
                                              )) > 0)

        new_dq |= DQ_BITS['satur'] * ((dq & maskvals['SAT']) > 0)
        new_dq |= DQ_BITS['cr'] * ((dq & maskvals['CR']) > 0)
        return new_dq

    def get_zeropoint(self, primhdr, hdr):
        return hdr['FPA.ZP']

    def get_airmass(self, primhdr, imghdr, ra, dec):
        airmass = imghdr.get('AIRMASS', None)
        if airmass is None:
            airmass = self.recompute_airmass(primhdr, ra, dec)
        return airmass

    def get_mjd(self, primhdr):
        key = 'MJD-OBS'
        if key in primhdr:
            return primhdr[key]
        hdr = self.read_image_header()
        return hdr.get(key)

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        from legacypipe.image import estimate_sky_from_pixels
        skymed, skyrms = estimate_sky_from_pixels(img)
        return 0., skymed, skyrms
    
    def check_image_header(self, imghdr):
        pass

    def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
        # Before passing files to SourceExtractor / PsfEx, filter our mask image
        # because we want to ignore the STARCORE mask bit
        tmpimgfn,tmpmaskfn = super().funpack_files(imgfn, maskfn, imghdu, maskhdu, todelete)
        #print('Dropping mask bit 5 before running SE')
        m,mhdr = fitsio.read(tmpmaskfn, header=True)
        maskvals = self.get_mask_names(mhdr)
        # Ignore STARCORE
        val = maskvals['STARCORE']
        val = np.uint16(val)
        nset = np.sum(m & val > 0)
        #print('Mask:', m.shape, m.dtype)
        #print('STARCORE bit value:', val, type(val))
        m &= ~val
        print('Ignoring STARCORE mask bit on', nset, 'pixels')
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpmaskfn)
        fitsio.write(tmpmaskfn, m, clobber=True, header=mhdr)

        # Also need to open the image file and unapply the arcsinh scaling!
        img,imhdr = fitsio.read(tmpimgfn, header=True)
        alpha = 2.5 * np.log10(np.e)
        boff  = imhdr['BOFFSET']
        bsoft = imhdr['BSOFTEN']
        img = boff + bsoft * 2. * np.sinh(img / alpha)
        tmpimgfn = create_temp(suffix='.fits')
        todelete.append(tmpimgfn)
        fitsio.write(tmpimgfn, img, clobber=True, header=imhdr)

        return tmpimgfn, tmpmaskfn
