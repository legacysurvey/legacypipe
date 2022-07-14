import warnings
import numpy as np

from legacypipe.image import LegacySurveyImage, validate_version

import logging
logger = logging.getLogger('legacypipe.decam')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

'''
Code specific to images from the Dark Energy Camera (DECam).
'''

class DecamImage(LegacySurveyImage):

    ZP0 =  dict(g = 25.001,
                r = 25.209,
                z = 24.875,
                # u from Arjun 2021-03-17, based on DECosmos-to-SDSS
                u = 23.3205,
                # i,Y from DESY1_Stripe82 95th percentiles
                i = 25.149,
                Y = 23.712,
                # N419 is hacked to just = N501
                N419 = 23.812,
                N501 = 23.812,
                N673 = 24.151,
    )

    K_EXT = dict(g = 0.173,
                 r = 0.090,
                 i = 0.054,
                 z = 0.060,
                 Y = 0.058,
                 # From Arjun 2021-03-17 based on DECosmos (calib against SDSS)
                 u = 0.63,
                 # HACK - == broadband
                 N419 = 0.173,
                 N501 = 0.173,
                 N673 = 0.090,
    )

    '''
    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.
    '''
    def __init__(self, survey, t, image_fn=None, image_hdu=0):
        super(DecamImage, self).__init__(survey, t, image_fn=image_fn, image_hdu=image_hdu)

        # Nominal zeropoints
        # These are used only for "ccdskybr", so are not critical.
        self.zp0 = DecamImage.ZP0.copy()

        # extinction per airmass, from Arjun's 2019-02-27
        # "Zeropoint variations with MJD for DECam data".
        self.k_ext = DecamImage.K_EXT.copy()

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.262

    def get_site(self):
        from astropy.coordinates import EarthLocation
        # zomg astropy's caching mechanism is horrific
        # return EarthLocation.of_site('ctio')
        from astropy.units import m
        from astropy.utils import iers
        iers.conf.auto_download = False
        return EarthLocation(1814304. * m, -5214366. * m, -3187341. * m)

    def calibration_good(self, primhdr):
        '''Did the CP processing succeed for this image?  If not, no need to process further.
        '''
        wcscal = primhdr.get('WCSCAL', '').strip().lower()
        if wcscal.startswith('success'):  # success / successful
            return True
        # Frank's work-around for some with incorrect WCSCAL=Failed (DR9 re-reductions)
        return primhdr.get('SCAMPFLG') == 0

    def colorterm_sdss_to_observed(self, sdssstars, band):
        from legacypipe.ps1cat import sdss_to_decam
        return sdss_to_decam(sdssstars, band)
    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1_to_decam
        return ps1_to_decam(cat, band)

    def get_gain(self, primhdr, hdr):
        return np.average((hdr['GAINA'],hdr['GAINB']))

    def get_expnum(self, primhdr):
        # Some images (eg decam/DECam_CP-DR10c/CP20200224/c4d_200225_063620_ooi_i_v1.fits.fz)
        # have it as a string!
        return int(primhdr['EXPNUM'])

    def get_sky_template_filename(self, old_calibs_ok=False):
        import os
        from astrometry.util.fits import fits_table
        dirnm = os.environ.get('SKY_TEMPLATE_DIR', None)
        if dirnm is None:
            warnings.warn('decam: no SKY_TEMPLATE_DIR environment variable set.')
            return None
        '''
        # Create this sky-scales.kd.fits file via:
        python legacypipe/create-sky-template-kdtree.py skyscales_ccds.fits \
        sky-scales.kd.fits
        '''
        fn = os.path.join(dirnm, 'sky-scales.kd.fits')
        if not os.path.exists(fn):
            warnings.warn('decam: no $SKY_TEMPLATE_DIR/sky-scales.kd.fits file.')
            return None
        from astrometry.libkd.spherematch import tree_open
        kd = tree_open(fn, 'expnum')
        I = kd.search(np.array([self.expnum]), 0.5, 0, 0)
        if len(I) == 0:
            warnings.warn('decam: expnum %i not found in file %s' % (self.expnum, fn))
            return None
        # Read only the CCD-table rows within range.
        S = fits_table(fn, rows=I)
        if 'ccdname' in S.get_columns():
            # DR9: table was per-CCD.
            S.cut(np.array([c.strip() == self.ccdname for c in S.ccdname]))
            imgid = 'expnum %i, ccdname %s' % (self.expnum, self.ccdname)
            if len(S) == 0:
                warnings.warn('decam: %s not found in file %s' % (imgid, fn))
                return None
        else:
            # DR10: table is per-exposure.
            imgid = 'expnum %i' % (self.expnum)
        assert(len(S) == 1)
        sky = S[0]
        if sky.run == -1:
            debug('sky template: run=-1 for %s' % (imgid))
            return None
        # Check PLPROCID only
        if not validate_version(
                fn, 'table', self.expnum, None, self.plprocid, data=S):
            txt = ('Sky template for %s did not pass consistency validation (EXPNUM, PLPROCID) -- image %i,%s vs template table %i,%s' %
                   (imgid, self.expnum, self.plprocid, sky.expnum, sky.plprocid))
            if old_calibs_ok:
                warnings.warn(txt + '-- but old_calibs_ok, so using sky template anyway')
            else:
                warnings.warn(txt + '-- not subtracting sky template for this CCD')
                return None

        #assert(self.band == sky.filter)
        tfn = os.path.join(dirnm, 'sky_templates',
                           'sky_template_%s_%i.fits' % (self.band, sky.run))
        if not os.path.exists(tfn):
            warnings.warn('Sky template file %s does not exist' % tfn)
            return None
        return dict(template_filename=tfn, sky_template_dir=dirnm, sky_obj=sky, skyscales_fn=fn)

    def get_sky_template(self, slc=None, old_calibs_ok=False):
        import fitsio
        d = self.get_sky_template_filename(old_calibs_ok=old_calibs_ok)
        if d is None:
            return None
        skyscales_fn = d['skyscales_fn']
        sky_template_dir = d['sky_template_dir']
        tfn = d['template_filename']
        sky = d['sky_obj']
        #info('Reading', tfn, 'ext', self.ccdname)
        F = fitsio.FITS(tfn)
        if not self.ccdname in F:
            warnings.warn('Sky template file %s does not contain extension %s' % (tfn, self.ccdname))
            return None
        f = F[self.ccdname]
        if slc is None:
            template = f.read()
        else:
            template = f[slc]
        hdr = F[0].read_header()
        ver = hdr.get('SKYTMPL', -1)
        meta = dict(sky_scales_fn=skyscales_fn, template_fn=tfn, sky_template_dir=sky_template_dir,
                    run=sky.run, scale=sky.skyscale, version=ver)
        return template * sky.skyscale, meta

    def get_good_image_subregion(self):
        x0,x1,y0,y1 = None,None,None,None

        glow_expnum = 298251

        # Handle 'glowing' edges in early r-band images
        if self.band == 'r' and self.expnum < glow_expnum:
            # Northern chips: drop 100 pix off the bottom
            if 'N' in self.ccdname:
                debug('Clipping bottom part of northern DES r-band chip')
                y0 = 100
            else:
                # Southern chips: drop 100 pix off the top
                debug('Clipping top part of southern DES r-band chip')
                y1 = self.height - 100

        # Clip the bad half of chip S7.
        # The left half is OK.
        if self.ccdname == 'S7':
            debug('Clipping the right half of chip S7')
            x1 = 1023

        return x0,x1,y0,y1

    def read_invvar(self, slc=None, header=False, dq=None, **kwargs):
        # See email of 2021-08-18 "Unexpected different-sized ooi
        # and oow images" These OOW images have one extra pixel on
        # each side (they are supposed to be trimmed off of the
        # ooi, ood, and oow images at the end, but in this range
        # of plprocids, the oow images missed out on this processing.
        if not self.plprocid >= '98bf8a6' and self.plprocid <= '98ed7fa':
            return super().read_invvar(slc=slc, header=header, dq=dq, **kwargs)

        debug('DECam image', self, 'with PLPROCID', self.plprocid, 'may have weird-shaped OOW')
        # We could try to be clever and adjust the limits of the "slc"
        # arg if given...  or we could keep it simple: read without "slc",
        # trim one pixel around the edges, and then apply slc.
        iv = super().read_invvar(slc=None, dq=None, header=header, **kwargs)
        if header:
            iv,hdr = iv
        # If image is unexpectedly 2 pixels too big on each side,
        # Trim one pixel off each edge!
        debug('DECam image', self, 'with PLPROCID', self.plprocid, 'OOW shape:', iv.shape)
        if iv.shape == (4096, 2048):
            info('DECam image', self, 'with PLPROCID', self.plprocid, 'has weird-shaped OOW; trimming')
            iv = iv[1:-1, 1:-1]
        # Apply slice if present
        if slc:
            iv = iv[slc]
        # Zero out masked pixels (this would be done by super().read_invvar, but we can't do
        # it until the iv is the right shape!
        if dq is not None:
            iv[dq != 0] = 0.

        if header:
            return iv,hdr
        return iv

    def read_dq(self, header=None, **kwargs):
        # Reduce DQ size
        dq = super().read_dq(header=header, **kwargs)
        if header:
            # unpack
            dq,hdr = dq
        # Downsize type
        dq = dq.astype(self.dq_type)
        if header:
            # repack
            dq = dq,hdr
        return dq

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        import fitsio
        primhdr = fitsio.read_header(self.dqfn)
        plver = primhdr['PLVER']
        if decam_has_dq_codes(plver):
            from legacypipe.image import remap_dq_cp_codes
            # Always ignore the satellite masker code.
            ignore = [8]
            if not decam_use_dq_cr(plver):
                # In some runs of the pipeline (not captured by plver)
                # the CR and Satellite masker codes got confused, so ignore
                # both.
                ignore.append(5)
            dq = remap_dq_cp_codes(dq, ignore_codes=ignore)

            # In some runs (not captured by plver), the DES
            # star mask (circular mask around saturated stars) got
            # copied into the BLEED mask.
            # Try to undo this by demanding that BLEED pixels be vertically
            # connected to SATUR pixels.
            from scipy.ndimage.morphology import binary_dilation
            from legacypipe.bits import DQ_BITS
            sat = ((dq & DQ_BITS['satur']) > 0)
            # dilated saturated
            disat = binary_dilation(sat, iterations=2)
            bleed = (dq & DQ_BITS['bleed']) > 0
            outbleed = np.zeros_like(bleed)
            # We're going to start from the dilated-SAT pixels and keep vertical runs
            # of BLEED pixels.
            Y,X = np.nonzero(disat)
            H,W = outbleed.shape
            # ALSO add in any pixels that are BLEED at the image edge, because they
            # might have been connected to a SAT pixel that is not in our subimage.
            # (this does mean that we'll keep any circular BLEED masks that happen to hit
            # the edge of the CCD subimage we're looking at)
            Xtop = np.flatnonzero(bleed[-1,:])
            Xbot = np.flatnonzero(bleed[0,:])
            if len(Xtop)+len(Xbot):
                X = np.hstack((X, Xtop, Xbot))
                Y = np.hstack((Y, np.zeros(len(Xtop), int)+(H-1), np.zeros(len(Xbot), int)))
            for x,y in zip(X,Y):
                # keep the region we dilated the SAT mask into
                # (this also catches the top/bottom BLEED pixels)
                if not sat[y,x] and bleed[y,x]:
                    outbleed[y,x] = True
                for yy in range(y+1, H):
                    # hit a neighbor -- we'll process this column in the neighbor
                    if disat[yy,x]:
                        break
                    if not bleed[yy,x]:
                        break
                    outbleed[yy,x] = True
                for yy in range(y-1, -1, -1):
                    if disat[yy,x]:
                        break
                    if not bleed[yy,x]:
                        break
                    outbleed[yy,x] = True

            # Update BLEED bit
            dq &= ~(self.dq_type(DQ_BITS['bleed']))
            dq |= self.dq_type(DQ_BITS['bleed']*outbleed)

        else:
            from legacypipe.bits import DQ_BITS
            # Un-set the SATUR flag for pixels that also have BADPIX set.
            bothbits = DQ_BITS['badpix'] | DQ_BITS['satur']
            I = np.flatnonzero((dq & bothbits) == bothbits)
            if len(I):
                debug('Warning: un-setting SATUR for', len(I),
                      'pixels with SATUR and BADPIX set.')
                dq.flat[I] &= ~(self.dq_type(DQ_BITS['satur']))
                assert(np.all((dq & bothbits) != bothbits))
        # should already be this type...
        dq = dq.astype(self.dq_type)
        return dq

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        plver = primhdr['PLVER']
        if decam_s19_satur_ok(plver):
            return
        if self.ccdname != 'S19':
            return
        I,J = np.nonzero((img > 46000) * (dq == 0) * (invvar > 0))
        info('Masking %i additional saturated pixels in DECam expnum %i S19 CCD, %s' %
             (len(I), self.expnum, self.print_imgpath), 'slice', slc)
        if len(I) == 0:
            return
        from legacypipe.bits import DQ_BITS
        if dq is not None:
            dq[I,J] |= DQ_BITS['satur']
        invvar[I,J] = 0.

    # S30, N14, S19, S16, S10
    def get_tractor_sky_model(self, img, goodpix):
        from tractor.splinesky import SplineSky
        from legacypipe.jumpsky import JumpSky
        boxsize = self.splinesky_boxsize
        # For DECam chips where we drop half the chip, spline becomes
        # underconstrained
        if min(img.shape) / boxsize < 4:
            boxsize /= 2

        if (self.band in ['g','r','i'] and
            self.ccdname.strip() in ['S30', 'N14', 'S19', 'S16', 'S10']):
            H,W = img.shape
            xbreak = W//2
            skyobj = JumpSky.BlantonMethod(img, goodpix, boxsize, xbreak)
        else:
            skyobj = SplineSky.BlantonMethod(img, goodpix, boxsize, min_fraction=0.25)

        return skyobj

def decam_cp_version_after(plver, after):
    from distutils.version import StrictVersion
    # The format of the DQ maps changed as of version 3.5.0 of the
    # Community Pipeline.  Handle that here...
    plver = plver.strip()
    plver = plver.replace('V','')
    plver = plver.replace('DES ', '')
    plver = plver.replace('+1', 'a1')
    #'4.8.2a'
    if plver.endswith('2a'):
        plver = plver.replace('.2a', '.2a1')
    # 5.2.2LS
    if plver.endswith('LS'):
        plver = plver.replace('LS', '')
    # 5.0beta
    if plver.endswith('beta'):
        plver = plver.replace('beta', 'b1')
    return StrictVersion(plver) >= StrictVersion(after)

def decam_s19_satur_ok(plver):
    return decam_cp_version_after(plver, '4.9.0')

def decam_use_dq_cr(plver):
    return decam_cp_version_after(plver, '4.8.0')

def decam_has_dq_codes(plver):
    # The format of the DQ maps changed as of version 3.5.0 of the CP
    return decam_cp_version_after(plver, '3.5.0')
