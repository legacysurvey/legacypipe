import os
import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS

'''
This is for the "pitcairn" reductions for CFIS-r data.

eg, search for data from here,
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/cfis/csky.html

and use "get data", then download a URL list, or grab a search box (here for u and r-band images)

http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/community/cfis/mcut.pl?&ra1=251&ra2=254.75&dec1=34.2&dec2=35.7&grid=true&images=true&tiles=false&fils=u2&fils=r2

and retrieve with:

wget -N --content-disposition -i ../urls.txt --http-user=dstn --http-password=$CADC_PASSWORD --auth-no-challenge

Zeropoints like:
python legacyzpts/legacy_zeropoints.py --psf --splinesky --calibdir cfis/calib --run-calibs --camera megaprime --image pitcairn/2106094p.fits.fz --not_on_proj --outdir cfis/zpts/ > 12.log 2>&1 &
ls cfis/zpts/2??????p-legacypipe.fits > zpts.txt
python legacyzpts/legacy_zeropoints_merge.py --nproc 0 --outname cfis/survey-ccds-cfis-pitcairn.fits --file_list zpts.txt

# Deep3 region:
fitscopy ~/cosmo/data/legacysurvey/dr6/survey-ccds-dr6plus.kd.fits+1"[(abs(ra-215)<2) && (abs(dec-52.75)<1) && ((filter=='g') || (filter=='z'))]" dr6-deep3.fits
fitsgetext -i dr6-deep3.fits -e 0 -e 1 -o cfis/survey-ccds-dr6-gz.fits
gzip cfis/survey-ccds-dr6-gz.fits

# Deep2-Field2 region:
fitscopy ~/cosmo/data/legacysurvey/dr6/survey-ccds-dr6plus.kd.fits+1"[(ra>215.0) && (ra<254.75) && (dec>34.2) && (dec<35.7) && ((filter=='g') || (filter=='z'))]" dr6-deep2f2.fits
fitsgetext -i dr6-deep2f2.fits -e 0 -e 1 -o cfis/survey-ccds-dr6-deep2f2-gz.fits
# CFIS search as above: RA 215.5 to 254.25 plus 0.5-deg margin, Dec 34.7 to 35.2 plus 0.5-deg margin
# zpts like:
python legacyzpts/legacy_zeropoints.py --psf --splinesky --calibdir cfis/calib --run-calibs --camera megaprime --image pitcairn/$img --not_on_proj --outdir cfis/zpts/ --threads 8 > $log 2>&1


'''


class MegaPrimeImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the MegaPrime
    camera on CFHT.
    '''
    def __init__(self, survey, t, image_fn=None, image_hdu=0, **kwargs):
        super(MegaPrimeImage, self).__init__(survey, t, image_fn=image_fn, image_hdu=image_hdu,
                                             **kwargs)
        # print('MegaPrimeImage: CCDs table entry', t)
        # for x in dir(t):
        #     if x.startswith('_'):
        #         continue
        #     print('  ', x, ':', getattr(t,x))
        ### HACK!!!
        self.zp0 =  dict(g = 26.610,
                         r = 26.818,
                         z = 26.484,
                         # Totally made up
                         u = 26.610,
                         )
                         #                  # i,Y from DESY1_Stripe82 95th percentiles
                         #                  i=26.758, Y=25.321) # e/sec
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06,
                          # Totally made up
                          u = 0.24)
        #                   #i, Y totally made up
        #                   i=0.08, Y=0.06)
        # --> e/sec

        ##### UGH they contain duplicate EXTNAME header cards.
        # if image_hdu is not None:
        #     print('image_hdu', image_hdu, 'hdu', self.hdu)
        #     self.ccdname = 'ccd%i' % (self.hdu - 1)
        #     print('Reset CCDNAME to', self.ccdname)

        # Try grabbing fwhm from PSFEx file, if it exists.
        if hasattr(self, 'fwhm') and not np.isfinite(self.fwhm):
            try:
                # PSF model file may not have been created yet...
                self.fwhm = self.get_fwhm(None, None)
            except:
                pass

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.185

    def override_ccd_table_types(self):
        # "ccd00"
        return {'ccdname':'S5'}

    def get_extension_list(self, debug=False):
        # duplicate EXTNAME cards in the headers?! trips up fitsio;
        # https://github.com/esheldon/fitsio/issues/324
        F = fitsio.FITS(self.imgfn)
        exts = []
        for i,f in enumerate(F[1:]):
            #exts.append(f.get_extname())
            exts.append(i+1)
            if debug:
                break
        return exts

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        self.dqfn = self.imgfn.replace('p.fits.fz', 'p.flag.fits.fz')
        assert(self.dqfn != self.imgfn)
        self.wtfn = None
        #self.wtfn = self.imgfn.replace('p.fits.fz', 'p.weight.fits.fz')
        #assert(self.wtfn != self.imgfn)

    #def compute_filenames(self):
    #    self.dqfn = 'cfis/test.mask.0.40.01.fits'

    def read_image_header(self, **kwargs):
        hdr = super().read_image_header(**kwargs)
        ##### UGH they contain duplicate EXTNAME header cards.
        hdr['EXTNAME'] = 'ccd%02i' % (self.hdu - 1)
        print('Reset EXTNAME to', hdr['EXTNAME'])
        return hdr

    def get_radec_bore(self, primhdr):
        return primhdr['RA_DEG'], primhdr['DEC_DEG']

    def photometric_calibrator_to_observed(self, name, cat):
        from legacypipe.ps1cat import ps1cat
        ps1band_map = ps1cat.ps1band
        if name == 'ps1':
            # u->g
            ps1band = dict(u='g').get(self.band, self.band)
            ps1band_index = ps1band_map[ps1band]
            colorterm = self.colorterm_ps1_to_observed(cat.median, self.band)
            return cat.median[:, ps1band_index] + np.clip(colorterm, -1., +1.)
        elif name == 'sdss':
            from legacypipe.ps1cat import sdsscat
            colorterm = self.colorterm_sdss_to_observed(cat.psfmag, self.band)
            band = sdsscat.sdssband[self.band]
            return cat.psfmag[:, band] + np.clip(colorterm, -1., +1.)
        else:
            raise RuntimeError('No photometric conversion from %s to CFHT' % name)

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacypipe.ps1cat import ps1_to_decam
        print('HACK -- using DECam color term for CFHT!!')
        if band == 'u':
            print('HACK -- using g-band color term for u band!')
            band = 'g'
        return ps1_to_decam(ps1stars, band)

    def colorterm_sdss_to_observed(self, sdssstars, band):
        from legacypipe.ps1cat import sdss_to_decam
        return sdss_to_decam(sdssstars, band)

    def get_band(self, primhdr):
        # u.MP9302
        band = primhdr['FILTER'][0]
        return band

    def get_propid(self, primhdr):
        return primhdr['RUNID']

    def get_gain(self, primhdr, hdr):
        return hdr['GAIN']

    def get_fwhm(self, primhdr, imghdr):
        # Nothing in the image headers...
        # There's a get_fwhm() call early in the constructor... this will return NaN.
        if not hasattr(self, 'merged_psffn'):
            return super().get_fwhm(primhdr, imghdr)
        psf = self.read_psf_model(0, 0, pixPsf=True)
        fwhm = psf.fwhm
        print('Got FWHM from PSF model:', fwhm)
        return fwhm

    # Used during zeropointing
    def scale_image(self, img):
        return img.astype(np.float32)

    # def get_wcs(self, hdr=None):
    #     ### FIXME -- no distortion solution in here
    #     # from astrometry.util.util import Tan
    #     # return Tan(self.hdr)
    # 
    #     # "pitcairn" reductions have PV header cards (CTYPE is still RA---TAN)
    #     from astrometry.util.util import wcs_pv2sip_hdr
    #     if hdr is None:
    #         hdr = self.read_image_header()
    #     return wcs_pv2sip_hdr(self.hdr)

    # def remap_dq(self, dq, header):
    #     '''
    #     Called by get_tractor_image() to map the results from read_dq
    #     into a bitmask.  We only have a 0/1 bad pixel mask.
    #     '''
    #     dqbits = np.zeros(dq.shape, np.int16)
    #     dqbits[dq == 0] = DQ_BITS['badpix']
    #     return dqbits

    def read_image(self, header=False, **kwargs):
        img = super(MegaPrimeImage, self).read_image(header=header, **kwargs)
        if header:
            img,hdr = img
        img = img.astype(np.float32)
        if header:
            return img,hdr
        return img

    def read_invvar(self, **kwargs):
        # The "weight" maps given are 0/1, apparently == flags.
        print('MegaPrimeImage.read_invvar')
        img = self.read_image(**kwargs)
        dq = self.read_dq(**kwargs)
        if self.sig1 is None or self.sig1 == 0.:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            ok = (dq[slice1] == 0) * (dq[slice2] == 0)
            mad = np.median(np.abs(img[slice1][ok] - img[slice2][ok]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            # self.sig1 must be in calibrated units
            #self.sig1 = sig1
            print('Computed sig1 by Blanton method:', sig1, '(MAD:', mad, ')')
        else:
            from tractor import NanoMaggies
            print('sig1 from CCDs file:', self.sig1)
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            print('scaled to image units:', sig1)

        iv = np.empty_like(img)
        iv[:,:] = 1./sig1**2
        iv[dq != 0] = 0.
        return iv

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        # SATURATE header keyword is ~65536, but actual saturation in the images is
        # 32760.
        I,J = np.nonzero(img > 32700)
        from legacypipe.bits import DQ_BITS
        if len(I):
            dq[I,J] |= DQ_BITS['satur']
            invvar[I,J] = 0

    # def read_invvar(self, **kwargs):
    #     ## FIXME -- at the very least, apply mask
    #     print('MegaPrimeImage.read_invvar')
    #     img = self.read_image(**kwargs)
    #     if self.sig1 is None:
    #         # Estimate per-pixel noise via Blanton's 5-pixel MAD
    #         slice1 = (slice(0,-5,10),slice(0,-5,10))
    #         slice2 = (slice(5,None,10),slice(5,None,10))
    #         mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
    #         sig1 = 1.4826 * mad / np.sqrt(2.)
    #         self.sig1 = sig1
    #         print('Computed sig1 by Blanton method:', self.sig1)
    #     else:
    #         print('sig1 from CCDs file:', self.sig1)
    # 
    #     iv = np.zeros_like(img) + (1./self.sig1**2)
    #     return iv

    # calibs

    # def run_se(self, imgfn, maskfn):
    #     from astrometry.util.file import trymakedirs
    #     sedir = self.survey.get_se_dir()
    #     trymakedirs(self.sefn, dir=True)
    #     # We write the SE catalog to a temp file then rename, to avoid
    #     # partially-written outputs.
    # 
    #     from legacypipe.survey import create_temp
    #     import fitsio
    # 
    #     tmpmaskfn = create_temp(suffix='.fits')
    #     # # The test.mask file has 1 for good pix, 0 for bad... invert for SE
    #     goodpix = fitsio.read(maskfn)
    #     fitsio.write(tmpmaskfn, (1-goodpix).astype(np.uint8), clobber=True)
    #     #tmpmaskfn = maskfn
    # 
    #     tmpfn = os.path.join(os.path.dirname(self.sefn),
    #                          'tmp-' + os.path.basename(self.sefn))
    #     cmd = ' '.join([
    #         'sex',
    #         '-c', os.path.join(sedir, self.camera + '.se'),
    #         '-PARAMETERS_NAME', os.path.join(sedir, self.camera + '.param'),
    #         '-FILTER_NAME %s' % os.path.join(sedir, self.camera + '.conv'),
    #         '-FLAG_IMAGE %s' % tmpmaskfn,
    #         '-CATALOG_NAME %s' % tmpfn,
    #         '-SEEING_FWHM %f' % 0.8,
    #         '-FITS_UNSIGNED N',
    #         #'-VERBOSE_TYPE FULL',
    #         #'-PIXEL_SCALE 0.185',
    #         #'-SATUR_LEVEL 100000',
    #         imgfn])
    #     print(cmd)
    #     rtn = os.system(cmd)
    #     if rtn:
    #         raise RuntimeError('Command failed: ' + cmd)
    #     os.rename(tmpfn, self.sefn)
    #     os.unlink(tmpmaskfn)



# For CFIS images processed with Elixir
class MegaPrimeElixirImage(MegaPrimeImage):
    def compute_filenames(self):
        self.dqfn = None
        self.wtfn = None
    # don't need overridden read_image_header
    def read_dq(self, header=False, **kwargs):
        from legacypipe.bits import DQ_BITS
        # Image pixels to be ignored have value 0.0
        img = self._read_fits(self.imgfn, self.hdu, header=header, **kwargs)
        if header:
            img,hdr = img
        dq = np.zeros(img.shape, np.int16)
        dq[img == 0] = DQ_BITS['badpix']
        # There are also (small) values in the u-band images (-7e-12 eg)...
        n = np.sum(img < 0)
        print('Flagged', n, 'pixels in the DQ map with negative image values')
        dq[img < 0] = DQ_BITS['badpix']
        # Ugh there are ALSO small POSITIVE values around the u-band image edges (+7e-12)
        n = np.sum((img > 0) * (img < 0.5))
        print('Flagged', n, 'additional pixels in the DQ map with small positive image values')
        dq[img < 0.5] = DQ_BITS['badpix']

        if header:
            dq = dq,hdr
        return dq
    def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
        from legacypipe.survey import create_temp
        tmpimgfn,_ = super().funpack_files(imgfn, maskfn, imghdu, maskhdu, todelete)
        img = fitsio.read(tmpimgfn)
        print('Funpack_files: image minimum value %g, max %g' % (img.min(), img.max()), 'type', img.dtype)
        mask = (img < 0.5).astype(np.int16)
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpmaskfn)
        fitsio.write(tmpmaskfn, mask, clobber=True)

        #
        fitsio.write(tmpimgfn, img.astype(np.float32), clobber=True)
        return tmpimgfn, tmpmaskfn

    def get_good_image_subregion(self):
        '''
        Returns x0,x1,y0,y1 of the good region of this chip,

        DATASEC = '[33:2080,1:4612]'
        '''
        hdr = self.read_image_header()
        datasec = hdr.get('DATASEC')
        nil = None,None,None,None
        if datasec is None:
            return nil
        datasec = datasec.strip()
        if not (datasec.startswith('[') and datasec.endswith(']')):
                return nil
        words = datasec[1:-1].split(',')
        if len(words) != 2:
            return nil
        xx,yy = words
        xx = xx.split(':')
        yy = yy.split(':')
        if len(xx) != 2 or len(yy) != 2:
            return nil
        try:
            rtn = int(xx[0]), int(xx[1]), int(yy[0]), int(yy[1])
            print('Returning good image subregion', rtn)
            return rtn
        except:
            pass
        return nil
