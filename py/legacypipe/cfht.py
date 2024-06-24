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
                         CaHK = 26.610,
                         )
                         #                  # i,Y from DESY1_Stripe82 95th percentiles
                         #                  i=26.758, Y=25.321) # e/sec
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06,
                          # Totally made up
                          u = 0.24,
                          CaHK = 0.24)
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
        # The old filter set - now called "uS", "gS" etc, where the new ones are just called
        # "u", "g", etc - vignet the "ears" of the camera.
        # The naming scheme decoder ring is:
        # Old set:
        #   U.MP9301
        #   G.MP9401
        #   R.MP9601
        #   I.MP9701
        #   Z.MP9801
        # Second generation:
        #   I.MP9702
        # Third generation / new set:
        #   U.MP9302
        #   G.MP9402
        #   R.MP9602
        #   I.MP9703
        #   Z.MP9901
        F = fitsio.FITS(self.imgfn)

        primhdr = F[0].read_header()
        filt = primhdr['FILTER']
        old_filter = (filt in ['u.MP9301', 'g.MP9401', 'r.MP9601', 'i.MP9701', 'z.MP9801'])

        # duplicate EXTNAME cards in the headers?! trips up fitsio;
        # https://github.com/esheldon/fitsio/issues/324
        exts = []
        for i,f in enumerate(F[1:]):
            # Drop vignetted CCDs!
            if old_filter and i >= 36:
                print('Image uses an old filter (%s) - dropping vignetted "ears" CCDs' % filt)
                break
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

    def clip_colorterm(self, c):
        # Note larger range than usual!
        return np.clip(c, -1., +4.)

    def photometric_calibrator_to_observed(self, name, cat):
        from legacypipe.ps1cat import ps1cat
        ps1band_map = ps1cat.ps1band
        if name == 'ps1':
            # u->g, CaHK->g
            ps1band = dict(u='g', CaHK='g').get(self.band, self.band)
            ps1band_index = ps1band_map[ps1band]
            colorterm = self.colorterm_ps1_to_observed(cat.median, self.band)
            colorterm = self.clip_colorterm(colorterm)
            # Note larger range of color term than usual!
            return cat.median[:, ps1band_index] + colorterm
        elif name == 'sdss':
            from legacypipe.ps1cat import sdsscat
            colorterm = self.colorterm_sdss_to_observed(cat.psfmag, self.band)
            colorterm = self.clip_colorterm(colorterm)
            sdssbands = sdsscat.sdssband.copy()
            sdssbands.update(CaHK=0)
            band = sdssbands[self.band]
            return cat.psfmag[:, band] + colorterm
        else:
            raise RuntimeError('No photometric conversion from %s to CFHT' % name)

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacypipe.ps1cat import ps1_to_decam
        print('HACK -- using DECam color term for CFHT!!')
        #if band == 'u':
        #    print('HACK -- using g-band color term for u band!')
        #    band = 'g'
        return ps1_to_decam(ps1stars, band)

    def colorterm_sdss_to_observed(self, sdssstars, band):
        from legacypipe.ps1cat import sdss_to_decam
        return sdss_to_decam(sdssstars, band)

    def get_band(self, primhdr):
        # u.MP9302
        band = primhdr['FILTER'].split('.')[0]
        return band

    def get_propid(self, primhdr):
        return primhdr['RUNID']

    def get_gain(self, primhdr, hdr):
        return hdr['GAIN']

    def get_ha_deg(self, primhdr):
        from astrometry.util.starutil_numpy import hmsstring2ra
        # HA header is missing from some images (eg, XMM u-band cfht-xmm-u/776654p.fits.fz)
        hastr = primhdr.get('HA')
        if hastr is None:
            return np.nan
        return hmsstring2ra(hastr)

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
            #print('sig1 from CCDs file:', self.sig1)
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            print('sig1 from CCDs file:', self.sig1, 'scaled by ccdzpt', self.ccdzpt, 'to image units: sig1=', sig1)

        iv = np.empty_like(img)
        iv[:,:] = 1./sig1**2
        iv[dq != 0] = 0.
        return iv

    def get_satur(self):
        # SATURATE header keyword is ~65536, but actual saturation in the images is
        # 32760. [ref needed]
        return 32700

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        I,J = np.nonzero(img > self.get_satur())
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scamp_wcs = None
        # Run sky calib first (for patching...)
        self.sky_before_psfex = True

        self.do_solve_field = (self.band in ['CaHK', 'u'])

        self.do_lacosmic = (self.band in ['CaHK', 'u'])

        # Should we cut the SE detections to only Gaia stars?
        # Helps on images with lots of cosmic rays and few sources (eg, CaHK)
        self.cut_to_gaia = (self.band in ['CaHK', 'u'])

        # Just create a constant PSF model, not a polynomially-varying one!
        self.constant_psfex = (self.band == 'CaHK')


    def run_se(self, imgfn, maskfn):
        # For some images (whyyyyy), interpolation just doesn't seem to be happening, so we
        # end up with zero-valued pixels in the image, that get sky-subtracted and turn into
        # highly-negative pixels in the vignets.
        # Try instantiating the sky model and explicitly patching the image before calling SE.
        from legacypipe.survey import create_temp
        tmpimgfn = create_temp(suffix='.fits')
        print('Patching image before running SE: image', imgfn, '--> patched file', tmpimgfn)
        sky = self.read_sky_model()
        mask = fitsio.read(maskfn)
        img = fitsio.read(imgfn)
        hdr = fitsio.read_header(imgfn)
        skyimg = np.zeros(img.shape, np.float32)
        sky.addTo(skyimg)
        from collections import Counter
        img[mask != 0] = skyimg[mask != 0]

        print('Image type:', img.dtype, 'min, median, max', img.min(), np.median(img.ravel()), img.max())
        
        # FIXME -- Replace the header with the WCS from our initial
        # WCS solution, so that the SE catalog's alpha_j2000,
        # delta_j2000 columns are correct??

        fitsio.write(tmpimgfn, img, header=hdr, clobber=True)
        print('Running SE on temp image and mask files', tmpimgfn, maskfn)

        if self.cut_to_gaia:
            tmpsefn = create_temp(suffix='.fits')
            filt_sefn = self.sefn
            self.sefn = tmpsefn
            print('Writing SE results to temp file', tmpsefn)
        super().run_se(tmpimgfn, maskfn)
        #print('cfht.py not removing patched image file', tmpimgfn)
        os.remove(tmpimgfn)

        if self.cut_to_gaia:
            # Filter SE detections to Gaia stars
            from astrometry.util.fits import fits_table
            from astrometry.util.util import Sip
            from astrometry.libkd.spherematch import match_radec
            from astrometry.util.file import trymakedirs
            from legacypipe.gaiacat import GaiaCatalog

            self.sefn = filt_sefn
            print('Filtering SE detections...')
            print('Reading temp SE catalog', tmpsefn)
            S = fits_table(tmpsefn, hdu=2, lower=False)
            print('Got', len(S), 'detections')
            wcs = Sip(self.wcs_initial_fn)

            gaiacat = GaiaCatalog()
            gaia = gaiacat.get_catalog_in_wcs(wcs)
            print('Got', len(gaia), 'Gaia stars within WCS')
            S.ra, S.dec = wcs.pixelxy2radec(S.X_IMAGE, S.Y_IMAGE)
            I,J,d = match_radec(S.ra, S.dec, gaia.ra, gaia.dec, 2.5/3600., nearest=True)
            print('Matched', len(I), 'Gaia stars')

            Fin = fitsio.FITS(tmpsefn, 'r')
            # Copy the first two HDUs unchanged...
            trymakedirs(self.sefn, dir=True)
            Fout = fitsio.FITS(self.sefn, 'rw', clobber=True)
            data = Fin[0].read()
            hdr  = Fin[0].read_header()
            Fout.write(data, header=hdr)
            data = Fin[1].read()
            hdr  = Fin[1].read_header()
            Fout.write(data, header=hdr, extname='LDAC_IMHEAD')
            #data = Fin[2].read()
            hdr  = Fin[2].read_header()
            Fin.close()
            Fout.close()
            S.cut(I)
            S.rename('ra',  'ALPHA_J2000')
            S.rename('dec', 'DELTA_J2000')
            S.writeto(self.sefn, append=True, header=hdr, extname='LDAC_OBJECTS')
            print('Wrote', len(I), 'filtered stars to', self.sefn)
            os.remove(tmpsefn)

    def get_psfex_conf(self):
        if self.constant_psfex:
            return '-PSFVAR_DEGREES 0 -VERBOSE_TYPE FULL'
        return super().get_psfex_conf()

    def read_scamp_wcs(self, hdr=None):
        from astrometry.util.util import wcs_pv2sip_hdr
        import tempfile

        print('Reading Scamp file', self.scamp_fn, 'HDU', self.hdu)
        lines = open(self.scamp_fn,'rb').readlines()
        lines = [line.strip() for line in lines]
        iline = 0
        header = []
        # find my HDU in the header
        for i in range(1, self.hdu+1):
            header = []
            while True:
                if iline >= len(lines):
                    raise RuntimeError('Failed to find HDU %i in Scamp header file %s' %
                                       (self.hdu, self.scamp_fn))
                line = lines[iline]
                header.append(line)
                iline += 1
                if line == b'END':
                    break

        # print('Keeping Scamp header:')
        # for line in header:
        #     print(line)

        # Write to a temp file and then read w/ fitsio!
        tmp = tempfile.NamedTemporaryFile(delete=False)
        preamble = [b'SIMPLE  =                    T / file does conform to FITS standard',
                    b'BITPIX  =                   16 / number of bits per data pixel',
                    b'NAXIS   =                    0 / number of data axes',
                    b'EXTEND  =                    T / FITS dataset may contain extensions'
                    ]
        hdrstr = b''.join([s + b' '*(80-len(s)) for s in preamble + header])
        tmp.write(hdrstr + b' '*(2880-(len(hdrstr)%2880)))
        tmp.close()
        scamp_hdr = fitsio.read_header(tmp.name)
        del tmp
        #print('Parsed scamp header:', scamp_hdr)

        # Read original WCS header
        if hdr is None:
            hdr = self.read_image_header()

        # Verify that we're looking at the right HDU (because the Scamp output format SUCKS)
        # by looking at how much the image center has moved in RA,Dec.
        from astrometry.util.starutil_numpy import arcsec_between
        if os.path.exists(self.wcs_initial_fn):
            print('Comparing Scamp solution to Astrometry.net')
            from astrometry.util.util import Sip
            oldwcs = Sip(self.wcs_initial_fn)
        else:
            print('Comparing Scamp solution against original image header')
            oldwcs = wcs_pv2sip_hdr(hdr)

        # print('Reading Scamp header: values of   original  -->  scamp:')
        # dist1 = np.hypot(hdr['CRPIX1'] - scamp_hdr['CRPIX1'], hdr['CRPIX2'] - scamp_hdr['CRPIX2'])
        # print('   CRPIX (%.1f, %.1f)  -->   (%.1f, %.1f)    dist %.1f pix' %
        #       (hdr['CRPIX1'], hdr['CRPIX2'], scamp_hdr['CRPIX1'], scamp_hdr['CRPIX2'], dist1))
        # dist2 = arcsec_between(hdr['CRVAL1'], hdr['CRVAL2'],
        #                        scamp_hdr['CRVAL1'], scamp_hdr['CRVAL2']) / 0.186
        # print('   CRVAL (%.4f, %.4f)  -->   (%.4f, %.4f)    dist %.1f pix' %
        #       (hdr['CRVAL1'], hdr['CRVAL2'], scamp_hdr['CRVAL1'], scamp_hdr['CRVAL2'], dist2))

        # Copy Scamp header cards in...
        for key in ['EQUINOX', 'RADESYS', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                    'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'PV1_0', 'PV1_1', 'PV1_2', 'PV1_4', 'PV1_5', 'PV1_6',
                    'PV1_7', 'PV1_8', 'PV1_9', 'PV1_10',
                    'PV2_0', 'PV2_1', 'PV2_2', 'PV2_4', 'PV2_5', 'PV2_6',
                    'PV2_7', 'PV2_8', 'PV2_9', 'PV2_10']:
            hdr[key] = scamp_hdr[key]
        wcs = wcs_pv2sip_hdr(hdr)

        #print('Old WCS:', oldwcs)
        #print('New (scamp) WCS:', wcs)
        xx,yy = 2112/2., 4644/2.
        r1,d1 = oldwcs.pixelxy2radec(xx, yy)
        r2,d2 =    wcs.pixelxy2radec(xx, yy)
        dist3 = arcsec_between(r1, d1, r2, d2)
        print('Image center distance: %.1f arcsec  (~ %.1f pixels)' % (dist3, dist3/0.186))
        if dist3 > 200.:
            raise RuntimeError('Scamp WCS is more than 200 arcsec away from initial astrometry')

        return wcs

    def compute_filenames(self):
        self.dqfn = None
        self.wtfn = None

    def set_calib_filenames(self):
        super().set_calib_filenames()
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        calname = self.name
        self.scamp_fn = os.path.join(calibdir, 'wcs-scamp', imgdir, basename + '-scamp.head')
        self.wcs_initial_fn = os.path.join(calibdir, 'wcs-initial', imgdir, basename,
                                           calname + '.wcs')
        self.lacosmic_fn = os.path.join(calibdir, 'lacosmic', imgdir, basename,
                                        calname + '-cr.fits')
        #if not os.path.exists(self.scamp_fn):
        #    print('Warning: Scamp header', self.scamp_fn, 'does not exist, using default WCS')

    def get_wcs(self, hdr=None):
        if self.scamp_wcs is not None:
            return self.scamp_wcs
        # Look for Scamp "head" file
        if os.path.exists(self.scamp_fn):
            # Load Scamp WCS
            self.scamp_wcs = self.read_scamp_wcs(hdr=hdr)
        if self.scamp_wcs is not None:
            return self.scamp_wcs

        if self.do_solve_field:
            if not os.path.exists(self.wcs_initial_fn):
                self.run_solve_field()
            from astrometry.util.util import Sip
            return Sip(self.wcs_initial_fn)

        return super().get_wcs(hdr=hdr)

    def run_solve_field(self):
        from pkg_resources import resource_filename
        from astrometry.util.file import trymakedirs
        from legacypipe.survey import create_temp
        # Initial astrometry -- using solve-field on the image
        dirname = resource_filename('legacypipe', 'data')
        configfn = os.path.join(dirname, 'an-cfht.cfg')
        primhdr = self.read_image_primary_header()
        hdr = self.read_image_header()
        r,d = self.get_radec_bore(primhdr)

        imgfn = self.imgfn
        ext = self.hdu
        tmpimgfn = None
        if self.do_lacosmic:
            print('Masking out CRs using Lacosmic map')
            tmpimgfn = create_temp(suffix='.fits')
            dq = self.read_dq()
            img = self.read_image()
            med = np.median(img[dq == 0])
            img[dq != 0] = med
            fitsio.write(tmpimgfn, img, clobber=True)
            imgfn = tmpimgfn
            ext = 0

        for ds in [2, 4]:
            args = ['--config', configfn,
                    '--downsample', ds,
                    '--objs', 130,
                    '--tweak-order', 1,
                    '--scale-low', self.pixscale * 0.8,
                    '--scale-high', self.pixscale * 1.2,
                    '--scale-units', 'app',
                    '--width', 2112, '--height', 4644,
                    '--no-plots',
                    '--no-remove-lines',
                    '--continue',
                    '--crpix-x', hdr['CRPIX1'],
                    '--crpix-y', hdr['CRPIX2'],
                    '--new-fits', 'none',
                    '--temp-axy',
                    '--solved', 'none',
                    '--match', 'none',
                    '--corr', 'none',
                    '--index-xyls', 'none',
                    '--rdls', 'none',
                    '--wcs', self.wcs_initial_fn,
                    '--extension', ext]
            if r is not None and d is not None:
                args.extend(['--ra', r, '--dec', d, '--radius', 5])
            print('Creating initial WCS using solve-field...')
            trymakedirs(self.wcs_initial_fn, dir=True)
            cmd = ' '.join([str(x) for x in ['solve-field'] + args + [imgfn]])
            print('Running:', cmd)
            rtn = os.system(cmd)
            print('solve-field return value:', rtn)
            if os.path.exists(self.wcs_initial_fn):
                break

        if tmpimgfn is not None:
            os.remove(tmpimgfn)

    def run_lacosmic(self):
        import lacosmic
        from astrometry.util.file import trymakedirs
        from legacypipe.bits import DQ_BITS

        img = self.read_image()
        print('run_lacosmic: got img, range', img.min(), img.max())
        dq = self.read_dq(use_lacosmic=False)
        # SATUR
        dq[img > self.get_satur()] |= DQ_BITS['satur']
        mask = (dq != 0)

        # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        ok = (dq[slice1] == 0) * (dq[slice2] == 0)
        mad = np.median(np.abs(img[slice1][ok] - img[slice2][ok]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)

        del dq

        err = np.empty(img.shape, np.float32)
        err[:,:] = sig1

        contrast = 2.
        threshold = 6.
        neighbor_threshold = 1.
        print('run_lacosmic: running lacosmic')
        _,crmask = lacosmic.lacosmic(img, contrast, threshold, neighbor_threshold,
                                     error=err, mask=mask)
        print('run_lacosmic: masked', np.sum(crmask), 'pixels')
        tmpfn = self.lacosmic_fn.replace('-cr.fits', '-cr-temp.fits')
        trymakedirs(self.lacosmic_fn, dir=True)
        fits = fitsio.FITS(tmpfn, 'rw', clobber=True)
        fits.write(crmask.astype(np.uint8), compress='rice')
        fits.close()
        os.rename(tmpfn, self.lacosmic_fn)
        print('Wrote', self.lacosmic_fn)

    def run_calibs(self, **kwargs):
        # Check for all zero image pixel values (eg 730710p.fits (XMM-u) [4] [CCD03])
        # and bail out early.
        img = self.read_image()
        if np.all(img == 0):
            print('All image pixel values are zero!')
            from legacypipe.utils import ZeroWeightError
            raise ZeroWeightError('All image pixels are zero in CFHT expnum %i ext %s' %
                                  (self.expnum, self.ccdname))
        mn = img.min()
        mx = img.max()
        print('Image range:', mn, mx)
        if mn == mx:
            print('All image pixels have the same value: %f!' % mn)
            from legacypipe.utils import ZeroWeightError
            raise ZeroWeightError('All image pixels have the same value in CFHT expnum %i ext %s' %
                                  (self.expnum, self.ccdname))
        if self.do_lacosmic:
            if not os.path.exists(self.lacosmic_fn):
                self.run_lacosmic()
        if self.do_solve_field:
            if not os.path.exists(self.wcs_initial_fn):
                self.run_solve_field()

        super().run_calibs(**kwargs)

    def get_crpixcrval(self, primhdr, hdr):
        wcs = self.get_wcs()
        p1,p2 = wcs.get_crpix()
        v1,v2 = wcs.get_crval()
        return p1,p2,v1,v2

    def get_cd_matrix(self, primhdr, hdr):
        wcs = self.get_wcs()
        return wcs.get_cd()

    # don't need overridden read_image_header
    def read_dq(self, header=False, use_lacosmic=None, **kwargs):
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

        if use_lacosmic is None:
            use_lacosmic = self.do_lacosmic
        if use_lacosmic:
            if not os.path.exists(self.lacosmic_fn):
                self.run_lacosmic()
            slc = kwargs.pop('slc', None)
            if slc is not None:
                crmask = fitsio.FITS(self.lacosmic_fn)[1][slc]
            else:
                crmask=  fitsio.read(self.lacosmic_fn, ext=1)
            dq[(crmask != 0)] |= DQ_BITS['cr']

        if header:
            dq = dq,hdr
        return dq

    def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
        from legacypipe.survey import create_temp
        tmpimgfn,_ = super().funpack_files(imgfn, maskfn, imghdu, maskhdu, todelete)
        img = fitsio.read(tmpimgfn)
        hdr = fitsio.read_header(tmpimgfn)
        print('Funpack_files: image minimum value %g, max %g' % (img.min(), img.max()), 'type', img.dtype)
        mask = (img < 0.5).astype(np.int16)
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpmaskfn)
        fitsio.write(tmpmaskfn, mask, clobber=True)

        img = img.astype(np.float32)
        if 'BZERO' in hdr:
            print('Removing BZERO =', hdr['BZERO'])
            hdr.delete('BZERO')
        if 'BSCALE' in hdr:
            print('Removing BSCALE =', hdr['BSCALE'])
            hdr.delete('BSCALE')
        print('SATUR is', hdr['SATURATE'], 'max val in image is', img.max())
        #
        fitsio.write(tmpimgfn, img, clobber=True, header=hdr)
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
            #print('Returning good image subregion', rtn)
            return rtn
        except:
            pass
        return nil

    def get_tractor_sky_model(self, img, goodpix):
        from legacypipe.jumpsky import JumpSky
        boxsize = self.splinesky_boxsize
        _,W = img.shape
        xbreak = W//2
        skyobj = JumpSky.BlantonMethod(img, goodpix, boxsize, xbreak)
        return skyobj
