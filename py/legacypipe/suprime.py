import os
import numpy as np
from legacypipe.image import LegacySurveyImage

import logging
logger = logging.getLogger('legacypipe.suprime')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

'''
A class to handle images from the SuprimeCam camera (old prime-focus
camera on the Subaru telescope), specifically a data set using
intermediate-band filters that were used to observe the COSMOS field,
which we are using for DESI-2 Lyman-alpha-emitter selection studies.

Currently, this class only handles the intermediate-band filters that
are within the g broad-band filter!
'''
class SuprimeImage(LegacySurveyImage):

    zp0 = {
        'I-A-L427': 25,
        'I-A-L464': 25,
        'I-A-L484': 25,
        'I-A-L505': 25,
        'I-A-L527': 25,
    }

    k_ext = {
        'I-A-L427': 0.173,   # made up!
        'I-A-L464': 0.173,
        'I-A-L484': 0.173,
        'I-A-L505': 0.173,
        'I-A-L527': 0.173,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # "sig1" is in nanomaggy units, but at some points we also
        # need it in raw image units, which is misnamed sig1_blanton
        # here!
        self.sig1_blanton = None

    def get_ps1_band(self):
        from legacypipe.ps1cat import ps1cat
        # Returns the integer index of the band in Pan-STARRS1 to use
        # for an image in filter self.band.  eg, g=0, r=1, i=2, z=3,
        # Y=4.  Here, we're assuming g band.
        return ps1cat.ps1band['g']

    def read_image_primary_header(self, **kwargs):
        # SuprimeCam images have an empty primary header, with a bunch
        # of duplicated cards in the image HDUs, so we'll hack that
        # here!
        self._primary_header = self.read_image_fits()[1].read_header()
        return self._primary_header

    def get_band(self, primhdr):
        # FILTER01, but not in the primary header!
        # read from first extension!
        band = primhdr['FILTER01']
        band = band.split()[0]
        return band

    def get_propid(self, primhdr):
        return primhdr.get('PROP-ID', '')

    def get_expnum(self, primhdr):
        s = primhdr['EXP-ID']
        s = s.replace('SUPE','')
        return int(s, 10)

    def get_mjd(self, primhdr):
        return primhdr.get('MJD-STR')

    def get_ccdname(self, primhdr, hdr):
        # DET-ID  =                    0 / ID of the detector used for this data
        return 'det%i' % hdr['DET-ID']

    def check_image_header(self, imghdr):
        # check consistency between the CCDs table and the image header
        e = 'det%i' % imghdr['DET-ID']
        if e.strip() != self.ccdname.strip():
            warnings.warn('Expected "det" + header DET-ID="%s" to match self.ccdname="%s", self.imgfn=%s' % (e.strip(), self.ccdname, self.imgfn))

    def compute_filenames(self):
        # Compute data quality and weight-map filenames... we have the
        # "p.weight.fits.fz" files that have partial DQ information,
        # but we create our own "dq" calibration product that
        # incorporates saturation and bleed trails.
        self.dqfn = None
        self.wtfn = None
        self.dq_hdu = 1

    def set_calib_filenames(self):
        super().set_calib_filenames()
        # Add our custom "dq" product.
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        calname = self.name
        self.dqfn = os.path.join(calibdir, 'dq', imgdir, basename,
                                 calname + '-dq.fits.fz')

    def read_invvar(self, slc=None, **kwargs):
        if self.sig1 is None or self.sig1 == 0.:
            if self.sig1_blanton is None:
                img = self.read_image(**kwargs)
                # Estimate per-pixel noise via Blanton's 5-pixel MAD
                slice1 = (slice(0,-5,10),slice(0,-5,10))
                slice2 = (slice(5,None,10),slice(5,None,10))
                mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
                self.sig1_blanton = 1.4826 * mad / np.sqrt(2.)
                debug('Computed sig1 by Blanton method:', self.sig1_blanton)
            sig1 = self.sig1_blanton
        else:
            from tractor import NanoMaggies
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            debug('read_invvar: sig1 from CCDs file:', self.sig1, '-> image units', sig1)

        if slc is None:
            shape = self.shape
        else:
            sy,sx = slc
            W = sx.stop - sx.start
            H = sy.stop - sy.start
            shape = (H,W)

        iv = np.empty(shape, np.float32)
        iv[:,:] = 1./sig1**2
        return iv

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        if dq is not None:
            invvar[dq > 0] = 0.

    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1cat
        g_index = ps1cat.ps1band['g']
        #r_index = ps1cat.ps1band['r']
        i_index = ps1cat.ps1band['i']
        gmag = cat[:,g_index]
        #rmag = cat[:,r_index]
        imag = cat[:,i_index]
        gi = gmag - imag
        colorterm = np.zeros(len(gmag))

        coeffs = {
            'I-A-L427': [-0.7496, 0.8574,-0.1317],
            'I-A-L464': [-0.2358, 1.0048,-1.4035, 0.7257,-0.1200],
            'I-A-L484': [ 0.2545,-0.6517, 0.4007,-0.0664],
            'I-A-L505': [ 0.2422,-1.1041, 1.4401,-0.6947, 0.1103],
            'I-A-L527': [ 0.2007,-0.1745,-0.0020],
        }[band]

        '''
        coeffs = {
            ('I-A-L427', 'det0'): [-0.28424705, 0.82685339, -0.11524449],
            ('I-A-L427', 'det1'): [-0.75325532, 0.85490214, -0.12558040],
            ('I-A-L427', 'det2'): [-0.75637723, 0.79008085, -0.10552436],
            ('I-A-L427', 'det3'): [-0.78159808, 0.80512430, -0.11295731],
            ('I-A-L427', 'det4'): [-0.71297356, 0.79428127, -0.10296920],
            ('I-A-L427', 'det5'): [-0.84320759, 0.87572098, -0.13101296],
            ('I-A-L427', 'det6'): [-0.68329719, 0.78425943, -0.09968866],
            ('I-A-L427', 'det7'): [-0.82159570, 0.89193908, -0.13738897],
            ('I-A-L427', 'det8'): [-0.68616224, 0.83322286, -0.12044040],
            ('I-A-L427', 'det9'): [-0.74135645, 0.85037165, -0.12584055],

            ('I-A-L464', 'det0'): [ 0.16721592,  1.0240256, -1.4248714, 0.73881510, -0.12151509],
            ('I-A-L464', 'det1'): [-0.25788172,  1.0796685, -1.4960659, 0.77449375, -0.12874674],
            ('I-A-L464', 'det2'): [-0.19818093, 0.77259546, -1.1509697, 0.61700267, -0.10377371],
            ('I-A-L464', 'det3'): [-0.23667510, 0.81385095, -1.1822467, 0.62427989, -0.10394285],
            ('I-A-L464', 'det4'): [-0.28498758,  1.1403747, -1.6122600, 0.85914350, -0.14853811],
            ('I-A-L464', 'det5'): [-0.28369011,  1.0125591, -1.4419947, 0.75356121, -0.12527917],
            ('I-A-L464', 'det6'): [-0.16343730, 0.85664462, -1.2338599, 0.64654038, -0.10641947],
            ('I-A-L464', 'det7'): [-0.31020500,  1.1510306, -1.5828711, 0.81760253, -0.13634334],
            ('I-A-L464', 'det8'): [-0.14667440, 0.82502549, -1.1378496, 0.57465170, -0.09050647],
            ('I-A-L464', 'det9'): [-0.22115253, 0.96904387, -1.3394605, 0.68545881, -0.11148692],

            ('I-A-L484', 'det0'): [0.56623436, -0.46684735, 0.28723995, -0.042132945],
            ('I-A-L484', 'det1'): [0.31068532, -0.79086707, 0.49918099, -0.087169289],
            ('I-A-L484', 'det2'): [0.22405387, -0.69220742, 0.44491582, -0.077782046],
            ('I-A-L484', 'det3'): [0.22615498, -0.70722840, 0.44299891, -0.075952822],
            ('I-A-L484', 'det4'): [0.17879048, -0.54449126, 0.34089165, -0.054861936],
            ('I-A-L484', 'det5'): [0.20155096, -0.71276788, 0.45891697, -0.080426247],
            ('I-A-L484', 'det6'): [0.16120666, -0.40949649, 0.24023223, -0.031625251],
            ('I-A-L484', 'det7'): [0.11763535, -0.46142051, 0.28414022, -0.042724078],
            ('I-A-L484', 'det8'): [0.18347510, -0.36264566, 0.20113764, -0.022468228],
            ('I-A-L484', 'det9'): [0.17453148, -0.46639761, 0.27986103, -0.039640822],

            ('I-A-L505', 'det0'): [ 0.50373080, -0.8559560, 1.2598590, -0.64413976, 0.10739586],
            ('I-A-L505', 'det1'): [ 0.32277551, -1.4659562, 1.9486049, -0.98047265, 0.16476331],
            ('I-A-L505', 'det2'): [ 0.17912342, -1.0843317, 1.4304244, -0.69948633, 0.11379497],
            ('I-A-L505', 'det3'): [ 0.11757860, -0.8626978, 1.1161089, -0.52100995, 0.07913526],
            ('I-A-L505', 'det4'): [ 0.27868774, -1.3850005, 1.7415611, -0.81373014, 0.12612835],
            ('I-A-L505', 'det5'): [ 0.04673695, -0.6334728, 0.8515035, -0.40652864, 0.06270727],
            ('I-A-L505', 'det6'): [ 0.18423355, -0.9538490, 1.3373032, -0.66530996, 0.10797023],
            ('I-A-L505', 'det7'): [ 0.12444852, -0.8564716, 1.1346299, -0.53790123, 0.08276072],
            ('I-A-L505', 'det8'): [ 0.21130280, -0.9415645, 1.3139283, -0.65251298, 0.10571869],
            ('I-A-L505', 'det9'): [ 0.10681874, -0.6856443, 0.9489984, -0.45276313, 0.06946127],

            ('I-A-L527', 'det0'): [ 0.53620829, -0.12187766, -0.014672495],
            ('I-A-L527', 'det1'): [ 0.17815916, -0.16554206, -0.001510648],
            ('I-A-L527', 'det2'): [ 0.10282139, -0.09439950, -0.020957809],
            ('I-A-L527', 'det3'): [ 0.09833594, -0.10809471, -0.019701545],
            ('I-A-L527', 'det4'): [ 0.12965203, -0.10689139, -0.020319229],
            ('I-A-L527', 'det5'): [ 0.05744854, -0.07383463, -0.029106456],
            ('I-A-L527', 'det6'): [ 0.19910593, -0.15223658, -0.005140478],
            ('I-A-L527', 'det7'): [ 0.07660945, -0.07283587, -0.027140544],
            ('I-A-L527', 'det8'): [ 0.20662955, -0.12101095, -0.017464684],
            ('I-A-L527', 'det9'): [ 0.18526426, -0.15797659, -0.003059741],

        }[band, self.ccdname]
        '''
        colorterm = np.zeros(len(gmag))
        for power,coeff in enumerate(coeffs):
            colorterm += coeff * gi**power
            #colorterm += coeff * gr**power

        return colorterm

    def get_extension_list(self, debug=False):
        if debug:
            return [1]
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def get_good_image_subregion(self):
        x0,x1,y0,y1 = None,None,None,None
        # Clip 50 pixels off the left/right sides of images to avoid biases near the ragged edges
        if self.ccdname in ['det0', 'det5', 'det6', 'det7', 'det8']:
            x0 = 50
        elif self.ccdname in ['det1', 'det2', 'det3', 'det4', 'det9']:
            x1 = 2048-50
        return x0,x1,y0,y1

    def override_ccd_table_types(self):
        return {'camera':'S10',
                'filter': 'S8',
                }

    def get_fwhm(self, primhdr, imghdr):
        # If PsfEx file exists, read FWHM from there
        if not hasattr(self, 'merged_psffn'):
            return super().get_fwhm(primhdr, imghdr)
        psf = self.read_psf_model(0, 0, pixPsf=True)
        fwhm = psf.fwhm
        return fwhm

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    def run_calibs(self, **kwargs):
        if not os.path.exists(self.dqfn):
            import fitsio
            from scipy.ndimage import label, find_objects
            from legacypipe.bits import DQ_BITS
            from astrometry.util.file import trymakedirs
            debug('Creating DQ map.')
            img = self.read_image()
            invvar = self.read_invvar()
            debug('image max:', np.max(img))
            med = np.median(img[invvar > 0])
            debug('image median:', med)
            # in image pixel units (not nanomaggies)
            sig1 = self.sig1_blanton
            assert(sig1 is not None)

            mskfn = self.imgfn.replace('p.fits.fz', 'p.weight.fits.fz')
            msk = self._read_fits(mskfn, self.hdu)
            # "weight.fits.fz" file has 1 = good; flip
            dq = np.zeros(img.shape, np.int16)
            dq[msk == 0] = DQ_BITS['badpix']

            # Around the margins, the image pixels are (very)
            # negative, while the good regions have offsets of, say,
            # 200 or 400 counts.  Zero out the ivar in the margins!
            dq[img < 0] = DQ_BITS['badpix']

            SATUR = 30000
            BLEED = med + 2 * self.sig1
            H,W = img.shape
            # Find SATUR pixels, and mark as BLEED any
            # vertically-connected runs of pixels with value 2 sigma
            # above the median!
            for col in range(W):
                pixcol = img[:,col]
                colsat = (pixcol > SATUR)
                i, = np.nonzero(colsat)
                if len(i) == 0:
                    continue
                # MARK satur
                dq[i,col] |= DQ_BITS['satur']
                # Find connected BLEED pixels
                blobs,nblobs = label(colsat)
                slcs = find_objects(blobs)
                for sy, in slcs:
                    # Downward
                    for y in range(sy.start-1, -1, -1):
                        if pixcol[y] < BLEED:
                            # end of the run of BLEED pixels
                            break
                        if colsat[y]:
                            # hit another SATUR region
                            break
                    # Mark it!
                    dq[y+1 : sy.start, col] |= DQ_BITS['bleed']
                    # Upward
                    for y in range(sy.stop, H):
                        if pixcol[y] < BLEED:
                            break
                        if colsat[y]:
                            break
                    # Mark it!
                    dq[sy.stop : y-1, col] |= DQ_BITS['bleed']

            fn = self.dqfn
            tmpfn = os.path.join(os.path.dirname(fn),
                                 'tmp-'+os.path.basename(fn))
            dirnm = os.path.dirname(tmpfn)
            trymakedirs(dirnm)
            F = fitsio.FITS(tmpfn, 'rw', clobber=True)
            F.write(dq, extname=self.ccdname, compress='HCOMPRESS',
                    tile_dims=(100,100))
            F.close()
            os.rename(tmpfn, fn)
            info('Wrote', fn)
        super().run_calibs(**kwargs)

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.2
