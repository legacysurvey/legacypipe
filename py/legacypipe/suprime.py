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
        self.sig1_blanton = None
    
    def get_ps1_band(self):
        from legacypipe.ps1cat import ps1cat
        # Returns the integer index of the band in Pan-STARRS1 to use for an image in filter
        # self.band.
        # eg, g=0, r=1, i=2, z=3, Y=4
        # FIXME
        return ps1cat.ps1band['g']
    
    def read_image_primary_header(self, **kwargs):
        # SuprimeCam images have an empty primary header, with a bunch of duplicated cards
        # in the image HDUs, so we'll hack that here!
        self._primary_header = self.read_image_fits()[1].read_header()
        return self._primary_header

    def get_band(self, primhdr):
        # FILTER01, but not in the primary header!
        # read from first extension!
        #self.hdu = 1
        #imghdr = self.read_image_header()
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

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        #self.dqfn = self.imgfn.replace('p.fits.fz', 'p.weight.fits.fz')
        #assert(self.dqfn != self.imgfn)
        self.dqfn = None
        self.wtfn = None
        self.dq_hdu = 1

    def set_calib_filenames(self):
        super().set_calib_filenames()

        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        calname = self.name
        self.dqfn = os.path.join(calibdir, 'dq', imgdir, basename, calname + '-dq.fits.fz')
        
    def read_invvar(self, slc=None, **kwargs):
        if self.sig1 is None or self.sig1 == 0.:
            if self.sig1_blanton is None:
                img = self.read_image(**kwargs)
                # Estimate per-pixel noise via Blanton's 5-pixel MAD
                slice1 = (slice(0,-5,10),slice(0,-5,10))
                slice2 = (slice(5,None,10),slice(5,None,10))
                mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
                self.sig1_blanton = 1.4826 * mad / np.sqrt(2.)
                print('Computed sig1 by Blanton method:', self.sig1_blanton)
            sig1 = self.sig1_blanton
        else:
            from tractor import NanoMaggies
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            print('Suprime read_invvar: sig1 from CCDs file:', self.sig1, '-> image units', sig1)

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

    # def read_dq(self, header=None, **kwargs):
    #     dq = super().read_dq(header=header, **kwargs)
    #     if header:
    #         dq,hdr = dq
    #     # .weight.fits.fz files: 1 = good
    #     dq = 1 - dq
    #     if header:
    #         dq = dq,hdr
    #     return dq

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        invvar[dq > 0] = 0.

    #     import numpy as np
    #     from scipy.ndimage import label, find_objects
    #     from legacypipe.bits import DQ_BITS
    #     print('suprime fix_saturation.  sig1 =', self.sig1)
    #     print('image max:', max(img))
    #     #sat[i,j] = True
    #     H,W = img.shape
    #     # Find SATUR pixels, and mark as BLEED any vertically-connected runs of pixels
    #     # with value 2 sigma above the median!
    #     for col in range(W):
    #         pixcol = img[:,col]
    #         colsat = (pixcol > SATUR)
    #         i, = np.nonzero(colsat)
    #         if len(i) == 0:
    #             continue
    #         # MARK satur
    #         dq[i,col] |= DQ_BITS['satur']
    #         invvar[i,col] = 0.
    # 
    #         blobs,nblobs = label(colsat)
    #         slcs = find_objects(blobs)
    #         for sy, in slcs:
    #             # Downward
    #             for y in range(sy.start-1, -1, -1):
    #                 if pixcol[y] < BLEED:
    #                     # end of the run of BLEED pixels
    #                     break
    #                 if colsat[y]:
    #                     # hit another SATUR region
    #                     break
    #             # MARK IT!
    #             dq[y+1 : sy.start, col] |= DQ_BITS['bleed']
    #             invvar[y+1 : sy.start, col] = 0.
    # 
    #             # Upward
    #             for y in range(sy.stop, H):
    #                 if pixcol[y] < BLEED:
    #                     break
    #                 if colsat[y]:
    #                     break
    #             # MARK IT!
    #             dq[sy.stop : y-1, col] |= DQ_BITS['bleed']
    #             invvar[sy.stop : y-1, col] = 0.

    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1cat
        g_index = ps1cat.ps1band['g']
        r_index = ps1cat.ps1band['r']
        #i_index = ps1cat.ps1band['i']
        gmag = cat[:,g_index]
        rmag = cat[:,r_index]
        #imag = cat[:,i_index]
        colorterm = np.zeros(len(gmag))
        return colorterm

    def get_extension_list(self, debug=False):
        if debug:
            return [1]
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def override_ccd_table_types(self):
        return {'camera':'S10',
                'filter': 'S8',
                #'ccdname': 'S4',
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
        print('run_calibs(): dqfn = ', self.dqfn)
        if not os.path.exists(self.dqfn):
            import fitsio
            from scipy.ndimage import label, find_objects
            from legacypipe.bits import DQ_BITS
            from astrometry.util.file import trymakedirs

            print('suprime creating DQ map.')
            img = self.read_image()
            invvar = self.read_invvar()
            print('image max:', np.max(img))
            med = np.median(img[invvar > 0])
            print('image median:', med)
            print('ccdname', self.ccdname)
            # in image pixel units (not nanomaggies)
            sig1 = self.sig1_blanton
            assert(sig1 is not None)

            mskfn = self.imgfn.replace('p.fits.fz', 'p.weight.fits.fz')
            msk = self._read_fits(mskfn, self.hdu)
            # "weight.fits.fz" file has 1 = good; flip
            dq = np.zeros(img.shape, np.int16)
            dq[msk == 0] = DQ_BITS['badpix']

            # Around the margins, the image pixels are (very) negative, while the good regions
            # have offsets of, say, 200 or 400 counts.  Zero out the ivar in the margins!
            dq[img < 0] = DQ_BITS['badpix']
            
            SATUR = 30000
            BLEED = med + 2 * self.sig1
            H,W = img.shape
            # Find SATUR pixels, and mark as BLEED any vertically-connected runs of pixels
            # with value 2 sigma above the median!
            for col in range(W):
                pixcol = img[:,col]
                colsat = (pixcol > SATUR)
                i, = np.nonzero(colsat)
                if len(i) == 0:
                    continue
                # MARK satur
                dq[i,col] |= DQ_BITS['satur']
    
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
                    # MARK IT!
                    dq[y+1 : sy.start, col] |= DQ_BITS['bleed']
    
                    # Upward
                    for y in range(sy.stop, H):
                        if pixcol[y] < BLEED:
                            break
                        if colsat[y]:
                            break
                    # MARK IT!
                    dq[sy.stop : y-1, col] |= DQ_BITS['bleed']

            fn = self.dqfn
            tmpfn = os.path.join(os.path.dirname(fn),
                                 'tmp-'+os.path.basename(fn))
            dirnm = os.path.dirname(tmpfn)
            trymakedirs(dirnm)
            F = fitsio.FITS(tmpfn, 'rw', clobber=True)
            F.write(dq, extname=self.ccdname, compress='HCOMPRESS', tile_dims=(100,100))
            F.close()
            os.rename(tmpfn, fn)
            print('Wrote', fn)

        super().run_calibs(**kwargs)

    # Flip the weight map (1=good) to a flag map (1=bad)
    # def run_se(self, imgfn, maskfn):
    #     import fitsio
    #     import os
    #     from collections import Counter
    #     from legacypipe.survey import create_temp
    #     tmpmaskfn  = create_temp(suffix='.fits')
    #     print('run_se: maskfn', maskfn)
    #     #mask,hdr = self.read_dq(maskfn, header=True)
    #     mask,hdr = self.read_dq(header=True)
    #     print('Mask values:', Counter(mask.ravel()))
    #     fitsio.write(tmpmaskfn, mask, header=hdr)
    #     R = super().run_se(imgfn, tmpmaskfn)
    #     os.unlink(tmpmaskfn)
    #     return R

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.2

