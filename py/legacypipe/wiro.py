import os
from datetime import datetime
import numpy as np

from legacypipe.image import LegacySurveyImage

import logging
logger = logging.getLogger('legacypipe.wiro')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class WiroImage(LegacySurveyImage):

    zp0 = dict(
        g = 25.0,
        NB_A = 25.0,
        NB_B = 25.0,
        NB_C = 25.0,
        NB_D = 25.0,
        NB_E = 25.0,
        NB_F = 25.0,
        )

    k_ext = dict(
        g = 0.173,
        NB_A = 0.173,
        NB_B = 0.173,
        NB_C = 0.173,
        NB_D = 0.173,
        NB_E = 0.173,
        NB_F = 0.173,
    )

    splinesky_boxsize = 256

    def __init__(self, survey, ccd, image_fn=None, image_hdu=0, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        self.dq_hdu = 1
        self.wt_hdu = 2
        self.SATUR = 32700

    def set_ccdzpt(self, ccdzpt):
        # Adjust zeropoint for exposure time
        self.ccdzpt = ccdzpt + 2.5 * np.log10(self.exptime)

    def get_band(self, primhdr):
        f = primhdr['FILTER']
        filtmap = {
            'Filter 1: g 1736'  : 'g',
            'Filter 2: C 14859' : 'NB_C',
            'Filter 3: D 27981' : 'NB_D',
            'Filter 4: E 41102' : 'NB_E',
            'Filter 5: A 54195' : 'NB_A',
        }
        # ASSUME that the filter is one of the above!
        return filtmap[f]

    def get_radec_bore(self, primhdr):
        # Some TELDEC header cards (eg 20221030/a276) have a bug:
        # TELDEC  = '-4:-50:-23.-9'
        try:
            return super.get_radec_bore(primhdr)
        except:
            return None,None

    def get_expnum(self, primhdr):
        d = self.get_date(primhdr)
        expnum = d.second + 100*(d.minute + 100*(d.hour + 100*(d.day + 100*(d.month + 100*d.year))))
        return expnum

    def get_mjd(self, primhdr):
        from astrometry.util.starutil_numpy import datetomjd
        d = self.get_date(primhdr)
        return datetomjd(d)

    def get_date(self, primhdr):
        date = primhdr['DATE-OBS']
        # DATE-OBS= '2022-10-04T05:20:19.335'
        return datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")

    def get_camera(self, primhdr):
        cam = super().get_camera(primhdr)
        cam = {'wiroprime':'wiro'}.get(cam, cam)
        return cam

    def get_ccdname(self, primhdr, hdr):
        # return 'CCD'
        return ''

    def get_pixscale(self, primhdr, hdr):
        return 0.58

    @classmethod
    def get_nominal_pixscale(cls):
        return 0.58

    def get_fwhm(self, primhdr, imghdr):
        # If PsfEx file exists, read FWHM from there
        if not hasattr(self, 'merged_psffn'):
            return super().get_fwhm(primhdr, imghdr)
        psf = self.read_psf_model(0, 0, pixPsf=True)
        fwhm = psf.fwhm
        return fwhm

    def get_gain(self, primhdr, hdr):
        # from https://iopscience.iop.org/article/10.1088/1538-3873/128/969/115003/ampdf
        return 2.6

    def get_object(self, primhdr):
        return primhdr.get('OBJNAME', '')

    def compute_filenames(self):
        # Masks and weight-maps are in HDUs following the image
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    def get_extension_list(self, debug=False):
        return [0]

    def read_invvar(self, **kwargs):
        # The reduced WIRO images have an Uncertainty HDU, but this only counts dark current
        # and readout noise only.
        img = self.read_image(**kwargs)
        if self.sig1 is None or self.sig1 == 0.:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            self.sig1 = sig1
            print('Computed sig1 by Blanton method:', self.sig1)
        else:
            from tractor import NanoMaggies
            print('WIRO read_invvar: sig1 from CCDs file:', self.sig1)
            # sig1 in the CCDs file is in nanomaggy units --
            # but here we need to return in image units.
            zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
            sig1 = self.sig1 * zpscale
            print('scaled to image units:', sig1)
        iv = np.empty_like(img)
        iv[:,:] = 1./sig1**2
        return iv

    def fix_saturation(self, img, dq, invvar, primhdr, imghdr, slc):
        # SATURATE header keyword is ~65536, but actual saturation in the images is
        # 32760.
        I,J = np.nonzero(img > self.SATUR)
        from legacypipe.bits import DQ_BITS
        if len(I):
            dq[I,J] |= DQ_BITS['satur']
            invvar[I,J] = 0

    def set_calib_filenames(self):
        super().set_calib_filenames()
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        calname = self.name
        self.wcs_initial_fn = os.path.join(calibdir, 'wcs-initial', imgdir, basename,
                                           calname + '.wcs')
        self.wcs_final_fn = os.path.join(calibdir, 'wcs-final', imgdir, basename,
                                         calname + '.wcs')

    def get_wcs(self, hdr=None):
        from astrometry.util.util import Sip
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        fn = self.wcs_initial_fn
        print('Initial WCS filename:', fn, 'exists?', os.path.exists(fn))
        fn = self.wcs_final_fn
        print('Final WCS filename:', fn, 'exists?', os.path.exists(fn))
        if os.path.exists(fn):
            return Sip(fn)
        return Sip(self.wcs_initial_fn)

    def get_crpixcrval(self, primhdr, hdr):
        wcs = self.get_wcs()
        p1,p2 = wcs.get_crpix()
        v1,v2 = wcs.get_crval()
        return p1,p2,v1,v2

    def get_cd_matrix(self, primhdr, hdr):
        wcs = self.get_wcs()
        return wcs.get_cd()

    def get_ps1_band(self):
        # Returns the integer index of the band in Pan-STARRS1 to use for an image in filter
        # self.band.
        # eg, g=0, r=1, i=2, z=3, Y=4
        # A known filter?
        from legacypipe.ps1cat import ps1cat
        if self.band in ps1cat.ps1band:
            return ps1cat.ps1band[self.band]
        # Narrow-band filters -- calibrate to PS1 g band.
        return dict(
            NB_A = 0,
            NB_B = 0,
            NB_C = 0,
            NB_D = 0,
            NB_E = 0,
            NB_F = 0,
            )[self.band]

    def get_photometric_calibrator_cuts(self, name, cat):
        good = super().get_photometric_calibrator_cuts(name, cat)
        if self.band == 'NB_C' and name == 'ps1':
            grcolor= cat.median[:,0] - cat.median[:,1]
            good *= (grcolor > 0.2) * (grcolor < 1.0)
        return good

    def colorterm_ps1_to_observed(self, cat, band):
        from legacypipe.ps1cat import ps1cat
        # See, eg, ps1cat.py's ps1_to_decam.
        # "cat" is a table of PS1 stars;
        # Grab the g-i color:

        # Arjun's color terms are polynomial (well, linear!) in g-r.
        g_index = ps1cat.ps1band['g']
        r_index = ps1cat.ps1band['r']
        #i_index = ps1cat.ps1band['i']
        gmag = cat[:,g_index]
        rmag = cat[:,r_index]
        #imag = cat[:,i_index]
        #gi = gmag - imag
        gr = gmag - rmag

        # NB_A : -23.5806,  0.9274
        # NB_C : -22.7565,  1.0497
        # NB_D : -23.1234,  0.8103
        # NG_E : -23.1617,  1.0552
        # WIRO_g :-25.0186,  0.1758
        
        coeffs = dict(
            g = [ 0.,  0.1758],
            NB_A = [ 0., 0.9274 ],
            NB_B = [ 0., ],
            NB_C = [ 0., 1.0497 ],
            NB_D = [ 0., 0.8103 ],
            NB_E = [ 0., 1.0552 ],
            NB_F = [ 0., ],
            )[band]
        colorterm = np.zeros(len(gmag))
        for power,coeff in enumerate(coeffs):
            #colorterm += coeff * gi**power
            colorterm += coeff * gr**power
        return colorterm

    def zeropointing_completed(self, annfn, photomfn, ann, photom, hdr):
        if not os.path.exists(self.wcs_final_fn):
            from pkg_resources import resource_filename
            from astrometry.util.file import trymakedirs
            from astrometry.util.fits import fits_table
            from legacypipe.survey import create_temp
            # Final astrometry -- using solve-field on the "photom" results!
            dirname = resource_filename('legacypipe', 'data')
            configfn = os.path.join(dirname, 'an-wiro.cfg')
            primhdr = self.read_image_primary_header()
            r,d = self.get_radec_bore(primhdr)
            pixscale = self.get_pixscale(None,None)
            args = ['--config', configfn,
                    '--tweak-order', 3,
                    '--scale-low', pixscale * 0.8,
                    '--scale-high', pixscale * 1.2,
                    '--scale-units', 'app',
                    '--width', 4096, '--height', 4096,
                    '--no-plots',
                    '--no-remove-lines',
                    '--continue',
                    '--crpix-center',
                    '--new-fits', 'none',
                    '--temp-axy',
                    '--solved', 'none',
                    '--match', 'none',
                    '--corr', 'none',
                    '--index-xyls', 'none',
                    '--rdls', 'none',
                    '--wcs', self.wcs_final_fn]
            if r is not None and d is not None:
                args.extend(['--ra', r, '--dec', d, '--radius', 5])
            print('Creating final WCS using solve-field...')
            trymakedirs(self.wcs_final_fn, dir=True)
            tmpxy = create_temp(suffix='.fits')
            xy = fits_table()
            xy.x = 1. + photom.x_fit
            xy.y = 1. + photom.y_fit
            xy.mag = photom.psfmag
            xy.cut(np.argsort(xy.mag))
            xy.cut(xy.mag != 0)
            xy.writeto(tmpxy)

            cmd = ' '.join([str(x) for x in ['solve-field'] + args + [tmpxy]])
            print('Running:', cmd)
            os.system(cmd)
            os.unlink(tmpxy)

    def run_calibs(self, **kwargs):
        if not os.path.exists(self.wcs_initial_fn):
            from pkg_resources import resource_filename
            from astrometry.util.file import trymakedirs
            # Initial astrometry -- using solve-field on the image??
            dirname = resource_filename('legacypipe', 'data')
            configfn = os.path.join(dirname, 'an-wiro.cfg')
            primhdr = self.read_image_primary_header()
            r,d = self.get_radec_bore(primhdr)
            args = ['--config', configfn,
                    '--tweak-order', 3,
                    '--scale-low', self.pixscale * 0.8,
                    '--scale-high', self.pixscale * 1.2,
                    '--scale-units', 'app',
                    '--width', 4096, '--height', 4096,
                    '--no-plots',
                    '--no-remove-lines',
                    '--continue',
                    '--crpix-center',
                    '--new-fits', 'none',
                    '--temp-axy',
                    '--solved', 'none',
                    '--match', 'none',
                    '--corr', 'none',
                    '--index-xyls', 'none',
                    '--rdls', 'none',
                    '--wcs', self.wcs_initial_fn]
            if r is not None and d is not None:
                args.extend(['--ra', r, '--dec', d, '--radius', 5])
            print('Creating initial WCS using solve-field...')
            trymakedirs(self.wcs_initial_fn, dir=True)
            cmd = ' '.join([str(x) for x in ['solve-field'] + args + [self.imgfn]])
            print('Running:', cmd)
            os.system(cmd)
        super().run_calibs(**kwargs)

    def run_se(self, imgfn, maskfn):
        import fitsio
        from collections import Counter
        # Add SATUR to the mask before running SE
        from legacypipe.survey import create_temp
        tmpmaskfn  = create_temp(suffix='.fits')
        img = fitsio.read(imgfn)
        mask,hdr = fitsio.read(maskfn, header=True)
        print('Mask values:', Counter(mask.ravel()))
        mask[img > self.SATUR] |= 1
        fitsio.write(tmpmaskfn, mask, header=hdr)
        R = super().run_se(imgfn, tmpmaskfn)
        os.unlink(tmpmaskfn)
        return R

    def run_sky(self, **kwargs):
        # Sky variations are large enough that we don't want to do simple
        # brightness masking of pixels in this image!  Be sure to set
        # --blob-mask-dir -- eg to the DR10 results.
        kwargs.update(boxcar_mask=False)
        return super().run_sky(**kwargs)

    def check_image_header(self, imghdr):
        pass

if __name__ == '__main__':
    from legacypipe.survey import LegacySurveyData
    from astrometry.util.fits import fits_table
    survey = LegacySurveyData(survey_dir='wiro-dir')
    wiro = WiroImage(survey, None, 'wiro/20221031/a114_zbf.fit')
    annfn = 'wiro-dir/zpt/wiro/20221031/a114_zbf.fit-annotated.fits'
    photomfn = 'wiro-dir/zpt/wiro/20221031/a114_zbf.fit-photom.fits'
    ann = fits_table(annfn)
    photom = fits_table(photomfn)
    hdr = ann.get_header()
    wiro.zeropointing_completed(annfn, photomfn, ann, photom, hdr)
