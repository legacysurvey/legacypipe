from __future__ import print_function

import os
import fitsio

import numpy as np
from glob import glob

from legacypipe.cpimage import CPImage
from legacypipe.survey import LegacySurveyData    

class MosaicImage(CPImage):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOAO Community Pipeline.
    '''

    mjd_third_pixel_fixed = 57674.

    # this is defined here for testing purposes (to handle small images)
    splinesky_boxsize = 512

    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)
        # convert FWHM into pixel units
        #self.fwhm /= self.pixscale

    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        print('Reading weight map image', self.wtfn, 'ext', self.hdu)
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        return invvar

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def run_calibs(self, psfex=True, sky=True, se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False, git_version=None,
                   splinesky=False):
        se = False
        if psfex and os.path.exists(self.psffn) and (not force):
            if self.check_psf(self.psffn):
                psfex = False
        # dependency
        if psfex:
            se = True

        if se and os.path.exists(self.sefn) and (not force):
            if self.check_se_cat(self.sefn):
                se = False
        # dependency
        if se:
            funpack = True

        if sky and (not force) and (
            (os.path.exists(self.skyfn) and not splinesky) or
            (os.path.exists(self.splineskyfn) and splinesky)):
            fn = self.skyfn
            if splinesky:
                fn = self.splineskyfn

            if os.path.exists(fn):
                try:
                    hdr = fitsio.read_header(fn)
                except:
                    print('Failed to read sky file', fn, '-- deleting')
                    os.unlink(fn)
            if os.path.exists(fn):
                print('File', fn, 'exists -- skipping')
                sky = False

        if just_check:
            return (se or psfex or sky)

        todelete = []
        if funpack:
            # The image & mask files to process (funpacked if necessary)
            imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn, self.hdu, todelete)
        else:
            imgfn,maskfn = self.imgfn,self.dqfn
    
        if se:
            self.run_se('mosaic', imgfn, maskfn)
        if psfex:
            self.run_psfex('mosaic')
        if sky:
            self.run_sky('mosaic', splinesky=splinesky, git_version=git_version)

        for fn in todelete:
            os.unlink(fn)


