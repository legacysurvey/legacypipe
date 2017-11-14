from __future__ import print_function

import os
import fitsio

import numpy as np

from legacypipe.image import LegacySurveyImage
from legacypipe.survey import LegacySurveyData

# From: http://www.noao.edu/noao/staff/fvaldes/CPDocPrelim/PL201_3.html
# 1   -- detector bad pixel           InstCal
# 1   -- detector bad pixel/no data   Resampled
# 1   -- No data                      Stacked
# 2   -- saturated                    InstCal/Resampled
# 4   -- interpolated                 InstCal/Resampled
# 16  -- single exposure cosmic ray   InstCal/Resampled
# 64  -- bleed trail                  InstCal/Resampled
# 128 -- multi-exposure transient     InstCal/Resampled 
CP_DQ_BITS = dict(badpix=1, satur=2, interp=4, cr=16, bleed=64,
                  trans=128,
                  edge = 256,
                  edge2 = 512,
                  ## masked by stage_mask_junk
                  longthin = 1024,
                  )

class CPImage(LegacySurveyImage):
    '''
    A mix-in class for common code between NOAO Community Pipeline-processed
    data from DECam and Mosaic3.
    '''

    def __init__(self, *args, **kwargs):
        super(CPImage, self).__init__(*args, **kwargs)
        '''
        Note, this assumes the "self.imgfn" parameter has been set; this can
        require the inheritance order and order of calling super.__init__()
        to be just right.
        '''
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_').replace(
            '_oki_','_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_').replace(
            '_oki_','_oow_')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue
            if fn.endswith('.fz'):
                fun = fn[:-3]
                if os.path.exists(fun):
                    print('Using      ', fun)
                    print('rather than', fn)
                    setattr(self, attr, fun)
                    fn = fun
            # Workaround: exposure numbers 330667 through 330890 at least have some of the
            # files named "v1" and some named "v2".  Try both.
            if 'v1' in fn:
                fnother = fn.replace('v1', 'v2')
                if os.path.exists(fnother):
                    print('Using', fnother, 'rather than', fn)
                    setattr(self, attr, fnother)
                    fn = fnother
            elif 'v2' in fn:
                fnother = fn.replace('v2', 'v1')
                if os.path.exists(fnother):
                    print('Using', fnother, 'rather than', fn)
                    setattr(self, attr, fnother)
                    fn = fnother

        expstr = '%08i' % self.expnum
        self.calname = '%s/%s/%s-%s-%s' % (expstr[:5], expstr, self.camera,
                                           expstr, self.ccdname)
        self.name = '%s-%s' % (expstr, self.ccdname)

        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn  = os.path.join(calibdir, 'se',    self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')

        self.merged_psffn = os.path.join(calibdir, 'psfex-merged', expstr[:5],
                                         '%s-%s.fits' % (self.camera, expstr))
        self.merged_splineskyfn = os.path.join(calibdir, 'splinesky-merged', expstr[:5],
                                         '%s-%s.fits' % (self.camera, expstr))

        self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        self.splineskyfn = os.path.join(calibdir, 'splinesky', self.calname + '.fits')
        self.dq_saturation_bits = CP_DQ_BITS['satur']

    def get_cacheable_filename_variables(self):
        '''
        These are names of self.X variables that are filenames that
        could be cached.
        '''
        keys = super(CPImage, self).get_cacheable_filename_variables()
        keys.extend(['dqfn', 'wtfn', 'psffn', 'merged_psffn',
                     'merged_splineskyfn', 'splineskyfn', 'skyfn',])
        return keys

    def check_image_header(self, imghdr):
        # check consistency between the CCDs table and the image header
        e = imghdr['EXTNAME']
        if e.strip() != self.ccdname.strip():
            print('WARNING: Expected header EXTNAME="%s" to match self.ccdname="%s", self.imgfn=%s' % (e.strip(), self.ccdname,self.imgfn))

    def get_wcs(self, hdr=None):
        from astrometry.util.util import wcs_pv2sip_hdr
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        if hdr is None:
            hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        # Correction: ccd ra,dec offsets from zeropoints/CCDs file
        dra,ddec = self.dradec
        # print('Applying astrometric zeropoint:', (dra,ddec))
        r,d = wcs.get_crval()
        wcs.set_crval((r + dra / np.cos(np.deg2rad(d)), d + ddec))
        wcs.version = ''
        phdr = self.read_image_primary_header()
        wcs.plver = phdr.get('PLVER', '').strip()
        return wcs

    def remap_dq_codes(self, dq):
        # Integer codes, not bit masks.
        dqbits = np.zeros(dq.shape, np.int16)
        '''
        1 = bad
        2 = no value (for remapped and stacked data)
        3 = saturated
        4 = bleed mask
        5 = cosmic ray
        6 = low weight
        7 = diff detect (multi-exposure difference detection from median)
        8 = long streak (e.g. satellite trail)
        '''
        dqbits[dq == 1] |= CP_DQ_BITS['badpix']
        dqbits[dq == 2] |= CP_DQ_BITS['badpix']
        dqbits[dq == 3] |= CP_DQ_BITS['satur']
        dqbits[dq == 4] |= CP_DQ_BITS['bleed']
        dqbits[dq == 5] |= CP_DQ_BITS['cr']
        dqbits[dq == 6] |= CP_DQ_BITS['badpix']
        dqbits[dq == 7] |= CP_DQ_BITS['trans']
        dqbits[dq == 8] |= CP_DQ_BITS['trans']
        return dqbits

    # A function that can be called by a subclasser's remap_invvar() method
    def remap_invvar_shotnoise(self, invvar, primhdr, img, dq):
        print('Remapping weight map for', self.name)
        const_sky = primhdr['SKYADU'] # e/s, Recommended sky level keyword from Frank 
        expt = primhdr['EXPTIME'] # s
        with np.errstate(divide='ignore'):
            var_SR = 1./invvar # e/s 
        var_Astro = np.abs(img - const_sky) / expt # e/s 
        wt = 1./(var_SR + var_Astro) # s/e

        # Zero out NaNs and masked pixels 
        wt[np.isfinite(wt) == False] = 0.
        wt[dq != 0] = 0.

        return wt
    
