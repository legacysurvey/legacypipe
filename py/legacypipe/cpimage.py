from __future__ import print_function

import os
import fitsio

import numpy as np

from astrometry.util.util import wcs_pv2sip_hdr

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
        self.calname = '%s/%s/decam-%s-%s' % (expstr[:5], expstr, expstr, self.ccdname)
        self.name = '%s-%s' % (expstr, self.ccdname)

        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        self.splineskyfn = os.path.join(calibdir, 'splinesky', self.calname + '.fits')
        self.dq_saturation_bits = CP_DQ_BITS['satur']

    def check_image_header(self, imghdr):
        # check consistency... something of a DR1 hangover
        e = imghdr['EXTNAME']
        assert(e.strip() == self.ccdname.strip())

    def get_wcs(self,hdr=None):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        if hdr is None:
            hdr = self.read_image_header()
        #if self.camera == '90prime':
            # WCS is in myriad of formats
            # Don't support TNX yet, use TAN for now
        #    hdr = self.read_image_header()
        #    hdr['CTYPE1'] = 'RA---TAN'
        #    hdr['CTYPE2'] = 'DEC--TAN'
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        # Correctoin: ccd,ccdraoff, decoff from zeropoints file
        dra,ddec = self.dradec
        print('Applying astrometric zeropoint:', (dra,ddec))
        r,d = wcs.get_crval()
        wcs.set_crval((r + dra, d + ddec))
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

def newWeightMap(wtfn=None,imgfn=None,dqfn=None,clobber=False):
    '''MZLS or BASS
    Converts the oow weight map: 1 / var(sky, read)
    to a 1 / var(sky, read, astrophysical) weight map,\
    Creates the new map if it doesn't exist

    Returns: new_wtfn
    '''
    from astropy.io import fits
    newfn= wtfn.replace('oow','oow_wshot')
    make_wtmap=True
    # Skip if exists AND has all 4 hdus + primary
    if os.path.exists(newfn):
        hdulist = fits.open(newfn) 
        if len(hdulist) == 5:
            make_wtmap=False
    if make_wtmap: 
        print('Creating new weightmap: %s' % newfn)
        imgobj= fits.open(imgfn) 
        wtobj = fits.open(wtfn) 
        dqobj = fits.open(dqfn) 
        hdr = imgobj[0].header 
        #read_noise= hdr['RDNOISE'] # e 
        #gain=1.8 # fixed 
        const_sky= hdr['SKYADU'] # e/s, Recommended sky level keyword from Frank 
        expt= hdr['EXPTIME'] # s 
        for hdu in range(1,len(imgobj)): 
            cpwt= wtobj[hdu].data # s/e, 1 / [var(sky) + var(read)] 
            cpimg= imgobj[hdu].data # e/s, img - median bkgrd  - var(sky) - var(read) + const sky 
            cpbad= dqobj[hdu].data # bitmask, 0 is good 
            var_SR= 1./cpwt # e/s 
            var_Astro= np.abs(cpimg - const_sky) / expt # e/s 
            wt= 1./(var_SR + var_Astro) # s/e 
            # Zero out NaNs and masked pixels 
            wt[np.isfinite(wt) == False]= 0. 
            wt[cpbad != 0]= 0. 
            # Overwrite w new weight 
            wtobj[hdu].data= wt 
        wtobj.writeto(newfn,clobber=True) 
        print('Wrote new weightmap: %s' % newfn)
    return newfn


