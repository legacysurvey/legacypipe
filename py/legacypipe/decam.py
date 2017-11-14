from __future__ import print_function
import os
import numpy as np
import fitsio
from legacypipe.cpimage import CPImage
from legacypipe.survey import LegacySurveyData

import astropy.time

'''
Code specific to images from the Dark Energy Camera (DECam).
'''

class DecamImage(CPImage):
    '''
    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.
    '''
    # this is defined here for testing purposes (to handle the small
    # images used in unit tests)
    splinesky_boxsize = 512

    def __init__(self, survey, t):
        super(DecamImage, self).__init__(survey, t)

        # DEBUG
        if not os.path.exists(self.imgfn):
            fn = self.imgfn.replace('/decam/', '/decam/DECam_CP/')
            if os.path.exists(fn):
                print('Replaced image path', self.imgfn, 'with', fn)
                self.imgfn = fn
        if not os.path.exists(self.dqfn):
            fn = self.dqfn.replace('/decam/', '/decam/DECam_CP/')
            if os.path.exists(fn):
                print('Replaced image path', self.dqfn, 'with', fn)
                self.dqfn = fn
        if not os.path.exists(self.wtfn):
            fn = self.wtfn.replace('/decam/', '/decam/DECam_CP/')
            if os.path.exists(fn):
                print('Replaced image path', self.wtfn, 'with', fn)
                self.wtfn = fn

        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)

    def __str__(self):
        return 'DECam ' + self.name

    glowmjd = astropy.time.Time('2014-08-01').utc.mjd

    def get_good_image_subregion(self):
        x0,x1,y0,y1 = None,None,None,None

        # Handle 'glowing' edges in DES r-band images
        # aww yeah
        if self.band == 'r' and (
                ('DES' in self.imgfn) or ('COSMOS' in self.imgfn) or
                (self.mjdobs < DecamImage.glowmjd)):
            # Northern chips: drop 100 pix off the bottom
            if 'N' in self.ccdname:
                print('Clipping bottom part of northern DES r-band chip')
                y0 = 100
            else:
                # Southern chips: drop 100 pix off the top
                print('Clipping top part of southern DES r-band chip')
                y1 = self.height - 100

        # Clip the bad half of chip S7.
        # The left half is OK.
        if self.ccdname == 'S7':
            print('Clipping the right half of chip S7')
            x1 = 1023

        return x0,x1,y0,y1

    def read_dq(self, header=False, **kwargs):
        from distutils.version import StrictVersion
        print('Reading data quality from', self.dqfn, 'hdu', self.hdu)
        dq,hdr = self._read_fits(self.dqfn, self.hdu, header=True, **kwargs)
        # The format of the DQ maps changed as of version 3.5.0 of the
        # Community Pipeline.  Handle that here...
        primhdr = self.read_primary_header(self.dqfn)
        plver = primhdr['PLVER'].strip()
        plver = plver.replace('V','')
        plver = plver.replace('DES ', '')
        plver = plver.replace('+1', 'a1')
        if StrictVersion(plver) >= StrictVersion('3.5.0'):
            dq = self.remap_dq_codes(dq)
        else:
            from legacypipe.cpimage import CP_DQ_BITS
            dq = dq.astype(np.int16)
            # Un-set the SATUR flag for pixels that also have BADPIX set.
            both = CP_DQ_BITS['badpix'] | CP_DQ_BITS['satur']
            I = np.flatnonzero((dq & both) == both)
            if len(I):
                print('Warning: un-setting SATUR for', len(I),
                      'pixels with SATUR and BADPIX set.')
                dq.flat[I] &= ~CP_DQ_BITS['satur']
                assert(np.all((dq & both) != both))

        if header:
            return dq,hdr
        else:
            return dq

    def read_invvar(self, clip=True, clipThresh=0.2, **kwargs):
        print('Reading inverse-variance from', self.wtfn, 'hdu', self.hdu)
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            # Clamp near-zero (incl negative!) invvars to zero.
            # These arise due to fpack.
            if clipThresh > 0.:
                med = np.median(invvar[invvar > 0])
                thresh = clipThresh * med
            else:
                thresh = 0.
            invvar[invvar < thresh] = 0
        return invvar

