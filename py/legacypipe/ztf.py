from __future__ import print_function

from legacypipe.image import LegacySurveyImage
import fitsio
import os
import numpy as np

'''
Code specific to images from the Zwicky Transient Facility (ZTF).
'''

class ZtfImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 
    Zwicky Transient Facility (ZTF) at Palomar Observatory.
    '''
    def __init__(self, survey, ccd):
        super(ZtfImage, self).__init__(survey, ccd)


    def compute_filenames(self):
        """Assume your image filename is here: self.imgfn"""
        self.dqfn = self.imgfn.replace('sciimg', 'mskimg')
        self.wtfn = self.imgfn.replace('sciimg', 'weight')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue

    def read_dq(self, **kwargs):

        '''return bit mask which Tractor calls "data quality" image
        ZTF DMASK BIT DEFINITIONS
        BIT00 = 0 / AIRCRAFT/SATELLITE TRACK
        BIT01 = 1 / OBJECT (detected by SExtractor)
        BIT02 = 2 / LOW RESPONSIVITY
        BIT03 = 3 / HIGH RESPONSIVITY
        BIT04 = 4 / NOISY
        BIT05 = 5 / GHOST ** not yet implemented **
        BIT06 = 6 / CCD BLEED ** not yet implemented **
        BIT07 = 7 / PIXEL SPIKE (POSSIBLE RAD HIT)
        BIT08 = 8 / SATURATED
        BIT09 = 9 / DEAD (UNRESPONSIVE)
        BIT10 = 10 / NAN (not a number)
        BIT11 = 11 / CONTAINS PSF-EXTRACTED SOURCE POSITION
        BIT12 = 12 / HALO ** not yet implemented **
        BIT13 = 13 / RESERVED FOR FUTURE USE
        BIT14 = 14 / RESERVED FOR FUTURE USE
        BIT15 = 15 / RESERVED FOR FUTURE USE
        '''
        print('Reading data quality image from', self.dqfn, 'hdu', self.hdu)
        mask, header = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return mask.astype(np.int16), header

    def remap_dq_cp_codes(self, dq, header):
        '''
        Some versions of the CP use integer codes, not bit masks.
        This converts them.
        '''
        from legacypipe.image import CP_DQ_BITS
        dqbits = np.zeros(dq.shape, np.int16)

        dqbits[dq & 2**0 != 0] |= CP_DQ_BITS['trans']
        dqbits[dq & 2**2 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**3 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**4 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**5 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**6 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**7 != 0] |= CP_DQ_BITS['cr']
        dqbits[dq & 2**8 != 0] |= CP_DQ_BITS['satur']
        dqbits[dq & 2**9 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**10 != 0] |= CP_DQ_BITS['badpix']
        dqbits[dq & 2**12 != 0] |= CP_DQ_BITS['badpix']
        dqbits[:,0] != CP_DQ_BITS['edge']
        dqbits[:,-1] != CP_DQ_BITS['edge']
        dqbits[0,:] != CP_DQ_BITS['edge']
        dqbits[-1,:] != CP_DQ_BITS['edge']
        return dqbits
