from __future__ import print_function

from legacypipe.image import LegacySurveyImage
import fitsio
import os

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
        print('Reading data quality image from', dqfn, 'hdu', hdu)
        mask = fitsio.read(dqfn, ext=hdu, header=False)
        cond1 = mask & 6141 != 0
        mask[cond1] = 1
        mask[~cond1] = 0
        return mask.astype(np.int16)
