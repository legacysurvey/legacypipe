from __future__ import print_function
import os
import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS
from legacypipe.survey import create_temp

class HscImage(LegacySurveyImage):
    def __init__(self, survey, ccd):
        ccd.plver = 'xxx'
        ccd.procdate = 'xxx'
        ccd.plprocid = 'xxx'
        super().__init__(survey, ccd)
        self.dq_hdu = 2
        self.wt_hdu = 3
        
    def compute_filenames(self):
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    # def read_dq(self, **kwargs):
    #     print('Read dq')
    #     dq = self._read_fits(self.dqfn, self.dq_hdu, **kwargs)
    #     return dq

    def remap_dq(self, dq, header):
        new_dq = np.zeros(dq.shape, np.int16)
        # MP_BAD  =                    0
        # MP_SUSPECT =        7
        # MP_NO_DATA =        8
        # MP_CROSSTALK =      9
        # MP_UNMASKEDNAN =   11
        new_dq |= (DQ_BITS['badpix'] * ((dq & ((1<<0) | (1<<7) | (1<<8) | (1<<9) | (1<<11))) != 0))
        # MP_SAT  =                    1
        new_dq |= (DQ_BITS['satur' ] * ((dq & (1<<1)) != 0))
        # MP_INTRP=                    2
        new_dq |= (DQ_BITS['interp'] * ((dq & (1<<2)) != 0))
        # MP_CR   =                    3
        new_dq |= (DQ_BITS['cr'] * ((dq & (1<<3)) != 0))
        #MP_EDGE =                    4
        new_dq |= (DQ_BITS['edge'] * ((dq & (1<<4)) != 0))
        '''
        MP_DETECTED =       5
        MP_DETECTED_NEGATIVE = 6
        MP_NOT_DEBLENDED = 10
        '''
        return new_dq

    def read_invvar(self, dq=None, **kwargs):
        print('Read iv')
        v = self._read_fits(self.wtfn, self.wt_hdu, **kwargs)
        iv = 1./v
        iv[v==0] = 0.
        iv[np.logical_not(np.isfinite(iv))] = 0.
        return iv

    def funpack_files(self, imgfn, maskfn, imghdu, maskhdu, todelete):
        # Before passing files to SourceExtractor / PsfEx, filter our mask image
        # because it marks DETECTED pixels with a mask bit.
        tmpimgfn,tmpmaskfn = super().funpack_files(imgfn, maskfn, imghdu, maskhdu, todelete)
        print('Dropping mask bit 5 before running SE')
        m = fitsio.read(tmpmaskfn)
        m &= ~(1 << 5)
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpmaskfn)
        fitsio.write(tmpmaskfn, m, clobber=True)
        return tmpimgfn, tmpmaskfn

    def validate_version(self, *args, **kwargs):
        return True
