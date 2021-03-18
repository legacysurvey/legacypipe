from __future__ import print_function
import numpy as np
import os
from legacypipe.image import LegacySurveyImage
from legacypipe.bits import DQ_BITS

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

    def read_invvar(self, dq=None, **kwargs):
        print('Read iv')
        v = self._read_fits(self.wtfn, self.wt_hdu, **kwargs)
        iv = 1./v
        iv[v==0] = 0.
        iv[np.logical_not(np.isfinite(iv))] = 0.
        return iv
