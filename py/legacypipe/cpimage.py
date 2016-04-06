from __future__ import print_function

import os
import fitsio

import numpy as np

from astrometry.util.util import wcs_pv2sip_hdr

from legacypipe.image import LegacySurveyImage
from legacypipe.common import LegacySurveyData

class CPMixin(object):
    '''
    A mix-in class for common code between NOAO Community Pipeline-processed
    data from DECam and Mosaic3.
    '''

    def __init__(self):
        #print('CPMixin __init__')
        super(CPMixin, self).__init__()
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

        expstr = '%08i' % self.expnum
        self.calname = '%s/%s/decam-%s-%s' % (expstr[:5], expstr, expstr, self.ccdname)
        self.name = '%s-%s' % (expstr, self.ccdname)

        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        self.splineskyfn = os.path.join(calibdir, 'splinesky', self.calname + '.fits')
        
    def get_wcs(self):
        hdr = fitsio.read_header(self.imgfn, self.hdu)
        wcs = wcs_pv2sip_hdr(hdr)
        dra,ddec = self.survey.get_astrometric_zeropoint_for(self)
        r,d = wcs.get_crval()
        print('Applying astrometric zeropoint:', (dra,ddec))
        wcs.set_crval((r + dra, d + ddec))
        wcs.version = ''
        phdr = fitsio.read_header(self.imgfn, 0)
        wcs.plver = phdr.get('PLVER', '').strip()
        return wcs

    
    
