from __future__ import print_function
#import sys
import os
import fitsio
import numpy as np

from astrometry.util.util import wcs_pv2sip_hdr

from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPImage, newWeightMap
from legacypipe.survey import LegacySurveyData
#from survey import create_temp
#from astrometry.util.util import Tan, Sip, anwcs_t

#from astrometry.util.file import trymakedirs

from tractor.sky import ConstantSky
#from tractor.basics import NanoMaggies, ConstantFitsWcs, LinearPhotoCal
#from tractor.image import Image
#from tractor.tractortime import TAITime


'''
Code specific to images from the 90prime camera on the Bok telescope.
'''
 
#class BokImage(LegacySurveyImage):
#class BokImage(LegacySurveyImage, CalibMixin):
class BokImage(CPImage, CalibMixin):
    '''
    Class for handling images from the 90prime camera processed by the
    NOAO Community Pipeline.
    '''
    def __init__(self, survey, t):
        super(BokImage, self).__init__(survey, t)
        self.pixscale= 0.455
        #self.dqfn= None #self.read_dq() #array of 0s for now
        #self.whtfn= self.imgfn.replace('.fits','.wht.fits')
        ##self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        self.dq_saturation_bits = 0 #not used so set to 0
        
        self.fwhm = t.fwhm
        self.arawgain = t.arawgain
        self.name = self.imgfn
        # Add poisson noise to weight map
        self.wtfn= newWeightMap(wtfn=self.wtfn,imgfn=self.imgfn,dqfn=self.dqfn)
        
    def __str__(self):
        return 'Bok ' + self.name

    def read_sky_model(self, **kwargs):
        ## HACK -- create the sky model on the fly
        img = self.read_image()
        sky = np.median(img)
        print('Median "sky" model:', sky)
        sky = ConstantSky(sky)
        sky.version = '0'
        sky.plver = '0'
        return sky

    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq

    def read_invvar(self, **kwargs):
        print('Reading the 90Prime oow weight map as Inverse Varianc')
        X = self._read_fits(self.wtfn, self.hdu, **kwargs)
        return X

    # read the TPV header, convert it to SIP, and apply an offset from the
    # CCDs table
#    def get_wcs(self):
#        # Make sure the PV-to-SIP converter samples enough points for small
#        # images
#        stepsize = 0
#        if min(self.width, self.height) < 600:
#            stepsize = min(self.width, self.height) / 10.
#        hdr = fitsio.read_header(self.imgfn, self.hdu)
#
#        # WORKAROUND bug in astrometry.net when CTYPEx don't have a comment string! Yuk
#        for r in hdr.records():
#            if not r['name'] in ['CTYPE1','CTYPE2']:
#                continue
#            r['comment'] = 'Hello'
#            r['card'] = hdr._record2card(r)
#
#        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
#        print('wcs bounds=:',wcs.radec_bounds())
#        raise ValueError
#        dra,ddec = self.dradec
#        r,d = wcs.get_crval()
#        print('Applying astrometric zeropoint:', (dra,ddec))
#        wcs.set_crval((r + dra, d + ddec))
#        wcs.version = ''
#        wcs.plver = ''
#        return wcs


    def run_calibs(self, psfex=True, sky=True, se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False, git_version=None,
                   splinesky=False,**kwargs):

        '''
        Run calibration pre-processing steps.
        '''
        print('run_calibs for', self.name, 'kwargs', kwargs)
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
 
        #if just_check:
        #    return (se or psfex)

        todelete = []
        if funpack:
            # The image & mask files to process (funpacked if necessary)
            imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn, self.hdu, todelete)
        else:
            imgfn,maskfn = self.imgfn,self.dqfn
        
        if se:
            # CAREFUL no mask given to SE
            self.run_se('90prime', imgfn, 'junkname')
        if psfex:
            self.run_psfex('90prime')

        for fn in todelete:
            os.unlink(fn)


