from __future__ import print_function
import sys
import os
import fitsio
import numpy as np

from image import LegacySurveyImage
from common import create_temp
from astrometry.util.util import Tan, Sip, anwcs_t

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.file import trymakedirs

from tractor.sky import ConstantSky
from tractor.basics import NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.image import Image
from tractor.tractortime import TAITime

from legacypipe.ptf import zeropoint_for_ptf


'''
Code specific to images from the 90prime camera on the Bok telescope.
'''

class BokImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 90prime
    camera on the Bok telescope.
    '''
    def __init__(self, survey, t):
        super(BokImage, self).__init__(survey, t)
        self.pixscale= 0.455
        self.fwhm = t.fwhm
        self.arawgain = t.arawgain
        
        self.whtfn= self.imgfn.replace('.fits','.wht.fits')

        self.calname = os.path.basename(self.imgfn).replace('.fits','') 
        self.name = self.imgfn

        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        #self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        self.dq_saturation_bits = -1 #junk b/c currently don't have dq images for bok so read_dq returns array of 0s
        print('in BokImage init, calibdir=%s,self.calname=%s,self.imgfn=%s, self.whtfn=%s, self.sefn=%s, self.psffn=%s' % (calibdir,self.calname,self.imgfn,self.whtfn,self.sefn, self.psffn))
        
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
        # already account for bad pixels in wht, so return array of 0s
        # (good everywhere)
        dq = np.zeros(self.shape, np.uint8)
        return dq

    def read_invvar(self, **kwargs):
        print('Reading inv=%s for image=%s, hdu=' % (self.whtfn,self.imgfn),
              self.hdu)
        X = self._read_fits(self.whtfn, self.hdu, **kwargs)
        return X

    # read the TPV header, convert it to SIP, and apply an offset from the
    # CCDs table
    def get_wcs(self):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.
        hdr = fitsio.read_header(self.imgfn, self.hdu)

        # WORKAROUND bug in astrometry.net when CTYPEx don't have a comment string! Yuk
        for r in hdr.records():
            if not r['name'] in ['CTYPE1','CTYPE2']:
                continue
            r['comment'] = 'Hello'
            r['card'] = hdr._record2card(r)

        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        dra,ddec = self.dradec
        r,d = wcs.get_crval()
        print('Applying astrometric zeropoint:', (dra,ddec))
        wcs.set_crval((r + dra, d + ddec))
        wcs.version = ''
        wcs.plver = ''
        return wcs

    def run_calibs(self, psfex=True, sky=True, funpack=False, git_version=None,
                       force=False,
                       **kwargs):
        print('run_calibs for', self.name, ': sky=', sky, 'kwargs', kwargs) 
        se = False
        if psfex and os.path.exists(self.psffn) and (not force):
            psfex = False
        if psfex:
            se = True

        if se and os.path.exists(self.sefn) and (not force):
            se = False
        if se:
            sedir = self.survey.get_se_dir()
            trymakedirs(self.sefn, dir=True)
            ####
            trymakedirs('junk',dir=True) #need temp dir for mask-2 and invvar map
            hdu=0
            maskfn= self.imgfn.replace('_scie_','_mask_')
            #invvar= self.read_invvar(self.imgfn,maskfn,hdu) #note, all post processing on image,mask done in read_invvar
            invvar= self.read_invvar()
            mask= self.read_dq()
            maskfn= os.path.join('junk',os.path.basename(maskfn))
            invvarfn= maskfn.replace('_mask_','_invvar_')
            fitsio.write(maskfn, mask)
            fitsio.write(invvarfn, invvar)
            print('wrote mask-2 to %s, invvar to %s' % (maskfn,invvarfn))
            #run se 
            hdr=fitsio.read_header(self.imgfn,ext=hdu)
            #magzp  = zeropoint_for_ptf(hdr)
            magzp = self.ccdzpt
            seeing = self.pixscale * self.fwhm
            gain= self.arawgain
            cmd = ' '.join(['sex','-c', os.path.join(sedir,'ptf.se'),
                            '-WEIGHT_IMAGE %s' % invvarfn, '-WEIGHT_TYPE MAP_WEIGHT',
                            '-GAIN %f' % gain,
                            '-FLAG_IMAGE %s' % maskfn,
                            '-FLAG_TYPE OR',
                            '-SEEING_FWHM %f' % seeing,
                            '-DETECT_MINAREA 3',
                            '-PARAMETERS_NAME', os.path.join(sedir,'ptf.param'),
                            '-FILTER_NAME', os.path.join(sedir, 'ptf_gauss_3.0_5x5.conv'),
                            '-STARNNW_NAME', os.path.join(sedir, 'ptf_default.nnw'),
                            '-PIXEL_SCALE 0',
                            # SE has a *bizarre* notion of "sigma"
                            '-DETECT_THRESH 1.0',
                            '-ANALYSIS_THRESH 1.0',
                            '-MAG_ZEROPOINT %f' % magzp,
                            '-CATALOG_NAME', self.sefn,
                            self.imgfn])
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
        if psfex:
            sedir = self.survey.get_se_dir()
            trymakedirs(self.psffn, dir=True)
            # If we wrote *.psf instead of *.fits in a previous run...
            oldfn = self.psffn.replace('.fits', '.psf')
            if os.path.exists(oldfn):
                print('Moving', oldfn, 'to', self.psffn)
                os.rename(oldfn, self.psffn)
            else: 
                cmd= ' '.join(['psfex',self.sefn,'-c', os.path.join(sedir,'ptf.psfex'),
                    '-PSF_DIR',os.path.dirname(self.psffn)])
                print(cmd)
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)
    
