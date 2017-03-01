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

    # this is defined here for testing purposes (to handle small images)
    splinesky_boxsize = 256

    def __init__(self, survey, t, makeNewWeightMap=True):
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


    @classmethod
    def nominal_zeropoints(self):
        return dict(g = 25.74,
                    r = 25.52,)

    @classmethod
    def photometric_ccds(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds'
        that are photometric.

        This recipe is adapted from the DECam one.
        '''
        # See legacypipe/ccd_cuts.py
        z0 = self.nominal_zeropoints()
        z0 = np.array([z0[f[0]] for f in ccds.filter])
        good = np.ones(len(ccds), bool)
        n0 = sum(good)
        # This is our list of cuts to remove non-photometric CCD images
        # These flag too many: ('zpt < 0.5 mag of nominal',(ccds.zpt < (z0 - 0.5))),
        # And ('zpt > 0.25 mag of nominal', (ccds.zpt > (z0 + 0.25))),
        for name,crit in [
            ('exptime < 30 s', (ccds.exptime < 30)),
            ('ccdnmatch < 20', (ccds.ccdnmatch < 20)),
            ('abs(zpt - ccdzpt) > 0.1',
             (np.abs(ccds.zpt - ccds.ccdzpt) > 0.1)),
            ('zpt < 0.5 mag of nominal',
             (ccds.zpt < (z0 - 0.5))),
            ('zpt > 0.18 mag of nominal',
             (ccds.zpt > (z0 + 0.18))),
        ]:
            good[crit] = False
            #continue as usual
            n = sum(good)
            print('Flagged', n0-n, 'more non-photometric using criterion:',
                  name)
            n0 = n
        return np.flatnonzero(good)

    @classmethod
    def bad_exposures(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds'
        that are good exposures (NOT flagged) in the bad_expid file.
        '''
        good = np.ones(len(ccds), bool)
        print('WARNING: camera: %s not using bad_expid file' % '90prime')
        return np.flatnonzero(good)

    @classmethod
    def has_third_pixel(self, survey, ccds):
        '''
        For mosaic only, ensures ccds are 1/3 pixel interpolated. Nothing for other cameras
        '''
        good = np.ones(len(ccds), bool)
        return np.flatnonzero(good)

    @classmethod
    def ccdname_hdu_match(self, survey, ccds):
        '''
        Mosaic + Bok, ccdname and hdu number must match. If not, IDL zeropoints files has
        duplicated zeropoint info from one of the other four ccds
        '''
        good = np.ones(len(ccds), bool)
        n0 = sum(good)
        ccdnum= np.char.replace(ccds.ccdname,'ccd','').astype(ccds.image_hdu.dtype)
        flag= ccds.image_hdu - ccdnum != 0
        good[flag]= False
        n = sum(good)
        print('Flagged', n0-n, 'ccdname_hdu_match')
        n0 = n 
        return np.flatnonzero(good)


#    def read_sky_model(self, imghdr=None, **kwargs):
#        ''' Bok CP does same sky subtraction as Mosaic CP, so just
#        use a constant sky level with value from the header.
#        '''
#        from tractor.sky import ConstantSky
#        # Frank reocmmends SKYADU 
#        phdr = self.read_image_primary_header()
#        sky = ConstantSky(phdr['SKYADU'])
#        sky.version = ''
#        sky.plver = phdr.get('PLVER', '').strip()
#        return sky

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
                   splinesky=False):
    #def run_calibs(self, psfex=True, sky=True, se=False,
    #               funpack=False, fcopy=False, use_mask=True,
    #               force=False, just_check=False, git_version=None,
    #               splinesky=False,**kwargs):

        '''
        Run calibration pre-processing steps.
        '''
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

        if sky and (not force) and (
            (os.path.exists(self.skyfn) and not splinesky) or
            (os.path.exists(self.splineskyfn) and splinesky)):
            fn = self.skyfn
            if splinesky:
                fn = self.splineskyfn

            if os.path.exists(fn):
                try:
                    hdr = fitsio.read_header(fn)
                except:
                    print('Failed to read sky file', fn, '-- deleting')
                    os.unlink(fn)
            if os.path.exists(fn):
                print('File', fn, 'exists -- skipping')
                sky = False

        if just_check:
            return (se or psfex or sky)

        todelete = []
        if funpack:
            # The image & mask files to process (funpacked if necessary)
            imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn, self.hdu, todelete)
        else:
            imgfn,maskfn = self.imgfn,self.dqfn
        
        if se:
            # CAREFUL no mask given to SE
            self.run_se('90prime', imgfn, maskfn) #'junkname')
        if psfex:
            self.run_psfex('90prime')

        if sky:
            self.run_sky('90prime', splinesky=splinesky,\
                         git_version=git_version)

        for fn in todelete:
            os.unlink(fn)


