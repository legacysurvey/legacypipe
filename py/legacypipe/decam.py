from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.file import trymakedirs
from astrometry.util.fits import fits_table
from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPMixin
from legacypipe.common import *

import astropy.time

'''
Code specific to images from the Dark Energy Camera (DECam).
'''

class DecamImage(LegacySurveyImage, CalibMixin, CPMixin):
    '''

    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.

    '''
    def __init__(self, survey, t):
        super(DecamImage, self).__init__(survey, t)

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
        primhdr = fitsio.read_header(self.dqfn)
        plver = primhdr['PLVER'].strip()
        plver = plver.replace('V','')
        if StrictVersion(plver) >= StrictVersion('3.5.0'):
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

            dq = dqbits

        else:
            dq = dq.astype(np.int16)

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

    def run_calibs(self, psfex=True, sky=True, se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False, git_version=None,
                   splinesky=False):

        '''
        Run calibration pre-processing steps.

        Parameters
        ----------
        just_check: boolean
            If True, returns True if calibs need to be run.
        '''
        from .common import (create_temp, get_version_header,
                             get_git_version)
        
        if psfex and os.path.exists(self.psffn) and (not force):
            if self.check_psf(self.psffn):
                psfex = False
        if psfex:
            se = True
            
        if se and os.path.exists(self.sefn) and (not force):
            if self.check_se_cat(self.sefn):
                se = False
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
            self.run_se('DECaLS', imgfn, maskfn)
        if psfex:
            self.run_psfex('DECaLS', sefn)

        if sky:
            #print('Fitting sky for', self)

            hdr = get_version_header(None, self.survey.get_survey_dir(),
                                     git_version=git_version)
            primhdr = self.read_image_primary_header()
            plver = primhdr.get('PLVER', '')
            hdr.delete('PROCTYPE')
            hdr.add_record(dict(name='PROCTYPE', value='ccd',
                                comment='NOAO processing type'))
            hdr.add_record(dict(name='PRODTYPE', value='skymodel',
                                comment='NOAO product type'))
            hdr.add_record(dict(name='PLVER', value=plver,
                                comment='CP ver of image file'))

            slc = self.get_good_image_slice(None)
            #print('Good image slice is', slc)

            img = self.read_image(slice=slc)
            wt = self.read_invvar(slice=slc)

            if splinesky:
                from tractor.splinesky import SplineSky
                from scipy.ndimage.morphology import binary_dilation

                # Start by subtracting the overall median
                med = np.median(img[wt>0])
                # Compute initial model...
                skyobj = SplineSky.BlantonMethod(img - med, wt>0, 512)
                skymod = np.zeros_like(img)
                skyobj.addTo(skymod)
                # Now mask bright objects in (image - initial sky model)
                sig1 = 1./np.sqrt(np.median(wt[wt>0]))
                masked = (img - med - skymod) > (5.*sig1)
                masked = binary_dilation(masked, iterations=3)
                masked[wt == 0] = True

                sig1b = 1./np.sqrt(np.median(wt[masked == False]))
                print('Sig1 vs sig1b:', sig1, sig1b)
                
                # Now find the final sky model using that more extensive mask
                skyobj = SplineSky.BlantonMethod(
                    img - med, np.logical_not(masked), 512)
                # add the overall median back in
                skyobj.offset(med)
    
                if slc is not None:
                    sy,sx = slc
                    y0 = sy.start
                    x0 = sx.start
                    skyobj.shift(-x0, -y0)

                hdr.add_record(dict(name='SIG1', value=sig1,
                                    comment='Median stdev of unmasked pixels'))
                hdr.add_record(dict(name='SIG1B', value=sig1,
                                    comment='Median stdev of unmasked pixels+'))
                    
                trymakedirs(self.splineskyfn, dir=True)
                skyobj.write_fits(self.splineskyfn, primhdr=hdr)
                print('Wrote sky model', self.splineskyfn)
    
            else:
                img = img[wt > 0]
                try:
                    skyval = estimate_mode(img, raiseOnWarn=True)
                    skymeth = 'mode'
                except:
                    skyval = np.median(img)
                    skymeth = 'median'
                tsky = ConstantSky(skyval)

                hdr.add_record(dict(name='SKYMETH', value=skymeth,
                                    comment='estimate_mode, or fallback to median?'))
                sig1 = 1./np.sqrt(np.median(wt[wt>0]))
                masked = (img - skyval) > (5.*sig1)
                masked = binary_dilation(masked, iterations=3)
                masked[wt == 0] = True
                sig1b = 1./np.sqrt(np.median(wt[masked == False]))
                print('Sig1 vs sig1b:', sig1, sig1b)

                hdr.add_record(dict(name='SIG1', value=sig1,
                                    comment='Median stdev of unmasked pixels'))
                hdr.add_record(dict(name='SIG1B', value=sig1,
                                    comment='Median stdev of unmasked pixels+'))
                
                trymakedirs(self.skyfn, dir=True)
                tsky.write_fits(self.skyfn, hdr=hdr)
                print('Wrote sky model', self.skyfn)

        for fn in todelete:
            os.unlink(fn)

