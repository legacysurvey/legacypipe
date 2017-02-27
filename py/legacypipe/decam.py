from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.file import trymakedirs
from astrometry.util.fits import fits_table
from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPImage, CP_DQ_BITS
from legacypipe.survey import *

import astropy.time

'''
Code specific to images from the Dark Energy Camera (DECam).
'''

class DecamImage(CPImage, CalibMixin):
    '''

    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.

    '''
    # this is defined here for testing purposes (to handle small images)
    splinesky_boxsize = 512

    def __init__(self, survey, t):
        super(DecamImage, self).__init__(survey, t)

        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)
        
    def __str__(self):
        return 'DECam ' + self.name

    @classmethod
    def nominal_zeropoints(self):
        return dict(g = 25.08,
                    r = 25.29,
                    z = 24.92,)
    
    @classmethod
    def photometric_ccds(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds' that are
        photometric.

        Slightly revised recipe by DJS in Re: [decam-data 828] 2015-07-31:
        
        * CCDNMATCH >= 20 (At least 20 stars to determine zero-pt)
        * abs(ZPT - CCDZPT) < 0.10  (Loose agreement with full-frame zero-pt)
        * ZPT within [25.08-0.50, 25.08+0.25] for g-band
        * ZPT within [25.29-0.50, 25.29+0.25] for r-band
        * ZPT within [24.92-0.50, 24.92+0.25] for z-band
        * DEC > -20 (in DESI footprint)
        * EXPTIME >= 30
        * CCDNUM = 31 (S7) should mask outside the region [1:1023,1:4094]
        '''
        # Nominal zeropoints (DECam)
        z0 = self.nominal_zeropoints()
        z0 = np.array([z0[f[0]] for f in ccds.filter])
        good = np.ones(len(ccds), bool)
        n0 = sum(good)
        # This is our list of cuts to remove non-photometric CCD images
        for name,crit in [
            ('exptime < 30 s', (ccds.exptime < 30)),
            ('ccdnmatch < 20', (ccds.ccdnmatch < 20)),
            ('abs(zpt - ccdzpt) > 0.1',
             (np.abs(ccds.zpt - ccds.ccdzpt) > 0.1)),
            ('zpt < 0.5 mag of nominal',
             (ccds.zpt < (z0 - 0.5))),
            ('zpt > 0.25 mag of nominal',
             (ccds.zpt > (z0 + 0.25))),
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
        print('WARNING: camera: %s not using bad_expid file' % 'decam')
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
        return np.flatnonzero(good)
 

    @classmethod
    def apply_blacklist(self, survey, ccds):
        propid_blacklist = [
            '2012B-0003', # labeled as "DES SV", but appears to
                          # exclusively be DES deep fields taken
                          # during SV, through Jan 2013.
            '2012B-0624', # 1 field, GAMA09
            '2012B-0625', # 1 field, NGC2872
            '2012B-3011', # 1 field
            '2012B-3012', # 1 field, Pan-STARRS MD01/
            '2012B-9999', # Engineering?

            '2013A-0351', # lots of deep data on COSMOS
            '2013A-0360', # 9 fields total
            '2013A-0529', # 2 fields
            '2013A-0611', # many 900-sec exposures in EDR region
            '2013A-0613', # 40 exposures of 600 sec in g,r and nothing else in DR2
            '2013A-0614', # 2 fields
            '2013A-0716', # 3 fields
            '2013A-0717', # 2 fields
            '2013A-0719', # 8 fields (more in DR3)
            '2013A-0737', # 10 fields
            '2013A-9999', # Engineering?

            '2013B-0502', # 3 fields
            '2013B-0531', # Neptune

            '2014A-0073', # one deep field at (135.8,-4.6), g
            '2014A-0239', # 1 field (SUDSS X3)
            '2014A-0313', # 3 fields
            '2014A-0339', # two strips of sky + one deep field (Her)
            '2014A-0415', # 4 fields
            '2014A-0429', # 2 fields
            '2014A-0632', # 2 fields

            '2014B-0146', # COSMOS + one other field
            '2014B-0244', # 6 fields
            '2014B-0608', # 2 fields

            # To bring exposure times below 1 hour:
            '2014A-0321', # Pal 3+5
            '2014A-0610', # Pal 3+5
            '2014A-0611', # Pal 3+5
            '2014A-0621', # Pal 3+5
            '2014A-0622', # HCG
            '2013B-0438', # SHELA
            '2014A-0608', # Blind 1-hour pointings in g-band
            '2014A-0386', # A2029
        ]

        object_blacklist = [
            'DES supernova hex',
            ]

        if survey.version == 'dr1' or survey.version == 'dr2':
            keep = np.array([(propid not in propid_blacklist) for propid in ccds.propid])
        else:
            keep = np.array([(propid not in propid_blacklist) and
                             not any([obj.startswith(x) 
                                      for x in object_blacklist])
                                      for propid,obj in zip(ccds.propid, ccds.object)])
        return np.flatnonzero(keep)
    
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
            dq = self.remap_dq_codes(dq)

        else:
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
        from .survey import (create_temp, get_version_header,
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
            self.run_psfex('DECaLS')

        if sky:
            #print('Fitting sky for', self)
            self.run_sky('DECaLS', splinesky=splinesky,\
                         git_version=git_version)


        for fn in todelete:
            os.unlink(fn)

