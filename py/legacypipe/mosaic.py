from __future__ import print_function

import os
import fitsio

import numpy as np
from glob import glob

from astrometry.util.util import wcs_pv2sip_hdr

from tractor.basics import ConstantFitsWcs

from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPImage, newWeightMap
from legacypipe.survey import LegacySurveyData    

class MosaicImage(CPImage, CalibMixin):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOAO Community Pipeline.
    '''

    @classmethod
    def nominal_zeropoints(self):
        # HACK
        return dict(z = 26.5)
    
    @classmethod
    def photometric_ccds(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds'
        that are photometric.

        This recipe is adapted from the DECam one.
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

    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)
        # Add poisson noise to weight map
        self.wtfn= newWeightMap(wtfn=self.wtfn,imgfn=self.imgfn,dqfn=self.dqfn)
        # convert FWHM into pixel units
        self.fwhm /= self.pixscale

    def read_sky_model(self, imghdr=None, **kwargs):
        ''' The Mosaic CP does a good job of sky subtraction, so just
        use a constant sky level with value from the header.
        '''
        from tractor.sky import ConstantSky
        # Frank reocmmends SKYADU 
        sky = ConstantSky(imghdr['SKYADU'])
        sky.version = ''
        phdr = self.read_image_primary_header()
        sky.plver = phdr.get('PLVER', '').strip()
        return sky
        
    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        print('Reading weight map image', self.wtfn, 'ext', self.hdu)
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        return invvar

    def get_wcs(self):
        '''cpimage.py get_wcs() but wcs comes from interpolated image if this is an
        uninterpolated image'''
        prim= self.read_image_primary_header()
        if 'YSHIFT' in prim.keys():
            # Interpolated image, use its wcs
            hdr = self.read_image_header()
        else:
            # Non-interpolated, use WCS of interpolated instead
            # Temporarily set imgfn to Interpolated image
            imgfn_backup= self.imgfn
            # Change CP*v3 --> CP*v2
            cpdir=os.path.basename(os.path.dirname(imgfn_backup)).replace('v3','v2')
            dirnm= os.path.dirname(os.path.dirname(imgfn_backup))
            i=os.path.basename(imgfn_backup).find('_ooi_')
            searchnm= os.path.basename(imgfn_backup)[:i+5]+'*.fits.fz'
            self.imgfn= np.array( glob(os.path.join(dirnm,cpdir,searchnm)) )
            assert(self.imgfn.size == 1)
            self.imgfn= self.imgfn[0]
            newprim= self.read_image_primary_header()
            assert('YSHIFT' in newprim.keys())
            hdr = self.read_image_header()
            self.imgfn= imgfn_backup
            # Continue with wcs using the interpolated hdr
        # First child of MosaicImage is CPImage
        return super(MosaicImage,self).get_wcs(hdr=hdr)
        ## Make sure the PV-to-SIP converter samples enough points for small
        ## images
        #stepsize = 0
        #if min(self.width, self.height) < 600:
        #    stepsize = min(self.width, self.height) / 10.;
        ##if self.camera == '90prime':
        #    # WCS is in myriad of formats
        #    # Don't support TNX yet, use TAN for now
        ##    hdr = self.read_image_header()
        ##    hdr['CTYPE1'] = 'RA---TAN'
        ##    hdr['CTYPE2'] = 'DEC--TAN'
        #wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        ## Correctoin: ccd,ccdraoff, decoff from zeropoints file
        #dra,ddec = self.dradec
        #print('Applying astrometric zeropoint:', (dra,ddec))
        #r,d = wcs.get_crval()
        #wcs.set_crval((r + dra, d + ddec))
        #wcs.version = ''
        #phdr = self.read_image_primary_header()
        #wcs.plver = phdr.get('PLVER', '').strip()
        #return wcs

    def get_tractor_wcs(self, wcs, x0, y0,
                        primhdr=None, imghdr=None):
        '''1/3 pixel shift if nont-interpolated image'''
        prim= self.read_image_primary_header()
        if 'YSHIFT' in prim.keys():
            # Use Default wcs class, this is an interpolated image
            return super(MosaicImage, self).get_tractor_wcs(wcs, x0, y0)
        else:
            # IDENTICAL to image.py get_tractor_wcs() except uses OneThirdPixelShiftWcs() 
            # Instead of ConstantFitsWcs()
            # class OneThirdPixelShiftWcs is a ConstantFitsWcs class with1/3 pixel function
            twcs= OneThirdPixelShiftWcs(wcs)
            if x0 or y0:
                twcs.setX0Y0(x0,y0)
            return twcs

    def run_calibs(self, psfex=True, funpack=False, git_version=None,
                   force=False, **kwargs):
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

        todelete = []
        if funpack:
            # The image & mask files to process (funpacked if necessary)
            imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn, self.hdu, todelete)
        else:
            imgfn,maskfn = self.imgfn,self.dqfn
    
        if se:
            self.run_se('mzls', imgfn, maskfn)
        if psfex:
            self.run_psfex('mzls')

        for fn in todelete:
            os.unlink(fn)


class OneThirdPixelShiftWcs(ConstantFitsWcs):
    def __init__(self,wcs):
        super(OneThirdPixelShiftWcs,self).__init__(wcs)

    def positionToPixel(self, pos, src=None):
        '''
        Converts an :class:`tractor.RaDecPos` to a pixel position.
        Returns: tuple of floats ``(x, y)``
        '''
        x,y = super(OneThirdPixelShiftWcs, self).positionToPixel(pos, src=src)
        # Top half of CCD needs be shifted up by 1./3 pixel
        if (y + self.y0 > 2048):
            #y += 1./3
            y -= 1./3
        return x,y


def main():

    from astrometry.util.fits import fits_table, merge_tables
    from legacypipe.survey import exposure_metadata
    # Fake up a survey-ccds.fits table from MzLS_CP
    from glob import glob
    #fns = glob('/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202/k4m_160203_*oki*')
    fns = glob('/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20160202/k4m_160203_08*oki*')

    print('Filenames:', fns)
    T = exposure_metadata(fns)

    # HACK
    T.fwhm = T.seeing / 0.262

    # FAKE
    T.ccdnmatch = np.zeros(len(T), np.int32) + 50
    T.zpt = np.zeros(len(T), np.float32) + 26.518
    T.ccdzpt = T.zpt.copy()
    T.ccdraoff = np.zeros(len(T), np.float32)
    T.ccddecoff = np.zeros(len(T), np.float32)
    
    fmap = {'zd':'z'}
    T.filter = np.array([fmap[f] for f in T.filter])
    
    T.writeto('mzls-ccds.fits')

    os.system('cp mzls-ccds.fits ~/legacypipe-dir/survey-ccds.fits')
    os.system('gzip -f ~/legacypipe-dir/survey-ccds.fits')
    
    import sys
    sys.exit(0)
    
    import logging
    import sys
    from legacypipe.runbrick import run_brick, get_runbrick_kwargs, get_parser

    parser = get_parser()
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    survey, kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1, 0]:
        return kwargs

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    kwargs.update(splinesky=True, pixPsf=True)

    run_brick(opt.brick, survey, **kwargs)
    
if __name__ == '__main__':
    main()
