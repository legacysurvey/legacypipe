from __future__ import print_function

import os
import fitsio

import numpy as np

from astrometry.util.util import wcs_pv2sip_hdr

from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPImage
from legacypipe.common import LegacySurveyData

class MosaicImage(CPImage, CalibMixin):
    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)
        # convert FWHM into pixel units
        self.fwhm /= self.pixscale

    def read_sky_model(self, imghdr=None, **kwargs):
        ''' The Mosaic CP does a good job of sky subtraction, so just
        use a constant sky level with value from the header.
        '''
        from tractor.sky import ConstantSky
        sky = ConstantSky(imghdr['AVSKY'])
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
        #print('Reading weight map image', self.wtfn, 'ext', self.hdu)
        #invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        #return invvar

        print('HACK -- not reading weight map, estimating from image')
        ##### HACK!  No weight-maps available?
        img = self.read_image(**kwargs)
        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('sig1 estimate:', sig1)
        invvar = np.ones_like(img) / sig1**2
        # assume this is going to be masked by the DQ map.
        return invvar

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

def main():

    from astrometry.util.fits import fits_table, merge_tables
    from legacypipe.common import exposure_metadata
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
    kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1, 0]:
        return kwargs

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    kwargs.update(splinesky=True, pixPsf=True)

    run_brick(opt.brick, **kwargs)
    
if __name__ == '__main__':
    main()
