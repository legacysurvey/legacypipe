from __future__ import print_function

import os
import fitsio

import numpy as np
from glob import glob

from astrometry.util.util import wcs_pv2sip_hdr

from tractor.basics import ConstantFitsWcs

from legacypipe.image import LegacySurveyImage, CalibMixin
from legacypipe.cpimage import CPImage
from legacypipe.survey import LegacySurveyData    

class MosaicImage(CPImage, CalibMixin):
    '''
    Class for handling images from the Mosaic3 camera processed by the
    NOAO Community Pipeline.
    '''

    mjd_third_pixel_fixed = 57674.

    # this is defined here for testing purposes (to handle small images)
    splinesky_boxsize = 512

    def __init__(self, survey, t):
        super(MosaicImage, self).__init__(survey, t)
        # convert FWHM into pixel units
        #self.fwhm /= self.pixscale

    @classmethod
    def get_bad_expids(self):
        import legacyccds
        fn = os.path.join(os.path.dirname(legacyccds.__file__),
                          'bad_expid_mzls.txt')
        bad_expids = np.loadtxt(fn, dtype=int, usecols=(0,))
        return bad_expids

    @classmethod
    def nominal_zeropoints(self):
        # See legacypipe/ccd_cuts.py and Photometric cuts email 12/21/2016
        return dict(z = 26.20)
    
    @classmethod
    def photometric_ccds(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds'
        that are photometric.

        This recipe is adapted from the DECam one.
        '''
        # Nominal zeropoints (DECam)
        z0 = self.nominal_zeropoints()
        z0 = np.array([z0.get(f[0], 0) for f in ccds.filter])
        good = np.ones(len(ccds), bool)
        n0 = sum(good)
        # See Photometric cuts email 12/21/2016
        # This is our list of cuts to remove non-photometric CCD images
        for name,crit in [
            ('exptime < 30 s', (ccds.exptime < 30)),
            ('ccdnmatch < 20', (ccds.ccdnmatch < 20)),
            ('sky too bright: ccdskycounts >= 150', (ccds.ccdskycounts >= 150)),
            ('abs(zpt - ccdzpt) > 0.1',
             (np.abs(ccds.zpt - ccds.ccdzpt) > 0.1)),
            ('zpt < 0.6 mag of nominal',
             (ccds.zpt < (z0 - 0.6))),
            ('zpt > 0.6 mag of nominal',
             (ccds.zpt > (z0 + 0.6))),
            ('z band',
             ccds.filter != 'z'),
        ]:
            good[crit] = False
            #continue as usual
            n = sum(good)
            print('Flagged', n0-n, 'more non-photometric using criterion:',
                  name)
            n0 = n
        return np.flatnonzero(good)

    @classmethod
    def ccd_cuts(self, survey, ccds):
        ccdcuts = super(MosaicImage, self).ccd_cuts(survey, ccds)
        bits = LegacySurveyData.ccd_cut_bits
        I = self.bad_third_pixel(survey, ccds)
        print(np.sum(I), 'CCDs have bad THIRD_PIXEL')
        ccdcuts[I] += bits['THIRD_PIXEL']
        return ccdcuts
    
    @classmethod
    def bad_third_pixel(self, survey, ccds):
        '''
        For mosaic this is inconsistent YSHIFT header 
        Ensures that ccds are 1/3 pixel interpolated.
        '''
        from astropy.io import fits
        bad = np.zeros(len(ccds), bool)
        # Remove if primary header does NOT have keyword YSHIFT
        rootdir = survey.get_image_dir()
        # The 1/3-pixel shift problem was fixed in hardware on MJD 57674,
        # so only check for problems in data before then.
        I = np.flatnonzero(ccds.mjd_obs < MosaicImage.mjd_third_pixel_fixed)
        for i in I:
            fn = ccds.image_filename[i]
            fn = fn.strip()
            fn = os.path.join(rootdir, fn)
            hdulist = fits.open(fn)
            if not 'YSHIFT' in hdulist[0].header:
                print('Did not find YSHIFT in primary header of', fn)
                bad[i] = True
        return bad

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

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def get_wcs(self):
        '''cpimage.py get_wcs() but wcs comes from interpolated image if this is an
        uninterpolated image'''
        prim= self.read_image_primary_header()
        if 'YSHIFT' in prim.keys() or self.mjdobs > MosaicImage.mjd_third_pixel_fixed:
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
            from astropy.io import fits
            hdulist = fits.open(self.imgfn)
            print("'YSHIFT' in hdulist[0].header=",'YSHIFT' in hdulist[0].header)
            assert('YSHIFT' in newprim.keys())
            hdr = self.read_image_header()
            self.imgfn= imgfn_backup
            # Continue with wcs using the interpolated hdr
        # First child of MosaicImage is CPImage
        return super(MosaicImage,self).get_wcs(hdr=hdr)
        
    def get_tractor_wcs(self, wcs, x0, y0,
                        primhdr=None, imghdr=None):
        '''1/3 pixel shift if non-interpolated image'''
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

    def run_calibs(self, psfex=True, sky=True, se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False, git_version=None,
                   splinesky=False):
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
            self.run_se('mosaic', imgfn, maskfn)
        if psfex:
            self.run_psfex('mosaic')
        if sky:
            self.run_sky('mosaic', splinesky=splinesky, git_version=git_version)

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
