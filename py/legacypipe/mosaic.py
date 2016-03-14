from __future__ import print_function

import os
import fitsio

import numpy as np

from astrometry.util.util import wcs_pv2sip_hdr

from legacypipe.image import LegacySurveyImage
from legacypipe.common import Decals

class MosaicImage(LegacySurveyImage):
    def __init__(self, decals, t):
        super(MosaicImage, self).__init__(decals, t)

        # convert FWHM into pixel units
        self.fwhm /= self.pixscale

        self.dqfn = self.imgfn.replace('_ooi_', '_ood_').replace('_oki_','_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_').replace('_oki_','_oow_')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        expstr = '%08i' % self.expnum
        self.name = '%s-%s' % (expstr, self.ccdname)
        self.calname = '%s/%s/mosaic-%s-%s' % (expstr[:5], expstr, expstr, self.ccdname)
        calibdir = os.path.join(self.decals.get_calib_dir(), self.camera)
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        # hack, for image.py : read_sky_model
        self.skyfn = self.splineskyfn

    def read_sky_model(self, imghdr=None, **kwargs):
        from tractor.sky import ConstantSky
        return ConstantSky(imghdr['AVSKY'])
        
    def get_wcs(self):
        hdr = fitsio.read_header(self.imgfn, self.hdu)
        wcs = wcs_pv2sip_hdr(hdr)

        phdr = fitsio.read_header(self.imgfn, 0)

        dra,ddec = self.decals.get_astrometric_zeropoint_for(self)
        r,d = wcs.get_crval()
        print('Applying astrometric zeropoint:', (dra,ddec))
        wcs.set_crval((r + dra, d + ddec))

        wcs.version = ''
        wcs.plver = phdr.get('PLVER', '').strip()
        
        return wcs

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

    def run_calibs(self, psfex=True, funpack=False, git_version=None,
                   force=False,
                   **kwargs):
        from astrometry.util.file import trymakedirs
        from legacypipe.common import (create_temp, get_version_header,
                                       get_git_version)

        print('run_calibs for', self.name, ': sky=', sky, 'kwargs', kwargs)

        se = False
        if psfex and os.path.exists(self.psffn) and (not force):
            psfex = False
        if psfex:
            se = True

        if se and os.path.exists(self.sefn) and (not force):
            se = False
        if se:
            funpack = True

        tmpimgfn = None
        tmpmaskfn = None

        # Unpacked image file
        funimgfn = self.imgfn
        funmaskfn = self.dqfn
        
        if funpack:
            # For FITS files that are not actually fpack'ed, funpack -E
            # fails.  Check whether actually fpacked.
            hdr = fitsio.read_header(self.imgfn, ext=self.hdu)
            if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
                print('Image', self.imgfn, 'HDU', self.hdu, 'is not actually fpacked; not funpacking, just imcopying.')
                fcopy = True

            tmpimgfn  = create_temp(suffix='.fits')
            tmpmaskfn = create_temp(suffix='.fits')
    
            cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpimgfn, self.imgfn)
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
            funimgfn = tmpimgfn
            
            cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpmaskfn, self.dqfn)
            print(cmd)
            if os.system(cmd):
                print('Command failed: ' + cmd)
                M,hdr = self._read_fits(self.dqfn, ext=self.hdu, header=True)
                print('Read', M.dtype, M.shape)
                fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)
                print('Wrote', tmpmaskfn, 'with fitsio')
            funmaskfn = tmpmaskfn
    
        if se:
            # grab header values...
            primhdr = self.read_image_primary_header()
            magzp  = primhdr['MAGZERO']
            seeing = self.pixscale * self.fwhm

            print('FWHM', self.fwhm, 'pix')
            print('pixscale', self.pixscale, 'arcsec/pix')
            print('Seeing', seeing, 'arcsec')
    
        if se:
            maskstr = '-FLAG_IMAGE ' + funmaskfn
            sedir = self.decals.get_se_dir()

            trymakedirs(self.sefn, dir=True)

            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS.se'),
                maskstr,
                '-SEEING_FWHM %f' % seeing,
                '-PARAMETERS_NAME', os.path.join(sedir, 'DECaLS.param'),
                '-FILTER_NAME', os.path.join(sedir, 'gauss_5.0_9x9.conv'),
                '-STARNNW_NAME', os.path.join(sedir, 'default.nnw'),
                '-PIXEL_SCALE 0',
                # SE has a *bizarre* notion of "sigma"
                '-DETECT_THRESH 1.0',
                '-ANALYSIS_THRESH 1.0',
                '-MAG_ZEROPOINT %f' % magzp,
                '-CATALOG_NAME', self.sefn,
                funimgfn])
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
        if psfex:
            sedir = self.decals.get_se_dir()
            trymakedirs(self.psffn, dir=True)

            # If we wrote *.psf instead of *.fits in a previous run...
            oldfn = self.psffn.replace('.fits', '.psf')
            if os.path.exists(oldfn):
                print('Moving', oldfn, 'to', self.psffn)
                os.rename(oldfn, self.psffn)
            else:
                primhdr = self.read_image_primary_header()
                plver = primhdr.get('PLVER', '')
                verstr = get_git_version()
                cmds = ['psfex -c %s -PSF_DIR %s %s' %
                        (os.path.join(sedir, 'DECaLS.psfex'),
                         os.path.dirname(self.psffn), self.sefn),
                        'modhead %s LEGPIPEV %s "legacypipe git version"' %
                        (self.psffn, verstr),
                        'modhead %s PLVER %s "CP ver of image file"' %
                        (self.psffn, plver)]
                for cmd in cmds:
                    print(cmd)
                    rtn = os.system(cmd)
                    if rtn:
                        raise RuntimeError('Command failed: ' + cmd + ': return value: %i' % rtn)

        if tmpimgfn is not None:
            os.unlink(tmpimgfn)
        if tmpmaskfn is not None:
            os.unlink(tmpmaskfn)


def main():
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
