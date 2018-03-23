from __future__ import print_function
import numpy as np
import os
from legacypipe.image import LegacySurveyImage, CP_DQ_BITS

class MegaPrimeImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the MegaPrime
    camera on CFHT.
    '''
    def __init__(self, survey, t):
        super(MegaPrimeImage, self).__init__(survey, t)
        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)
        print('MegaPrimeImage: CCDs table entry', t)
        for x in dir(t):
            if x.startswith('_'):
                continue
            print('  ', x, ':', getattr(t,x))

    def compute_filenames(self):
        self.dqfn = 'cfis/test.mask.0.40.01.fits'

    def read_image(self, **kwargs):
        img = super(MegaPrimeImage, self).read_image(**kwargs)
        img = img.astype(np.float32)
        return img
        
    def read_invvar(self, **kwargs):
        ## FIXME -- at the very least, apply mask
        print('MegaPrimeImage.read_invvar')
        img = self.read_image(**kwargs)
        if self.sig1 is None:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            self.sig1 = sig1
            print('Computed sig1 by Blanton method:', self.sig1)

        iv = np.zeros_like(img) + (1./self.sig1**1)
        return iv



    # calibs



    def run_se(self, imgfn, maskfn):
        from astrometry.util.file import trymakedirs
        sedir = self.survey.get_se_dir()
        trymakedirs(self.sefn, dir=True)
        # We write the SE catalog to a temp file then rename, to avoid
        # partially-written outputs.

        from legacypipe.survey import create_temp
        import fitsio
        tmpmaskfn = create_temp(suffix='.fits')
        # The test.mask file has 1 for good pix, 0 for bad... invert for SE
        goodpix = fitsio.read(maskfn)
        fitsio.write(tmpmaskfn, (1-goodpix).astype(np.uint8), clobber=True)

        tmpfn = os.path.join(os.path.dirname(self.sefn),
                             'tmp-' + os.path.basename(self.sefn))
        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, self.camera + '.se'),
            '-PARAMETERS_NAME', os.path.join(sedir, self.camera + '.param'),
            '-FILTER_NAME %s' % os.path.join(sedir, self.camera + '.conv'),
            '-FLAG_IMAGE %s' % tmpmaskfn,
            '-CATALOG_NAME %s' % tmpfn,
            '-SEEING_FWHM %f' % 0.8,
            '-VERBOSE_TYPE FULL',
            '-PIXEL_SCALE 0.185',
            imgfn])
        print(cmd)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: ' + cmd)
        os.rename(tmpfn, self.sefn)

        os.unlink(tmpmaskfn)

    # def funpack_files(self, imgfn, dqfn, hdu, todelete):
    #     ''' Source Extractor can't handle .fz files, so unpack them.'''
    #     from legacypipe.survey import create_temp
    #     tmpimgfn = None
    #     # For FITS files that are not actually fpack'ed, funpack -E
    #     # fails.  Check whether actually fpacked.
    #     fcopy = False
    #     hdr = fitsio.read_header(imgfn, ext=hdu)
    #     if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
    #         print('Image %s, HDU %i is not fpacked; just imcopying.' %
    #               (imgfn,  hdu))
    #         fcopy = True
    # 
    #     tmpimgfn  = create_temp(suffix='.fits')
    #     todelete.append(tmpimgfn)
    #     
    #     if fcopy:
    #         cmd = 'imcopy %s"+%i" %s' % (imgfn, hdu, tmpimgfn)
    #     else:
    #         cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
    #     print(cmd)
    #     if os.system(cmd):
    #         raise RuntimeError('Command failed: ' + cmd)
    #     
    #     if fcopy:
    #         cmd = 'imcopy %s"+%i" %s' % (maskfn, hdu, tmpmaskfn)
    #     else:
    #         cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, maskfn)
    #     print(cmd)
    #     if os.system(cmd):
    #         print('Command failed: ' + cmd)
    #         M,hdr = self._read_fits(maskfn, hdu, header=True)
    #         print('Read', M.dtype, M.shape)
    #         fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)
    #         print('Wrote', tmpmaskfn, 'with fitsio')
    # 
    #     return tmpimgfn,tmpmaskfn
    #     #imgfn,maskfn = self.
