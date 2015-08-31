from __future__ import print_function
from image import LegacySurveyImage

'''
Code specific to images from the 90prime camera on the Bok telescope,
processed by the NOAO pipeline.  This is currently just a sketch.
'''

class BokImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 90prime
    camera on the Bok telescope.

    Currently, there are several hacks and shortcuts in handling the
    calibration; this is a sketch, not a final working solution.

    '''
    def __init__(self, decals, t):
        super(BokImage, self).__init__(decals, t)

        self.dqfn = self.imgfn.replace('_oi.fits', '_od.fits')

        expstr = '%10i' % self.expnum
        self.calname = '%s/%s/bok-%s-%s' % (expstr[:5], expstr, expstr, self.ccdname)
        self.name = '%s-%s' % (expstr, self.ccdname)

        calibdir = os.path.join(self.decals.get_calib_dir(), self.camera)
        self.pvwcsfn = os.path.join(calibdir, 'astrom-pv', self.calname + '.wcs.fits')
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')

    def __str__(self):
        return 'Bok ' + self.name

    def read_sky_model(self):
        ## HACK -- create the sky model on the fly
        img = self.read_image()
        sky = np.median(img)
        print('Median "sky" model:', sky)
        sky = ConstantSky(sky)
        sky.version = '0'
        sky.plver = '0'
        return sky

    def read_dq(self, **kwargs):
        print('Reading data quality from', self.dqfn, 'hdu', self.hdu)
        X = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return X

    def read_invvar(self, **kwargs):
        print('Reading inverse-variance for image', self.imgfn, 'hdu', self.hdu)
        ##### HACK!  No weight-maps available?
        img = self.read_image(**kwargs)

        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('sig1 estimate:', sig1)
        invvar = np.ones_like(img) / sig1**2
        return invvar

    def get_wcs(self):
        ##### HACK!  Ignore the distortion solution in the headers,
        ##### converting to straight TAN.
        hdr = fitsio.read_header(self.imgfn, self.hdu)
        print('Converting CTYPE1 from', hdr.get('CTYPE1'), 'to RA---TAN')
        hdr['CTYPE1'] = 'RA---TAN'
        print('Converting CTYPE2 from', hdr.get('CTYPE2'), 'to DEC--TAN')
        hdr['CTYPE2'] = 'DEC--TAN'
        H,W = self.get_image_shape()
        hdr['IMAGEW'] = W
        hdr['IMAGEH'] = H
        tmphdr = create_temp(suffix='.fits')
        fitsio.write(tmphdr, None, header=hdr, clobber=True)
        print('Wrote fake header to', tmphdr)
        wcs = Tan(tmphdr)
        print('Returning', wcs)
        wcs.version = '0'
        wcs.plver = '0'
        return wcs
