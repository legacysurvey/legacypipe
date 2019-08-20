from legacyzpts.legacy_zeropoints import *

import logging
logger = logging.getLogger('legacypipe.image')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class MosaicRawMeasurer(Mosaic3Measurer):
    def __init__(self, *args, **kwargs):
        super(MosaicRawMeasurer, self).__init__(*args, **kwargs)
        self.plver = 'V0.0'
        self.procdate = 'xxxx'
        self.plprocid = 'xxxxx'

    def get_extension_list(self, fn, debug=False):
        if debug:
            return ['im4']
        return ['im%i'%(i+1) for i in range(16)]

    def get_fwhm(self, hdr, hdu):
        ### HACK
        return 4.0

    def good_wcs(self, primhdr):
        return True

    def set_hdu(self, ext):
        super(MosaicRawMeasurer, self).set_hdu(ext)
        img,hdr = super(MosaicRawMeasurer, self).read_image()
        self.img_data,self.img_hdr = img,hdr
        # Subtract median overscan and multiply by gains
        dataA = parse_section(hdr['DATASEC'], slices=True)
        biasA = parse_section(hdr['BIASSEC'], slices=True)
        gainA = hdr['GAIN']
        b = np.median(img[biasA])
        img[dataA] = (img[dataA] - b) * gainA
        # Trim the image
        trimA = parse_section(hdr['TRIMSEC'], slices=True)
        # zero out all but the trim section
        trimg = img[trimA].copy().astype(np.float32)
        img[:,:] = 0
        img[trimA] = trimg

        # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        pix1 = img[slice1].ravel()
        pix2 = img[slice2].ravel()
        I = np.flatnonzero((pix1 != 0) * (pix2 != 0))
        mad = np.median(np.abs(pix1[I] - pix2[I]))
        sig1 = 1.4826 * mad / np.sqrt(2.)
        invvar = (1. / sig1**2)
        self.invvar_data = np.zeros(img.shape, np.float32)
        self.invvar_data[trimA] = invvar

    def read_image(self):
        return self.img_data, self.img_hdr

    def scale_image(self, img):
        return img
    def scale_weight(self, img):
        return img

    def remap_bitmask(self, mask):
        return mask
    def remap_invvar(self, invvar, primhdr, img, dq):
        return invvar

    def read_bitmask(self):
        return np.zeros((self.height, self.width), np.int16)

    def read_weight(self, bitmask=None):
        return self.invvar_data

from legacypipe.mosaic import MosaicImage
class MosaicRawImage(MosaicImage):
    def compute_filenames(self):
        self.dqfn = self.imgfn
        self.wtfn = self.imgfn

    def read_image_primary_header(self):
        hdr = super(MosaicRawImage, self).read_image_primary_header()
        hdr['PLPROCID'] = 'xxxxx'
        hdr['DATE'] = 'xxxx'
        return hdr

    def read_image(self, **kwargs):
        img = super(MosaicRawImage, self).read_image(**kwargs)
        img = img.astype(np.float32)
        return img
    
    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        debug('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = np.zeros((self.height, self.width), np.int16)
        return dq
        
    def read_invvar(self, clip=True, clipThresh=0.1, dq=None, slice=None,
                    **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        debug('Reading weight map image', self.wtfn, 'ext', self.hdu)
        iv = np.ones((self.height, self.width), np.float32)
        return iv
    
def main():
    from astrometry.util.multiproc import multiproc
    mp = multiproc()
    imgfn = 'k4m_160504_030532_ori.fits.fz'
    debug = True
    measureargs = dict(measureclass=MosaicRawMeasurer, debug=debug, choose_ccd=False,
                       splinesky=True, calibdir='calib', image_dir='.', camera='mosaic')

    from legacyzpts.legacy_zeropoints import FakeLegacySurveyData
    survey = FakeLegacySurveyData()
    survey.imagedir = '.'
    survey.calibdir = measureargs.get('calibdir')
    survey.image_typemap.update({'mosaic': MosaicRawImage})
    measureargs.update(survey=survey)
    
    #measure = measure_image(imgfn, mp, image_dir='.',
    #                        camera='mosaic', **measureargs)
    #just_measure=True, 
    photomfn = 'photom.fits'
    surveyfn = 'survey.fits'
    annfn = 'ann.fits'
    # from astrometry.util.multiproc import multiproc
    # mp = multiproc()
    # 
    runit(imgfn, photomfn, surveyfn, annfn, mp, **measureargs)

if __name__ == '__main__':
    main()
