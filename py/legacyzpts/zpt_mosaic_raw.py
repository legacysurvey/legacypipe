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
        self.plver = 'xxx'
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

    # From image.py : remove PLVER etc checking & propagation
    def run_psfex(self, git_version=None, ps=None):
        from astrometry.util.file import trymakedirs
        from legacypipe.survey import get_git_version
        sedir = self.survey.get_se_dir()
        trymakedirs(self.psffn, dir=True)
        primhdr = self.read_image_primary_header()
        imghdr = self.read_image_header()
        if git_version is None:
            git_version = get_git_version()
        # We write the PSF model to a .fits.tmp file, then rename to .fits
        psfdir = os.path.dirname(self.psffn)
        psfoutfn = os.path.join(psfdir, os.path.basename(self.sefn).replace('.fits','') + '.fits')
        psftmpfn = psfoutfn + '.tmp'
        cmd = 'psfex -c %s -PSF_DIR %s -PSF_SUFFIX .fits.tmp -VERBOSE_TYPE QUIET %s' % (os.path.join(sedir, self.camera + '.psfex'), psfdir, self.sefn)
        debug(cmd)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: %s: return value: %i' % (cmd,rtn))
        
        # Update the header
        hlist = [
            {'name': 'LEGPIPEV', 'value': git_version, 'comment': "legacypipe git version"},
            {'name': 'EXPNUM',   'value': self.expnum, 'comment': "exponsure number"},
            ]
        F = fitsio.FITS(psftmpfn, 'rw')
        F[0].write_keys(hlist)
        F.close()

        cmd = 'mv %s %s' % (psftmpfn, psfoutfn)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: %s: return value: %i' % (cmd,rtn))

        
if __name__ == '__main__':
    from astrometry.util.multiproc import multiproc
    mp = multiproc()
    imgfn = 'k4m_160504_030532_ori.fits.fz'
    measureargs = dict(measureclass=MosaicRawMeasurer, debug=False, choose_ccd=False,
                       splinesky=True, calibdir='calib')

    from legacyzpts.legacy_zeropoints import FakeLegacySurveyData
    survey = FakeLegacySurveyData()
    survey.imagedir = '.'
    survey.calibdir = measureargs.get('calibdir')
    survey.image_typemap.update({'mosaic': MosaicRawImage})
    measureargs.update(survey=survey)
    
    measure = measure_image(imgfn, mp, image_dir='.',
                            camera='mosaic', **measureargs)
    #just_measure=True, 
    # photomfn = 'photom.fits'
    # surveyfn = 'survey.fits'
    # annfn = 'ann.fits'
    # from astrometry.util.multiproc import multiproc
    # mp = multiproc()
    # 
    # runit(F.imgfn, photomfn, surveyfn, annfn, mp, **measureargs)

