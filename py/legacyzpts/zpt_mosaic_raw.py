import matplotlib
matplotlib.use('Agg')

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
        img,invvar = read_mosaic_raw_image(img, hdr)
        self.img_data,self.img_hdr = img, hdr
        self.invvar_data = invvar

    def get_wcs(self):
        wcs = wcs_pv2sip_hdr(self.hdr)

        # Find astrometric offset!!
        import sqlite3
        conn = sqlite3.connect('obsdb/mosaic3.sqlite3')
        c = conn.cursor()
        print('Expnum is', self.expnum)
        for dx,dy in c.execute('select dx,dy from obsdb_measuredccd where expnum=?',
                             (self.expnum,)):
            print('Found obsdb astrometric offset:', dx,dy)
            px,py = wcs.get_crpix()
            wcs.set_crpix((px-dx, py-dy))
            break
        return wcs
        #return self.wcs_data

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

    def read_weight(self, bitmask=None, scale=False):
        return self.invvar_data

def read_mosaic_raw_image(img, hdr):
    # Subtract median overscan and multiply by gains
    dataA = parse_section(hdr['DATASEC'], slices=True)
    biasA = parse_section(hdr['BIASSEC'], slices=True)
    gainA = hdr['GAIN']
    b = np.median(img[biasA])
    img[dataA] = (img[dataA] - b) * gainA
    # Trim the image
    trimA = parse_section(hdr['TRIMSEC'], slices=True)
    # zero out all but the trim section
    trimg = img[trimA].copy()
    img[:,:] = 0
    img[trimA] = trimg

    # Zero out some edge pixels too!
    edge = 20
    img[:edge,:] = 0.
    img[-edge:,:] = 0.
    img[:,:edge] = 0.
    img[:,-edge:] = 0.

    img = img.astype(np.float32)
        
    # Estimate per-pixel noise via Blanton's 5-pixel MAD
    slice1 = (slice(0,-5,10),slice(0,-5,10))
    slice2 = (slice(5,None,10),slice(5,None,10))
    pix1 = img[slice1].ravel()
    pix2 = img[slice2].ravel()
    I = np.flatnonzero((pix1 != 0) * (pix2 != 0))
    mad = np.median(np.abs(pix1[I] - pix2[I]))
    sig1 = 1.4826 * mad / np.sqrt(2.)
    iv = (1. / sig1**2)
    invvar = np.zeros_like(img)
    invvar[img != 0.0] = iv
    return img,invvar

# from obsbot.measure_raw
def parse_section(s, slices=False):
    '''
    parse '[57:2104,51:4146]' into integers; also subtract 1.
    '''
    s = s.replace('[','').replace(']','').replace(',',' ').replace(':',' ')
    #print('String', s)
    i = [int(si)-1 for si in s.split()]
    assert(len(i) == 4)
    if not slices:
        return i
    slc = slice(i[2], i[3]+1), slice(i[0], i[1]+1)
    #print('Slice', slc)
    return slc

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
        img,hdr = super(MosaicRawImage, self).read_image(header=True)
        img,invvar = read_mosaic_raw_image(img, hdr)
        self.invvar_data = invvar
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
        #debug('Reading weight map image', self.wtfn, 'ext', self.hdu)
        #iv = np.ones((self.height, self.width), np.float32)
        #return iv
        return self.invvar_data
        
def main():
    from astrometry.util.multiproc import multiproc
    mp = multiproc()
    #imgfn = '20160503/k4m_160504_030532_ori.fits.fz'
    #imgfn = '20160503/k4m_160504_031306_ori.fits.fz'
    #imgfn = '20160503/k4m_160504_031425_ori.fits.fz'
    for imgfn in [
        '20160503/k4m_160504_030532_ori.fits.fz',
        '20160503/k4m_160504_031306_ori.fits.fz',
        '20160503/k4m_160504_031345_ori.fits.fz',
        '20160503/k4m_160504_031425_ori.fits.fz',
        '20160503/k4m_160504_031512_ori.fits.fz',
        '20160503/k4m_160504_031552_ori.fits.fz',
        '20160503/k4m_160504_031632_ori.fits.fz',
        '20160503/k4m_160504_031722_ori.fits.fz',
        '20160503/k4m_160504_031802_ori.fits.fz',
        '20160503/k4m_160504_031843_ori.fits.fz',
        '20160503/k4m_160504_031927_ori.fits.fz',
        '20160503/k4m_160504_032007_ori.fits.fz',
        '20160503/k4m_160504_032058_ori.fits.fz',
        '20160503/k4m_160504_032137_ori.fits.fz',
        '20160503/k4m_160504_032223_ori.fits.fz',
    ]:
        debug = False
        plots = False

        basefn = imgfn.replace('.fits.fz', '')
    
        photomfn = basefn + '-photom.fits'
        surveyfn = basefn + '-survey.fits'
        annfn = basefn + '-ann.fits'
    
        if os.path.exists(annfn):
            continue
        
        image_dir = '/global/project/projectdirs/cosmo/staging/mosaic/MZLS_Raw/'
        measureargs = dict(measureclass=MosaicRawMeasurer, debug=debug, choose_ccd=False,
                           splinesky=True, calibdir='calib', image_dir=image_dir, camera='mosaic',
                           plots=plots)
    
        from legacyzpts.legacy_zeropoints import FakeLegacySurveyData
        survey = FakeLegacySurveyData()
        survey.imagedir = image_dir
        survey.calibdir = measureargs.get('calibdir')
        survey.image_typemap.update({'mosaic': MosaicRawImage})
        measureargs.update(survey=survey)
        
        #measure = measure_image(imgfn, mp, image_dir='.',
        #                        camera='mosaic', **measureargs)
    
        runit(imgfn, photomfn, surveyfn, annfn, mp, **measureargs)
    
        if plots:
            from astrometry.util.fits import fits_table
            P = fits_table(photomfn)
            S = fits_table(surveyfn)
            zpt = S.ccdzpt[0]
            I = np.flatnonzero((P.legacy_survey_mag > 10.) * (P.legacy_survey_mag < 25.))
            import pylab as plt
            plt.clf()
            plt.plot(P.legacy_survey_mag[I], P.instpsfmag[I] + zpt - P.legacy_survey_mag[I], 'b.')
            plt.xlabel('PS1 predicted mag')
            plt.ylabel('Calibrated mag - PS1 predicted mag')
            plt.savefig('zpt.png')

    
if __name__ == '__main__':
    main()
