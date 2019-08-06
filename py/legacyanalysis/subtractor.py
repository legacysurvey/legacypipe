import tempfile
import os
import sys

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import fitsio

from legacypipe.survey import LegacySurveyData
from legacyzpts.legacy_zeropoints import runit, DecamMeasurer
from legacypipe.decam import DecamImage
from legacypipe.forced_photom import get_catalog_in_wcs
from legacypipe.catalog import read_fits_catalog

from tractor import Tractor

from astrometry.util.file import trymakedirs
from astrometry.util.multiproc import multiproc


class GoldsteinDecamImage(DecamImage):
    def compute_filenames(self):
        self.dqfn = self.imgfn.replace('.fits', '.bpm.fits')
        self.wtfn = self.imgfn.replace('.fits', '.weight.fits')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

    def funpack_files(self, imgfn, dqfn, hdu, todelete):
        return imgfn, dqfn

    def read_image_primary_header(self):
        hdr = super(GoldsteinDecamImage, self).read_image_primary_header()
        hdr['PLPROCID'] = 'xxx'
        hdr['DATE'] = 'xxx'
        return hdr

    def remap_dq(self, dq, hdr):
        return dq
    
class GoldsteinDecamMeasurer(DecamMeasurer):
    def __init__(self, *args, **kwargs):
        super(GoldsteinDecamMeasurer, self).__init__(*args, **kwargs)
        self.plver = 'V0.0'
        self.procdate = 'xxx'
        self.plprocid = 'xxx'

    def read_primary_header(self):
        hdr = super(GoldsteinDecamMeasurer, self).read_primary_header()
        hdr['WCSCAL'] = 'success'
        return hdr

    def read_header(self, ext):
        hdr = fitsio.read_header(self.fn, ext=ext)
        hdr['FWHM'] = hdr['SEEING'] / self.pixscale
        return hdr

    def get_bitmask_fn(self, imgfn):
        return imgfn.replace('.fits', '.bpm.fits')

    def read_bitmask(self):
        dqfn = self.get_bitmask_fn(self.fn)
        #### No extension
        if self.slc is not None:
            mask = fitsio.FITS(dqfn)[self.slc]
        else:
            mask = fitsio.read(dqfn)
        mask = self.remap_bitmask(mask)
        return mask

    def remap_bitmask(self, mask):
        return mask

    def get_weight_fn(self, imgfn):
        return imgfn.replace('.fits', '.weight.fits')


        
def main():
    fn = '/global/cscratch1/sd/dstn/c4d_190730_024955_ori/c4d_190730_024955_ori.52.fits'
    survey_dir = '/global/cscratch1/sd/dstn/subtractor-survey-dir'

    imagedir = os.path.join(survey_dir, 'images')
    trymakedirs(imagedir)
    calibdir = os.path.join(survey_dir, 'calib')

    psfexdir = os.path.join(calibdir, 'decam', 'psfex-merged')
    trymakedirs(psfexdir)
    skydir = os.path.join(calibdir, 'decam', 'splinesky-merged')
    trymakedirs(skydir)
    
    basename = os.path.basename(fn)
    basename = basename.replace('.fits', '')
    
    f,photfn = tempfile.mkstemp()
    os.close(f)
    surveyfn = os.path.join(survey_dir, 'survey-ccds-%s.fits.gz' % basename)
    annfn = os.path.join(survey_dir, 'annotated-%s.fits' % basename)
    mp = multiproc()

    survey = LegacySurveyData(survey_dir)
    #survey.calibdir = calibdir
    survey.image_typemap.update(decam=GoldsteinDecamImage)
    
    hdr = fitsio.read_header(fn)
    
    #ccdname = 'N21'
    expnum = hdr['EXPNUM']
    ccdname = hdr['EXTNAME'].strip()

    print('Exposure', expnum, 'CCD', ccdname)
    
    import logging
    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if not os.path.exists(surveyfn):
        # Run calibrations and zeropoints
        runit(fn, photfn, surveyfn, annfn, mp, survey=survey, camera='decam', debug=False,
              choose_ccd=ccdname, splinesky=True, calibdir=calibdir,
              measureclass=GoldsteinDecamMeasurer)

    # Find catalog sources touching this CCD
    ccds = survey.find_ccds(expnum=expnum, ccdname=ccdname)
    assert(len(ccds) == 1)
    ccd = ccds[0]
    print('Got CCD', ccd)

    im = survey.get_image_object(ccd)
    print('Got image:', im)

    #zoomslice=None
    zoomslice = (slice(0, 1000), slice(0, 1000))
    
    tim = im.get_tractor_image(slc=zoomslice, pixPsf=True, splinesky=True,
                               hybridPsf=True,
                               normalizePsf=True,
                               old_calibs_ok=True)
    print('Got tim:', tim)

    catsurvey = LegacySurveyData('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/south')
    T = get_catalog_in_wcs(tim.subwcs, catsurvey)
    print('Got', len(T), 'DR8 catalog sources within CCD')

    # Gaia stars: move RA,Dec to the epoch of this image.
    I = np.flatnonzero(T.ref_epoch > 0)
    if len(I):
        from legacypipe.survey import radec_at_mjd
        print('Moving', len(I), 'Gaia stars to MJD', tim.time.toMjd())
        ra,dec = radec_at_mjd(T.ra[I], T.dec[I], T.ref_epoch[I].astype(float),
                              T.pmra[I], T.pmdec[I], T.parallax[I],
                              tim.time.toMjd())
        T.ra [I] = ra
        T.dec[I] = dec

    cat = read_fits_catalog(T, bands=tim.band)
    print('Created', len(cat), 'source objects')

    tr = Tractor([tim], cat)
    mod = tr.getModelImage(0)
    
    ima = dict(interpolation='nearest', origin='lower', vmin=-2*tim.sig1, vmax=10*tim.sig1,
               cmap='gray')
    plt.clf()
    plt.imshow(tim.getImage(), **ima)
    plt.title('Image')
    plt.savefig('img.jpg')

    plt.clf()
    plt.imshow(mod, **ima)
    plt.title('Model')
    plt.savefig('mod.jpg')

    plt.clf()
    plt.imshow(tim.getImage() - mod, **ima)
    plt.title('Residual')
    plt.savefig('res.jpg')
    
if __name__ == '__main__':
    main()
    







