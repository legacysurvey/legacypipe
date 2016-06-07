from __future__ import print_function
from glob import glob
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import PlotSequence
import fitsio

from legacypipe.runbrick import run_brick, rgbkwargs, rgbkwargs_resid
from legacypipe.common import LegacySurveyData
from legacypipe.image import LegacySurveyImage

from tractor.sky import ConstantSky
from tractor.sfd import SFDMap

rgbscales = dict(I=(0, 0.01))
rgbkwargs      .update(scales=rgbscales)
rgbkwargs_resid.update(scales=rgbscales)

SFDMap.extinctions.update({'DES I': 1.592})


def make_zeropoints():
    C = fits_table()
    C.image_filename = []
    C.image_hdu = []
    C.camera = []
    C.expnum = []
    C.filter = []
    C.exptime = []
    C.crpix1 = []
    C.crpix2 = []
    C.crval1 = []
    C.crval2 = []
    C.cd1_1 = []
    C.cd1_2 = []
    C.cd2_1 = []
    C.cd2_2 = []
    C.width = []
    C.height = []
    C.ra = []
    C.dec = []
    C.ccdzpt = []
    C.ccdname = []
    C.ccdraoff = []
    C.ccddecoff = []
    C.fwhm = []
    C.propid = []
    C.mjd_obs = []
    
    base = 'euclid/images/'
    fns = glob(base + 'acs-vis/*_sci.VISRES.fits')
    fns.sort()
    for fn in fns:
        hdu = 0
        hdr = fitsio.read_header(fn, ext=hdu)
        C.image_hdu.append(hdu)
        C.image_filename.append(fn.replace(base,''))
        C.camera.append('acs-vis')
        words = fn.split('_')
        expnum = int(words[-2], 10)
        C.expnum.append(expnum)
        filt = words[-4]
        C.filter.append(filt)
        C.exptime.append(hdr['EXPTIME'])
        C.ccdzpt.append(hdr['PHOTZPT'])

        for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
            C.get(k.lower()).append(hdr[k])
        C.width.append(hdr['NAXIS1'])
        C.height.append(hdr['NAXIS2'])
            
        wcs = wcs_pv2sip_hdr(hdr)
        rc,dc = wcs.radec_center()
        C.ra.append(rc)
        C.dec.append(dc)

        psffn = fn.replace('_sci.VISRES.fits', '_sci.VISRES_psfex.psf')
        psfhdr = fitsio.read_header(psffn, ext=1)
        fwhm = psfhdr['PSF_FWHM']
        
        C.ccdname.append('0')
        C.ccdraoff.append(0.)
        C.ccddecoff.append(0.)
        #C.fwhm.append(0.18 / 0.1)
        C.fwhm.append(fwhm)
        C.propid.append('0')
        C.mjd_obs.append(0.)
        
    C.to_np_arrays()
    fn = 'euclid/survey-ccds-acsvis.fits.gz'
    C.writeto(fn)
    print('Wrote', fn)


class AcsVisImage(LegacySurveyImage):
    def __init__(self, *args, **kwargs):
        super(AcsVisImage, self).__init__(*args, **kwargs)

        self.psffn = self.imgfn.replace('_sci.VISRES.fits', '_sci.VISRES_psfex.psf')
        assert(self.psffn != self.imgfn)

        self.wtfn = self.imgfn.replace('_sci', '_wht')
        assert(self.wtfn != self.imgfn)

        self.dqfn = self.imgfn.replace('_sci', '_flg')
        assert(self.dqfn != self.imgfn)
        
        self.name = 'AcsVisImage: expnum %i' % self.expnum

        self.dq_saturation_bits = 0
        
    def __str__(self):
        return self.name

    def get_wcs(self):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        return wcs

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        #return self._read_fits(self.wtfn, self.hdu, **kwargs)

        img = self.read_image(**kwargs)
        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('sig1 estimate:', sig1)
        invvar = np.ones_like(img) / sig1**2
        return invvar
        
    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq
        
    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        sky = ConstantSky(0.)
        return sky
    
if __name__ == '__main__':
    make_zeropoints()

    survey = LegacySurveyData(survey_dir='euclid', output_dir='euclid-out')
    survey.image_typemap['acs-vis'] = AcsVisImage
    
    run_brick(None, survey, radec=(150.64, 1.71), pixscale=0.1,
              width=1000, height=1000, bands=['I'], wise=False, do_calibs=False,
              pixPsf=True, coadd_bw=True, ceres=False,
              blob_image=True, allbands='I',
              forceAll=True, writePickles=False)
    
    #plots=True, plotbase='euclid',
        
