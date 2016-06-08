from __future__ import print_function
from glob import glob
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import PlotSequence
import fitsio

from legacypipe.runbrick import run_brick, rgbkwargs, rgbkwargs_resid
from legacypipe.common import LegacySurveyData
from legacypipe.image import LegacySurveyImage

from legacypipe.desi_common import read_fits_catalog

from tractor.sky import ConstantSky
from tractor.sfd import SFDMap
from tractor import Tractor, NanoMaggies
from tractor.galaxy import disable_galaxy_cache
from tractor.ellipses import EllipseE

rgbscales = dict(I=(0, 0.01))
rgbkwargs      .update(scales=rgbscales)
rgbkwargs_resid.update(scales=rgbscales)

SFDMap.extinctions.update({'DES I': 1.592})

allbands = 'I'


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
    fns = glob(base + 'megacam/???????p.fits')
    fns.sort()
    for fn in fns:
        F = fitsio.FITS(fn)
        print(len(F), 'FITS extensions in', fn)
        print(F)
        phdr = F[0].read_header()
        expnum = phdr['EXPNUM']
        filt = phdr['FILTER']
        print('Filter', filt)
        filt = filt[0]
        exptime = phdr['EXPTIME']
        for hdu in range(1, len(F)):
            hdr = fitsio.read_header(fn, ext=hdu)
            C.image_filename.append(fn.replace(base,''))
            C.image_hdu.append(hdu)
            C.camera.append('megacam')
            C.expnum.append(expnum)
            C.filter.append(filt)
            C.exptime.append(exptime)
            C.ccdzpt.append(hdr['PHOTZPT'])

            for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
                C.get(k.lower()).append(hdr[k])
            C.width.append(hdr['NAXIS1'])
            C.height.append(hdr['NAXIS2'])
            
            wcs = wcs_pv2sip_hdr(hdr)
            rc,dc = wcs.radec_center()
            C.ra.append(rc)
            C.dec.append(dc)

            psffn = fn.replace('p.fits', 'p_psfex.psf')
            psfhdr = fitsio.read_header(psffn, ext=1)
            fwhm = psfhdr['PSF_FWHM']
        
            C.ccdname.append(hdr['EXTNAME'])
            C.ccdraoff.append(0.)
            C.ccddecoff.append(0.)
            C.fwhm.append(fwhm)
            C.propid.append('0')
            C.mjd_obs.append(hdr['MJD-OBS'])
        
    C.to_np_arrays()
    fn = 'euclid/survey-ccds-megacam.fits.gz'
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

class MegacamImage(LegacySurveyImage):
    def __init__(self, *args, **kwargs):
        super(MegacamImage, self).__init__(*args, **kwargs)

        self.psffn = self.imgfn.replace('p.fits', 'p_psfex.psf')
        assert(self.psffn != self.imgfn)

        self.wtfn = self.imgfn.replace('p.fits', 'p_weight.fits')
        assert(self.wtfn != self.imgfn)

        self.dqfn = self.imgfn.replace('p.fits', 'p_flag.fits')
        assert(self.dqfn != self.imgfn)
        
        self.name = 'MegacamImage: %i-%s' % (self.expnum, self.ccdname)
        self.dq_saturation_bits = 0
        
    def get_wcs(self):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        return wcs

    def read_image(self, header=None, **kwargs):
        print('Reading image from', self.imgfn, 'hdu', self.hdu)
        R = self._read_fits(self.imgfn, self.hdu, header=header, **kwargs)
        if header:
            img,header = R
        else:
            img = R
        img = img.astype(np.float32)
        if header:
            return img,header
        return img
    
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
        print('Got', dq.dtype, dq.shape)
        return dq
        
    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        sky = ConstantSky(0.)
        return sky

    
if __name__ == '__main__':
    if False:
        make_zeropoints()

    survey = LegacySurveyData(survey_dir='euclid', output_dir='euclid-out')
    survey.image_typemap['acs-vis'] = AcsVisImage
    survey.image_typemap['megacam'] = MegacamImage

    if False:
        # CUT to just the ACS-VIS ones!
        ccds = survey.get_ccds_readonly()
        ccds.cut(ccds.camera == 'acs-vis')
        print('Cut to', len(ccds), 'from ACS-VIS')
    
        run_brick(None, survey, radec=(150.64, 1.71), pixscale=0.1,
                  width=1000, height=1000, bands=['I'], wise=False, do_calibs=False,
                  pixPsf=True, coadd_bw=True, ceres=False,
                  blob_image=True, allbands=allbands,
                  forceAll=True, writePickles=False)
        #plots=True, plotbase='euclid',

    else:

        opti = None
        forced_kwargs = {}
        #if opt.ceres:
        if True:
            from tractor.ceres_optimizer import CeresOptimizer
            B = 8
            opti = CeresOptimizer(BW=B, BH=B)
        
        T = fits_table('euclid-out/tractor/cus/tractor-custom-150640p01710.fits')
        print(len(T), 'sources in catalog')

        #declo,dechi = cat.dec.min(), cat.dec.max()
        #ralo , rahi = cat.ra .min(), cat.ra .max()

        T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
        T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
        cat = read_fits_catalog(T, ellipseClass=EllipseE, allbands=allbands,
                                bands=allbands)

        
        ccds = survey.get_ccds_readonly()
        I = np.flatnonzero(ccds.camera == 'megacam')

        bands = np.unique(ccds.filter[I])
        print('Unique bands:', bands)
        for src in cat:
            src.brightness = NanoMaggies(**dict([(b,1.) for b in bands]))

        for i in I:
            ccd = ccds[i]
            im = survey.get_image_object(ccd)

            wcs = im.get_wcs()
            ok,x,y = wcs.radec2pixelxy(T.ra, T.dec)
            x = (x-1).astype(np.float32)
            y = (y-1).astype(np.float32)
            J = np.flatnonzero((x >= 0) * (x < ccd.width) *
                               (y >= 0) * (y < ccd.height))
            if len(J) == 0:
                print('No sources within image.')
                continue

            tim = im.get_tractor_image(pixPsf=True)
            print('Forced photometry for', tim.name)

            for src in cat:
                # Limit sizes of huge models
                from tractor.galaxy import ProfileGalaxy
                if isinstance(src, ProfileGalaxy):
                    px,py = tim.wcs.positionToPixel(src.getPosition())
                    h = src._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
                    MAXHALF = 128
                    if h > MAXHALF:
                        print('halfsize', h,'for',src,'-> setting to',MAXHALF)
                        src.halfsize = MAXHALF
            
            tr = Tractor([tim], cat, optimizer=opti)
            tr.freezeParam('images')
            for src in cat:
                src.freezeAllBut('brightness')
                src.getBrightness().freezeAllBut(tim.band)
            disable_galaxy_cache()
            # Reset fluxes
            nparams = tr.numberOfParams()
            tr.setParams(np.zeros(nparams, np.float32))
            
            F = fits_table()
            F.brickid   = T.brickid
            F.brickname = T.brickname
            F.objid     = T.objid
            
            F.filter  = np.array([tim.band]               * len(T))
            F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(T))
            F.exptime = np.array([tim.primhdr['EXPTIME']] * len(T)).astype(np.float32)

            ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
            F.x = (x-1).astype(np.float32)
            F.y = (y-1).astype(np.float32)
            
            R = tr.optimize_forced_photometry(variance=True, fitstats=True,
                                              shared_params=False, priors=False,
                                              **forced_kwargs)

            F.flux = np.array([src.getBrightness().getFlux(tim.band)
                               for src in cat]).astype(np.float32)
            F.flux_ivar = R.IV.astype(np.float32)
            #F.fracflux = R.fitstats.profracflux.astype(np.float32)
            #F.rchi2    = R.fitstats.prochi2    .astype(np.float32)

            hdr = fitsio.FITSHDR()
            units = {'exptime':'sec', 'flux':'nanomaggy', 'flux_ivar':'1/nanomaggy^2'}
            columns = F.get_columns()
            for i,col in enumerate(columns):
                if col in units:
                    hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

            outfn = 'euclid-out/forced/megacam-%i-%s.fits' % (im.expnum, im.ccdname)
            #fitsio.write(outfn, None, header=hdr, clobber=True)
            F.writeto(outfn, header=hdr, append=True)
            
