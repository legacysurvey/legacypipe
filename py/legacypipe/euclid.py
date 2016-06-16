from __future__ import print_function
import sys
import os

# YUCK!  scinet bonkers python setup
paths = os.environ['PYTHONPATH']
sys.path = paths.split(':') + sys.path
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import astropy

from glob import glob
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import PlotSequence, plothist
import fitsio
import pylab as plt

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

    if False:
        # Regular tiling with small overlaps; RA,Dec aligned
        survey = LegacySurveyData(survey_dir='euclid', output_dir='euclid-out')
        T = fits_table('euclid/survey-ccds-acsvis.fits.gz')
        plt.clf()
        for ccd in T:
            wcs = survey.get_approx_wcs(ccd)
            h,w = wcs.shape
            rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
            plt.plot(rr, dd, 'b-')
        #plt.savefig('acs-outlines.png')

        T = fits_table('euclid/survey-ccds-megacam.fits.gz')
        #plt.clf()
        for ccd in T:
            wcs = survey.get_approx_wcs(ccd)
            h,w = wcs.shape
            rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
            plt.plot(rr, dd, 'r-')
        plt.savefig('acs-outlines.png')

        TT = []
        fns = glob('euclid-out/tractor/*/tractor-*.fits')
        for fn in fns:
            T = fits_table(fn)
            print(len(T), 'from', fn)
            TT.append(T)
        T = merge_tables(TT)
        plt.clf()
        plothist(T.ra, T.dec, 200)
        plt.savefig('acs-sources1.png')
        T = fits_table('euclid/survey-ccds-acsvis.fits.gz')
        for ccd in T:
            wcs = survey.get_approx_wcs(ccd)
            h,w = wcs.shape
            rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
            plt.plot(rr, dd, 'b-')
        plt.savefig('acs-sources2.png')

        sys.exit(0)

        # It's a 7x7 grid... hackily define RA,Dec boundaries.
        T = fits_table('euclid/survey-ccds-acsvis.fits.gz')
        ras = T.ra.copy()
        ras.sort()
        decs = T.dec.copy()
        decs.sort()
        print('RAs:', ras)
        print('Decs:', decs)
        ras  =  ras.reshape((-1, 7)).mean(axis=1)
        decs = decs.reshape((-1, 7)).mean(axis=1)
        print('RAs:', ras)
        print('Decs:', decs)
        rasplits = (ras[:-1] + ras[1:])/2.
        print('RA boundaries:', rasplits)
        decsplits = (decs[:-1] + decs[1:])/2.
        print('Dec boundaries:', decsplits)



    rasplits = np.array([ 149.72716429, 149.89394223,  150.06073352,
                          150.22752888,  150.39431559, 150.56110037])
                          
    decsplits = np.array([ 1.79290318,  1.95956698,  2.12623253,
                           2.2929002,   2.45956215,  2.62621403])

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expnum', type=int,
                        help='ACS exposure number to run')
    parser.add_argument('--threads', type=int, help='Run multi-threaded')

    parser.add_argument('--forced', type=int, help='Run forced photometry for given MegaCam CCD index')

    parser.add_argument('--ceres', action='store_true', help='Use Ceres?')

    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')

    opt = parser.parse_args()
    if opt.expnum is None and opt.forced is None:
        print('Need --expnum or --forced')
        sys.exit(-1)

    survey = LegacySurveyData(survey_dir='euclid', output_dir='euclid-out')
    survey.image_typemap['acs-vis'] = AcsVisImage
    survey.image_typemap['megacam'] = MegacamImage

    if opt.expnum is not None:
        # Run pipeline on a single ACS image.
        ccds = survey.get_ccds_readonly()
        ccds.cut(ccds.camera == 'acs-vis')
        print('Cut to', len(ccds), 'from ACS-VIS')

        ccds.cut(ccds.expnum == opt.expnum)
        print('Cut to', len(ccds), 'with expnum', opt.expnum)
        allccds = ccds
        
        for iccd in range(len(allccds)):
            # Process just this single CCD.
            survey.ccds = allccds[np.array([iccd])]
            ccd = survey.ccds[0]
            brickname = 'acsvis-%03i' % ccd.expnum
            run_brick(brickname, survey, radec=(ccd.ra, ccd.dec), pixscale=0.1,
                      #width=200, height=200,
                      width=ccd.width, height=ccd.height, 
                      bands=['I'],
                      threads=opt.threads,
                      wise=False, do_calibs=False,
                      pixPsf=True, coadd_bw=True, ceres=False,
                      blob_image=True, allbands=allbands,
                      forceAll=True, writePickles=False)
        #plots=True, plotbase='euclid',

    else:

        # Read all ACS catalogs
        mfn = 'euclid-out/merged-catalog.fits'
        if not os.path.exists(mfn):
            TT = []
            fns = glob('euclid-out/tractor/*/tractor-*.fits')
            for fn in fns:
                T = fits_table(fn)
                print(len(T), 'from', fn)
    
                mra  = np.median(T.ra)
                mdec = np.median(T.dec)
    
                print(np.sum(T.brick_primary), 'PRIMARY')
                I = np.flatnonzero(rasplits > mra)
                if len(I) > 0:
                    T.brick_primary &= (T.ra < rasplits[I[0]])
                    print(np.sum(T.brick_primary), 'PRIMARY after RA high cut')
                I = np.flatnonzero(rasplits < mra)
                if len(I) > 0:
                    T.brick_primary &= (T.ra >= rasplits[I[-1]])
                    print(np.sum(T.brick_primary), 'PRIMARY after RA low cut')
                I = np.flatnonzero(decsplits > mdec)
                if len(I) > 0:
                    T.brick_primary &= (T.dec < decsplits[I[0]])
                    print(np.sum(T.brick_primary), 'PRIMARY after DEC high cut')
                I = np.flatnonzero(decsplits < mdec)
                if len(I) > 0:
                    T.brick_primary &= (T.dec >= decsplits[I[-1]])
                    print(np.sum(T.brick_primary), 'PRIMARY after DEC low cut')
    
                TT.append(T)
            T = merge_tables(TT)
            del TT
            T.writeto(mfn)
        else:
            T = fits_table(mfn)

        # plt.clf()
        # I = T.brick_primary
        # plothist(T.ra[I], T.dec[I], 200)
        # plt.savefig('acs-sources3.png')

        opti = None
        forced_kwargs = {}
        if opt.ceres:
        #if True:
            from tractor.ceres_optimizer import CeresOptimizer
            B = 8
            opti = CeresOptimizer(BW=B, BH=B)
        
        #T = fits_table('euclid-out/tractor/cus/tractor-custom-150640p01710.fits')
        #print(len(T), 'sources in catalog')
        #declo,dechi = cat.dec.min(), cat.dec.max()
        #ralo , rahi = cat.ra .min(), cat.ra .max()
        T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
        T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T

        ccds = survey.get_ccds_readonly()
        I = np.flatnonzero(ccds.camera == 'megacam')
        print(len(I), 'MegaCam CCDs')

        #bands = np.unique(ccds.filter[I])
        #print('Unique bands:', bands)
        #for src in cat:
        #    src.brightness = NanoMaggies(**dict([(b,1.) for b in bands]))

        print('Cut to a single CCD: index', opt.forced)
        I = I[np.array([opt.forced])]

        slc = None
        if opt.zoom:
            x0,x1,y0,y1 = opt.zoom
            zw = x1-x0
            zh = y1-y0
            slc = slice(y0,y1), slice(x0,x1)

        for i in I:
            ccd = ccds[i]
            im = survey.get_image_object(ccd)
            print('CCD', im)

            wcs = im.get_wcs()
            if opt.zoom:
                wcs = wcs.get_subimage(x0, y0, zw, zh)

            ok,x,y = wcs.radec2pixelxy(T.ra, T.dec)
            x = (x-1).astype(np.float32)
            y = (y-1).astype(np.float32)
            h,w = wcs.shape
            J = np.flatnonzero((x >= 0) * (x < w) *
                               (y >= 0) * (y < h))
            if len(J) == 0:
                print('No sources within image.')
                continue

            Ti = T[J]
            print('Cut to', len(Ti), 'sources within image')

            cat = read_fits_catalog(Ti, ellipseClass=EllipseE, allbands=allbands,
                                    bands=allbands)
            for src in cat:
                src.brightness = NanoMaggies(**{ ccd.filter: 1. })

            tim = im.get_tractor_image(pixPsf=True, slc=slc)
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
            F.brickid   = Ti.brickid
            F.brickname = Ti.brickname
            F.objid     = Ti.objid
            
            F.filter  = np.array([tim.band]               * len(Ti))
            F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(Ti))
            F.exptime = np.array([tim.primhdr['EXPTIME']] * len(Ti)).astype(np.float32)

            ok,x,y = tim.sip_wcs.radec2pixelxy(Ti.ra, Ti.dec)
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

            primhdr = fitsio.FITSHDR()
            primhdr.add_record(dict(name='EXPNUM', value=im.expnum,
                                    comment='Exposure number'))
            primhdr.add_record(dict(name='CCDNAME', value=im.ccdname,
                                    comment='CCD name'))
            primhdr.add_record(dict(name='CAMERA', value=im.camera,
                                    comment='Camera'))

            outfn = 'euclid-out/forced/megacam-%i-%s.fits' % (im.expnum, im.ccdname)
            fitsio.write(outfn, None, header=primhdr, clobber=True)
            F.writeto(outfn, header=hdr, append=True)
            #F.writeto(outfn, header=hdr)
            
