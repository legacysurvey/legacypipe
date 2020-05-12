import fitsio
import logging
import sys
import pylab as plt
import numpy as np
import tractor
from astrometry.util.fits import fits_table
from tractor import GaussianMixturePSF, ConstantFitsWcs, ConstantSky, LinearPhotoCal
from tractor import RaDecPos, NanoMaggies, ExpGalaxy, Tractor
from tractor.ellipses import EllipseE
from tractor.tractortime import TAITime
from legacypipe.survey import BrickDuck, wcs_for_brick, LegacySurveyData
from legacypipe.image import LegacySurveyImage
from legacypipe.runbrick import run_brick

def main():
    # I read a DESI DR8 target catalog, cut to ELGs, then took a narrow
    # r-mag slice around the peak of that distribution;
    # desi/target/catalogs/dr8/0.31.1/targets/main/resolve/targets-dr8-hp-1,5,11,50,55,60,83,84,86,89,91,98,119,155,158,174,186,187,190.fits')
    # Then took the median g and z mags
    # And opened the coadd invvar mags for a random brick in that footprint
    # (0701p000) to find the median per-pixel invvars.

    r = 23.0
    g = 23.4
    z = 22.2

    # Selecting EXPs, the peak of the shapeexp_r was ~ 0.75".
    r_e = 0.75

    # Image properties:
    
    giv = 77000.
    riv = 27000.
    ziv  = 8000.

    # PSF sizes were roughly equal, 1.16, 1.17, 1.19"
    # -> sigma = 1.9 DECam pixels
    psf_sigma = 1.9

    H,W = 1000,1000

    seed = 42
    np.random.seed(seed)
    
    ra,dec = 70., 1.
    brick = BrickDuck(ra, dec, 'custom-0700p010')
    wcs = wcs_for_brick(brick, W=W, H=H)
    bands = 'grz'

    tims = []
    for band,iv in zip(bands, [giv, riv, ziv]):
        img = np.zeros((H,W), np.float32)
        tiv = np.zeros_like(img) + iv
        s = psf_sigma**2
        psf = GaussianMixturePSF(1., 0., 0., s, s, 0.)
        twcs = ConstantFitsWcs(wcs)
        sky = ConstantSky(0.)
        photocal = LinearPhotoCal(1., band=band)

        tai = TAITime(None, mjd=55700.)
        
        tim = tractor.Image(data=img, invvar=tiv, psf=psf, wcs=twcs,
                            sky=sky, photocal=photocal,
                            name='fake %s' % band, time=tai)
        tim.skyver = ('1','1')
        tim.psfver = ('1','1')
        tim.plver = '1'
        tim.x0 = tim.y0 = 0
        tim.subwcs = wcs
        tim.psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        tim.galnorm = tim.psfnorm
        tim.propid = '2020A-000'
        tim.band = band
        tim.dq = None
        tim.sig1 = 1./np.sqrt(iv)
        tim.psf_sigma = psf_sigma
        tim.primhdr = fitsio.FITSHDR()

        tims.append(tim)



    # Simulated catalog
    gflux = NanoMaggies.magToNanomaggies(g)
    rflux = NanoMaggies.magToNanomaggies(r)
    zflux = NanoMaggies.magToNanomaggies(z)
    box = 50
    CX,CY = np.meshgrid(np.arange(box//2, W, box),
                        np.arange(box//2, H, box))
    ny,nx = CX.shape
    BA,PHI = np.meshgrid(np.linspace(0.1, 1.0, nx),
                         np.linspace(0., 180., ny))
    cat = []
    for cx,cy,ba,phi in zip(CX.ravel(),CY.ravel(),BA.ravel(),PHI.ravel()):
        #print('x,y %.1f,%.1f, ba %.2f, phi %.2f' % (cx, cy, ba, phi))
        r,d = wcs.pixelxy2radec(cx+1, cy+1)
        src = ExpGalaxy(RaDecPos(r, d),
                        NanoMaggies(order=bands, g=gflux, r=rflux, z=zflux),
                        EllipseE.fromRAbPhi(r_e, ba, phi))
        cat.append(src)

    from legacypipe.catalog import prepare_fits_catalog
    TT = fits_table()
    TT.bx = CX.ravel()
    TT.by = CY.ravel()
    TT.ba = BA.ravel()
    TT.phi = PHI.ravel()
    
    tcat = tractor.Catalog(*cat)
    T2 = prepare_fits_catalog(tcat, None, TT, bands, save_invvars=False)
    T2.writeto('sim-cat.fits')
        
    tr = Tractor(tims, cat)

    for band,tim in zip(bands, tims):
        mod = tr.getModelImage(tim)
        mod += np.random.standard_normal(size=tim.shape) * 1./tim.getInvError()
        fitsio.write('sim-%s.fits' % band, mod, clobber=True)
        tim.data = mod

    ccds = fits_table()
    ccds.filter = np.array([f for f in bands])
    ccds.ccd_cuts = np.zeros(len(ccds), np.int16)
    ccds.imgfn = np.array([tim.name for tim in tims])
    ccds.propid = np.array(['2020A-000'] * len(ccds))
    ccds.fwhm = np.zeros(len(ccds), np.float32) + psf_sigma * 2.35
    ccds.mjd_obs = np.zeros(len(ccds))
    ccds.camera = np.array(['fake'] * len(ccds))

    survey = FakeLegacySurvey(ccds, tims)

    import logging
    verbose = False
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)
    
    run_brick(None, survey, radec=(ra,dec), width=W, height=H,
              do_calibs=False,
              gaia_stars=False, large_galaxies=False, tycho_stars=False,
              forceall=True, outliers=False)#, stages=['image_coadds'])

class FakeImage(LegacySurveyImage):
    def __init__(self, ccd, tim):
        self.ccd = ccd
        self.tim = tim
        self.imgfn = ccd.imgfn
        self.band = ccd.filter
        self.exptime = 42.
        self.pixscale = 0.262
        self.name = tim.name
        self.camera = ccd.camera
        self.tim.imobj = self
        self.fwhm = ccd.fwhm

    def get_tractor_image(self, **kwargs):
        return self.tim

class FakeLegacySurvey(LegacySurveyData):
    def __init__(self, ccds, tims, **kwargs):
        super().__init__(**kwargs)
        self.image_typemap['fake'] = FakeImage
        assert(len(ccds) == len(tims))
        self.tims = tims
        self.ccds = ccds
        self.ccds.index = np.arange(len(self.ccds))

    def ccds_touching_wcs(self, targetwcs, **kwargs):
        return self.ccds

    def get_image_object(self, ccd):
        return FakeImage(ccd, self.tims[ccd.index])

if __name__ == '__main__':
    main()
