from __future__ import print_function
import numpy as np

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.util import Tan
from astrometry.util.miscutils import polygon_area
from legacypipe.common import Decals

def main():
    decals = Decals()
    ccds = decals.get_ccds()

    print("HACK!")
    ccds.cut(np.array([name in ['N15', 'N16', 'N21', 'N9']
                       for name in ccds.ccdname]) *
                       ccds.expnum == 229683)
    
    I = decals.photometric_ccds(ccds)
    ccds.photometric = np.zeros(len(ccds), bool)
    ccds.photometric[I] = True

    I = decals.apply_blacklist(ccds)
    ccds.blacklist_ok = np.zeros(len(ccds), bool)
    ccds.blacklist_ok[I] = True

    ccds.good_region = np.empty((len(ccds), 4), np.int16)
    ccds.good_region[:,:] = -1

    ccds.ra0  = np.zeros(len(ccds), np.float64)
    ccds.dec0 = np.zeros(len(ccds), np.float64)
    ccds.ra1  = np.zeros(len(ccds), np.float64)
    ccds.dec1 = np.zeros(len(ccds), np.float64)
    ccds.ra2  = np.zeros(len(ccds), np.float64)
    ccds.dec2 = np.zeros(len(ccds), np.float64)
    ccds.ra3  = np.zeros(len(ccds), np.float64)
    ccds.dec3 = np.zeros(len(ccds), np.float64)

    ccds.dra  = np.zeros(len(ccds), np.float32)
    ccds.ddec = np.zeros(len(ccds), np.float32)
    ccds.ra_center  = np.zeros(len(ccds), np.float64)
    ccds.dec_center = np.zeros(len(ccds), np.float64)

    ccds.meansky = np.zeros(len(ccds), np.float32)
    ccds.stdsky  = np.zeros(len(ccds), np.float32)
    ccds.maxsky  = np.zeros(len(ccds), np.float32)
    ccds.minsky  = np.zeros(len(ccds), np.float32)

    ccds.pixscale_mean = np.zeros(len(ccds), np.float32)
    ccds.pixscale_std  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_max  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_min  = np.zeros(len(ccds), np.float32)
    
    for iccd,ccd in enumerate(ccds):
        im = decals.get_image_object(ccd)

        X = im.get_good_image_subregion()
        for i,x in enumerate(X):
            if x is not None:
                ccds.good_region[iccd,i] = x

        psf = None
        try:
            psf = im.read_psf_model(0, 0, pixPsf=True)
        except:
            import traceback
            traceback.print_exc()

        if psf is not None:
            print('Got PSF', psf)
            # Instantiate PSF on a grid
            

        sky = None
        try:
            sky = im.read_sky_model(splinesky=True)
        except:
            import traceback
            traceback.print_exc()

        if sky is not None:
            print('Got sky', sky)
            mod = np.zeros((ccd.height, ccd.width), np.float32)
            sky.addTo(mod)
            print('Instantiated')
            ccds.meansky[iccd] = np.mean(mod)
            ccds.stdsky[iccd]  = np.std(mod)
            ccds.maxsky[iccd]  = mod.max()
            ccds.minsky[iccd]  = mod.min()
            
        wcs = None
        try:
            wcs = im.read_pv_wcs()
        except:
            import traceback
            traceback.print_exc()

        if wcs is not None:
            print('Got WCS', wcs)
            W,H = ccd.width, ccd.height
            ccds.ra0[iccd],ccds.dec0[iccd] = wcs.pixelxy2radec(1, 1)
            ccds.ra1[iccd],ccds.dec1[iccd] = wcs.pixelxy2radec(1, H)
            ccds.ra2[iccd],ccds.dec2[iccd] = wcs.pixelxy2radec(W, H)
            ccds.ra3[iccd],ccds.dec3[iccd] = wcs.pixelxy2radec(W, 1)

            midx, midy = (W+1)/2., (H+1)/2.
            rc,dc  = wcs.pixelxy2radec(midx, midy)
            ra,dec = wcs.pixelxy2radec([1,W,midx,midx], [midy,midy,1,H])
            ccds.dra [iccd] = max(degrees_between(ra, dc+np.zeros_like(ra),
                                                  rc, dc))
            ccds.ddec[iccd] = max(degrees_between(rc+np.zeros_like(dec), dec,
                                                  rc, dc))
            ccds.ra_center [iccd] = rc
            ccds.dec_center[iccd] = dc

            # Compute scale change across the chip
            # how many pixels to step
            step = 10
            xx = np.linspace(1+step, W-step, 5)
            yy = np.linspace(1+step, H-step, 5)
            xx,yy = np.meshgrid(xx, yy)
            pixscale = []
            for x,y in zip(xx.ravel(), yy.ravel()):
                sx = [x-step, x-step, x+step, x+step, x-step]
                sy = [y-step, y+step, y+step, y-step, y-step]
                sr,sd = wcs.pixelxy2radec(sx, sy)
                rc,dc = wcs.pixelxy2radec(x, y)
                # project around a tiny little TAN WCS at (x,y), with 1" pixels
                locwcs = Tan(rc, dc, 0., 0., 1./3600, 0., 0., 1./3600, 1., 1.)
                ok,lx,ly = locwcs.radec2pixelxy(sr, sd)
                #print('local x,y:', lx, ly)
                A = polygon_area((lx, ly))
                pixscale.append(np.sqrt(A / (2*step)**2))
            print('Pixel scales:', pixscale)
            ccds.pixscale_mean[iccd] = np.mean(pixscale)
            ccds.pixscale_min[iccd] = min(pixscale)
            ccds.pixscale_max[iccd] = max(pixscale)
            ccds.pixscale_std[iccd] = np.std(pixscale)
            
    ccds.writeto('ccds-annotated.fits')


if __name__ == '__main__':
    import sys
    sys.exit(main())
