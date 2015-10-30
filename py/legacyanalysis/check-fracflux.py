import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import *
from tractor.galaxy import *
from astrometry.util.util import *
from astrometry.util.fits import *

from legacypipe.runbrick import _compute_source_metrics

if __name__ == '__main__':

    W,H = 50,50
    cx,cy = W/2., H/2.

    gala = ExpGalaxy(RaDecPos(0., 0.), Flux(200.), EllipseESoft(1., 0., 0.5))
    galb = ExpGalaxy(RaDecPos(0., 0.), Flux(100.), EllipseESoft(1., 0., 0.5))
    halfsize = 25
    
    gpsf = NCircularGaussianPSF([2.], [1.])
    gpsf.radius = halfsize
    psfimg = gpsf.getPointSourcePatch(0., 0., radius=15)
    print 'PSF image size', psfimg.shape
    pixpsf = PixelizedPSF(psfimg.patch)
    data=np.zeros((H,W), np.float32)

    wcs = ConstantFitsWcs(Tan(0., 0., W/2., H/2.,
                              1e-4/3600., 0.262/3600., 0.263/3600., 1e-5/3600.,
                              float(W), float(H)))

    img = Image(data=data, invvar=np.ones_like(data), psf=gpsf, wcs=wcs)
    img.band = 'r'

    srcs = [gala, galb]
    tims = [img]
    tr = Tractor(tims, srcs)

    B = fits_table()
    B.sources = srcs
    print 'B.sources:', B.sources
    
    bands = 'r'

    allfracs = []

    img.inverr[H/2:,:] = 0.
    
    for ra in np.linspace(-0.001, 0.001, 7):
        gala.pos.ra = ra

        M = _compute_source_metrics(srcs, tims, bands, tr)
        for k,v in M.items():
            B.set(k, v)

        allfracs.append(B.fracflux)

    allfracs = np.hstack(allfracs).T
    print 'Allfracs:', allfracs.shape

    plt.clf()
    plt.plot(allfracs[:,0], 'b-')
    plt.plot(allfracs[:,1], 'r-')
    plt.savefig('fracs.png')
    

    # Put one galaxy on the edge of the image.
    galb.pos = wcs.pixelToPosition(0, H/2.)
    print 'Gal b:', wcs.positionToPixel(galb.pos)
    allfracs = []
    for ra in np.linspace(-0.002, 0.002, 15):
        gala.pos.ra = ra
        print 'Gal a:', wcs.positionToPixel(gala.pos)
        M = _compute_source_metrics(srcs, tims, bands, tr)
        for k,v in M.items():
            B.set(k, v)
        allfracs.append(B.fracflux)
    allfracs = np.hstack(allfracs).T
    plt.clf()
    plt.plot(allfracs[:,0], 'b-')
    plt.plot(allfracs[:,1], 'r-')
    plt.savefig('fracs2.png')

    
