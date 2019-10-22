import numpy as np
import pylab as plt

import tractor
from legacypipe.survey import RexGalaxy, LogRadius
from astrometry.util.util import Tan

'''
An example of rendering a Rex galaxy in an image.

'''
if __name__ == '__main__':
    H,W = 100,100

    pixscale = 0.262
    ra,dec = 40., 10.
    psf_sigma = 1.4 # pixels
    v = psf_sigma**2

    ps = pixscale / 3600.
    wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5, -ps, 0., 0., ps, float(W), float(H))

    tim = tractor.Image(data=np.zeros((H,W), np.float32),
                        inverr=np.ones((H,W), np.float32),
                        psf=tractor.GaussianMixturePSF(1., 0., 0., v, v, 0.),
                        wcs=tractor.ConstantFitsWcs(wcs))
    src = RexGalaxy(tractor.RaDecPos(ra, dec), tractor.Flux(100.),
                    LogRadius(0.))

    tr = tractor.Tractor([tim], [src])
    mod = tr.getModelImage(0)

    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('rex.png')

    # add noise with std 1.
    noisy = mod + np.random.normal(size=mod.shape)
    # make that the tim's data
    tim.data = noisy

    # reset the source params
    src.brightness.setParams([1.])

    tr.freezeParam('images')
    print('Fitting:')
    tr.printThawedParams()
    tr.optimize_loop()
    print('Fit:', src)
    R = tr.optimize(variance=True)
    v = R[-1]
    print('Variance:', v)

    plt.clf()
    plt.imshow(noisy, interpolation='nearest', origin='lower')
    plt.savefig('noisy.png')

    # best fit mod
    fitmod = tr.getModelImage(0)
    plt.clf()
    plt.imshow(fitmod, interpolation='nearest', origin='lower')
    plt.savefig('fitmod.png')
    
