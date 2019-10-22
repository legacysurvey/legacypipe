import numpy as np
import pylab as plt

import tractor
from legacypipe.survey import RexGalaxy, LogRadius

'''
An example of rendering a Rex galaxy in an image.

'''
if __name__ == '__main__':
    H,W = 100,100
    psf_sigma = 1.4 # pixels
    v = psf_sigma**2
    tim = tractor.Image(data=np.zeros((H,W), np.float32),
                        inverr=np.ones((H,W), np.float32),
                        psf=tractor.GaussianMixturePSF(1., 0., 0., v, v, 0.))
    src = RexGalaxy(tractor.PixPos(W/2, H/2), tractor.Flux(100.),
                    LogRadius(0.))

    tr = tractor.Tractor([tim], [src])
    mod = tr.getModelImage(0)

    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('rex.png')
    
