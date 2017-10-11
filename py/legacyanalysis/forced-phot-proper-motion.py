from __future__ import print_function
import sys
import os
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence, plothist, loghist
from astrometry.util.util import Tan
from legacypipe.survey import LegacySurveyData, imsave_jpeg, get_rgb
from tractor import *


def main():
    W = H = 50
    pixscale = 0.262 / 3600.
    
    truewcs = Tan(0., 0., W/2., H/2., -pixscale, 0., 0., pixscale,
                  float(W), float(H))

    forced_cat = [PointSource(RaDecPos(0., 0.,), Flux(1.))]

    sig1 = 0.25
    flux = 100.
    psf_sigma = 2.0

    psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
    nsigma = flux * psfnorm / sig1
    print('S/N:', nsigma)

    v = psf_sigma**2
    # Create a pixelized PSF model by rendering the Gaussian on a stamp
    xx,yy = np.meshgrid(np.arange(-12,13), np.arange(-12,13))
    pp = np.exp(-0.5 * (xx**2 + yy**2) / psf_sigma**2)
    pp /= np.sum(pp)
    psf = PixelizedPSF(pp)
        
    tim = Image(data=np.zeros((H,W), np.float32),
                inverr=np.ones((H,W), np.float32) * 1./sig1,
                wcs=ConstantFitsWcs(truewcs),
                photocal=LinearPhotoCal(1.),
                psf=psf)

    dra_pix = np.linspace(-5, 5, 100)
    ddec_pix = np.zeros_like(dra)

    for dras,ddecs in [(dra_pix, ddec_pix),]:
        for dra,ddec in zip(dras,ddecs):
            src = PointSource(RaDecPos(0.+dra*pixscale, 0.+ddec*pixscale), Flux(flux))
            tr = Tractor([tim], [src])
            mod = tr.getModelImage(0)
            mod += np.random.normal(size=mod.shape) * sig1
            tim.data = mod
            tr.setCatalog(forced_cat)
            


if __name__ == '__main__':
    main()
    
