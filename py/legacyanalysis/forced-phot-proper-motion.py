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

from legacypipe.forced_photom_decam import run_forced_phot

def main():
    W = H = 50
    pixscale = 0.262 / 3600.
    band = 'r'
    
    truewcs = Tan(0., 0., W/2., H/2., -pixscale, 0., 0., pixscale,
                  float(W), float(H))

    src = PointSource(RaDecPos(0., 0.,),
                      NanoMaggies(**{band: 1.}))
    src.symmetric_derivs = True
    forced_cat = [src]

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
                photocal=LinearPhotoCal(1., band=band),
                sky=ConstantSky(0.),
                psf=psf)
    tim.band = band
    tim.sig1 = sig1

    #dra_pix = np.linspace(-5, 5, 21)
    dra_pix = np.linspace(-2, 2, 21)
    ddec_pix = np.zeros_like(dra_pix)

    dra2_pix  = np.linspace(-5, 5, 21)
    ddec2_pix = -0.5 * dra2_pix
    
    ps = PlotSequence('pm')

    for dras,ddecs,srcflux in [(dra_pix * pixscale, ddec_pix * pixscale, flux),
                               (dra_pix * pixscale, ddec_pix * pixscale, flux/2),
                               (dra2_pix * pixscale, ddec2_pix * pixscale, flux),]:
        FF = []
        slicex = []
        slicey = []
        residx = []
        residy = []
        for dra,ddec in zip(dras,ddecs):
            src = PointSource(RaDecPos(0.+dra, 0.+ddec),
                              NanoMaggies(**{band: srcflux}))
            tr = Tractor([tim], [src])
            truemod = tr.getModelImage(0)
            noise = np.random.normal(size=truemod.shape) * sig1
            tim.data = truemod + noise

            F = run_forced_phot(forced_cat, tim, ceres=False, derivs=True, do_apphot=False, fixed_also=True) #, ps=ps)
            #print('Src:', forced_cat)
            t = Tractor([tim], forced_cat)
            m = t.getModelImage(0)
            mh,mw = m.shape
            slicex.append(m[mh//2,:])
            slicey.append(m[:,mw//2])
            residx.append((m - truemod)[mh//2,:])
            residy.append((m - truemod)[:,mw//2])
            
            #F.about()
            F.true_dra  = dra  + np.zeros(len(F))
            F.true_ddec = ddec + np.zeros(len(F))
            FF.append(F)
        F = merge_tables(FF)

        plt.clf()
        plt.plot(F.true_dra  * 3600., F.flux_dra  / F.flux * 3600., 'b.', label='RA')
        plt.plot(F.true_ddec * 3600., F.flux_ddec / F.flux * 3600., 'g.', label='Dec')
        mx = max(max(np.abs(F.true_dra)), max(np.abs(F.true_ddec)))
        mx *= 3600.
        plt.plot([-mx,mx],[-mx,mx], 'k-', alpha=0.1)
        plt.xlabel('True offset (arcsec)')
        plt.ylabel('Flux deriv / Flux * 3600 (arcsec)')
        plt.legend()
        ps.savefig()

        plt.clf()
        plt.plot(np.hypot(F.true_dra, F.true_ddec)  * 3600., F.flux, 'b.',
                 label='Flux (dra/dec)')
        plt.plot(np.hypot(F.true_dra, F.true_ddec)  * 3600., F.flux_fixed,
                 'g.', label='Flux (fixed)')
        plt.xlabel('True offset (arcsec)')
        plt.ylabel('Flux')
        ps.savefig()

        plt.clf()
        N = len(slicex)
        cc = float(N-1)
        for i,s in enumerate(slicex):
            #plt.plot(s + i*10, 'b-')
            rgb = (0,i/cc,1.-i/cc)
            #print('rgb', rgb)
            plt.plot(s, '-', color=rgb, alpha=0.5)
        ps.savefig()

        plt.clf()
        N = len(residx)
        cc = float(N-1)
        for i,s in enumerate(residx):
            rgb = (0,i/cc,1.-i/cc)
            plt.plot(s, '-', color=rgb, alpha=0.5)
        ps.savefig()
        
if __name__ == '__main__':
    main()
    
