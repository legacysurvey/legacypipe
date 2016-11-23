from __future__ import print_function
import pylab as plt
import numpy as np
import os
from astrometry.util.plotutils import PlotSequence
from astrometry.util.util import Tan
from tractor.galaxy import *
from tractor import Tractor, Image, PixPos, Flux, PointSource, NullWCS, EllipseE, ConstantFitsWcs, LinearPhotoCal, RaDecPos, NanoMaggies
from tractor.psf import GaussianMixturePSF
from legacypipe.survey import SimpleGalaxy
from scipy.ndimage.filters import gaussian_filter

from legacypipe.oneblob import one_blob

'''
Investigate proposed change to our model selection cut of PSF/SIMP vs
EXP/DEV:

- for true EXP sources in the r_e vs S/N plane, what fraction of the
  time do we classify them as EXP / PSF / SIMP?  (As a function of
  seeing.)

- where do true EXP and PSF sources land in the DCHISQ(PSF) vs
  DCHISQ(EXP) plane?

'''

if __name__ == '__main__':
    ps = PlotSequence('morph')

    seeing = 1.3

    pixscale = 0.262
    psfsigma = seeing / pixscale / 2.35
    print('PSF sigma:', psfsigma, 'pixels')
    psf = GaussianMixturePSF(1., 0., 0., psfsigma**2, psfsigma**2, 0.)

    sig1 = 0.01
    psfnorm = 1./(2. * np.sqrt(np.pi) * psfsigma)
    detsig1 = sig1 / psfnorm

    #sn_vals = np.linspace(8., 20., 5)
    #re_vals = np.logspace(-1., 0.5, 5)
    sn_vals = np.logspace(1.2, 2.5, 5)
    re_vals = np.logspace(-1., -0.5, 5)

    Nper = 10

    Nexp  = np.zeros((len(sn_vals), len(re_vals)), int)
    Nsimp = np.zeros_like(Nexp)
    Npsf  = np.zeros_like(Nexp)
    Nother= np.zeros_like(Nexp)

    np.random.seed(42)

    sz = 50
    cd = pixscale / 3600.
    wcs = Tan(0., 0., float(sz/2), float(sz/2), -cd, 0., 0., cd,
              float(sz), float(sz))
    band = 'r'

    all_dchisqs = []
    
    for i,sn in enumerate(sn_vals):
        for j,re in enumerate(re_vals):
    
            tim = Image(data=np.zeros((sz,sz)), inverr=np.ones((sz,sz)) / sig1,
                        psf=psf,
                        wcs = ConstantFitsWcs(wcs),
                        photocal = LinearPhotoCal(1., band=band))
            #wcs = NullWCS(pixscale=pixscale))

            ## HACK -- this is the flux required for a PSF to be
            ## detected at target S/N... adjust for galaxy?
            flux = sn * detsig1
            # Create round EXP galaxy
            #PixPos(sz/2, sz/2),
            true_src = ExpGalaxy(RaDecPos(0., 0.),
                                 NanoMaggies(**{band: flux}),
                                 EllipseE(re, 0., 0.))
            
            tr = Tractor([tim], [true_src])
            true_mod = tr.getModelImage(0)

            ima = dict(interpolation='nearest', origin='lower',
                       vmin=-2.*sig1, vmax=5.*sig1, cmap='hot')

            this_dchisqs = []
            flux_sns = []
            
            for k in range(Nper):
                noise = np.random.normal(scale=sig1, size=true_mod.shape)

                tim.data = true_mod + noise

                if k == 0:
                    plt.clf()
                    plt.subplot(1,2,1)
                    plt.imshow(true_mod, **ima)
                    plt.subplot(1,2,2)
                    plt.imshow(tim.data, **ima)
                    plt.title('S/N %f, r_e %f' % (sn, re))
                    ps.savefig()
                
                ## run one_blob code?  Or shortcut?
                src = PointSource(RaDecPos(0., 0.),
                                  NanoMaggies(**{band: flux}))

                nblob,iblob,Isrcs = 0, 1, np.array([0])
                brickwcs = wcs
                bx0, by0, blobw, blobh = 0, 0, sz, sz
                blobmask = np.ones((sz,sz), bool)
                timargs = [(tim.data, tim.getInvError(), tim.wcs, tim.wcs.wcs,
                            tim.getPhotoCal(), tim.getSky(), tim.psf,
                            'tim', 0, sz, 0, sz, band, sig1,
                            tim.modelMinval, None)]
                srcs = [src]
                bands = band
                plots,psx = False, None
                simul_opt, use_ceres, hastycho = False, False, False
                
                X = (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh,
                     blobmask, timargs, srcs, bands, plots, psx, simul_opt,
                     use_ceres, hastycho)
                R = one_blob(X)
                #print('Got:', R)

                print('Sources:', R.sources)
                print('Dchisqs:', R.dchisqs)
                #R.about()
                if len(R.sources) == 0:
                    continue

                assert(len(R.sources) == 1)

                src = R.sources[0]
                dchisq = R.dchisqs[0]

                #print('srcs', src)
                #print('ivs:', R.srcinvvars[0])

                # HACK...
                
                this_dchisqs.append(dchisq)
                #flux_sns.append(
                flux = src.getParams()[2]
                fluxiv = R.srcinvvars[0][2]
                flux_sns.append(flux * np.sqrt(fluxiv))
                
                if isinstance(src, PointSource):
                    Npsf[i, j] += 1
                    # note, SimpleGalaxy is a subclass of ExpGalaxy
                elif isinstance(src, SimpleGalaxy):
                    Nsimp[i, j] += 1
                elif isinstance(src, ExpGalaxy):
                    Nexp[i, j] += 1
                else:
                    Nother[i, j] += 1
                    print('Other:', src)

            d = np.array(this_dchisqs)
            print('this_dchisqs shape', d.shape)
            if len(d):
                plt.clf()
                plt.plot(d[:,0], d[:,1], 'b.')
                plt.xlabel('dchisq(PSF)')
                plt.ylabel('dchisq(SIMP)')
                ax = plt.axis()
                xx = np.array([0, 100000])
                plt.plot(xx, xx, 'b-', alpha=0.5)
                plt.axis(ax)
                plt.title('S/N %f, r_e %f' % (sn, re))
                ps.savefig()
    
                plt.clf()
                plt.plot(d[:,0], d[:,3], 'b.')
                plt.xlabel('dchisq(PSF)')
                plt.ylabel('dchisq(EXP)')
                ax = plt.axis()
                xx = np.array([0, 100000])
                fcut = 0.02
                plt.plot(xx, xx, 'b-', alpha=0.5)
                plt.plot(xx, xx + 3, 'r-', alpha=0.5)
                plt.plot(xx, (1. + fcut) * xx, 'r-', alpha=0.5)
                plt.axis(ax)
                plt.title('S/N %f, r_e %f' % (sn, re))
                ps.savefig()
            
            all_dchisqs.append(this_dchisqs)

            print('Flux S/N values:', flux_sns)
            
    ima = dict(interpolation='nearest', origin='lower',
               extent=[np.log10(min(re_vals)), np.log10(max(re_vals)),
                       min(sn_vals), max(sn_vals),],
               aspect='auto', cmap='hot', vmin=0, vmax=Nper)

    plt.clf()
    plt.imshow(Nexp, **ima)
    plt.colorbar()
    plt.xlabel('log_10 r_e (arcsec)')
    plt.ylabel('S/N (psf)')
    plt.title('Nexp')
    ps.savefig()

    plt.clf()
    plt.imshow(Nsimp, **ima)
    plt.colorbar()
    plt.xlabel('log_10 r_e (arcsec)')
    plt.ylabel('S/N (psf)')
    plt.title('Nsimp')
    ps.savefig()

    plt.clf()
    plt.imshow(Npsf, **ima)
    plt.colorbar()
    plt.xlabel('log_10 r_e (arcsec)')
    plt.ylabel('S/N (psf)')
    plt.title('Npsf')
    ps.savefig()
    
    plt.clf()
    plt.imshow(Nother, **ima)
    plt.colorbar()
    plt.xlabel('log_10 r_e (arcsec)')
    plt.ylabel('S/N (psf)')
    plt.title('Nother')
    ps.savefig()
