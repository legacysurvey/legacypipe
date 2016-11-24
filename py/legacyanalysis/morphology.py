from __future__ import print_function
import sys
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

def scan_dchisq(seeing, target_dchisq, ps, e1=0.):
    pixscale = 0.262
    psfsigma = seeing / pixscale / 2.35
    print('PSF sigma:', psfsigma, 'pixels')
    psf = GaussianMixturePSF(1., 0., 0., psfsigma**2, psfsigma**2, 0.)

    sig1 = 0.01
    psfnorm = 1./(2. * np.sqrt(np.pi) * psfsigma)
    detsig1 = sig1 / psfnorm

    sz = 50
    cd = pixscale / 3600.
    wcs = Tan(0., 0., float(sz/2), float(sz/2), -cd, 0., 0., cd,
              float(sz), float(sz))
    band = 'r'

    tim = Image(data=np.zeros((sz,sz)), inverr=np.ones((sz,sz)) / sig1,
                psf=psf,
                wcs = ConstantFitsWcs(wcs),
                photocal = LinearPhotoCal(1., band=band))
    
    re_vals = np.logspace(-1., 0., 50)
    
    all_runs = []

    mods = []
    
    for i,re in enumerate(re_vals):
        true_src = ExpGalaxy(RaDecPos(0., 0.),
                             NanoMaggies(**{band: 1.}),
                             EllipseE(re, e1, 0.))
        print('True source:', true_src)
        tr = Tractor([tim], [true_src])
        tr.freezeParams('images')
        true_mod = tr.getModelImage(0)

        dchisq_none = np.sum((true_mod * tim.inverr)**2)
        scale = np.sqrt(target_dchisq / dchisq_none)

        true_src.brightness.setParams([scale])

        true_mod = tr.getModelImage(0)
        dchisq_none = np.sum((true_mod * tim.inverr)**2)

        mods.append(true_mod)
        
        tim.data = true_mod
        
        exp_src = true_src.copy()
        psf_src = PointSource(true_src.pos.copy(), true_src.brightness.copy())
        simp_src = SimpleGalaxy(true_src.pos.copy(), true_src.brightness.copy())

        dchisqs = []
        #for src in [psf_src, simp_src, exp_src]:
        for src in [psf_src, simp_src]:
            src.freezeParam('pos')
            #print('Fitting source:', src)
            #src.printThawedParams()
            tr.catalog[0] = src
            tr.optimize_loop()
            #print('Fitted:', src)
            mod = tr.getModelImage(0)
            dchisqs.append(dchisq_none - np.sum(((true_mod - mod) * tim.inverr)**2))
            #print('dchisq:', dchisqs[-1])
        dchisqs.append(dchisq_none)
        
        all_runs.append([re,] + dchisqs)

    all_runs = np.array(all_runs)

    re = all_runs[:,0]
    dchi_psf  = all_runs[:,1]
    dchi_simp = all_runs[:,2]
    dchi_exp  = all_runs[:,3]

    dchi_ps = np.maximum(dchi_psf, dchi_simp)
    dchi_cut1 = dchi_ps + 3+9
    dchi_cut2 = dchi_ps + dchi_psf * 0.02
    dchi_cut3 = dchi_ps + dchi_psf * 0.008
    
    plt.clf()
    plt.plot(re, dchi_psf, 'k-', label='PSF')
    plt.plot(re, dchi_simp, 'b-', label='SIMP')
    plt.plot(re, dchi_exp, 'r-', label='EXP')

    plt.plot(re, dchi_cut2, 'm--', alpha=0.5, lw=2, label='Cut: 2%')
    plt.plot(re, dchi_cut3, 'm:',  alpha=0.5, lw=2, label='Cut: 0.08%')
    plt.plot(re, dchi_cut1, 'm-',  alpha=0.5, lw=2, label='Cut: 12')

    plt.xlabel('True r_e (arcsec)')
    plt.ylabel('dchisq')
    #plt.legend(loc='lower left')
    plt.legend(loc='upper right')
    tt = 'Seeing = %g arcsec, S/N ~ %i' % (seeing, int(np.round(np.sqrt(target_dchisq))))
    if e1 != 0.:
        tt += ', Ellipticity %g' % e1
    plt.title(tt)
    plt.ylim(0.90 * target_dchisq, 1.05 * target_dchisq)

    # aspect = 1.2
    # ax = plt.axis()
    # dre  = (ax[1]-ax[0]) / 20 / aspect
    # dchi = (ax[3]-ax[2]) / 20
    # I = np.linspace(0, len(re_vals)-1, 8).astype(int)
    # for mod,re in [(mods[i], re_vals[i]) for i in I]:
    #     print('extent:', [re-dre, re+dre, ax[2], ax[2]+dchi])
    #     plt.imshow(mod, interpolation='nearest', origin='lower', aspect='auto',
    #                extent=[re-dre, re+dre, ax[2], ax[2]+dchi], cmap='gray')
    # plt.axis(ax)
        
    ps.savefig()


if __name__ == '__main__':
    ps = PlotSequence('morph')

    dchisq = 1000.
    scan_dchisq(1.5, dchisq, ps)
    scan_dchisq(1.0, dchisq, ps)
    scan_dchisq(0.8, dchisq, ps)

    dchisq = 2500.
    scan_dchisq(1.0, dchisq, ps)
    dchisq = 10000.
    scan_dchisq(1.0, dchisq, ps)

    dchisq = 1000.
    scan_dchisq(1.0, dchisq, ps, e1=0.25)
    scan_dchisq(1.0, dchisq, ps, e1=0.5)
    
    sys.exit(0)
    
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

    #all_dchisqs = []

    all_runs = []

    tim = Image(data=np.zeros((sz,sz)), inverr=np.ones((sz,sz)) / sig1,
                psf=psf,
                wcs = ConstantFitsWcs(wcs),
                photocal = LinearPhotoCal(1., band=band))
    
    for i,sn in enumerate(sn_vals):
        for j,re in enumerate(re_vals):
            ## HACK -- this is the flux required for a PSF to be
            ## detected at target S/N... adjust for galaxy?
            flux = sn * detsig1
            # Create round EXP galaxy
            #PixPos(sz/2, sz/2),
            true_src = ExpGalaxy(RaDecPos(0., 0.),
                                 NanoMaggies(**{band: flux}),
                                 EllipseE(re, 0., 0.))
            
            tr = Tractor([tim], [true_src])
            tr.freezeParams('images')
            true_mod = tr.getModelImage(0)

            ima = dict(interpolation='nearest', origin='lower',
                       vmin=-2.*sig1, vmax=5.*sig1, cmap='hot')

            this_dchisqs = []
            flux_sns = []
            
            for k in range(Nper):
                noise = np.random.normal(scale=sig1, size=true_mod.shape)

                tim.data = true_mod + noise

                if k == 0 and False:
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
                psfflux_sn = 0.
                srctype = 'N'

                dchi_psf  = 0.
                dchi_simp = 0.
                dchi_exp  = 0.
                
                if len(R.sources) > 0:
                    assert(len(R.sources) == 1)
                    src = R.sources[0]
                    dchisq = R.dchisqs[0]
                    #print('srcs', src)
                    #print('ivs:', R.srcinvvars[0])

                    dchi_psf  = dchisq[0]
                    dchi_simp = dchisq[1]
                    dchi_exp  = dchisq[3]
                    
                    allmods = R.all_models[0]
                    allivs = R.all_model_ivs[0]
                    #print('All mods:', allmods)
                    psfmod = allmods['ptsrc']
                    psfflux = psfmod.getParams()[2]
                    psfiv = allivs['ptsrc'][2]
                    psfflux_sn = psfflux * np.sqrt(psfiv)
                    
                    # HACK...
                    
                    this_dchisqs.append(dchisq)
                    #flux_sns.append(
                    flux = src.getParams()[2]
                    fluxiv = R.srcinvvars[0][2]
                    flux_sns.append(flux * np.sqrt(fluxiv))
                    
                    if isinstance(src, PointSource):
                        Npsf[i, j] += 1
                        srctype = 'P'
                        # note, SimpleGalaxy is a subclass of ExpGalaxy
                    elif isinstance(src, SimpleGalaxy):
                        Nsimp[i, j] += 1
                        srctype = 'S'
                    elif isinstance(src, ExpGalaxy):
                        Nexp[i, j] += 1
                        srctype = 'E'
                    else:
                        Nother[i, j] += 1
                        print('Other:', src)
                        srctype = 'O'

                all_runs.append((srctype, sn, re, psfflux_sn,
                                 dchi_psf, dchi_simp, dchi_exp))
                        
            d = np.array(this_dchisqs)
            print('this_dchisqs shape', d.shape)
            if len(d) and False:
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
            
            #all_dchisqs.append(this_dchisqs)
            print('Flux S/N values:', flux_sns)
            
    ima = dict(interpolation='nearest', origin='lower',
               extent=[np.log10(min(re_vals)), np.log10(max(re_vals)),
                       min(sn_vals), max(sn_vals),],
               aspect='auto', cmap='hot', vmin=0, vmax=Nper)

    types = np.array([a[0]  for a in all_runs])
    runs  = np.array([a[1:] for a in all_runs])

    re = runs[:,1]
    psfsn = runs[:,2]
    dchi_psf = runs[:,3]
    dchi_simp = runs[:,4]
    dchi_exp = runs[:,5]
    
    print('re:', re)
    
    plt.clf()
    syms = dict(N='s', P='.', S='o', E='x')
    for t in np.unique(types):
        I = np.flatnonzero(types == t)
        if len(I) == 0:
            continue
        plt.plot(re, psfsn, 'b.', marker=syms[t], label=t,
                 mec='b', mfc='none')
    plt.xlabel('r_e (arcsec)')
    plt.ylabel('PSF S/N')
    plt.legend()
    ps.savefig()

    plt.clf()
    plt.scatter(dchi_psf, dchi_simp, c=re, edgecolors='face')
    plt.colorbar()
    plt.xlabel('dchisq_psf')
    plt.ylabel('dchisq_simp')
    xx = np.array([0, 1000000])
    ax = plt.axis()
    plt.plot(xx, xx, 'k-', alpha=0.1)
    plt.axis(ax)
    plt.title('color: r_e')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.axis([0,1e6,0,1e6])
    ps.savefig()    

    plt.clf()
    plt.scatter(dchi_psf, dchi_simp - dchi_psf, c=re, edgecolors='face')
    plt.colorbar()
    plt.xlabel('dchisq_psf')
    plt.ylabel('dchisq_simp - dchisq_psf')
    xx = np.array([0, 1000000])
    ax = plt.axis()
    plt.plot(xx, np.zeros_like(xx), 'k-', alpha=0.1)
    plt.axis(ax)
    plt.title('color: r_e')
    plt.xscale('symlog')
    plt.yscale('symlog')
    #plt.axis([1e1,1e6,0,1e6])
    plt.xlim(1e1, 1e6)
    ps.savefig()    

    plt.clf()
    plt.scatter(dchi_psf, dchi_exp - dchi_psf, c=re, edgecolors='face')
    plt.colorbar()
    plt.xlabel('dchisq_psf')
    plt.ylabel('dchisq_exp - dchisq_psf')
    xx = np.array([0, 1000000])
    ax = plt.axis()
    plt.plot(xx, np.zeros_like(xx), 'k-', alpha=0.1)
    plt.axis(ax)
    plt.title('color: r_e')
    plt.xscale('symlog')
    plt.yscale('symlog')
    #plt.axis([1e1,1e6,0,1e6])
    plt.xlim(1e1, 1e6)
    ps.savefig()    

    dchi_ps = np.maximum(dchi_psf, dchi_simp)
    
    plt.clf()
    plt.scatter(dchi_psf, dchi_exp - dchi_ps, c=re, edgecolors='face')
    plt.colorbar()
    plt.xlabel('dchisq_psf')
    plt.ylabel('dchisq_exp - max(dchisq_psf, dchisq_simp)')
    xx = np.array([0, 1000000])
    ax = plt.axis()
    plt.plot(xx, np.zeros_like(xx), 'k-', alpha=0.1)
    plt.axis(ax)
    plt.title('color: r_e')
    plt.xscale('symlog')
    plt.yscale('symlog')
    #plt.axis([1e1,1e6,0,1e6])
    plt.xlim(1e1, 1e6)
    ps.savefig()    

    xl,xh = plt.xlim()
    xx = np.logspace(np.log10(xl), np.log10(xh), 100)
    y1 = xx + 3
    y2 = xx * 1.02
    y3 = xx * 1.008
    plt.plot(xx, y1 - xx, 'r-', alpha=0.5)
    plt.plot(xx, y2 - xx, 'r--', alpha=0.5)
    plt.plot(xx, y3 - xx, 'r:', alpha=0.5)
    ps.savefig()    
    
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
