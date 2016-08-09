'''
Do some forced-photometry simulations to look at how errors in
astrometry affect the results.  Can we do anything with forced
photometry to measure astrometric offsets? (photometer PSF + its
derivatives?)
'''
from __future__ import print_function
import sys
import os
import numpy as np
import pylab as plt
import fitsio
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence, plothist, loghist
from astrometry.util.ttime import Time
from astrometry.util.util import Tan
from legacypipe.common import LegacySurveyData, imsave_jpeg, get_rgb
from scipy.ndimage.filters import gaussian_filter

from tractor import *

pixscale = 0.262 / 3600.

class TrackingTractor(Tractor):
    def __init__(self, *args, **kwargs):
        super(TrackingTractor, self).__init__(*args, **kwargs)
        self.reset_tracking()

    def reset_tracking(self):
        self.tracked_params = []
        self.tracked_lnprob = []
        
    def setParams(self, p):
        self.tracked_params.append(np.array(p).copy())
        super(TrackingTractor, self).setParams(p)
        self.tracked_lnprob.append(self.getLogProb())

def sim(nims, nsrcs, H,W, ps, dpix, nsamples, forced=True, ceres=False,
        alphas=None):

    wcs = Tan(0., 0., W/2., H/2., -pixscale, 0., 0., pixscale,
              float(W), float(H))

    #ngrid = int(np.ceil(np.sqrt(nsrcs)))
    #xx,yy = np.meshgrid(
    assert(nsrcs == 1)

    sig1 = 1.
    flux = 100.
    # PSF variance
    psf_sigma = 1.5

    psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
    nsigma = flux * psfnorm / sig1
    print('S/N:', nsigma)
    
    srcs = []
    for i in range(nsrcs):
        srcs.append(PointSource(RaDecPos(0., 0.), Flux(flux)))

    tims = []
    for i in range(nims):
        v = psf_sigma**2

        xx,yy = np.meshgrid(np.arange(-12,13), np.arange(-12,13))
        pp = np.exp(-0.5 * (xx**2 + yy**2) / psf_sigma**2)
        pp /= np.sum(pp)
        psf = PixelizedPSF(pp)

        #psf=GaussianMixturePSF(1., 0., 0., v, v, 0.)))
        
        tims.append(Image(data=np.zeros((H,W), np.float32),
                          inverr=np.ones((H,W), np.float32) * 1./sig1,
                          wcs=ConstantFitsWcs(wcs),
                          photocal=LinearPhotoCal(1.),
                          psf=psf))

    opt = None
    if ceres:
        from tractor.ceres_optimizer import *
        opt = CeresOptimizer()

    # Render "true" models, add noise
    tr = TrackingTractor(tims, srcs, optimizer=opt)
    mods = []
    for i,tim in enumerate(tims):
        mod = tr.getModelImage(i)
        mod += np.random.normal(size=mod.shape) * sig1
        tim.data = mod
        print('Mod type', mod.dtype)
        mods.append(mod)

    if ps is not None:
        plt.clf()
        plt.imshow(mods[0], interpolation='nearest', origin='lower')
        ps.savefig()
        
    tr.freezeParam('images')

    if forced:
        for src in srcs:
            src.freezeAllBut('brightness')

    print('Params:')
    tr.printThawedParams()

    p0 = tr.getParams()
    
    results = []
    for isamp in range(nsamples):
        print('Sample', isamp)
        if isamp % 100 == 0:
            print('Sample', isamp)

        tr.reset_tracking()
            
        # Scatter the tim WCS CRPIX values
        for i,tim in enumerate(tims):
            #dx = dpix * np.random.normal()
            #dy = dpix * np.random.normal()
            dx = dpix * np.random.uniform(low=-2., high=2.)
            dy = dpix * np.random.uniform(low=-2., high=2.)
            wcs = Tan(0., 0.,
                      W/2. + dx, H/2. + dy,
                      -pixscale, 0., 0., pixscale, float(W), float(H))
            tim.wcs = ConstantFitsWcs(wcs)

        tr.setParams(p0)

        track = []
        
        if forced:
            tr.optimize_forced_photometry()
        else:
            optargs = dict(priors=False, shared_params=False)
            if alphas is not None:
                optargs.update(alphas=alphas)
            #tr.optimize_loop()
            track.append(((None,None,None),tr.getParams(),tr.getLogProb()))
            for step in range(50):
                dlnp,X,alpha = tr.optimizer.optimize(tr, **optargs)
                track.append(((dlnp,X,alpha),tr.getParams(),tr.getLogProb()))
                print('dlnp,X,alpha', dlnp,X,alpha)
                if dlnp == 0:
                    break
                
        if forced:
            results.append((dx, dy, tr.getParams()))
        else:
            results.append((dx, dy, tr.getParams(), track, tr.tracked_params,
                            tr.tracked_lnprob,
                            tr.getLogProb()))
    return results
            
    
if __name__ == '__main__':
    import datetime
    
    ps = PlotSequence('sim')

    nims = 1
    nsrcs = 1
    #H,W = 50,50
    H,W = 21,21

    us = datetime.datetime.now().microsecond
    print('Setting random seed to', us)
    seed = us
    
    if False:
        results = sim(nims, nsrcs, H,W, ps, 1.0, 100)
        # Zoom in near zero
        np.random.seed(42)
        results2 = sim(nims, nsrcs, H,W, None, 0.1, 100)
    
        results.extend(results2)
    
        dx = np.array([x for x,y,p in results])
        dy = np.array([y for x,y,p in results])
        pp = np.array([p for x,y,p in results])
        print('Params:', pp.shape)
        
        flux = pp[:,0]
        
        plt.clf()
        plt.scatter(dx, dy, c=flux)
        plt.colorbar()
        plt.xlabel('WCS Scatter x (pix)')
        plt.ylabel('WCS Scatter y (pix)')
        plt.axis('equal')
        ax = plt.axis()
        mx = max(np.abs(ax))
        plt.axis([-mx,mx,-mx,mx])
        plt.axhline(0., color='k', alpha=0.2)
        plt.axvline(0., color='k', alpha=0.2)
        plt.axis([-2,2,-2,2])
        ps.savefig()
    
        r = np.hypot(dx, dy)
        plt.clf()
        plt.plot(r, flux, 'b.')
        plt.xlabel('WCS Scatter Distance (pix)')
        plt.ylabel('Flux')
        plt.title('Forced photometry: Astrometry sensitivity')
        ps.savefig()

    allfluxes = []
    allra = []
    alldec = []
    alldx = []
    alldy = []

    alltracks = []

    alllnprobtracks = []

    names = []

    bestlogprobs = None
    
    #for i in range(3):
    for i in range(3):

        np.random.seed(seed)

        name = ''
        
        nsamples = 200
        if i in [0,1]:
            print()
            print('LSQR Opt')
            print()
            alphas = None
            if i == 1:
                alphas = [0.1, 0.3, 1.0]
                name = 'LSQR, alphas'
            else:
                name = 'LSQR'
                
            results = sim(nims, nsrcs, H,W, None, 1.0, nsamples, forced=False,
                          alphas=alphas)
        else:
            print()
            print('Ceres Opt')
            print()
            name = 'Ceres'
            results = sim(nims, nsrcs, H,W, None, 1.0, nsamples, forced=False, ceres=True)
        #results = sim(nims, nsrcs, H,W, None, 1.0, 10, forced=False)

        names.append(name)
        
        dx = np.array([r[0] for r in results])
        dy = np.array([r[1] for r in results])
        pp = np.array([r[2] for r in results])
        #print('Params:', pp.shape)
        tracks = [r[3] for r in results]
        tracks2 = [r[4] for r in results]
        flux = pp[:,2]

        logprobs = np.array([r[6] for r in results])
        if bestlogprobs is None:
            bestlogprobs = logprobs
        else:
            bestlogprobs = np.maximum(bestlogprobs, logprobs)
        
        alltracks.append(tracks)
        allfluxes.append(flux)
        allra.append(pp[:,0])
        alldec.append(pp[:,1])
        alldx.append(dx)
        alldy.append(dy)

        alllnprobtracks.append([r[5] for r in results])
        
        ras  = pp[:,0] - dx * pixscale
        decs = pp[:,1] + dy * pixscale
        meanra  = np.mean(ras)
        meandec = np.mean(decs)
        
        plt.clf()
        plt.scatter(dx, dy, c=flux)
        plt.colorbar()
        plt.xlabel('WCS Scatter x (pix)')
        plt.ylabel('WCS Scatter y (pix)')
        plt.axis('equal')
        ax = plt.axis()
        mx = max(np.abs(ax))
        plt.axis([-mx,mx,-mx,mx])
        plt.axhline(0., color='k', alpha=0.2)
        plt.axvline(0., color='k', alpha=0.2)
        plt.axis([-2,2,-2,2])
        plt.title(name)
        ps.savefig()

        # plt.clf()
        # for dxi,dyi,track in zip(dx, dy, tracks):
        #     tp = np.array([t[1] for t in track])
        #     rapix  = tp[:,0] / pixscale  - dxi
        #     decpix = tp[:,1] / pixscale  + dyi
        #     flux = tp[:,2]
        #     plt.scatter(rapix, decpix, c=flux, zorder=20)
        #     plt.plot(rapix, decpix, 'k-', alpha=0.1, lw=2, zorder=10)
        # plt.colorbar()
        # plt.xlabel('RA (pix)')
        # plt.ylabel('Dec (pix)')
        # #plt.axis('equal')
        # #plt.axis('scaled')
        # ax = plt.axis()
        # mx = max(np.abs(ax))
        # plt.axis([-mx,mx,-mx,mx])
        # plt.axhline(0., color='k', alpha=0.2)
        # plt.axvline(0., color='k', alpha=0.2)
        # plt.axis([-2,2,-2,2])
        # plt.title(name)
        # ps.savefig()

        plt.clf()
        for dxi,dyi,track,track2 in zip(dx, dy, tracks, tracks2):
            #tp = np.array([t[1] for t in track])
            #print('track2', track2)
            tp = np.vstack(track2)
            rapix  = (tp[:,0] - dxi*pixscale - meanra ) / pixscale
            decpix = (tp[:,1] + dyi*pixscale - meandec) / pixscale
            #rapix  = tp[:,0] / pixscale  - dxi
            #decpix = tp[:,1] / pixscale  + dyi
            #flux = tp[:,2]
            #plt.scatter(rapix, decpix, c=flux, zorder=20)
            plt.scatter(rapix, decpix,
                        c=np.arange(len(rapix))/float(len(rapix)),zorder=20)
            plt.plot(rapix, decpix, 'k-', alpha=0.1, lw=2, zorder=10)
        plt.colorbar()
        plt.xlabel('RA (pix)')
        plt.ylabel('Dec (pix)')
        #plt.axis('equal')
        #plt.axis('scaled')
        ax = plt.axis()
        mx = max(np.abs(ax))
        plt.axis([-mx,mx,-mx,mx])
        plt.axhline(0., color='k', alpha=0.2)
        plt.axvline(0., color='k', alpha=0.2)
        plt.axis([-2,2,-2,2])
        plt.title(name)
        ps.savefig()
        
        # plt.xscale('symlog', linthreshx=1e-4)
        # plt.yscale('symlog', linthreshy=1e-4)
        # ps.savefig()

        # plt.axis([-0.2, 0.2, -0.2, 0.2])
        # ps.savefig()
        # plt.axis([-0.02, 0.02, -0.02, 0.02])
        # ps.savefig()
        
        # plt.clf()
        # plt.subplot(2,1,1)
        # for dxi,track in zip(dx,tracks):
        #     tp = np.array([t[1] for t in track])
        #     rapix = tp[:,0] / pixscale
        #     decpix = tp[:,1] / pixscale
        #     flux = tp[:,2]
        #     plt.plot(rapix - dxi, 'o-')
        #     #plt.axhline(dxi)
        # plt.ylabel('RA - dx (pix)')
        # plt.subplot(2,1,2)
        # for dyi,track in zip(dy,tracks):
        #     tp = np.array([t[1] for t in track])
        #     rapix = tp[:,0] / pixscale
        #     decpix = tp[:,1] / pixscale
        #     flux = tp[:,2]
        #     plt.plot(decpix + dyi, 'o-')
        #     #plt.axhline(dxi)
        # plt.ylabel('Dec + dy (pix)')
        # plt.xlabel('Opt Step')
        # ps.savefig()

        # plt.clf()
        # for dxi,dyi,track in zip(dx, dy, tracks):
        #     tp = np.array([t[1] for t in track])
        #     rapix  = tp[:,0] / pixscale  - dxi
        #     decpix = tp[:,1] / pixscale  + dyi
        #     #flux = tp[:,2]
        #     plt.plot(np.hypot(rapix, decpix), 'o-')
        # plt.xlabel('Opt Step')
        # plt.ylabel('Radius (pix)')
        # ps.savefig()

        plt.clf()
        for dxi,dyi,track,track2 in zip(dx, dy, tracks, tracks2):
            #tp = np.array([t[1] for t in track])
            #print('track2', track2)
            tp = np.vstack(track2)
            #for dxi,dyi,track in zip(dx, dy, tracks):
            #tp = np.array([t[1] for t in track])
            rapix  = (tp[:,0] - dxi*pixscale - meanra ) / pixscale
            decpix = (tp[:,1] + dyi*pixscale - meandec) / pixscale
            # print('Track ra', tp[:,0])
            # print('Mean RA', meanra)
            # print('Track dec', tp[:,1])
            # print('Mean Dec', meandec)
            plt.plot(np.hypot(rapix, decpix), '.-')
        plt.xlabel('Opt Step')
        plt.ylabel('Radius from mean (pix)')
        plt.yscale('symlog', linthreshy=1e-3)
        ps.savefig()

        # plt.clf()
        # for dxi,dyi,track in zip(dx, dy, tracks):
        #     tp = np.array([t[1] for t in track])
        #     rapix  = tp[:,0] / pixscale  - dxi
        #     decpix = tp[:,1] / pixscale  + dyi
        #     flux = tp[:,2]
        #     plt.plot(np.hypot(rapix, decpix), flux, 'o-')
        # plt.xlabel('Radius (pix)')
        # plt.ylabel('Flux')
        # ps.savefig()

    #for name,tracks in zip(names, alltracks):
    for name,tracks in zip(names, alllnprobtracks):
        plt.clf()
        for track,bestlnp in zip(tracks, bestlogprobs):
            lnprob = [-(t - bestlnp) for t in track]
            plt.plot(lnprob, '.-')
        plt.xlabel('Opt Step')
        plt.ylabel('Log-Prob gap vs best')
        plt.yscale('symlog', linthreshy=1e-4)
        plt.title(name)
        ps.savefig()
        
    plt.clf()
    for name,flux in zip(names, allfluxes):
        plt.hist(flux, bins=20, histtype='step', label=name)
    plt.xlabel('Flux')
    plt.legend()
    ps.savefig()

    plt.clf()
    for dx,ra in zip(alldx,allra):
        plt.plot(dx, ra, 'x')
    for dy,dec in zip(alldy,alldec):
        plt.plot(dy, dec, 's')
    plt.xlabel('Pixel shift')
    plt.ylabel('RA/Dec shift')
    ps.savefig()

    plt.clf()
    for dx,ra in zip(alldx,allra):
        A = np.empty((len(dx),2))
        A[:,0] = 1.
        A[:,1] = dx
        r = np.linalg.lstsq(A, ra)
        fit = r[0]
        fitline = fit[0] + dx*fit[1]
        plt.plot(dx, ra - fitline, 'x')
        
    for dy,dec in zip(alldy,alldec):
        A = np.empty((len(dy),2))
        A[:,0] = 1.
        A[:,1] = dy
        r = np.linalg.lstsq(A, dec)
        fit = r[0]
        fitline = fit[0] + dy*fit[1]
        plt.plot(dy, dec - fitline, 's', mfc='none')

    plt.xlabel('Pixel shift')
    plt.ylabel('RA/Dec shift - fit')
    ps.savefig()
    
