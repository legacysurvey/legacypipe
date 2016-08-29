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
from tractor.pointsource import BasicSource

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

class SourceDerivatives(MultiParams, BasicSource):
    def __init__(self, real, freeze, thaw, brights):
        '''
        *real*: The real source whose derivatives are my profiles.
        *freeze*: List of parameter names to freeze before taking derivs
        *thaw*: List of parameter names to thaw before taking derivs
        '''
        # This a subclass of MultiParams and we pass the brightnesses
        # as our params.
        super(SourceDerivatives,self).__init__(*brights)
        self.real = real
        self.freeze = freeze
        self.thaw = thaw
        self.brights = brights
        self.umods = None
        
    # forced photom calls getUnitFluxModelPatches
    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        self.real.freezeParamsRecursive(*self.freeze)
        self.real.thawParamsRecursive(*self.thaw)
        #print('SourceDerivatives: source has params:')
        #self.real.printThawedParams()
        # The derivatives will be scaled by the source brightness;
        # undo that scaling.
        #print('Brightness:', self.real.brightness)
        counts = img.getPhotoCal().brightnessToCounts(self.real.brightness)
        derivs = self.real.getParamDerivatives(img, modelMask=modelMask)
        #print('SourceDerivs: derivs', derivs)
        for d in derivs:
            if d is not None:
                d /= counts
                print('Deriv: abs max', np.abs(d.patch).max(), 'range', d.patch.min(), d.patch.max(), 'sum', d.patch.sum())
        # and revert...
        self.real.freezeParamsRecursive(*self.thaw)
        self.real.thawParamsRecursive(*self.freeze)
        self.umods = derivs
        return derivs

    def getModelPatch(self, img, minsb=0., modelMask=None):
        if self.umods is None:
            return None
        #print('getModelPatch()')
        #print('modelMask', modelMask)
        pc = img.getPhotoCal()
        #counts = [pc.brightnessToCounts(b) for b in self.brights]
        #print('umods', self.umods)
        return (self.umods[0] * pc.brightnessToCounts(self.brights[0]) +
                self.umods[1] * pc.brightnessToCounts(self.brights[1]))


        
def sim(nims, nsrcs, H,W, ps, dpix, nsamples, forced=True, ceres=False,
        alphas=None, derivs=False):

    truewcs = Tan(0., 0., W/2., H/2., -pixscale, 0., 0., pixscale,
                  float(W), float(H))

    #ngrid = int(np.ceil(np.sqrt(nsrcs)))
    #xx,yy = np.meshgrid(
    assert(nsrcs == 1)

    sig1 = 0.25
    flux = 100.
    # sig1 = 0.0025
    # flux = 1.
    #psf_sigma = 1.5
    psf_sigma = 2.0

    psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
    nsigma = flux * psfnorm / sig1
    print('S/N:', nsigma)
    
    realsrcs = []
    derivsrcs = []
    for i in range(nsrcs):
        src = PointSource(RaDecPos(0., 0.), Flux(flux))
        realsrcs.append(src)
        if forced:
            src.freezeAllBut('brightness')
        if derivs:
            realsrc = src
            dsrc = SourceDerivatives(realsrc, ['brightness'], ['pos'],
                                     [Flux(0.),Flux(0.)])
            derivsrcs.append(dsrc)
    
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
                          wcs=ConstantFitsWcs(truewcs),
                          photocal=LinearPhotoCal(1.),
                          psf=psf))

    opt = None
    if ceres:
        from tractor.ceres_optimizer import *
        opt = CeresOptimizer()

    # Render "true" models, add noise
    tr = TrackingTractor(tims, realsrcs, optimizer=opt)
    mods = []
    for i,tim in enumerate(tims):
        mod = tr.getModelImage(i)
        mod += np.random.normal(size=mod.shape) * sig1
        tim.data = mod
        mods.append(mod)

    if ps is not None:
        plt.clf()
        plt.imshow(mods[0], interpolation='nearest', origin='lower')
        ps.savefig()
        
    tr.freezeParam('images')

    if derivs:
        tr.catalog = Catalog(*(realsrcs + derivsrcs))
    
    print('Params:')
    tr.printThawedParams()

    p0 = tr.getParams()

    results = []
    for isamp in range(nsamples):
        #print('Sample', isamp)
        if isamp % 100 == 0:
            print('Sample', isamp)

        tr.reset_tracking()
            
        # Scatter the tim WCS CRPIX values
        dx = np.zeros(len(tims))
        dy = np.zeros(len(tims))
        
        for i,tim in enumerate(tims):
            # dx[i] = dpix * np.random.uniform(low=-1., high=1.)
            # dy[i] = dpix * np.random.uniform(low=-1., high=1.)
            dx[i] = dpix * np.random.normal()
            dy[i] = dpix * np.random.normal()
            wcs = Tan(0., 0.,
                      W/2. + dx[i], H/2. + dy[i],
                      -pixscale, 0., 0., pixscale, float(W), float(H))
            tim.wcs = ConstantFitsWcs(wcs)

        if ps is not None and isamp == 0:
            plt.clf()
            cols = int(np.ceil(np.sqrt(len(tims))))
            rows = int(np.ceil(len(tims) / float(cols)))
            for i,tim in enumerate(tims):
            #     from astrometry.util.resample import resample_with_wcs
            #     Yo,Xo,Yi,Xi,rims = resample_with_wcs(truewcs, tim.wcs.wcs,
            #                                          [tim.data])
            #     rimg = np.zeros(truewcs.shape)
            #     rimg[Yo,Xo] = rims[0]
            #     plt.subplot(rows, cols, i+1)
            #     plt.imshow(rimg, interpolation='nearest', origin='lower')
                plt.subplot(rows, cols, i+1)
                plt.imshow(tim.data, interpolation='nearest', origin='lower',
                           cmap='gray')
                x,y = tim.wcs.positionToPixel(realsrcs[0].pos)
                plt.axhline(y, color='r', alpha=0.5, lw=2)
                plt.axvline(x, color='r', alpha=0.5, lw=2)
                x,y = W/2, H/2
                plt.axhline(y, color='b', alpha=0.5, lw=2)
                plt.axvline(x, color='b', alpha=0.5, lw=2)
            plt.suptitle('Astrometric scatter: +- %g pix' % dpix)
            ps.savefig()

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

            if not ceres:
                for step in range(50):
                    dlnp,X,alpha = tr.optimizer.optimize(tr, **optargs)
                    track.append(((dlnp,X,alpha),tr.getParams(),tr.getLogProb()))
                    #print('dlnp,X,alpha', dlnp,X,alpha)
                    if dlnp == 0:
                        break
            else:
                tr.optimize_loop()

        if forced:
            results.append((dx, dy, tr.getParams()))
        else:
            results.append((dx, dy, tr.getParams(), track, tr.tracked_params,
                            tr.tracked_lnprob,
                            tr.getLogProb()))

        if ps is not None and isamp == 0:
            if derivs:
                plt.clf()
                tim = tims[0]
                mod1 = tr.getModelImage(tim, srcs=realsrcs)

                print('mod1 max value', mod1.max()/np.sum(mod1))
                
                # save derivative params
                pd = [d.getParams() for d in derivsrcs]
                # zero out the dDec coefficient
                for d,dp0 in zip(derivsrcs,pd):
                    p = dp0[:]
                    p[1] = 0.
                    d.setParams(p)
                modr = tr.getModelImage(tim, srcs=derivsrcs)
                # zero out the dRA coefficient, restore the dDec coeff
                for d,dp0 in zip(derivsrcs,pd):
                    p = dp0[:]
                    p[0] = 0.
                    d.setParams(p)
                modd = tr.getModelImage(tim, srcs=derivsrcs)
                # restore the dRA coeff
                for d,dp0 in zip(derivsrcs,pd):
                    d.setParams(dp0)

                mod = tr.getModelImage(tim)
                mx = mod.max()
                ima = dict(interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='gray')
                plt.subplot(2,3,1)
                plt.imshow(tim.getImage(), **ima)
                plt.title('data')
                plt.subplot(2,3,2)
                plt.imshow(mod1, **ima)
                plt.title('source')

                dscale = 5
                plt.subplot(2,3,3)
                plt.imshow(dscale * (tim.getImage() - mod1), **ima)
                plt.title('(data - source) x %g' % dscale)

                plt.subplot(2,3,4)
                plt.imshow(modr*dscale, **ima)
                plt.title('dRA x %g' % dscale)
                plt.subplot(2,3,5)
                plt.imshow(modd*dscale, **ima)
                plt.title('dDec x %g' % dscale)
                plt.subplot(2,3,6)
                plt.imshow(mod, **ima)
                plt.title('total')
                x1,y1 = tim.wcs.positionToPixel(realsrcs[0].pos)
                x2,y2 = W/2, H/2
                for i in [1,2,4,5,6]:
                    plt.subplot(2,3,i)
                    plt.axhline(y1, color='r', alpha=0.5, lw=2)
                    plt.axvline(x1, color='r', alpha=0.5, lw=2)
                    plt.axhline(y2, color='b', alpha=0.5, lw=2)
                    plt.axvline(x2, color='b', alpha=0.5, lw=2)
                ps.savefig()
            

            
    return results


def compare_optimizers():
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

    if True:
        nsamples = 400
        np.random.seed(seed)
        results = sim(nims, nsrcs, H,W, None, 1.0, nsamples)

        pp = np.array([p for x,y,p in results])
        flux0 = pp[:,0]

        np.random.seed(seed)
        results = sim(nims, nsrcs, H,W, ps, 1.0, nsamples, derivs=True)

        dx = np.array([x for x,y,p in results])
        dy = np.array([y for x,y,p in results])
        pp = np.array([p for x,y,p in results])
        print('Params:', pp.shape)
        
        flux = pp[:,0]
        fluxdx = pp[:,1]
        fluxdy = pp[:,2]

        r = np.hypot(dx, dy)
        plt.clf()
        plt.plot(r, flux0, 'k.', label='Flux (no derivs)')
        plt.plot(r, flux, 'b.', label='Flux')
        plt.xlabel('WCS Scatter Distance (pix)')
        plt.ylabel('Flux')
        plt.title('Forced photometry: Astrometry sensitivity')
        plt.legend(loc='lower left')
        ps.savefig()

        plt.clf()
        plt.plot(dx,  fluxdx / flux / pixscale, 'r.', label='RA deriv')
        plt.plot(dy, -fluxdy / flux / pixscale, 'g.', label='Dec deriv')
        ax = plt.axis()
        plt.plot([-10,10],[-10,10],'k-', alpha=0.1)
        mx = np.abs(np.array(ax)).max()
        plt.axis([-mx,mx,-mx,mx])
        plt.legend(loc='upper left')
        plt.xlabel('WCS scatter (pix)')
        plt.ylabel('Computed offset (pix)')
        plt.title('Forced photometry: Fitting derivatives to recover scatter')
        plt.axhline(0, color='k', alpha=0.1)
        plt.axvline(0, color='k', alpha=0.1)
        ps.savefig()

        plt.clf()
        plt.scatter(dx, dy, c=flux)
        plt.axhline(0, color='k', alpha=0.1)
        plt.axvline(0, color='k', alpha=0.1)
        ax = plt.axis()
        mx = np.abs(np.array(ax)).max()
        plt.axis([-mx,mx,-mx,mx])
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.title('Forced photometry: flux when fit w/derivatives')
        plt.colorbar()
        ps.savefig()

        plt.clf()
        plt.scatter(dx, dy, c=flux0)
        plt.axhline(0, color='k', alpha=0.1)
        plt.axvline(0, color='k', alpha=0.1)
        ax = plt.axis()
        mx = np.abs(np.array(ax)).max()
        plt.axis([-mx,mx,-mx,mx])
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.title('Forced photometry: flux when fit w/out derivatives')
        plt.colorbar()
        ps.savefig()

        
        
        # i = np.argmax(dx)
        # fluxdx = pp[:,1]
        # fluxdy = pp[:,2]
        # print('dx', dx[i], 'pixels')
        # print('fluxdx', fluxdx[i])
        # print('flux', flux[i])
        # print('d pix', fluxdx[i] / flux[i] / pixscale)
        
        sys.exit(0)
    
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

    if False:
        # How does scatter in the WCS (single image) affect flux measurements?
        # (this devolved into looking at differences between LSQR and Ceres)
        compare_optimizers()
        
        
    if True:
        # Look at how scatter in WCS solutions (multiple images) affects
        # flux measurements.

        nims = 4
        nsamples = 500

        allfluxes = []
        names = []

        dpixes = [1.5, 1., 0.5, 0.1, 0.]
        for dpix in dpixes:

            # Reset the seed -- same pixel noise instantiation for
            # each set, same directions of dpix scatter; all that
            # changes is the dpix scaling.
            np.random.seed(seed)

            ns = nsamples
            if dpix == 0:
                ns = 1
            alphas = [0.1, 0.3, 1.0]
            results = sim(nims, nsrcs, H, W, ps if dpix==1. else None,
                          dpix, ns,
                          forced=False,
                          alphas=alphas)
            #ceres=True)

            pp = np.array([r[2] for r in results])
            flux = pp[:,2]

            allfluxes.append(flux)
            names.append('+- %g pix' % dpix)
            
        plt.clf()
        bins = 50
        mn = min([min(flux) for flux in allfluxes])
        mx = max([max(flux) for flux in allfluxes])
        bins = np.linspace(mn, mx, bins)
        mx = 0
        for flux,name,dpix in zip(allfluxes, names, dpixes):
            if dpix == 0:
                plt.axvline(flux, color='k', alpha=0.5, lw=2, label=name)
            else:
                n,bins,p = plt.hist(flux, bins=bins, histtype='step',label=name)
                mx = max(mx, max(n))
        plt.ylim(0, mx*1.05)
        plt.xlabel('Flux')
        plt.legend(loc='upper left')
        plt.title('Astrometric scatter: %i images' % nims)
        ps.savefig()
