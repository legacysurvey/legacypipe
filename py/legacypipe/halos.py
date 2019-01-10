import numpy as np

def fit_halos(coimgs, cons, H, W, bands, gaia, plots, ps):
    rhaloimgs = [np.zeros((H,W),np.float32) for b in bands]
    residimgs = [co.copy() for co in coimgs]

    fitvalues = []
    
    for i,g in enumerate(gaia):
        print('Star w/ G=', g.G)
        # FIXME -- should do stars outside the brick too!
        ok,x,y = targetwcs.radec2pixelxy(g.ra, g.dec)
        if x <= 0 or y <= 0 or x > W or y > H:
            continue
        x -= 1.
        y -= 1.
        ix = int(np.round(x))
        iy = int(np.round(y))

        # Magnitude-based formula for the radius of this star, in brick pixels.
        pixrad = int(np.ceil(g.radius * 3600. / pixscale))
        print('Pixel radius:', pixrad)

        radii = np.arange(15, pixrad, 5)
        minr = int(radii[0])
        maxr = int(radii[-1])
        fitr = minr
        #fitr = 100.
        # Apodization fraction
        apr = maxr*0.8

        # percentile to fit
        segpct = 10
        
        ylo,yhi = max(0,iy-maxr), min(H,iy+maxr+1)
        xlo,xhi = max(0,ix-maxr), min(W,ix+maxr+1)
        if yhi-ylo <= 1 or xhi-xlo <= 1:
            # no overlap
            continue
        rsymms = []
        for iband,band in enumerate(bands):
            rsymms.append(residimgs[iband][ylo:yhi, xlo:xhi].copy())

        if False and plots:
            plt.clf()
            dimshow(get_rgb([co[ylo:yhi,xlo:xhi] for co in coimgs], bands, **rgbkwargs))
            ax = plt.axis()
            plt.plot((ix-xlo)+flipw*np.array([-1,1,1,-1,-1]),
                     (iy-ylo)+fliph*np.array([-1,-1,1,1,-1]), 'r-')
            plt.axis(ax)
            plt.title('zoom')
            ps.savefig()

        r2 = ((np.arange(ylo, yhi)[:,np.newaxis] - y)**2 +
              (np.arange(xlo, xhi)[np.newaxis,:] - x)**2)
        rads = np.sqrt(r2)
        # 
        apodize = np.clip((rads - maxr) / (apr - maxr), 0., 1.)

        rsegpros = []
        rprofiles = []
        fitpros3 = []

        fits = []

        #fixed_alpha = -2.7
        fixed_alpha = -3.0

        fit_fluxes = []
        
        for iband,band in enumerate(bands):
            rsymm = rsymms[iband]
            fitpro3 = np.zeros_like(rsymm)
            rpro = np.zeros_like(rsymm)
            rsegpro = np.zeros_like(rsymm)

            rsegpros.append(rsegpro)
            fitpros3.append(fitpro3)
            rprofiles.append(rpro)

            Nseg = 12
            segments = (Nseg * (np.arctan2(np.arange(ylo,yhi)[:,np.newaxis]-y,
                                           np.arange(xlo,xhi)[np.newaxis,:]-x) - -np.pi) / (2.*np.pi)).astype(int)

            r_rr = []
            r_mm = []
            r_dm = []

            for rlo,rhi in zip(radii, radii[1:]):
                IY,IX = np.nonzero((r2 >= rlo**2) * (r2 < rhi**2))
                ie = cons[iband][IY+ylo, IX+xlo]
                rseg = []
                for s in range(Nseg):
                    K = (ie > 0) * (segments[IY,IX] == s)
                    if np.sum(K):
                        rm = np.median(rsymm[IY[K],IX[K]])
                        rsegpro[IY[K],IX[K]] = rm
                        rseg.append(rm)
                rseg = np.array(rseg)
                rseg = rseg[np.isfinite(rseg)]
                if len(rseg):
                    mn,lo,quart,med = np.percentile(rseg, [0, segpct, 25, 50])
                    rpro[IY,IX] = lo
                    r_rr.append((rlo+rhi)/2.)
                    r_mm.append(lo)
                    r_dm.append(((med-quart)/2.))

            # Power-law fits??
            from scipy.optimize import minimize
            def powerlaw_model(offset, F, alpha, r):
                return offset + F * r**alpha
            def powerlaw_lnp(r, f, df, offset, F, alpha):
                mod = powerlaw_model(offset, F, alpha, r)
                return np.sum(((f - mod) / df)**2)
            rr = np.array(r_rr)
            mm = np.array(r_mm)
            dm = np.array(r_dm)
            dm = np.maximum(dm, 0.1*mm)
            I = np.flatnonzero(rr < fitr)
            def powerlaw_obj3(X):
                (F,) = X
                offset = 0.
                alpha = fixed_alpha
                return powerlaw_lnp(rr[I], mm[I], dm[I], offset, F, alpha)
            M3 = minimize(powerlaw_obj3, [1.])
            (F3,) = M3.x

            mod3 = powerlaw_model(0., F3, fixed_alpha, rads)
            K = (r2 >= minr**2) * (r2 <= maxr**2)
            fitpro3[K] += mod3[K]
            rhaloimgs[iband][ylo:yhi, xlo:xhi] += K * mod3 * apodize

            fit_fluxes.append(F3)
        fitvalues.append((fit_fluxes, fixed_alpha, minr, maxr, apr))
            
        if False and plots:
            plt.clf()
            for band,fit in zip(bands,fits):
                (F2,), rr, mm, dm, I,(F1,alpha1) = fit
                cc = dict(z='m').get(band,band)
                plt.loglog(rr, mm, '-', color=cc)
                plt.errorbar(rr, mm, yerr=dm, color=cc, fmt='.')
                #plt.plot(rr, powerlaw_model(offset, F, alpha, rr), '-', color=cc, lw=2, alpha=0.5)
                plt.plot(rr, powerlaw_model(0., F2, fixed_alpha, rr), '-', color=cc, lw=2, alpha=0.5)
                plt.plot(rr, powerlaw_model(0., F1, alpha1, rr), '-', color=cc, lw=3, alpha=0.3)
            ps.savefig()

            plt.clf()
            dimshow(get_rgb(rsymms, bands, **rgbkwargs))
            plt.title('rsymm')
            ps.savefig()
            
            plt.clf()
            dimshow(get_rgb(rsegpros, bands, **rgbkwargs))
            plt.title('rseg')
            ps.savefig()
            
            plt.clf()
            dimshow(get_rgb(rprofiles, bands, **rgbkwargs))
            plt.title('rpro')
            ps.savefig()
            
            plt.clf()
            dimshow(get_rgb(fitpros3, bands, **rgbkwargs))
            plt.title('r fit')
            ps.savefig()
            
            plt.clf()
            dimshow(get_rgb([co[ylo:yhi,xlo:xhi] - f for co,f in zip(residimgs,fitpros3)], bands, **rgbkwargs))
            plt.title('data - r fit (fixed)')
            ps.savefig()
            
        for res,fit in zip(residimgs,fitpros3):
            res[ylo:yhi, xlo:xhi] -= fit
        round1fits.append((ylo,yhi,xlo,xhi,fitpros3))

    return fitvalues,rhaloimgs
    
    

