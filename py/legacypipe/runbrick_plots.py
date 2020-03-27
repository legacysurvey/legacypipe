from __future__ import print_function
import pylab as plt
from astrometry.util.plotutils import dimshow
from legacypipe.survey import *
from legacypipe.coadds import quick_coadds

def detection_plots(detmaps, detivs, bands, saturated_pix, tims,
                    targetwcs, refstars, large_galaxies, ps):
    rgb = get_rgb(detmaps, bands)
    plt.clf()
    dimshow(rgb)
    plt.title('detmaps')
    ps.savefig()

    for i,satpix in enumerate(saturated_pix):
        rgb[:,:,2-i][satpix] = 1
    plt.clf()
    dimshow(rgb)
    plt.title('detmaps & saturated')
    ps.savefig()

    coimgs,cons = quick_coadds(tims, bands, targetwcs, fill_holes=False)

    if refstars:
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        lp,lt = [],[]
        tycho = refstars[refstars.isbright]
        if len(tycho):
            ok,ix,iy = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
            p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=14, mec='r', mfc='none')
            lp.append(p)
            lt.append('Tycho-2 only')
        if gaia_stars:
            gaia = refstars[refstars.isgaia]
        if gaia_stars and len(gaia):
            ok,ix,iy = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
            p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=10, mec='c', mfc='none')
            for x,y,g in zip(ix,iy,gaia.phot_g_mean_mag):
                plt.text(x, y, '%.1f' % g, color='k',
                         bbox=dict(facecolor='w', alpha=0.5))
            lp.append(p)
            lt.append('Gaia')
        # star_clusters?
        if large_galaxies:
            galaxies = refstars[refstars.islargegalaxy]
        if large_galaxies and len(galaxies):
            ok,ix,iy = targetwcs.radec2pixelxy(galaxies.ra, galaxies.dec)
            p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=14, mec=(0,1,0), mfc='none')
            lp.append(p)
            lt.append('Galaxies')
        plt.axis(ax)
        plt.title('Ref sources')
        plt.figlegend([p[0] for p in lp], lt)
        ps.savefig()

    for band, detmap,detiv in zip(bands, detmaps, detivs):
        plt.clf()
        plt.subplot(2,1,1)
        plt.hist((detmap * np.sqrt(detiv))[detiv>0], bins=50, range=(-5,8), log=True)
        plt.title('Detection map pixel values (sigmas): band %s' % band)
        plt.subplot(2,1,2)
        plt.hist((detmap * np.sqrt(detiv))[detiv>0], bins=50, range=(-5,8))
        ps.savefig()

def halo_plots_before(tims, bands, targetwcs, halostars, ps):
    coimgs,_ = quick_coadds(tims, bands, targetwcs)
    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ax = plt.axis()
    plt.plot(halostars.ibx, halostars.iby, 'o', mec='r', ms=15, mfc='none')
    plt.axis(ax)
    plt.title('Before star halo subtraction')
    ps.savefig()
    return coimgs

def halo_plots_after(tims, bands, targetwcs, halostars, coimgs, ps):
    coimgs2,cons = quick_coadds(tims, bands, targetwcs)
    plt.clf()
    dimshow(get_rgb(coimgs2, bands))
    ax = plt.axis()
    plt.plot(halostars.ibx, halostars.iby, 'o', mec='r', ms=15, mfc='none')
    plt.axis(ax)
    plt.title('After star halo subtraction')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb([co-co2 for co,co2 in zip(coimgs,coimgs2)],
                    bands))
    ax = plt.axis()
    plt.plot(halostars.ibx, halostars.iby, 'o', mec='r', ms=15, mfc='none')
    plt.axis(ax)
    plt.title('Subtracted halos')
    ps.savefig()

    for g in halostars[:10]:
        plt.clf()
        pixrad = int(g.radius * 3600. / pixscale)
        ax = [g.ibx-pixrad, g.ibx+pixrad, g.iby-pixrad, g.iby+pixrad]
        ima = dict(interpolation='nearest', origin='lower')
        plt.subplot(2,2,1)
        plt.imshow(get_rgb(coimgs, bands), **ima)
        plt.plot(halostars.ibx, halostars.iby, 'o', mec='r', ms=15, mfc='none')
        plt.axis(ax)
        plt.subplot(2,2,2)
        plt.imshow(get_rgb(coimgs2, bands), **ima)
        plt.axis(ax)
        plt.subplot(2,2,3)
        plt.imshow(get_rgb([co-co2 for co,co2 in zip(coimgs,coimgs2)],
                           bands), **ima)
        plt.axis(ax)
        ps.savefig()

def tim_plots(tims, bands, ps):
    # Pixel histograms of subimages.
    for b in bands:
        sig1 = np.median([tim.sig1 for tim in tims if tim.band == b])
        plt.clf()
        for tim in tims:
            if tim.band != b:
                continue
            # broaden range to encompass most pixels... only req'd
            # when sky is bad
            lo,hi = -5.*sig1, 5.*sig1
            pix = tim.getImage()[tim.getInvError() > 0]
            lo = min(lo, np.percentile(pix, 5))
            hi = max(hi, np.percentile(pix, 95))
            plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                     alpha=0.5, label=tim.name)
        plt.legend()
        plt.xlabel('Pixel values')
        plt.title('Pixel distributions: %s band' % b)
        ps.savefig()

        plt.clf()
        lo,hi = -5., 5.
        for tim in tims:
            if tim.band != b:
                continue
            ie = tim.getInvError()
            pix = (tim.getImage() * ie)[ie > 0]
            plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                     alpha=0.5, label=tim.name)
        plt.legend()
        plt.xlabel('Pixel values (sigma)')
        plt.xlim(lo,hi)
        plt.title('Pixel distributions: %s band' % b)
        ps.savefig()

    # Plot image pixels, invvars, masks
    for tim in tims:
        plt.clf()
        plt.subplot(2,2,1)
        dimshow(tim.getImage(), vmin=-3.*tim.sig1, vmax=10.*tim.sig1)
        plt.title('image')
        plt.subplot(2,2,2)
        dimshow(tim.getInvError(), vmin=0, vmax=1.1/tim.sig1)
        plt.title('inverr')
        if tim.dq is not None:
            plt.subplot(2,2,3)
            dimshow(tim.dq, vmin=0, vmax=tim.dq.max())
            plt.title('DQ')
            plt.subplot(2,2,3)
            dimshow(((tim.dq & tim.dq_saturation_bits) > 0),
                    vmin=0, vmax=1.5, cmap='hot')
            plt.title('SATUR')
        plt.subplot(2,2,4)
        dimshow(tim.getImage() * (tim.getInvError() > 0),
                vmin=-3.*tim.sig1, vmax=10.*tim.sig1)
        plt.title('image (masked)')
        plt.suptitle(tim.name)
        ps.savefig()

        if True and tim.dq is not None:
            from legacypipe.bits import DQ_BITS
            plt.clf()
            bitmap = dict([(v,k) for k,v in DQ_BITS.items()])
            k = 1
            for i in range(12):
                bitval = 1 << i
                if not bitval in bitmap:
                    continue
                # only 9 bits are actually used
                plt.subplot(3,3,k)
                k+=1
                plt.imshow((tim.dq & bitval) > 0,
                           vmin=0, vmax=1.5, cmap='hot')
                plt.title(bitmap[bitval])
            plt.suptitle('Mask planes: %s (%s %s)' % (tim.name, tim.imobj.image_filename, tim.imobj.ccdname))
            ps.savefig()

            im = tim.imobj
            if im.camera == 'decam':
                from legacypipe.decam import decam_has_dq_codes
                print(tim.name, ': plver "%s"' % im.plver, 'has DQ codes:', decam_has_dq_codes(im.plver))
            if im.camera == 'decam' and decam_has_dq_codes(im.plver):
                # Integer codes, not bitmask.  Re-read and plot.
                dq = im.read_dq(slice=tim.slice)
                plt.clf()
                plt.subplot(1,3,1)
                dimshow(tim.getImage(), vmin=-3.*tim.sig1, vmax=30.*tim.sig1)
                plt.title('image')
                plt.subplot(1,3,2)
                dimshow(tim.getInvError(), vmin=0, vmax=1.1/tim.sig1)
                plt.title('inverr')
                plt.subplot(1,3,3)
                plt.imshow(dq, interpolation='nearest', origin='lower',
                           cmap='tab10', vmin=-0.5, vmax=9.5)
                plt.colorbar()
                plt.title('DQ codes')
                plt.suptitle('%s (%s %s) PLVER %s' % (tim.name, im.image_filename, im.ccdname, im.plver))
                ps.savefig()

def _plot_mods(tims, mods, blobwcs, titles, bands, coimgs, cons, bslc,
               blobw, blobh, ps,
               chi_plots=True, rgb_plots=False, main_plot=True,
               rgb_format='%s'):
    import numpy as np

    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    
    make_coimgs = (coimgs is None)
    if make_coimgs:
        print('_plot_mods: blob shape', (blobh, blobw))
        coimgs = [np.zeros((blobh,blobw)) for b in bands]
        cons   = [np.zeros((blobh,blobw)) for b in bands]

    for iband,band in enumerate(bands):
        comods = [np.zeros((blobh,blobw)) for m in mods]
        cochis = [np.zeros((blobh,blobw)) for m in mods]
        comodn = np.zeros((blobh,blobw))
        mn,mx = 0,0
        sig1 = 1.
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, blobwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R

            rechi = np.zeros((blobh,blobw))
            chilist = []
            comodn[Yo,Xo] += 1
            for imod,mod in enumerate(mods):
                chi = ((tim.getImage()[Yi,Xi] - mod[itim][Yi,Xi]) *
                       tim.getInvError()[Yi,Xi])
                rechi[Yo,Xo] = chi
                chilist.append((rechi.copy(), itim))
                cochis[imod][Yo,Xo] += chi
                comods[imod][Yo,Xo] += mod[itim][Yi,Xi]
            chis[band].append(chilist)
            # we'll use 'sig1' of the last tim in the list below...
            mn,mx = -10.*tim.sig1, 30.*tim.sig1
            sig1 = tim.sig1
            if make_coimgs:
                nn = (tim.getInvError()[Yi,Xi] > 0)
                coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                cons  [iband][Yo,Xo] += nn
                
        if make_coimgs:
            coimgs[iband] /= np.maximum(cons[iband], 1)
            coimg  = coimgs[iband]
            coimgn = cons  [iband]
        else:
            coimg = coimgs[iband][bslc]
            coimgn = cons[iband][bslc]
            
        for comod in comods:
            comod /= np.maximum(comodn, 1)
        ima = dict(vmin=mn, vmax=mx, ticks=False)
        resida = dict(vmin=-5.*sig1, vmax=5.*sig1, ticks=False)
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi, resida))

    # Plot per-band image, model, and chi coadds, and RGB images
    rgba = dict(ticks=False)
    rgbs = []
    rgbnames = []
    plt.figure(1)
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        for ib,b in enumerate(bands):
            plt.subplot(rows,cols,ib+1)
            plt.title(b)
        plt.subplot(rows,cols,4)
        plt.title('RGB')
        plt.subplot(rows,cols,5)
        plt.title('RGB(stretch)')
        
        imgs = []
        themods = []
        resids = []
        for j,(img,imgn,mod,ima,chi,resida) in enumerate(subim):
            imgs.append(img)
            themods.append(mod)
            resid = img - mod
            resid[imgn == 0] = np.nan
            resids.append(resid)

            if main_plot:
                plt.subplot(rows,cols,1 + j + 0)
                dimshow(img, **ima)
                plt.subplot(rows,cols,1 + j + cols)
                dimshow(mod, **ima)
                plt.subplot(rows,cols,1 + j + cols*2)
                # dimshow(-chi, **imchi)
                # dimshow(imgn, vmin=0, vmax=3)
                dimshow(resid, nancolor='r', **resida)
        rgb = get_rgb(imgs, bands)
        if i == 0:
            rgbs.append(rgb)
            rgbnames.append(rgb_format % 'Image')
        if main_plot:
            plt.subplot(rows,cols, 4)
            dimshow(rgb, **rgba)
        rgb = get_rgb(themods, bands)
        rgbs.append(rgb)
        rgbnames.append(rgb_format % titles[i])
        if main_plot:
            plt.subplot(rows,cols, cols+4)
            dimshow(rgb, **rgba)
            plt.subplot(rows,cols, cols*2+4)
            dimshow(get_rgb(resids, bands, mnmx=(-10,10)), **rgba)

            mnmx = -5,300
            kwa = dict(mnmx=mnmx, arcsinh=1)
            plt.subplot(rows,cols, 5)
            dimshow(get_rgb(imgs, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols+5)
            dimshow(get_rgb(themods, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols*2+5)
            mnmx = -100,100
            kwa = dict(mnmx=mnmx, arcsinh=1)
            dimshow(get_rgb(resids, bands, **kwa), **rgba)
            plt.suptitle(titles[i])
            ps.savefig()

    if rgb_plots:
        # RGB image and model
        plt.figure(2)
        for rgb,tt in zip(rgbs, rgbnames):
            plt.clf()
            dimshow(rgb, **rgba)
            plt.title(tt)
            ps.savefig()

    if not chi_plots:
        return

    imchi = dict(cmap='RdBu', vmin=-5, vmax=5)

    plt.figure(1)
    # Plot per-image chis: in a grid with band along the rows and images along the cols
    cols = max(len(v) for v in chis.values())
    rows = len(bands)
    for imod in range(len(mods)):
        plt.clf()
        for row,band in enumerate(bands):
            sp0 = 1 + cols*row
            # chis[band] = [ (one for each tim:) [ (one for each mod:) (chi,itim), (chi,itim) ], ...]
            for col,chilist in enumerate(chis[band]):
                chi,itim = chilist[imod]
                plt.subplot(rows, cols, sp0 + col)
                dimshow(-chi, **imchi)
                plt.xticks([]); plt.yticks([])
                plt.title(tims[itim].name)
        #plt.suptitle(titles[imod])
        ps.savefig()

