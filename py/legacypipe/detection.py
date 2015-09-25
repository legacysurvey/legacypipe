from __future__ import print_function
import pylab as plt
import numpy as np
from astrometry.util.ttime import Time
from astrometry.util.fits import fits_table
from tractor.basics import PointSource, RaDecPos, NanoMaggies
from .common import tim_get_resamp

def _detmap(X):
    from scipy.ndimage.filters import gaussian_filter
    (tim, targetwcs, H, W) = X
    R = tim_get_resamp(tim, targetwcs)
    if R is None:
        return None,None,None,None
    ie = tim.getInvvar()
    psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    detim = tim.getImage().copy()

    detim[ie == 0] = 0.
    # Patch SATURATED pixels with the value saturated pixels would have??
    #detim[(tim.dq & tim.dq_bits['satur']) > 0] = tim.satval
    
    detim = gaussian_filter(detim, tim.psf_sigma) / psfnorm**2
    detsig1 = tim.sig1 / psfnorm
    subh,subw = tim.shape
    detiv = np.zeros((subh,subw), np.float32) + (1. / detsig1**2)
    detiv[ie == 0] = 0.
    (Yo,Xo,Yi,Xi) = R
    return Yo, Xo, detim[Yi,Xi], detiv[Yi,Xi]

def detection_maps(tims, targetwcs, bands, mp):
    # Render the detection maps
    H,W = targetwcs.shape
    detmaps = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    detivs  = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    for tim, (Yo,Xo,incmap,inciv) in zip(
        tims, mp.map(_detmap, [(tim, targetwcs, H, W) for tim in tims])):
        if Yo is None:
            continue
        detmaps[tim.band][Yo,Xo] += incmap * inciv
        detivs [tim.band][Yo,Xo] += inciv

    for band in bands:
        detmaps[band] /= np.maximum(1e-16, detivs[band])

    # back into lists, not dicts
    detmaps = [detmaps[b] for b in bands]
    detivs  = [detivs [b] for b in bands]
        
    return detmaps, detivs



def sed_matched_filters(bands):
    '''
    Determines which SED-matched filters to run based on the available
    bands.

    Returns
    -------
    SEDs : list of (name, sed) tuples
    
    '''

    if len(bands) == 1:
        return [(bands[0], (1.,))]

    # single-band filters
    SEDs = []
    for i,band in enumerate(bands):
        sed = np.zeros(len(bands))
        sed[i] = 1.
        SEDs.append((band, sed))

    if len(bands) > 1:
        flat = dict(g=1., r=1., z=1.)
        SEDs.append(('Flat', [flat[b] for b in bands]))
        red = dict(g=2.5, r=1., z=0.4)
        SEDs.append(('Red', [red[b] for b in bands]))

    return SEDs

def run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy,
                            targetwcs, nsigma=5, saturated_pix=None,
                            plots=False, ps=None, mp=None):
    '''
    Runs a given set of SED-matched filters.

    Parameters
    ----------
    SEDs : list of (name, sed) tuples
        The SEDs to run.  The `sed` values are lists the same length
        as `bands`.
    bands : list of string
        The band names of `detmaps` and `detivs`.
    detmaps : numpy array, float
        Detection maps for each of the listed `bands`.
    detivs : numpy array, float
        Inverse-variances of the `detmaps`.
    omit_xy : None, or (xx,yy) tuple
        Existing sources to avoid.
    targetwcs : WCS object
        WCS object to use to convert pixel values into RA,Decs for the
        returned Tractor PointSource objects.
    nsigma : float, optional
        Detection threshold
    saturated_pix : None or numpy array, boolean
        Passed through to sed_matched_detection.
        A map of pixels that are always considered "hot" when
        determining whether a new source touches hot pixels of an
        existing source.
    plots : boolean, optional
        Create plots?
    ps : PlotSequence object
        Create plots?
    mp : multiproc object
        Multiprocessing
        
    Returns
    -------
    Tnew : fits_table
        Table of new sources detected
    newcat : list of PointSource objects
        Newly detected objects, with positions and fluxes, as Tractor
        PointSource objects.
    hot : numpy array of bool
        "Hot pixels" containing sources.
        
    See also
    --------
    sed_matched_detection : run a single SED-matched filter.
    
    '''
    if omit_xy is not None:
        xx,yy = omit_xy
        n0 = len(xx)
    else:
        xx,yy = [],[]
        n0 = 0

    H,W = detmaps[0].shape
    hot = np.zeros((H,W), bool)

    peaksn = []
    apsn = []

    for sedname,sed in SEDs:
        print('SED', sedname)
        if plots:
            pps = ps
        else:
            pps = None
        t0 = Time()
        sedhot,px,py,peakval,apval = sed_matched_detection(
            sedname, sed, detmaps, detivs, bands, xx, yy,
            nsigma=nsigma, saturated_pix=saturated_pix, ps=pps)
        print('SED took', Time()-t0)
        if sedhot is None:
            continue
        print(len(px), 'new peaks')
        hot |= sedhot
        # With an empty xx, np.append turns it into a double!
        xx = np.append(xx, px).astype(int)
        yy = np.append(yy, py).astype(int)

        peaksn.extend(peakval)
        apsn.extend(apval)

    # New peaks:
    peakx = xx[n0:]
    peaky = yy[n0:]

    # Add sources for the new peaks we found
    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print('Adding', len(pr), 'new sources')
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    assert(len(peaksn) == len(Tnew))
    assert(len(apsn) == len(Tnew))
    Tnew.peaksn = np.array(peaksn)
    Tnew.apsn = np.array(apsn)

    Tnew.itx = np.clip(np.round(Tnew.tx), 0, W-1).astype(int)
    Tnew.ity = np.clip(np.round(Tnew.ty), 0, H-1).astype(int)
    newcat = []
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        fluxes = dict([(band, detmap[Tnew.ity[i], Tnew.itx[i]])
                       for band,detmap in zip(bands,detmaps)])
        newcat.append(PointSource(RaDecPos(r,d),
                                  NanoMaggies(order=bands, **fluxes)))

    return Tnew, newcat, hot

def sed_matched_detection(sedname, sed, detmaps, detivs, bands,
                          xomit, yomit,
                          nsigma=5.,
                          saturated_pix=None,
                          saddle=2.,
                          cutonaper=True,
                          ps=None):
    '''
    Runs a single SED-matched detection filter.

    Avoids creating sources close to existing sources.

    Parameters
    ----------
    sedname : string
        Name of this SED; only used for plots.
    sed : list of floats
        The SED -- a list of floats, one per band, of this SED.
    detmaps : list of numpy arrays
        The per-band detection maps.  These must all be the same size, the
        brick image size.
    detivs : list of numpy arrays
        The inverse-variance maps associated with `detmaps`.
    bands : list of strings
        The band names of the `detmaps` and `detivs` images.
    xomit, yomit : iterables (lists or numpy arrays) of int
        Previously known sources that are to be avoided.
    nsigma : float, optional
        Detection threshold.
    saturated_pix : None or numpy array, boolean
        A map of pixels that are always considered "hot" when
        determining whether a new source touches hot pixels of an
        existing source.
    saddle : float, optional
        Saddle-point depth from existing sources down to new sources.
    cutonaper : bool, optional
        Apply a cut that the source's detection strength must be greater
        than `nsigma` above the 16th percentile of the detection strength in
        an annulus (from 10 to 20 pixels) around the source.
    ps : PlotSequence object, optional
        Create plots?

    Returns
    -------
    hotblobs : numpy array of bool
        A map of the blobs yielding sources in this SED.
    px, py : numpy array of int
        The new sources found.
    aper : numpy array of float
        The detection strength in the annulus around the source, if
        `cutonaper` is set; else -1.
    peakval : numpy array of float
        The detection strength.

    See also
    --------
    sed_matched_filters : creates the `(sedname, sed)` pairs used here
    run_sed_matched_filters : calls this method
    
    '''
    from scipy.ndimage.measurements import label, find_objects
    from scipy.ndimage.morphology import binary_dilation, binary_fill_holes

    t0 = Time()
    H,W = detmaps[0].shape

    allzero = True
    for iband,band in enumerate(bands):
        if sed[iband] == 0:
            continue
        if np.all(detivs[iband] == 0):
            continue
        allzero = False
        break
    if allzero:
        print('SED', sedname, 'has all zero weight')
        return None,None,None,None,None

    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    for iband,band in enumerate(bands):
        if sed[iband] == 0:
            continue
        # We convert the detmap to canonical band via
        #   detmap * w
        # And the corresponding change to sig1 is
        #   sig1 * w
        # So the invvar-weighted sum is
        #    (detmap * w) / (sig1**2 * w**2)
        #  = detmap / (sig1**2 * w)
        sedmap += detmaps[iband] * detivs[iband] / sed[iband]
        sediv  += detivs [iband] / sed[iband]**2
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    del sedmap

    peaks = (sedsn > nsigma)
    print('SED sn:', Time()-t0)
    t0 = Time()

    def saddle_level(Y):
        # Require a saddle that drops by (the larger of) "saddle"
        # sigma, or 20% of the peak height
        drop = max(saddle, Y * 0.2)
        return Y - drop

    lowest_saddle = nsigma - saddle

    # zero out the edges -- larger margin here?
    peaks[0 ,:] = 0
    peaks[:, 0] = 0
    peaks[-1,:] = 0
    peaks[:,-1] = 0

    # Label the N-sigma blobs at this point... we'll use this to build
    # "sedhot", which in turn is used to define the blobs that we will
    # optimize simultaneously.  This also determines which pixels go
    # into the fitting!
    dilate = 8
    hotblobs,nhot = label(binary_fill_holes(
            binary_dilation(peaks, iterations=dilate)))

    # find pixels that are larger than their 8 neighbors
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,2:  ])
    print('Peaks:', Time()-t0)
    t0 = Time()

    if ps is not None:
        from astrometry.util.plotutils import dimshow
        crossa = dict(ms=10, mew=1.5)
        green = (0,1,0)

        def plot_boundary_map(X):
            bounds = binary_dilation(X) - X
            H,W = X.shape
            rgba = np.zeros((H,W,4), np.uint8)
            rgba[:,:,1] = bounds*255
            rgba[:,:,3] = bounds*255
            plt.imshow(rgba, interpolation='nearest', origin='lower')

        plt.clf()
        plt.subplot(1,2,2)
        dimshow(sedsn, vmin=-2, vmax=100, cmap='hot', ticks=False)
        plt.subplot(1,2,1)
        dimshow(sedsn, vmin=-2, vmax=10, cmap='hot', ticks=False)
        above = (sedsn > nsigma)
        plot_boundary_map(above)
        ax = plt.axis()
        y,x = np.nonzero(peaks)
        plt.plot(x, y, 'r+')
        plt.axis(ax)
        plt.title('SED %s: S/N & peaks' % sedname)
        ps.savefig()

        # plt.clf()
        # plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest',
        #            origin='lower', cmap='hot')
        # plot_boundary_map(sedsn > lowest_saddle)
        # plt.title('SED %s: S/N & lowest saddle point bounds' % sedname)
        # ps.savefig()

    # For each new source, compute the saddle value, segment at that
    # level, and drop the source if it is in the same blob as a
    # previously-detected source.  We dilate the blobs a bit too, to
    # catch slight differences in centroid vs SDSS sources.
    dilate = 2

    # For efficiency, segment at the minimum saddle level to compute
    # slices; the operations described above need only happen within
    # the slice.
    saddlemap = (sedsn > lowest_saddle)
    if saturated_pix is not None:
        saddlemap |= saturated_pix
    saddlemap = binary_dilation(saddlemap, iterations=dilate)
    allblobs,nblobs = label(saddlemap)
    allslices = find_objects(allblobs)
    ally0 = [sy.start for sy,sx in allslices]
    allx0 = [sx.start for sy,sx in allslices]

    # brightest peaks first
    py,px = np.nonzero(peaks)
    I = np.argsort(-sedsn[py,px])
    py = py[I]
    px = px[I]

    keep = np.zeros(len(px), bool)

    peakval = []
    aper = []
    apin = 10
    apout = 20
    
    # For each peak, determine whether it is isolated enough --
    # separated by a low enough saddle from other sources.  Need only
    # search within its "allblob", which is defined by the lowest
    # saddle.
    for i,(x,y) in enumerate(zip(px, py)):
        level = saddle_level(sedsn[y,x])
        ablob = allblobs[y,x]
        index = ablob - 1
        slc = allslices[index]

        saddlemap = (sedsn[slc] > level)
        if saturated_pix is not None:
            saddlemap |= saturated_pix[slc]
        saddlemap *= (allblobs[slc] == ablob)
        saddlemap = binary_fill_holes(saddlemap)
        saddlemap = binary_dilation(saddlemap, iterations=dilate)
        blobs,nblobs = label(saddlemap)
        x0,y0 = allx0[index], ally0[index]
        thisblob = blobs[y-y0, x-x0]

        # previously found sources:
        ox = np.append(xomit, px[:i][keep[:i]]) - x0
        oy = np.append(yomit, py[:i][keep[:i]]) - y0
        h,w = blobs.shape
        cut = False
        if len(ox):
            ox = ox.astype(int)
            oy = oy.astype(int)
            cut = any((ox >= 0) * (ox < w) * (oy >= 0) * (oy < h) *
                      (blobs[np.clip(oy,0,h-1), np.clip(ox,0,w-1)] == 
                       thisblob))

        if False and (not cut) and ps is not None:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(sedsn, vmin=-2, vmax=10, cmap='hot')
            plot_boundary_map((sedsn > nsigma))
            ax = plt.axis()
            plt.plot(x, y, 'm+', ms=12, mew=2)
            plt.axis(ax)

            plt.subplot(1,2,2)
            y1,x1 = [s.stop for s in slc]
            ext = [x0,x1,y0,y1]
            dimshow(saddlemap, extent=ext)
            #plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'c-')
            #ax = plt.axis()
            #plt.plot(ox+x0, oy+y0, 'rx')
            plt.plot(xomit, yomit, 'rx', ms=8, mew=2)
            plt.plot(px[:i][keep[:i]], py[:i][keep[:i]], '+',
                     color=green, ms=8, mew=2)
            plt.plot(x, y, 'mo', mec='m', mfc='none', ms=12, mew=2)
            plt.axis(ax)
            if cut:
                plt.suptitle('Cut')
            else:
                plt.suptitle('Keep')
            ps.savefig()

        if cut:
            # in same blob as previously found source
            continue

        # Measure in aperture...
        ap   =  sedsn[max(0, y-apout):min(H,y+apout+1),
                      max(0, x-apout):min(W,x+apout+1)]
        apiv = (sediv[max(0, y-apout):min(H,y+apout+1),
                      max(0, x-apout):min(W,x+apout+1)] > 0)
        aph,apw = ap.shape
        apx0, apy0 = max(0, x - apout), max(0, y - apout)
        R2 = ((np.arange(aph)+apy0 - y)[:,np.newaxis]**2 + 
              (np.arange(apw)+apx0 - x)[np.newaxis,:]**2)
        ap = ap[apiv * (R2 >= apin**2) * (R2 <= apout**2)]
        if len(ap):
            # 16th percentile ~ -1 sigma point.
            m = np.percentile(ap, 16.)
        else:
            # fake
            m = -1.
        if cutonaper:
            if sedsn[y,x] - m < nsigma:
                continue

        aper.append(m)
        peakval.append(sedsn[y,x])
        keep[i] = True

        if False and ps is not None:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(ap, vmin=-2, vmax=10, cmap='hot',
                    extent=[apx0,apx0+apw,apy0,apy0+aph])
            plt.subplot(1,2,2)
            dimshow(ap * ((R2 >= apin**2) * (R2 <= apout**2)),
                    vmin=-2, vmax=10, cmap='hot',
                    extent=[apx0,apx0+apw,apy0,apy0+aph])
            plt.suptitle('peak %.1f vs ap %.1f' % (sedsn[y,x], m))
            ps.savefig()

    print('New sources:', Time()-t0)
    t0 = Time()

    if ps is not None:
        pxdrop = px[np.logical_not(keep)]
        pydrop = py[np.logical_not(keep)]

    py = py[keep]
    px = px[keep]

    # Which of the hotblobs yielded sources?  Those are the ones to keep.
    hbmap = np.zeros(nhot+1, bool)
    hbmap[hotblobs[py,px]] = True
    if len(xomit):
        h,w = hotblobs.shape
        hbmap[hotblobs[np.clip(yomit, 0, h-1), np.clip(xomit, 0, w-1)]] = True
    # in case a source is (somehow) not in a hotblob?
    hbmap[0] = False
    hotblobs = hbmap[hotblobs]

    if ps is not None:
        plt.clf()
        dimshow(hotblobs, vmin=0, vmax=1, cmap='hot')
        ax = plt.axis()
        p1 = plt.plot(px, py, 'g+', ms=8, mew=2)
        p2 = plt.plot(pxdrop, pydrop, 'm+', ms=8, mew=2)
        p3 = plt.plot(xomit, yomit, 'r+', ms=8, mew=2)
        plt.axis(ax)
        plt.title('SED %s: hot blobs' % sedname)
        plt.figlegend((p3[0],p1[0],p2[0]), ('Existing', 'Keep', 'Drop'),
                      'upper left')
        ps.savefig()

    return hotblobs, px, py, aper, peakval
