from __future__ import print_function
import numpy as np

import logging
logger = logging.getLogger('legacypipe.detection')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def _detmap(X):
    from scipy.ndimage.filters import gaussian_filter
    from legacypipe.survey import tim_get_resamp
    (tim, targetwcs, apodize) = X
    R = tim_get_resamp(tim, targetwcs)
    if R is None:
        return None,None,None,None,None
    assert(tim.psf_sigma > 0)
    psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    ie = tim.getInvError()
    detim = tim.getImage().copy()
    # Zero out all masked pixels
    detim[ie == 0] = 0.
    tim.getSky().addTo(detim, scale=-1.)

    subh,subw = tim.shape
    detsig1 = tim.sig1 / psfnorm
    detiv = np.zeros((subh,subw), np.float32) + (1. / detsig1**2)
    detiv[ie == 0] = 0.

    (Yo,Xo,Yi,Xi) = R
    if tim.dq is None:
        sat = None
    else:
        sat = ((tim.dq[Yi,Xi] & tim.dq_saturation_bits) > 0)
        # Replace saturated pixels by the brightest (non-masked) pixel in the image
        if np.any(sat):
            I, = np.nonzero(sat)
            debug('Filling', len(I), 'saturated detmap pixels with max')
            detim[Yi[I],Xi[I]] = np.max(detim)
            # detection is based on S/N, so plug in values > 0 for iv
            detiv[Yi[I],Xi[I]] = 1./detsig1**2

    detim = gaussian_filter(detim, tim.psf_sigma) / psfnorm**2
    detiv = gaussian_filter(detiv, tim.psf_sigma)

    if apodize:
        apodize = int(apodize)
        ramp = np.arctan(np.linspace(-np.pi, np.pi, apodize+2))
        ramp = (ramp - ramp.min()) / (ramp.max()-ramp.min())
        # drop first and last (= 0 and 1)
        ramp = ramp[1:-1]
        detiv[:len(ramp),:] *= ramp[:,np.newaxis]
        detiv[:,:len(ramp)] *= ramp[np.newaxis,:]
        detiv[-len(ramp):,:] *= ramp[::-1][:,np.newaxis]
        detiv[:,-len(ramp):] *= ramp[::-1][np.newaxis,:]

    return Yo, Xo, detim[Yi,Xi], detiv[Yi,Xi], sat

def detection_maps(tims, targetwcs, bands, mp, apodize=None):
    # Render the detection maps
    H,W = targetwcs.shape
    H,W = np.int(H), np.int(W)
    ibands = dict([(b,i) for i,b in enumerate(bands)])

    detmaps = [np.zeros((H,W), np.float32) for b in bands]
    detivs  = [np.zeros((H,W), np.float32) for b in bands]
    satmaps = [np.zeros((H,W), bool)       for b in bands]
    for tim, (Yo,Xo,incmap,inciv,sat) in zip(
        tims, mp.map(_detmap, [(tim, targetwcs, apodize) for tim in tims])):
        if Yo is None:
            continue
        ib = ibands[tim.band]
        detmaps[ib][Yo,Xo] += incmap * inciv
        detivs [ib][Yo,Xo] += inciv
        if sat is not None:
            satmaps[ib][Yo,Xo] |= sat
    for detmap,detiv in zip(detmaps, detivs):
        detmap /= np.maximum(1e-16, detiv)
    return detmaps, detivs, satmaps

def sed_matched_filters(bands):
    '''
    Determines which SED-matched filters to run based on the available
    bands.

    Returns
    -------
    SEDs : list of (name, sed) tuples
    '''
    # single-band filters
    SEDs = []
    for i,band in enumerate(bands):
        sed = np.zeros(len(bands))
        sed[i] = 1.
        SEDs.append((band, sed))
    # Reverse the order -- run z-band detection filter *first*.
    SEDs = list(reversed(SEDs))

    if len(bands) > 1:
        flat = dict(g=1., r=1., i=1., z=1.)
        SEDs.append(('Flat', [flat[b] for b in bands]))
        red = dict(g=2.5, r=1., i=0.4, z=0.4)
        SEDs.append(('Red', [red[b] for b in bands]))

    info('SED-matched filters:', SEDs)

    return SEDs

def run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy,
                            targetwcs, nsigma=5,
                            saddle_fraction=0.1,
                            saddle_min=2.,
                            saturated_pix=None,
                            exclusion_radius=4.,
                            veto_map=None,
                            mp=None,
                            plots=False, ps=None, rgbimg=None):
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
    omit_xy : None, or (xx,yy,rr) tuple
        Existing sources to avoid: x, y, radius.
    targetwcs : WCS object
        WCS object to use to convert pixel values into RA,Decs for the
        returned Tractor PointSource objects.
    nsigma : float, optional
        Detection threshold
    saturated_pix : None or list of numpy arrays, booleans
        Passed through to sed_matched_detection.
        A map of pixels that are always considered "hot" when
        determining whether a new source touches hot pixels of an
        existing source.
    exclusion_radius: int
        How many pixels around an existing source to veto
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
    from astrometry.util.fits import fits_table
    from tractor import PointSource, RaDecPos, NanoMaggies

    if omit_xy is not None:
        xx,yy,rr = omit_xy
        n0 = len(xx)
    else:
        xx,yy,rr = [],[],[]
        n0 = 0

    H,W = detmaps[0].shape
    hot = np.zeros((H,W), bool)

    peaksn = []
    apsn = []

    for sedname,sed in SEDs:
        if plots:
            pps = ps
        else:
            pps = None
        sedhot,px,py,peakval,apval = sed_matched_detection(
            sedname, sed, detmaps, detivs, bands, xx, yy, rr,
            nsigma=nsigma, saddle_fraction=saddle_fraction, saddle_min=saddle_min,
            saturated_pix=saturated_pix, veto_map=veto_map,
            ps=pps, rgbimg=rgbimg)
        if sedhot is None:
            continue
        info('SED', sedname, ':', len(px), 'new peaks')
        hot |= sedhot
        # With an empty xx, np.append turns it into a double!
        xx = np.append(xx, px).astype(int)
        yy = np.append(yy, py).astype(int)
        rr = np.append(rr, np.zeros_like(px) + exclusion_radius).astype(int)
        peaksn.extend(peakval)
        apsn.extend(apval)

    # New peaks:
    peakx = xx[n0:]
    peaky = yy[n0:]

    Tnew = None
    newcat = []
    if len(peakx):
        # Add sources for the new peaks we found
        pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
        info('Adding', len(pr), 'new sources')
        # Also create FITS table for new sources
        Tnew = fits_table()
        Tnew.ra  = pr
        Tnew.dec = pd
        Tnew.ibx = peakx
        Tnew.iby = peaky
        assert(len(peaksn) == len(Tnew))
        assert(len(apsn) == len(Tnew))
        Tnew.peaksn = np.array(peaksn)
        Tnew.apsn = np.array(apsn)
        for r,d,x,y in zip(pr,pd,peakx,peaky):
            fluxes = dict([(band, detmap[y, x])
                           for band,detmap in zip(bands,detmaps)])
            newcat.append(PointSource(RaDecPos(r,d),
                                      NanoMaggies(order=bands, **fluxes)))
    return Tnew, newcat, hot

def plot_mask(X, rgb=(0,255,0), extent=None):
    import pylab as plt
    H,W = X.shape
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[:,:,0] = X*rgb[0]
    rgba[:,:,1] = X*rgb[1]
    rgba[:,:,2] = X*rgb[2]
    rgba[:,:,3] = X*255
    plt.imshow(rgba, interpolation='nearest', origin='lower', extent=extent)

def plot_boundary_map(X, rgb=(0,255,0), extent=None, iterations=1):
    from scipy.ndimage.morphology import binary_dilation
    H,W = X.shape
    it = iterations
    padded = np.zeros((H+2*it, W+2*it), bool)
    padded[it:-it, it:-it] = X.astype(bool)
    bounds = np.logical_xor(binary_dilation(padded), padded)
    if extent is None:
        extent = [-it, W+it, -it, H+it]
    else:
        x0,x1,y0,y1 = extent
        extent = [x0-it, x1+it, y0-it, y1+it]
    plot_mask(bounds, rgb=rgb, extent=extent)

def sed_matched_detection(sedname, sed, detmaps, detivs, bands,
                          xomit, yomit, romit,
                          nsigma=5.,
                          saddle_fraction=0.1,
                          saddle_min=2.,
                          saturated_pix=None,
                          veto_map=None,
                          cutonaper=True,
                          ps=None, rgbimg=None):
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
    xomit, yomit, romit : iterables (lists or numpy arrays) of int
        Previously known sources that are to be avoided; x,y +- radius
    nsigma : float, optional
        Detection threshold.
    saddle_fraction : float, optional
        Fraction of the peak heigh for selecting new sources.
    saddle_min : float, optional
        Saddle-point depth from existing sources down to new sources.
    saturated_pix : None or list of numpy arrays, boolean
        A map of pixels that are always considered "hot" when
        determining whether a new source touches hot pixels of an
        existing source.
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

    H,W = detmaps[0].shape
    allzero = True
    for iband in range(len(bands)):
        if sed[iband] == 0:
            continue
        if np.all(detivs[iband] == 0):
            continue
        allzero = False
        break
    if allzero:
        info('SED', sedname, 'has all zero weight')
        return None,None,None,None,None

    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    if saturated_pix is not None:
        satur  = np.zeros((H,W), bool)

    for iband in range(len(bands)):
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
        if saturated_pix is not None:
            satur |= saturated_pix[iband]
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    del sedmap
    peaks = (sedsn > nsigma)

    def saddle_level(Y):
        # Require a saddle that drops by (the larger of) "saddle"
        # sigma, or 10% of the peak height.
        # ("saddle" is passed in as an argument to the
        #  sed_matched_detection function)
        drop = max(saddle_min, Y * saddle_fraction)
        return Y - drop

    lowest_saddle = nsigma - saddle_min

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

    if ps is not None:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        plt.clf()
        plt.subplot(1,2,2)
        dimshow(sedsn, vmin=-2, vmax=100, cmap='hot', ticks=False)
        plt.subplot(1,2,1)
        dimshow(sedsn, vmin=-2, vmax=10, cmap='hot', ticks=False)
        above = (sedsn > nsigma)
        plot_boundary_map(above)
        ax = plt.axis()
        y,x = np.nonzero(peaks)
        plt.plot(xomit, yomit, 'm.')
        plt.plot(x, y, 'r+')
        plt.axis(ax)
        plt.title('SED %s: S/N & peaks' % sedname)
        ps.savefig()

        #import fitsio
        #fitsio.write('sed-sn-%s.fits' % sedname, sedsn)

        # plt.clf()
        # plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest',
        #            origin='lower', cmap='hot')
        # plot_boundary_map(sedsn > lowest_saddle)
        # plt.title('SED %s: S/N & lowest saddle point bounds' % sedname)
        # ps.savefig()

    # For each new source, compute the saddle value, segment at that
    # level, and drop the source if it is in the same blob as a
    # previously-detected source.

    # We dilate the blobs a bit too, to
    # catch slight differences in centroid positions.
    dilate = 1

    # For efficiency, segment at the minimum saddle level to compute
    # slices; the operations described above need only happen within
    # the slice.
    saddlemap = (sedsn > lowest_saddle)
    saddlemap = binary_dilation(saddlemap, iterations=dilate)
    if saturated_pix is not None:
        saddlemap |= satur
    allblobs,_ = label(saddlemap)
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

    # Map of pixels that are vetoed by sources found so far.  The veto
    # area is based on saddle height.  We go from brightest to
    # faintest pixels.  Thus the saddle level decreases, and the
    # saddlemap areas become larger; the saddlemap when a source is
    # found is a lower bound on the pixels that it will veto based on
    # the saddle heights of fainter sources.  Thus the vetomap isn't
    # the final word, it is just a quick veto of pixels we know for
    # sure will be vetoed.
    if veto_map is None:
        this_veto_map = np.zeros(sedsn.shape, bool)
    else:
        this_veto_map = veto_map.copy()

    for x,y,r in zip(xomit, yomit, romit):
        xlo = int(np.clip(np.floor(x - r), 0, W-1))
        xhi = int(np.clip(np.ceil (x + r), 0, W-1))
        ylo = int(np.clip(np.floor(y - r), 0, H-1))
        yhi = int(np.clip(np.ceil (y + r), 0, H-1))
        this_veto_map[ylo:yhi+1, xlo:xhi+1] |= (np.hypot(
            (x - np.arange(xlo, xhi+1))[np.newaxis, :],
            (y - np.arange(ylo, yhi+1))[:, np.newaxis]) < r)

    if ps is not None:
        plt.clf()
        plt.imshow(this_veto_map, interpolation='nearest', origin='lower', vmin=0, vmax=1, cmap='hot')
        plt.title('Veto map')
        ps.savefig()

        info('Peaks in initial veto map:', np.sum(this_veto_map[py,px]), 'of', len(px))

        plt.clf()
        plt.imshow(saddlemap, interpolation='nearest', origin='lower', vmin=0, vmax=1, cmap='hot')
        ax = plt.axis()
        for slc in allslices:
            sy,sx = slc
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
        plt.axis(ax)
        plt.title('Saddle map (lowest level): %i blobs' % len(allslices))
        ps.savefig()

    # For each peak, determine whether it is isolated enough --
    # separated by a low enough saddle from other sources.  Need only
    # search within its "allblob", which is defined by the lowest
    # saddle.
    info('Found', len(px), 'potential peaks')
    nveto = 0
    nsaddle = 0
    naper = 0
    for i,(x,y) in enumerate(zip(px, py)):
        if this_veto_map[y,x]:
            #info('  in veto map!')
            nveto += 1
            continue

        level = saddle_level(sedsn[y,x])
        ablob = allblobs[y,x]
        index = int(ablob - 1)
        slc = allslices[index]

        #info('source', i, 'of', len(px), 'at', x,y, 'S/N', sedsn[y,x], 'saddle', level)
        #info('  allblobs slice', slc)

        saddlemap = (sedsn[slc] > level)
        saddlemap = binary_dilation(saddlemap, iterations=dilate)
        if saturated_pix is not None:
            saddlemap |= satur[slc]
        saddlemap *= (allblobs[slc] == ablob)
        saddlemap = binary_fill_holes(saddlemap)
        blobs,_ = label(saddlemap)
        x0,y0 = allx0[index], ally0[index]
        thisblob = blobs[y-y0, x-x0]

        saddlemap *= (blobs == thisblob)

        # previously found sources:
        ox = np.append(xomit, px[:i][keep[:i]]) - x0
        oy = np.append(yomit, py[:i][keep[:i]]) - y0
        h,w = blobs.shape
        cut = False
        if len(ox):
            ox = ox.astype(int)
            oy = oy.astype(int)
            cut = any((ox >= 0) * (ox < w) * (oy >= 0) * (oy < h) *
                      (blobs[np.clip(oy,0,h-1), np.clip(ox,0,w-1)] == thisblob))

        # one plot per peak is a little excessive!
        if ps is not None and i<10:
            _peak_plot_1(this_veto_map, x, y, px, py, keep, i, xomit, yomit, sedsn, allblobs,
                         level, dilate, saturated_pix, satur, ps, rgbimg, cut)

        if False and cut and ps is not None:
            _peak_plot_2(ox, oy, w, h, blobs, thisblob, sedsn, x0, y0,
                         x, y, level, ps)
        if False and (not cut) and ps is not None:
            _peak_plot_3(sedsn, nsigma, x, y, x0, y0, slc, saddlemap,
                         xomit, yomit, px, py, keep, i, cut, ps)

        if cut:
            # in same blob as previously found source.
            #info('  cut')
            # update vetomap
            this_veto_map[slc] |= saddlemap
            nsaddle += 1
            #info('Added to vetomap:', np.sum(saddlemap), 'pixels set; now total of', np.sum(this_veto_map), 'pixels set')
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
                naper += 1
                continue

        aper.append(m)
        peakval.append(sedsn[y,x])
        keep[i] = True

        this_veto_map[slc] |= saddlemap
        #info('Added to vetomap:', np.sum(saddlemap), 'pixels set; now total of', np.sum(this_veto_map), 'pixels set')

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

    info('Of', len(px), 'potential peaks:', nveto, 'in veto map,', nsaddle, 'cut by saddle test,',
          naper, 'cut by aper test,', np.sum(keep), 'kept')

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
        dimshow(this_veto_map, vmin=0, vmax=1, cmap='hot')
        plt.title('SED %s: veto map' % sedname)
        ps.savefig()

        plt.clf()
        dimshow(hotblobs, vmin=0, vmax=1, cmap='hot')
        ax = plt.axis()
        p2 = plt.plot(pxdrop, pydrop, 'm+', ms=8, mew=2)
        p1 = plt.plot(px, py, 'g+', ms=8, mew=2)
        p3 = plt.plot(xomit, yomit, 'r+', ms=8, mew=2)
        plt.axis(ax)
        plt.title('SED %s: hot blobs' % sedname)
        plt.figlegend((p3[0],p1[0],p2[0]), ('Existing', 'Keep', 'Drop'),
                      'upper left')
        ps.savefig()

    return hotblobs, px, py, aper, peakval

def _peak_plot_1(vetomap, x, y, px, py, keep, i, xomit, yomit, sedsn, allblobs,
                 level, dilate, saturated_pix, satur, ps, rgbimg, cut):
    from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
    from scipy.ndimage.measurements import label
    import pylab as plt
    plt.clf()
    plt.suptitle('Peak at %i,%i (%s)' % (x,y, ('cut' if cut else 'kept')))
    newsym = '+'
    if cut:
        newsym = 'x'
    plt.subplot(2,3,1)
    plt.imshow(vetomap, interpolation='nearest', origin='lower',
               cmap='gray', vmin=0, vmax=1)
    ax = plt.axis()
    prevx = px[:i][keep[:i]]
    prevy = py[:i][keep[:i]]
    plt.plot(prevx, prevy, 'o', mec='r', mfc='none')
    plt.plot(xomit, yomit, 'm.')
    plt.plot(x, y, newsym, mec='r', mfc='r')
    plt.axis(ax)
    plt.title('veto map')

    ablob = allblobs[y,x]
    saddlemap = (sedsn > level)
    saddlemap = binary_dilation(saddlemap, iterations=dilate)
    if saturated_pix is not None:
        saddlemap |= satur
    saddlemap *= (allblobs == ablob)
    # plt.subplot(2,3,2)
    # plt.imshow(saddlemap, interpolation='nearest', origin='lower',
    #            vmin=0, vmax=1, cmap='gray')
    # ax = plt.axis()
    # plt.plot(x, y, newsym, mec='r', mfc='r')
    # plt.plot(prevx, prevy, 'o', mec='r', mfc='none')
    # plt.plot(xomit, yomit, 'm.')
    # plt.axis(ax)
    # plt.title('saddle map (1)')

    plt.subplot(2,3,2)
    saddlemap = binary_fill_holes(saddlemap)
    plt.imshow(saddlemap, interpolation='nearest', origin='lower',
               vmin=0, vmax=1, cmap='gray')
    ax = plt.axis()
    plt.plot(x, y, newsym, mec='r', mfc='r')
    plt.plot(prevx, prevy, 'o', mec='r', mfc='none')
    plt.plot(xomit, yomit, 'm.')
    plt.axis(ax)
    plt.title('saddle map (fill holes)')

    blobs,_ = label(saddlemap)
    thisblob = blobs[y, x]
    saddlemap *= (blobs == thisblob)

    plt.subplot(2,3,4)
    plt.imshow(saddlemap, interpolation='nearest', origin='lower',
               vmin=0, vmax=1, cmap='gray')
    ax = plt.axis()
    plt.plot(x, y, newsym, mec='r', mfc='r')
    plt.plot(prevx, prevy, 'o', mec='r', mfc='none')
    plt.plot(xomit, yomit, 'm.')
    plt.axis(ax)
    plt.title('saddle map (this blob)')

    nzy,nzx = np.nonzero(saddlemap)

    plt.subplot(2,3,5)
    plt.imshow(saddlemap, interpolation='nearest', origin='lower',
               vmin=0, vmax=1, cmap='gray')
    plt.plot(x, y, newsym, mec='r', mfc='r', label='New source')
    plt.plot(prevx, prevy, 'o', mec='r', mfc='none', label='Previous sources')
    plt.plot(xomit, yomit, 'm.', label='Omit sources')
    plt.axis([min(nzx)-1, max(nzx)+1, min(nzy)-1, max(nzy)+1])
    plt.legend()
    plt.title('saddle map (this blob)')

    if rgbimg is not None:
        for sp in [3,6]:
            plt.subplot(2,3,sp)
            plt.imshow(rgbimg, interpolation='nearest', origin='lower')
            ax = plt.axis()
            plt.plot(x, y, newsym, mec='r', mfc='r')
            plt.plot(prevx, prevy, 'o', mec='r', mfc='none')
            plt.plot(xomit, yomit, 'm.')
            if sp==3:
                plt.axis(ax)
            else:
                plt.axis([min(nzx)-1, max(nzx)+1, min(nzy)-1, max(nzy)+1])

    ps.savefig()

def _peak_plot_2(ox, oy, w, h, blobs, thisblob, sedsn, x0, y0,
                 x, y, level, ps):
    import pylab as plt
    I = np.flatnonzero((ox >= 0) * (ox < w) * (oy >= 0) * (oy < h) *
                       (blobs[np.clip(oy,0,h-1), np.clip(ox,0,w-1)] ==
                        thisblob))
    j = I[0]
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(sedsn, cmap='hot', interpolation='nearest', origin='lower')
    ax = plt.axis()
    plt.plot([ox[j]+x0, x], [oy[j]+y0, y], 'g-')
    plt.axis(ax)
    dx = ox[j]+x0 - x
    dy = oy[j]+y0 - y
    dist = max(1, np.hypot(dx, dy))
    ss = []
    steps = int(np.ceil(dist))
    H,W = sedsn.shape
    for s in range(-3, steps+3):
        ix = int(np.round(x + (dx / dist) * s))
        iy = int(np.round(y + (dy / dist) * s))
        ss.append(sedsn[np.clip(iy, 0, H-1), np.clip(ix, 0, W-1)])
    plt.subplot(1,2,2)
    plt.plot(ss)
    plt.axhline(sedsn[y,x], color='k')
    plt.axhline(sedsn[oy[j],ox[j]], color='r')
    plt.axhline(level)
    plt.xticks([])
    plt.title('S/N')
    ps.savefig()

def _peak_plot_3(sedsn, nsigma, x, y, x0, y0, slc, saddlemap,
                 xomit, yomit, px, py, keep, i, cut, ps):
    import pylab as plt
    green = (0,1,0)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(sedsn, vmin=-2, vmax=10, cmap='hot', interpolation='nearest',
               origin='lower')
    plot_boundary_map((sedsn > nsigma))
    ax = plt.axis()
    plt.plot(x, y, 'm+', ms=12, mew=2)
    plt.axis(ax)

    plt.subplot(1,2,2)
    y1,x1 = [s.stop for s in slc]
    ext = [x0,x1,y0,y1]
    plt.imshow(saddlemap, extent=ext, interpolation='nearest', origin='lower',
               cmap='gray')
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

def segment_and_group_sources(image, T, name=None, ps=None, plots=False):
    '''
    *image*: binary image that defines "blobs"
    *T*: source table; only ".ibx" and ".iby" elements are used (x,y integer
    pix pos).  Note: ".blob" field is added.
    *name*: for debugging only

    Returns: (blobs, blobsrcs, blobslices)

    *blobs*: image, values -1 = no blob, integer blob indices
    *blobsrcs*: list of np arrays of integers, elements in T within each blob
    *blobslices*: list of slice objects for blob bounding-boxes.
    '''
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import label, find_objects

    image = binary_fill_holes(image)
    blobs,nblobs = label(image)
    #info('Detected blobs:', nblobs)
    H,W = image.shape
    del image

    blobslices = find_objects(blobs)
    clipx = np.clip(T.ibx, 0, W-1)
    clipy = np.clip(T.iby, 0, H-1)
    T.blob = blobs[clipy, clipx]

    if plots:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        plt.clf()
        dimshow(blobs > 0, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1),
                     ha='center', va='bottom', color='r')
        plt.plot(T.ibx, T.iby, 'rx')
        for i,t in enumerate(T):
            plt.text(t.ibx, t.iby, 'src %i' % i, color='red',
                     ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    # Find sets of sources within blobs
    blobsrcs = []
    keepslices = []
    blobmap = {}
    for blob in range(1, nblobs+1):
        Isrcs, = np.nonzero(T.blob == blob)
        if len(Isrcs) == 0:
            blobmap[blob] = -1
            continue
        blobmap[blob] = len(blobsrcs)
        blobsrcs.append(Isrcs)
        bslc = blobslices[blob-1]
        keepslices.append(bslc)

    blobslices = keepslices

    # Find sources that do not belong to a blob and add them as
    # singleton "blobs"; otherwise they don't get optimized.
    # for sources outside the image bounds, what should we do?
    inblobs = np.zeros(len(T), bool)
    for Isrcs in blobsrcs:
        inblobs[Isrcs] = True
    del inblobs

    # Remap the "blobs" image so that empty regions are = -1 and the blob values
    # correspond to their indices in the "blobsrcs" list.
    if len(blobmap):
        maxblob = max(blobmap.keys())
    else:
        maxblob = 0
    maxblob = max(maxblob, blobs.max())
    bm = np.zeros(maxblob + 1, int)
    for k,v in blobmap.items():
        bm[k] = v
    bm[0] = -1
    # Remap blob numbers
    blobs = bm[blobs]

    if plots:
        from astrometry.util.plotutils import dimshow
        plt.clf()
        dimshow(blobs > -1, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1),
                     ha='center', va='bottom', color='r')
        plt.plot(T.ibx, T.iby, 'rx')
        for i,t in enumerate(T):
            plt.text(t.ibx, t.iby, 'src %i' % i, color='red',
                     ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    for j,Isrcs in enumerate(blobsrcs):
        for i in Isrcs:
            if (blobs[clipy[i], clipx[i]] != j):
                info('---------------------------!!!-------------------------')
                info('Blob', j, 'sources', Isrcs)
                info('Source', i, 'coords x,y', T.ibx[i], T.iby[i])
                info('Expected blob value', j, 'but got',
                      blobs[clipy[i], clipx[i]])

    T.blob = blobs[clipy, clipx]
    assert(len(blobsrcs) == len(blobslices))
    return blobs, blobsrcs, blobslices
