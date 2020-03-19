import numpy as np

class Duck(object):
    pass

def _build_objmask(img, ivar, skypix, boxcar=5, boxsize=1024):
    """Build an object mask by doing a quick estimate of the sky background on a
    given image.

    """
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import uniform_filter
    
    from tractor.splinesky import SplineSky
    
    # Get an initial guess of the sky using the mode, otherwise the median.
    skysig1 = 1.0 / np.sqrt(np.median(ivar[skypix]))
    skyval = np.median(img[skypix])
   
    # Mask objects in a boxcar-smoothed (image - initial sky model), smoothed by
    # a boxcar filter before cutting pixels above the n-sigma threshold.
    if min(img.shape) / boxsize < 4: # handle half-DECam chips
        boxsize /= 2

    # Compute initial model...
    skyobj = SplineSky.BlantonMethod(img - skyval, skypix, boxsize)
    skymod = np.zeros_like(img)
    skyobj.addTo(skymod)

    bskysig1 = skysig1 / boxcar # sigma of boxcar-smoothed image.
    objmask = np.abs(uniform_filter(img-skyval-skymod, size=boxcar,
                                    mode='constant')) > (3 * bskysig1)
    objmask = binary_dilation(objmask, iterations=3)

    return np.logical_and(~objmask, skypix) # True = object-free sky pixels

def largegalaxy_ubercal(fulltims, coaddtims=None, plots=False, ps=None, verbose=False):
    """Bring individual CCDs onto a common flux scale based on overlapping pixels.

    fulltims - full-CCD tims, used to derive the corrections
    coaddtims - tims sliced to just the pixels contributing to the output coadd

    Some notes on the procedure:

    A x = b
    A: weights 
    A: shape noverlap x nimg
    - entries have units of weights

    x_i: offset to apply to image i
    x: length nimg
    - entries will have values of image pixels

    b: (weighted) measured difference between image i and image j
    b: length -- "noverlap" number of overlapping pairs of images -- filled-in elements in your array
    - units of weighted image pixels

    """
    from astrometry.util.resample import resample_with_wcs

    band = fulltims[0].band
    
    nimg = len(fulltims)
    indx = np.arange(nimg)

    ## initialize A bigger than we will need, cut later
    A = np.zeros((nimg*nimg, nimg), np.float32)
    b = np.zeros((nimg*nimg), np.float32)
    ioverlap = 0

    for ii in indx:
        for jj in indx[ii+1:]:
            try:
                Yi, Xi, Yj, Xj, _ = resample_with_wcs(
                    fulltims[ii].subwcs, fulltims[jj].subwcs)
            except:
                continue

            imgI = fulltims[ii].getImage() [Yi, Xi]
            imgJ = fulltims[jj].getImage() [Yj, Xj]
            invI = fulltims[ii].getInvvar()[Yi, Xi]
            invJ = fulltims[jj].getInvvar()[Yj, Xj]
            good = (invI > 0) * (invJ > 0)
            diff = (imgI - imgJ)[good]
            iv = 1. / (1. / invI[good] + 1. / invJ[good])
            delta = np.sum(diff * iv)
            weight = np.sum(iv)

            A[ioverlap, ii] = -weight
            A[ioverlap, jj] =  weight

            b[ioverlap] = delta

            ioverlap += 1

    noverlap = ioverlap
    A = A[:noverlap, :]
    b = b[:noverlap]
    if verbose:
        print('A:')
        print(A)
        print('b:')
        print(b)

    R = np.linalg.lstsq(A, b, rcond=None)

    x = R[0]
    print('Delta offsets to each image:')
    print(x)

    if plots:
        import matplotlib.pyplot as plt
        plt.clf()
        for j, (correction, fulltim) in enumerate(zip(x, fulltims)):
            plt.subplot(nimg, 1, j+1)
            plt.hist(fulltim.data.ravel(), bins=50, histtype='step',
                     range=(-5, 5))
            plt.axvline(-correction)
        plt.title('Band %s: fulltim pix and -correction' % band)
        ps.savefig()

        if coaddtims is not None:
            plt.clf()
            for j,(correction,ii) in enumerate(zip(x, np.arange(len(coaddtims)))):
                plt.subplot(nimg, 1, j+1)
                plt.hist((coaddtims[ii].data + correction).ravel(), bins=50, histtype='step', range=(-5, 5))
            plt.title('Band %s: tim pix + correction' % band)
            ps.savefig()
    
    return x

def largegalaxy_sky(tims, targetwcs, survey, brickname, bands, mp, 
                    plots=False, ps=None, verbose=False):
    
    from tractor.sky import ConstantSky
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map
    from legacypipe.coadds import make_coadds
    from legacypipe.survey import get_rgb, imsave_jpeg
    from astropy.stats import sigma_clipped_stats

    if plots:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        refs, _ = get_reference_sources(survey, targetwcs, targetwcs.pixel_scale(), ['r'],
                                        tycho_stars=False, gaia_stars=False,
                                        large_galaxies=True, star_clusters=False)
        
        pixscale = targetwcs.pixel_scale()
        width, height = targetwcs.get_width() * pixscale / 3600, targetwcs.get_height() * pixscale / 3600 # [degrees]
        bb, bbcc = targetwcs.radec_bounds(), targetwcs.radec_center() # [degrees]
        pad = 0.5 * width # [degrees]

        delta = np.max( (np.diff(bb[0:2]), np.diff(bb[2:4])) ) / 2 + pad / 2
        xlim = bbcc[0] - delta, bbcc[0] + delta
        ylim = bbcc[1] - delta, bbcc[1] + delta

        plt.clf()
        fig, allax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)
        for ax, band in zip(allax, ('g', 'r', 'z')):
            ax.set_xlabel('RA (deg)')
            ax.text(0.9, 0.05, band, ha='center', va='bottom',
                    transform=ax.transAxes, fontsize=18)

            if band == 'g':
                ax.set_ylabel('Dec (deg)')
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            for gal in refs:
                ax.add_patch(patches.Circle((gal.ra, gal.dec), gal.radius, fill=False, edgecolor='black', lw=2))

            these = np.where([tim.band == band for tim in tims])[0]
            col = plt.cm.Set1(np.linspace(0, 1, len(tims)))
            for ii, indx in enumerate(these):
                tim = tims[indx]
                wcs = tim.subwcs
                cc = wcs.radec_bounds()
                ax.add_patch(patches.Rectangle((cc[0], cc[2]), cc[1]-cc[0],
                                               cc[3]-cc[2], fill=False, lw=2, 
                                               edgecolor=col[these[ii]],
                                               label='ccd{:02d}'.format(these[ii])))
                ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=10)

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.invert_xaxis()
            ax.set_aspect('equal')

        plt.subplots_adjust(bottom=0.12, wspace=0.05, left=0.12, right=0.97, top=0.95)
        ps.savefig()
        
    if plots:
        mods = []
        for tim in tims:
            imcopy = tim.getImage().copy()
            tim.sky.addTo(imcopy, -1)
            mods.append(imcopy)
        C = make_coadds(tims, bands, targetwcs, mods=mods, callback=None, mp=mp)
        imsave_jpeg('{}-before.jpg'.format(ps.basefn), get_rgb(C.comods, bands), origin='lower')

    allbands = np.array([tim.band for tim in tims])
    for band in sorted(set(allbands)):
        print('Working on band {}'.format(band))
        I = np.where(allbands == band)[0]

        bandtims = [tims[ii].imobj.get_tractor_image(
            gaussPsf=True, pixPsf=False, subsky=False, dq=True, apodize=False)
            for ii in I]

        #if plots:
        #    for bandtim in bandtims:
        #        C = make_coadds(bandtim, band, targetwcs, callback=None, sbscale=False, mp=mp)
        #        imsave_jpeg('{}-{}.jpg'.format(ps.basefn, bandtim.name.replace(' ', '-')),
        #                                       get_rgb(C.coimgs, band), origin='lower')

        # Derive the correction and then apply it.
        x = largegalaxy_ubercal(bandtims, coaddtims=[tims[ii] for ii in I], plots=plots, ps=ps)
        # Apply the correction and return the tims
        for jj, (correction, ii) in enumerate(zip(x, I)):
            tims[ii].data += correction
            tims[ii].sky = ConstantSky(0.0)
            # Also correct the full-field mosaics
            bandtims[jj].data += correction
            bandtims[jj].sky = ConstantSky(0.0)

        ## Check--
        #for jj, correction in enumerate(x):
        #    fulltims[jj].data += correction
        #newcorrection = largegalaxy_ubercal(fulltims)
        #print(newcorrection)

        ## Build a full-field coadd.
        #from astrometry.util.util import Tan
        #from astrometry.util.starutil_numpy import degrees_between
        #racen, deccen = targetwcs.crval[0], targetwcs.crval[1]
        #corners = [bandtim.imobj.get_wcs().radec_bounds() for bandtim in bandtims]
        #radius = np.max([degrees_between(racen, deccen, rr,  dd) for rr, dd in corners])
        #
        #bandpixscale = np.mean([bandtim.wcs.pixel_scale() for bandtim in bandtims])
        #print('Mean pixel scale', bandpixscale)
        #bandwcs = Tan(racen, deccen, radius+0.5, radius+0.5,
        #              -bandpixscale/3600.0, 0.0, 0.0, bandpixscale/3600.0,
        #              float(2*radius), float(2*radius))

    refs, _ = get_reference_sources(survey, targetwcs, targetwcs.pixel_scale(), ['r'],
                                    tycho_stars=True, gaia_stars=True,
                                    large_galaxies=True, star_clusters=True)
    refmask = (get_inblob_map(targetwcs, refs) == 0)

    C = make_coadds(tims, bands, targetwcs, callback=None, sbscale=False, mp=mp)
    for coimg,coiv,band in zip(C.coimgs, C.cowimgs, bands):
        # FIXME -- more extensive masking here?
        #cosky = np.median(coimg[skymask * (coiv > 0)])
        skypix = _build_objmask(coimg, coiv, refmask * (coiv>0))
        skymean, skymedian, skysig = sigma_clipped_stats(coimg, mask=~skypix, sigma=3.0)
        
        I = np.where(allbands == band)[0]
        #print('Band', band, 'Coadd sky:', skymedian)

        if plots:
            plt.clf()
            plt.hist(coimg.ravel(), bins=50, range=(-3,3), density=True)
            plt.axvline(skymedian, color='k')
            for ii in I:
                #print('Tim', tims[ii], 'median', np.median(tims[ii].data))
                plt.hist((tims[ii].data - skymedian).ravel(), bins=50, range=(-3,3), histtype='step', density=True)
            plt.title('Band %s: tim pix & skymedian' % band)
            ps.savefig()

            # Produce skymedian-subtracted, masked image for later RGB plot
            coimg -= skymedian
            coimg[~skypix] = 0.
            #coimg[np.logical_not(skymask * (coiv > 0))] = 0.

        for ii in I:
            tims[ii].data -= skymedian
            #print('Tim', tims[ii], 'after subtracting skymedian: median', np.median(tims[ii].data))

    if plots:
        plt.clf()
        plt.imshow(get_rgb(C.coimgs, bands), origin='lower', interpolation='nearest')
        ps.savefig()

        for band in bands:
            for tim in tims:
                if tim.band != band:
                    continue
                plt.clf()
                C = make_coadds([tim], bands, targetwcs, callback=None, sbscale=False, mp=mp)
                plt.imshow(get_rgb(C.coimgs, bands).sum(axis=2), cmap='gray',
                           interpolation='nearest', origin='lower')
                plt.title('Band %s: tim %s' % (band, tim.name))
                ps.savefig()

    if plots:
        C = make_coadds(tims, bands, targetwcs, callback=None,
                        mp=mp)
        imsave_jpeg('{}-after.jpg'.format(ps.basefn), get_rgb(C.coimgs, bands), origin='lower')
        if plots:
            plt.clf()
            for coimg,band in zip(C.coimgs, bands):
                plt.hist(coimg.ravel(), bins=50, range=(-0.5,0.5),
                         histtype='step', label=band)
            plt.legend()
            plt.title('After adjustment: coadds (sb scaled)')
            ps.savefig()

    return tims

def stage_largegalaxies(
        survey=None, targetwcs=None, pixscale=None, bands=None, tims=None,
        brickname=None, version_header=None,
        apodize=True,
        subsky=True,
        plots=False, ps=None, coadd_bw=False, W=None, H=None,
        brick=None, blobs=None, lanczos=True, ccds=None,
        write_metrics=True,
        mp=None, record_event=None,
        **kwargs):

    from legacypipe.coadds import make_coadds
    from legacypipe.bits import DQ_BITS
    from legacypipe.survey import get_rgb, imsave_jpeg, LegacySurveyWcs

    from tractor.image import Image
    from tractor.basics import NanoMaggies, LinearPhotoCal
    from tractor.sky import ConstantSky
    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    import astropy.time
    import fitsio

    # Custom sky-subtraction for large galaxies.
    if not subsky:
        plots = True
        if plots:
            from astrometry.util.plotutils import PlotSequence
            ps = PlotSequence('largegalaxy-{}'.format(brickname))
        tims = largegalaxy_sky(tims, targetwcs, survey, brickname, bands,
                               mp, plots=plots, ps=ps)
    import pdb ; pdb.set_trace()
    
    # Create coadds and then build custom tims from them.

    # We tried setting the invvars constant per tim -- this makes things worse, since we *remove*
    # the lowered invvars at the cores of galaxies.
    # for tim in tims:
    #     ie = tim.inverr
    #     newie = np.ones(ie.shape, np.float32) / (tim.sig1)
    #     newie[ie == 0] = 0.
    #     tim.inverr = newie

    # Here we're hacking the relative weights -- squaring the weights but then making the median
    # the same, ie, squaring the dynamic range or relative weights -- ie, downweighting the cores
    # even more than they already are from source Poisson terms.
    keeptims = []
    for tim in tims:
        ie = tim.inverr
        if not np.any(ie > 0):
            continue
        median_ie = np.median(ie[ie>0])
        #print('Num pix with ie>0:', np.sum(ie>0))
        #print('Median ie:', median_ie)
        # newie = (ie / median_ie)**2 * median_ie
        if median_ie > 0:
            newie = ie**2 / median_ie
            tim.inverr = newie
            assert(np.all(np.isfinite(tim.getInvError())))
            keeptims.append(tim)
    tims = keeptims

    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    allmasks=True, psf_images=True,
                    mp=mp, plots=plots, ps=ps,
                    callback=None)
    if True: # useful for quickly looking at the image coadd
        with survey.write_output('image-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, get_rgb(C.coimgs, bands), origin='lower')
    import pdb ; pdb.set_trace()

    if plots:
        import pylab as plt
        for band,iv in zip(bands, C.cowimgs):
            plt.clf()
            plt.imshow(np.sqrt(iv), interpolation='nearest', origin='lower')
            plt.title('Coadd Inverr: band %s' % band)
            ps.savefig()

        for band,psf in zip(bands, C.psf_imgs):
            plt.clf()
            plt.imshow(psf, interpolation='nearest', origin='lower')
            plt.title('Coadd PSF image: band %s' % band)
            ps.savefig()

        for band,img,iv in zip(bands, C.coimgs, C.cowimgs):
            from scipy.ndimage.filters import gaussian_filter
            plt.clf()
            plt.hist((img * np.sqrt(iv))[iv>0], bins=50, range=(-5,8), log=True)
            plt.title('Coadd pixel values (sigmas): band %s' % band)
            ps.savefig()

            psf_sigma = np.mean([(tim.psf_sigma * tim.imobj.pixscale / pixscale)
                                 for tim in tims if tim.band == band])
            gnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
            psfnorm = gnorm #np.sqrt(np.sum(psfimg**2))
            detim = gaussian_filter(img, psf_sigma) / psfnorm**2
            cosig1 = 1./np.sqrt(np.median(iv[iv>0]))
            detsig1 = cosig1 / psfnorm
            plt.clf()
            plt.subplot(2,1,1)
            plt.hist(detim.ravel() / detsig1, bins=50, range=(-5,8), log=True)
            plt.title('Coadd detection map values / sig1 (sigmas): band %s' % band)
            plt.subplot(2,1,2)
            plt.hist(detim.ravel() / detsig1, bins=50, range=(-5,8))
            ps.savefig()

            # # as in detection.py
            # detiv = np.zeros_like(detim) + (1. / detsig1**2)
            # detiv[iv == 0] = 0.
            # detiv = gaussian_filter(detiv, psf_sigma)
            # 
            # plt.clf()
            # plt.hist((detim * np.sqrt(detiv)).ravel(), bins=50, range=(-5,8), log=True)
            # plt.title('Coadd detection map values / detie (sigmas): band %s' % band)
            # ps.savefig()

    cotims = []
    for band,img,iv,mask,psfimg in zip(bands, C.coimgs, C.cowimgs, C.allmasks, C.psf_imgs):
        mjd = np.mean([tim.imobj.mjdobs for tim in tims if tim.band == band])
        mjd_tai = astropy.time.Time(mjd, format='mjd', scale='utc').tai.mjd
        tai = TAITime(None, mjd=mjd_tai)

        twcs = LegacySurveyWcs(targetwcs, tai)

        #print('PSF sigmas (in pixels) for band', band, ':',
        #      ['%.2f' % tim.psf_sigma for tim in tims if tim.band == band])
        print('PSF sigmas in coadd pixels:',
              ['%.2f' % (tim.psf_sigma * tim.imobj.pixscale / pixscale)
               for tim in tims if tim.band == band])
        psf_sigma = np.mean([(tim.psf_sigma * tim.imobj.pixscale / pixscale)
                             for tim in tims if tim.band == band])
        print('Using average PSF sigma', psf_sigma)

        psf = PixelizedPSF(psfimg)
        gnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)

        psfnorm = np.sqrt(np.sum(psfimg**2))
        print('Gaussian PSF norm', gnorm, 'vs pixelized', psfnorm)

        # if plots:
        #     from collections import Counter
        #     plt.clf()
        #     plt.imshow(mask, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.title('allmask')
        #     ps.savefig()
        #     print('allmask for band', band, ': values:', Counter(mask.ravel()))

        # Scale invvar to take into account that we have resampled (~double-counted) pixels
        cscale = np.mean([tim.imobj.pixscale / pixscale for tim in tims if tim.band == band])
        print('average tim pixel scale / coadd scale:', cscale)

        iv /= cscale**2

        cotim = Image(img, invvar=iv, wcs=twcs, psf=psf,
                      photocal=LinearPhotoCal(1., band=band),
                      sky=ConstantSky(0.), name='coadd-'+band)
        cotim.band = band
        cotim.subwcs = targetwcs
        cotim.psf_sigma = psf_sigma
        cotim.sig1 = 1./np.sqrt(np.median(iv[iv>0]))
        cotim.dq = mask
        cotim.dq_saturation_bits = DQ_BITS['satur']
        cotim.psfnorm = gnorm
        cotim.galnorm = 1.0 # bogus!
        cotim.imobj = Duck()
        cotim.imobj.fwhm = 2.35 * psf_sigma
        cotim.time = tai
        cotim.primhdr = fitsio.FITSHDR()

        cotims.append(cotim)

    # EVIL
    return dict(tims=cotims)

