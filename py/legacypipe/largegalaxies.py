import numpy as np

class Duck(object):
    pass

def _build_objmask(img, ivar, skypix, boxcar=5, boxsize=1024):
    """Build an object mask by doing a quick estimate of the sky background on a
    given CCD.

    """
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import uniform_filter
    
    from tractor.splinesky import SplineSky
    
    # Get an initial guess of the sky using the mode, otherwise the median.
    skysig1 = 1.0 / np.sqrt(np.median(ivar[skypix]))
    #try:
    #    skyval = estimate_mode(img[skypix], raiseOnWarn=True)
    #except:
    #    skyval = np.median(img[skypix])
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
                                    mode='constant') > (3 * bskysig1))
    objmask = binary_dilation(objmask, iterations=3)

    return objmask

def _get_sky(args):
    """Wrapper function for the multiprocessing."""
    return get_sky(*args)

def get_sky(survey, targetwcs, tim):
    """Perform custom sky-subtraction on a single CCD.

    """
    from astropy.stats import sigma_clipped_stats
    from astrometry.util.resample import resample_with_wcs
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    # Read the full-image tim (not the one restricted to targetwcs) and estimate
    # the sky background after aggressively masking.
    tim.imobj.get_tractor_image(slc=None, dq=None)
    
    skymodel = reftim.imobj.read_sky_model(slc=None)
    imh, imw = reftim.imobj.get_image_shape()
    refsky = np.zeros((imh, imw)).astype('f4')
    skymodel.addTo(refsky)

    img = tim.getImage()
    ivar = tim.getInvvar()

    # Add the original sky background back into the tim.
    origsky = np.zeros_like(img)
    tim.origsky.addTo(origsky)
    tim.setImage(img + origsky)

    (HH, WW), pixscale = targetwcs.shape, tim.subwcs.pixel_scale()
    overlapimg = np.zeros((HH, WW), np.float32)

    Yo, Xo, Yi, Xi, _ = resample_with_wcs(targetwcs, tim.subwcs)

    refs, _ = get_reference_sources(survey, tim.subwcs, pixscale, ['r'],
                                    tycho_stars=True, gaia_stars=True,
                                    large_galaxies=False, star_clusters=True)
    refmask = get_inblob_map(tim.subwcs, refs) != 0

    skypix = ~refmask * (ivar != 0)
    if np.sum(skypix) == 0:
        print('No pixels to estimate sky...fix me!')
        skymean, skymedian, skysig = 0., 0., 0.
    else:
        #objmask = _build_objmask(img, ivar, skypix)
        #skypix = np.logical_or(objmask != 0, skypix)
        skymean, skymedian, skysig = sigma_clipped_stats(img[Yi, Xi], mask=~skypix[Yi, Xi], sigma=3.0)
    #print(skymedian)
    overlapimg[Yo, Xo] = skymedian

    return overlapimg

def largegalaxy_sky(tims, targetwcs, survey, brickname):
    
    from astrometry.util.starutil_numpy import degrees_between
    from astrometry.util.resample import resample_with_wcs
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    if False:
        import os
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
                #ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), 2*radius * pixscale / 3600, # inner sky annulus
                #                            fill=False, edgecolor='black', lw=1))
                #ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), 5*radius * pixscale / 3600, # outer sky annulus
                #                            fill=False, edgecolor='black', lw=1))

            these = np.where([tim.band == band for tim in tims])[0]
            col = plt.cm.Set1(np.linspace(0, 1, len(tims)))
            for ii, indx in enumerate(these):
                tim = tims[indx]
                #wcs = tim.imobj.get_wcs()
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
            #print(ax.get_xlim(), ax.get_ylim())

        plt.subplots_adjust(bottom=0.12, wspace=0.05, left=0.12, right=0.97, top=0.95)
        pngfile = survey.write_output('image-jpeg', brick=brickname).fn.replace('.jpg', '-ccdpos.jpg')
        os.rename(pngfile, pngfile.replace('tmp-', ''))
        print('Writing {}'.format(pngfile))
        fig.savefig(pngfile)
        plt.close(fig)
        
    #overlapimg = np.stack(mp.map(_get_sky, [(survey, targetwcs, tim) for tim in tims]))

    # Add the original sky background back into the tim.
    #fullimg, fullwcs = [], []
    fullwcs = []
    for tim in tims:
        #img = tim.getImage()
        #origsky = np.zeros_like(img)
        #tim.origsky.addTo(origsky)
        #tim.setImage(img + origsky)
        #fullimg.append(tim.imobj.read_image())
        fullwcs.append(tim.imobj.get_wcs())
        #tims[0].imobj.read_sky_model()

    radec = np.array([wcs.crval for wcs in fullwcs])
    #radec = np.array([tim.subwcs.crval for tim in tims])
    ddeg = degrees_between(radec[:, 0], radec[:, 1], targetwcs.crval[0], targetwcs.crval[1])
    
    allbands = np.array([tim.band for tim in tims])
    for band in sorted(set(allbands)):
        print('Working on band {}'.format(band))
        I = np.where(allbands == band)[0]
        nn = len(I)

        # Build out the matrix of the number of overlapping pixels.
        noverlap = np.zeros((nn, nn)).astype(int)
        indx = np.arange(nn)
        for ii in indx:
            for jj in indx[ii:]:
                try:
                    Yo, Xo, Yi, Xi, _ = resample_with_wcs(fullwcs[I[ii]], fullwcs[I[jj]])
                    #Yo, Xo, Yi, Xi, _ = resample_with_wcs(tims[I[ii]].subwcs, tims[I[jj]].subwcs)
                    noverlap[jj, ii] = len(Yo)
                    #print(ii, jj, len(Yo), len(Xo))
                except:
                    pass

        print(noverlap)
        # Work from the inside out.
        JJ = np.argsort(ddeg[I])

        for J in JJ:
            # Sort the images that overlap with this image by increasing overlap
            # (but skip the image itself).
            print(noverlap[J, :])
            KK = np.argsort(noverlap[J, :])[::-1]
            KK = KK[noverlap[J, KK] > 0]
            for K in KK:
                try:
                    YJ, XJ, YK, XK, _ = resample_with_wcs(fullwcs[I[J]], fullwcs[I[K]])
                    #Yo, Xo, Yi, Xi, _ = resample_with_wcs(tims[I[J]].subwcs, tims[I[K]].subwcs)
                except:
                    import pdb ; pdb.set_trace()

                #slcJ = slice(YJ.min(), YJ.max() + 1), slice(XJ.min(), XJ.max() + 1)
                #slcK = slice(YK.min(), YK.max() + 1), slice(XK.min(), XK.max() + 1)

                imgK = tims[I[K]].imobj.read_image()[YK, XK]
                imgJ = tims[I[J]].imobj.read_image()[YJ, XJ]
                invJ = tims[I[J]].imobj.read_invvar()[YJ, XJ]
                
                #imgK = tims[I[K]].imobj.read_image(slc=slcK) # [Yi, Xi]
                #imgJ = tims[I[J]].imobj.read_image(slc=slcJ) # [Yo, Xo]
                #invJ = tims[I[J]].imobj.read_invvar(slc=slcJ) # [Yo, Xo]
                
                delta = np.sum(invJ * (imgJ - imgK)) / np.sum(invJ)
                #delta = np.sum(tims[I[J]].getInvvar()[Yo, Xo]*(tims[I[J]].getImage()[Yo, Xo] - tims[I[K]].getImage()[Yi, Xi])) / np.sum(tims[I[J]].getInvvar()[Yo, Xo])
                print(J, K, noverlap[J, K], delta)
                tims[I[K]].setImage(tims[I[K]].getImage() + delta)

        # Get the median sky background from the CCD that's furthest from the
        # center of the field.
        skytim = tims[I[JJ[-1]]]
        skyimg = skytim.imobj.read_sky_model(slc=None)
        skywcs = skytim.imobj.get_wcs()
        imh, imw = skytim.imobj.get_image_shape()
        refsky = np.zeros((imh, imw)).astype('f4')
        skyimg.addTo(refsky)

        refs, _ = get_reference_sources(survey, skywcs, skywcs.pixel_scale(), ['r'],
                                        tycho_stars=True, gaia_stars=True,
                                        large_galaxies=True, star_clusters=True)
        skymask = get_inblob_map(skywcs, refs) != 0 # True=unmasked
        medsky = np.median(refsky[skymask])

        for J in JJ:
            tims[I[J]].setImage(tims[I[J]].getImage() - medsky)
            
    #    # Choose the reference image, read the full tim, and render the original
    #    # sky image.
    #    indx = np.argsort(radec[I][:, 0])
    #    refindx = indx[0]
    #    reftim = tims[I[refindx]]
    #
    #    #fulltim = reftim.imobj.get_tractor_image(slc=None, dq=False, invvar=False, pixels=False)
    #    skymodel = reftim.imobj.read_sky_model(slc=None)
    #    imh, imw = reftim.imobj.get_image_shape()
    #    refsky = np.zeros((imh, imw)).astype('f4')
    #    skymodel.addTo(refsky)
    #
    #    tim.setImage(img + origsky)
    #
    #    
    #    reftim = im.get_tractor_image(splinesky=True, subsky=False, hybridPsf=True,
    #                                  normalizePsf=True, apodize=apodize)
    #    
    #
    #    
    #    refscale = np.max(overlapimg[I[indx[0]]])
    #    for ii in indx: # sort by RA
    #        scale = np.max(overlapimg[I[indx[ii]]])
    #        print(refscale, scale, refscale / scale)
    #        tims[I[ii]].setImage(tims[I[ii]].getImage() * refscale / scale)
        
    #ww = np.where([tim.band == 'r' for tim in tims])[0]
    #with fitsio.FITS('skytest.fits', 'rw') as ff:
    #    for w in ww:
    #        ff.write(overlapimg[w, :, :])

    #import pdb ; pdb.set_trace()

    return tims

def stage_largegalaxies(
        survey=None, targetwcs=None, bands=None, tims=None,
        brickname=None, version_header=None,
        apodize=True,
        plots=False, ps=None, coadd_bw=False, W=None, H=None,
        brick=None, blobs=None, lanczos=True, ccds=None,
        write_metrics=True,
        mp=None, record_event=None,
        **kwargs):

    from legacypipe.coadds import make_coadds, write_coadd_images
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
    tims = largegalaxy_sky(tims, targetwcs, survey, brickname)
    #import pdb ; pdb.set_trace()
    
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
    for tim in tims:
        ie = tim.inverr
        median_ie = np.median(ie[ie>0])
        # newie = (ie / median_ie)**2 * median_ie
        newie = ie**2 / median_ie
        tim.inverr = newie    

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
   
    cotims = []
    for band,img,iv,mask,psfimg in zip(bands, C.coimgs, C.cowimgs, C.allmasks, C.psf_imgs):
        mjd = np.mean([tim.imobj.mjdobs for tim in tims if tim.band == band])
        mjd_tai = astropy.time.Time(mjd, format='mjd', scale='utc').tai.mjd
        tai = TAITime(None, mjd=mjd_tai)

        twcs = LegacySurveyWcs(targetwcs, tai)

        print('PSF sigmas (in pixels?) for band', band, ':',
              ['%.2f' % tim.psf_sigma for tim in tims if tim.band == band])
        psf_sigma = np.mean([tim.psf_sigma for tim in tims if tim.band == band])
        print('Using average PSF sigma', psf_sigma)
        #psf = GaussianMixturePSF(1., 0., 0., psf_sigma**2, psf_sigma**2, 0.)

        psf = PixelizedPSF(psfimg)
        gnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)

        psfnorm = np.sqrt(np.sum(psfimg**2))
        print('Gaussian PSF norm', gnorm, 'vs pixelized', psfnorm)
        
        cotim = Image(img, invvar=iv, wcs=twcs, psf=psf,
                      photocal=LinearPhotoCal(1., band=band),
                      sky=ConstantSky(0.), name='coadd-'+band)
        cotim.band = band
        cotim.subwcs = targetwcs
        cotim.psf_sigma = psf_sigma
        cotim.sig1 = 1./np.sqrt(np.median(iv[iv>0]))
        #cotim.dq = mask # hmm, not what we think this is
        cotim.dq = np.zeros(cotim.shape, dtype=np.int16)
        cotim.dq_saturation_bits = DQ_BITS['satur']
        cotim.psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        cotim.galnorm = 1.0 # bogus!
        cotim.imobj = Duck()
        cotim.imobj.fwhm = 2.35 * psf_sigma
        cotim.time = tai
        cotim.primhdr = fitsio.FITSHDR()

        cotims.append(cotim)

    # EVIL
    return dict(tims=cotims)

