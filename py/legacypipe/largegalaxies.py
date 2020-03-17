import numpy as np

class Duck(object):
    pass

def largegalaxy_sky(tims, targetwcs, survey, brickname, qaplot=False):
    
    from astrometry.util.starutil_numpy import degrees_between
    from astrometry.util.resample import resample_with_wcs
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    if qaplot:
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
        
    # Read all the WCS objects and then build out the N**2/2 matrix of the
    # overlapping pixels.
    fullwcs = [tim.imobj.get_wcs() for tim in tims]

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
                    noverlap[jj, ii] = len(Yo)
                    #print(ii, jj, len(Yo), len(Xo))
                except:
                    pass
        print(noverlap)

        # Work from the inside, out.
        radec = np.array([wcs.crval for wcs in fullwcs])
        ddeg = degrees_between(radec[:, 0], radec[:, 1], targetwcs.crval[0], targetwcs.crval[1])
    
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
                except:
                    print('This should not happen...')
                    pass

                # Now read the images. This is stupidly slow because we read the
                # whole image and then slice it. Need to figure out the slices!
                
                #slcJ = slice(YJ.min(), YJ.max() + 1), slice(XJ.min(), XJ.max() + 1)
                #slcK = slice(YK.min(), YK.max() + 1), slice(XK.min(), XK.max() + 1)
                #imgK = tims[I[K]].imobj.read_image(slc=slcK) # [Yi, Xi]
                #imgJ = tims[I[J]].imobj.read_image(slc=slcJ) # [Yo, Xo]
                #invJ = tims[I[J]].imobj.read_invvar(slc=slcJ) # [Yo, Xo]
                
                imgK = tims[I[K]].imobj.read_image()[YK, XK]
                imgJ = tims[I[J]].imobj.read_image()[YJ, XJ]
                invJ = tims[I[J]].imobj.read_invvar()[YJ, XJ]

                # Get the inverse-variance weighted average of the *difference*
                # of the overlapping pixels.
                delta = np.sum(invJ * (imgJ - imgK)) / np.sum(invJ)

                # Apply the delta.
                print(J, K, noverlap[J, K], delta)
                tims[I[K]].setImage(tims[I[K]].getImage() + delta)

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
        survey=None, targetwcs=None, pixscale=None, bands=None, tims=None,
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
    from collections import Counter

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

