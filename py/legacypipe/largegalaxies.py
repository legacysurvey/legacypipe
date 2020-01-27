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

def _custom_sky(args):
    """Wrapper function for the multiprocessing."""
    return custom_sky(*args)

def custom_sky(survey, targetwcs, apodize, ccd): 
    """Perform custom sky-subtraction on a single CCD.

    """
    from astropy.stats import sigma_clipped_stats
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    # Why is this a DecamImage Class instead of a LegacySurveyImage Class
    im = survey.get_image_object(ccd)
    tim = im.get_tractor_image()
    #tim = im.get_tractor_image(splinesky=True, subsky=False, hybridPsf=True,
    #                           normalizePsf=True, apodize=apodize)
    img = tim.getImage()
    ivar = tim.getInvvar()

    refs, _ = get_reference_sources(survey, targetwcs, im.pixscale, ['r'],
                                    tycho_stars=True, gaia_stars=True,
                                    large_galaxies=True, star_clusters=True)
    refmask = get_inblob_map(tim.subwcs, refs) != 0

    skypix = ~refmask * (ivar != 0)
    if np.sum(skypix) == 0:
        print('No pixels to estimate sky...fix me!')
        skymean, skymedian, skysig = 0., 0., 0.
    else:
        objmask = _build_objmask(img, ivar, skypix)
        skypix = np.logical_or(objmask != 0, skypix)
        skymean, skymedian, skysig = sigma_clipped_stats(img, mask=~skypix, sigma=3.0)    

    return skymean, skymedian, skysig
        
def complicated_custom_sky(survey, brickname, targetwcs, apodize, ccd): 
    """Perform custom sky-subtraction on a single CCD.

    #onegal, radius_mask_arcsec, sky_annulus, 

    """
    from astrometry.util.resample import resample_with_wcs
    from legacypipe.reference import get_reference_sources
    from legacypipe.oneblob import get_inblob_map

    log = None

    # Preliminary stuff: read the full-field tim and parse it.
    im = survey.get_image_object(ccd)
    hdr = im.read_image_header()
    hdr.delete('INHERIT')
    hdr.delete('EXTVER')

    print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
          'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
          'object', getattr(ccd, 'object', None), file=log)

    #radius_mask = np.round(radius_mask_arcsec / im.pixscale).astype('int') # [pixels]
    print('Fix me!')
    tim = im.get_tractor_image()
    #tim = im.get_tractor_image(splinesky=True, subsky=False, hybridPsf=True,
    #                           normalizePsf=True, apodize=apodize)

    targetwcs, bands = tim.subwcs, tim.band
    H, W = targetwcs.shape
    H, W = np.int(H), np.int(W)

    img = tim.getImage()
    ivar = tim.getInvvar()

    ## Next, read the splinesky model (for comparison purposes).
    #T = fits_table(im.merged_splineskyfn)
    #I, = np.nonzero((T.expnum == im.expnum) * np.array([c.strip() == im.ccdname for c in T.ccdname]))
    #if len(I) != 1:
    #    print('Multiple splinesky models!', file=log)
    #    return 0
    #splineskytable = T[I]

    # Third, build up a mask consisting of (1) masked pixels in the inverse
    # variance map; (2) known bright stars; (3) astrophysical sources in the
    # image; and (4) the object of interest.
    ivarmask = ivar <= 0

    refs, _ = get_reference_sources(survey, targetwcs, im.pixscale, ['r'],
                                    tycho_stars=True, gaia_stars=True,
                                    large_galaxies=True, star_clusters=False)
    refmask = get_inblob_map(targetwcs, refs) != 0
    import pdb ; pdb.set_trace()    

    #http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    _, x0, y0 = targetwcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
    xcen, ycen = np.round(x0 - 1).astype('int'), np.round(y0 - 1).astype('int')
    ymask, xmask = np.ogrid[-ycen:H-ycen, -xcen:W-xcen]
    galmask = (xmask**2 + ymask**2) <= radius_mask**2

    skypix = (ivarmask*1 + refmask*1 + galmask*1) == 0
    objmask = _build_objmask(img, ivar, skypix)

    # Next, optionally define an annulus of sky pixels centered on the object of
    # interest.
    if sky_annulus:
        skyfactor_in = np.array([ 0.5, 0.5, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0], dtype='f4')
        skyfactor_out = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 5.0], dtype='f4')
        #skyrow_use = (skyfactor_in == 2.0) * (skyfactor_out == 4.0)
        skyrow_use = np.zeros(len(skyfactor_in)).astype(bool)
        skyrow_use[7] = True
        
        nsky = len(skyfactor_in)
        skymean = np.zeros(nsky, dtype='f4')
        skymedian, skysig, skymode = np.zeros_like(skymean), np.zeros_like(skymean), np.zeros_like(skymean)
        for ii in range(nsky):
            inmask = (xmask**2 + ymask**2) <= skyfactor_in[ii]*radius_mask**2
            outmask = (xmask**2 + ymask**2) <= skyfactor_out[ii]*radius_mask**2
            skymask = (outmask*1 - inmask*1 - galmask*1) == 1
            # There can be no sky pixels if the CCD is on the periphery.
            if np.sum(skymask) == 0:
                skymask = np.ones_like(img).astype(bool)

            skymean1, skymedian1, skysig1, skymode1 = _get_skystats(
                img, ivarmask, refmask, galmask, objmask, skymask, tim)
            skymean[ii], skymedian[ii], skysig[ii], skymode[ii] = skymean1, skymedian1, skysig1, skymode1
    else:
        nsky = 1
        skyfactor_in, skyfactor_out = np.array(0.0, dtype='f4'), np.array(0.0, dtype='f4')
        skyrow_use = np.array(False)
        
        skymask = np.ones_like(img).astype(bool)
        skymean, skymedian, skysig, skymode = _get_skystats(img, ivarmask, refmask, galmask, objmask, skymask, tim)

    # Final steps: 

    # (1) Build the final bit-mask image.
    #   0    = 
    #   2**0 = refmask  - reference stars and galaxies
    #   2**1 = objmask  - threshold-detected objects
    #   2**2 = galmask  - central galaxy & system
    mask = np.zeros_like(img).astype(np.int16)
    #mask[ivarmask] += 2**0
    mask[refmask]  += 2**0
    mask[objmask]  += 2**1
    mask[galmask]  += 2**2

    # (2) Resample the mask onto the final mosaic image.
    HH, WW = targetwcs.shape
    comask = np.zeros((HH, WW), np.int16)
    try:
        Yo, Xo, Yi, Xi, _ = resample_with_wcs(targetwcs, targetwcs)
        comask[Yo, Xo] = mask[Yi, Xi]
    except:
        pass

    # (3) Add the sky values and also the central pixel coordinates of the object of
    # interest (so we won't need the WCS object downstream, in QA).
    
    #for card, value in zip(('SKYMODE', 'SKYMED', 'SKYMEAN', 'SKYSIG'),
    #                       (skymode, skymed, skymean, skysig)):
    #    hdr.add_record(dict(name=card, value=value))
    hdr.add_record(dict(name='CAMERA', value=im.camera))
    hdr.add_record(dict(name='EXPNUM', value=im.expnum))
    hdr.add_record(dict(name='CCDNAME', value=im.ccdname))
    hdr.add_record(dict(name='RADMASK', value=np.array(radius_mask, dtype='f4'), comment='pixels'))
    hdr.add_record(dict(name='XCEN', value=x0-1, comment='zero-indexed'))
    hdr.add_record(dict(name='YCEN', value=y0-1, comment='zero-indexed'))

    customsky = fits_table()
    customsky.skymode = np.array(skymode)
    customsky.skymedian = np.array(skymedian)
    customsky.skymean = np.array(skymean)
    customsky.skysig = np.array(skysig)
    customsky.skyfactor_in = np.array(skyfactor_in)
    customsky.skyfactor_out = np.array(skyfactor_out)
    customsky.skyrow_use = np.array(skyrow_use)
    #customsky.xcen = np.repeat(x0 - 1, nsky) # 0-indexed
    #customsky.ycen = np.repeat(y0 - 1, nsky)
    customsky.to_np_arrays()

    # (4) Pack into a dictionary and return.
    out = dict()
    ext = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname.lower().strip())
    #ext = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
    out['{}-mask'.format(ext)] = mask
    out['{}-image'.format(ext)] = img
    out['{}-splinesky'.format(ext)] = splineskytable
    out['{}-header'.format(ext)] = hdr
    out['{}-customsky'.format(ext)] = customsky
    out['{}-comask'.format(ext)] = comask
    
    return out

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
    from tractor.wcs import ConstantFitsWcs
    from tractor import GaussianMixturePSF
    from tractor.tractortime import TAITime
    import astropy.time
    import fitsio
    
    # Custom sky-subtraction. TODO: subtract from the tims...
    #sky = list(zip(*mp.map(_custom_sky, [(survey, targetwcs, apodize, _ccd) for _ccd in ccds])))
    #import pdb ; pdb.set_trace()

    # Create coadds and then build custom tims from them.

    # We tried setting the invvars constant per tim -- this makes things worse, since we *remove*
    # the lowered invvars at the cores of galaxies.
    # for tim in tims:
    #     ie = tim.inverr
    #     newie = np.ones(ie.shape, np.float32) / (tim.sig1)
    #     newie[ie == 0] = 0.
    #     tim.inverr = newie

    #for tim in tims:
    #    print(tim.origsky)

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

        from tractor.psf import PixelizedPSF
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
        #import pdb ; pdb.set_trace()

        #return dict(cotims=cotims)
    # EVIL
    return dict(tims=cotims)

