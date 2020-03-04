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

    img = tim.getImage()
    ivar = tim.getInvvar()

    (HH, WW), pixscale = targetwcs.shape, tim.subwcs.pixel_scale()
    skyimg = np.zeros((HH, WW), np.float32)

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
    skyimg[Yo, Xo] = skymedian

    return skyimg
        
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
    #from tractor.wcs import ConstantFitsWcs
    #from tractor import GaussianMixturePSF
    from tractor.tractortime import TAITime
    import astropy.time
    import fitsio
    
    # Custom sky-subtraction. TODO: subtract from the tims...
    skyimg = np.stack(mp.map(_get_sky, [(survey, targetwcs, tim) for tim in tims]))
    ww = np.where([tim.band == 'r' for tim in tims])[0]
    with fitsio.FITS('skytest.fits', 'rw') as ff:
        for w in ww:
            ff.write(skyimg[w, :, :])

    import pdb ; pdb.set_trace()

    skyimg = []
    for tim in tims:
        skyimg.append(get_sky(survey, targetwcs, tim))
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

