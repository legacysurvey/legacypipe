import os
import sys
import logging
from collections import Counter
import numpy as np
from legacypipe.survey import LegacySurveyData
from legacypipe.ps1cat import ps1cat
from legacypipe.gaiacat import GaiaCatalog
#from legacypipe.gaiacat import GaiaCatalog
from tractor import ModelMask

logger = logging.getLogger('legacypipe.new-camera-setup')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)


def main():
    from legacyzpts.legacy_zeropoints import CAMERAS
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--camera', required=True)

    parser.add_argument('--image-hdu', default=0, type=int, help='Read image data from the given HDU number')

    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='More logging')

    parser.add_argument('--plots', action='store_true', default=False,
                        help='Make plots?')
    parser.add_argument('--gaia-photom', default=False, action='store_true',
                        help='Use Gaia rather than PS-1 for photometric cal.')

    parser.add_argument('image', metavar='image-filename', help='Image filename to read')

    opt = parser.parse_args()

    if opt.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    ps = None
    if opt.plots:
        from astrometry.util.plotutils import PlotSequence
        import pylab as plt
        ps = PlotSequence(opt.camera)

    if opt.camera not in CAMERAS:
        print('You must add your new camera to the list of known cameras at the top of the legacy_zeropoints.py script -- the CAMERAS variable.')
        return

    survey = LegacySurveyData(survey_dir=opt.survey_dir)

    clazz = None
    try:
        clazz = survey.image_class_for_camera(opt.camera)
    except KeyError:
        print('You must:')
        print(' - create a new legacypipe.image.LegacySurveyImage subclass for your new camera')
        print(' - add it to the dict in legacypipe/survey.py : LegacySurveyData : self.image_typemap')
        print(' - import your new class in LegacySurveyData.__init__().')
        return

    info('For camera "%s", found LegacySurveyImage subclass: %s' % (opt.camera, str(clazz)))

    info('Reading', opt.image, 'and trying to create new image object...')

    img = survey.get_image_object(None, camera=opt.camera,
                                  image_fn=opt.image, image_hdu=opt.image_hdu,
                                  camera_setup=True)

    info('Got image of type', type(img))

    # Here we're copying some code out of image.py...
    image_fn = opt.image
    image_hdu = opt.image_hdu
    img.image_filename = image_fn
    img.imgfn = os.path.join(img.survey.get_image_dir(), image_fn)

    info('Relative path to image file -- will be stored in the survey-ccds file --: ', img.image_filename)
    info('Filesystem path to image file:', img.imgfn)

    if not os.path.exists(img.imgfn):
        print('Filesystem path does not exist.  Should be survey-dir path + images (%s) + image-file-argument (%s)' % (survey.get_image_dir(), image_fn))
        return

    info('Reading primary FITS header from image file...')
    primhdr = img.read_image_primary_header()

    info('Reading a bunch of metadata from image primary header:')

    for k in ['band', 'propid', 'expnum', 'camera', 'exptime', 'object']:
        info('get_%s():' % k)
        v = getattr(img, 'get_'+k)(primhdr)
        info('  -> "%s"' % v)
        setattr(img, k, v)

    info('get_mjd():')
    img.mjdobs = img.get_mjd(primhdr)
    info('  -> "%s"' % img.mjdobs)

    namechange = {'date': 'procdate',}
    for key in ['HA', 'DATE', 'PLVER', 'PLPROCID']:
        info('get "%s" from primary header.' % key)
        val = primhdr.get(key)
        if isinstance(val, str):
            val = val.strip()
            if len(val) == 0:
                raise ValueError('Empty header card: %s' % key)
        key = namechange.get(key.lower(), key.lower())
        key = key.replace('-', '_')
        info('  -> "%s"' % val)
        setattr(img, key, val)

    img.hdu = image_hdu
    info('Will read image header from HDU', image_hdu)
    hdr = img.read_image_header(ext=image_hdu)
    info('Reading image metadata...')
    hinfo = img.read_image_fits()[image_hdu].get_info()
    #info('Got:', hinfo)
    img.height,img.width = hinfo['dims']
    info('Got image size', img.width, 'x', img.height, 'pixels')
    img.hdu = hinfo['hdunum'] - 1
    for key in ['ccdname', 'pixscale', 'fwhm']:
        info('get_%s():' % key)
        v = getattr(img, 'get_'+key)(primhdr, hdr)
        info('  -> "%s"' % v)
        setattr(img, key, v)

    for k,d in [('dq_hdu',img.hdu), ('wt_hdu',img.hdu), ('sig1',0.), ('ccdzpt',0.),
                ('dradec',(0.,0.))]:
        v = getattr(img, k, d)
        setattr(img, k, v)

    img.compute_filenames()
    info('Will read image pixels from file        ', img.imgfn, 'HDU', img.hdu)
    info('Will read inverse-variance map from file', img.wtfn,  'HDU', img.wt_hdu)
    info('Will read data-quality map from file    ', img.dqfn,  'HDU', img.dq_hdu)

    info('Will read images from these FITS HDUs:', img.get_extension_list())

    # test funpack_files?

    info('Source Extractor & PsfEx will read the following config files:')
    sedir = survey.get_se_dir()
    for (typ, suff) in [('SE config', '.se'),
                         ('SE params', '.param'),
                         ('SE convolution filter', '.conv'),
                         ('PsfEx config', '.psfex'),
                         ]:
        fn = os.path.join(sedir, img.camera + suff)
        ex = os.path.exists(fn)
        info('  %s: %s (%s)' % (typ, fn, 'exists' if ex else 'does not exist'))

    info('Special PsfEx flags for this CCD:', survey.get_psfex_conf(img.camera, img.expnum, img.ccdname))

    basename = img.get_base_name()
    if len(img.ccdname):
        calname = basename + '-' + img.ccdname
    else:
        calname = basename
    img.name = calname

    img.set_calib_filenames()

    #
    print('Re-creating image object using normal constructor...')
    img = survey.get_image_object(None, camera=opt.camera,
                                  image_fn=opt.image, image_hdu=opt.image_hdu)

    # Once legacy_zeropoints.py starts...
    ra_bore, dec_bore = img.get_radec_bore(primhdr)
    info('RA,Dec boresight:', ra_bore, dec_bore)
    info('Airmass:', img.get_airmass(primhdr, hdr, ra_bore, dec_bore))
    info('Gain:', img.get_gain(primhdr, hdr))
    p1,p2,v1,v2 = img.get_crpixcrval(primhdr, hdr)
    info('WCS Reference pixel:', p1, p2)
    info('WCS Reference pos:', v1, v2)
    info('WCS CD matrix:', img.get_cd_matrix(primhdr, hdr))

    wcs = img.get_wcs(hdr=hdr)
    info('Got WCS object:', wcs)

    H = img.height
    W = img.width
    ccdra, ccddec = wcs.pixelxy2radec((W+1) / 2.0, (H+1) / 2.0)
    info('With image size %i x %i, central RA,Dec is (%.4f, %.4f)' %
         (W, H, ccdra, ccddec))

    slc = img.get_good_image_slice(None)
    info('Good region in this image (slice):', slc)

    # PsfEx file?  FWHM?

    # Reading data...
    info('Reading data quality / mask file...')
    dq,dqhdr = img.read_dq(header=True, slc=slc)
    info('DQ file:', dq.shape, dq.dtype, 'min:', dq.min(), 'max', dq.max(),
         'number of pixels == 0:', np.sum(dq == 0))
    if dq is not None:
        info('Remapping data quality / mask file...')
        dq = img.remap_dq(dq, dqhdr)
    if dq is None:
        info('No DQ file')
    else:
        info('DQ file:', dq.shape, dq.dtype, 'min:', dq.min(), 'max', dq.max(),
             'number of pixels == 0:', np.sum(dq == 0))

    info('Reading inverse-variance / weight file...')
    invvar = img.read_invvar(dq=dq, slc=slc)
    info('Invvar map:', invvar.shape, invvar.dtype, 'min:', invvar.min(),
         'max', invvar.max(), 'median', np.median(invvar),
         'number of pixels == 0:', np.sum(invvar == 0), ', number >0:', np.sum(invvar>0))
    info('Reading image file...')
    impix = img.read_image(slc=slc)
    info('Image pixels:', impix.shape, impix.dtype, 'min:', impix.min(),
         'max', impix.max(), 'median', np.median(impix.ravel()))
    info('Running fix_saturation...')
    img.fix_saturation(impix, dq, invvar, primhdr, hdr, slc)
    info('Image pixels:', impix.shape, impix.dtype, 'min:', impix.min(),
         'max', impix.max(), 'median', np.median(impix.ravel()))
    info('Invvar map:', invvar.shape, invvar.dtype, 'min:', invvar.min(),
         'max', invvar.max(), 'median', np.median(invvar),
         'number of pixels == 0:', np.sum(invvar == 0), ', number >0:', np.sum(invvar>0))
    info('DQ file:', dq.shape, dq.dtype, 'min:', dq.min(), 'max', dq.max(),
         'number of pixels == 0:', np.sum(dq == 0))

    info('Calling estimate_sig1()...')
    img.sig1 = img.estimate_sig1(impix, invvar, dq, primhdr, hdr)
    info('Got sig1 / exptime =', img.sig1)

    info('Calling remap_invvar...')
    invvar = img.remap_invvar(invvar, primhdr, impix, dq)
    info('Blanking out', np.sum((invvar == 0) * (impix != 0)), 'image pixels with invvar=0')
    impix[invvar == 0] = 0.

    info('Image pixels:', impix.shape, impix.dtype, 'min:', impix.min(),
         'max', impix.max(), 'median', np.median(impix.ravel()))
    info('Invvar map:', invvar.shape, invvar.dtype, 'min:', invvar.min(),
         'max', invvar.max(), 'median', np.median(invvar),
         'number of pixels == 0:', np.sum(invvar == 0), ', number >0:', np.sum(invvar>0))
    info('DQ file:', dq.shape, dq.dtype, 'min:', dq.min(), 'max', dq.max(),
         'number of pixels == 0:', np.sum(dq == 0))

    info('Scaling weight(invvar) and image pixels...')
    invvar = img.scale_weight(invvar)
    impix = img.scale_image(impix)
    info('Image pixels:', impix.shape, impix.dtype, 'min:', impix.min(),
         'max', impix.max(), 'median', np.median(impix.ravel()))
    info('Invvar map:', invvar.shape, invvar.dtype, 'min:', invvar.min(),
         'max', invvar.max(), 'median', np.median(invvar),
         'number of pixels == 0:', np.sum(invvar == 0), ', number >0:', np.sum(invvar>0))

    info('Estimating sky level...')
    _, skymed, _ = img.estimate_sky(impix, invvar, dq, primhdr, hdr)
    info('Getting nominal zeropoint for band "%s"' % img.band)
    zp0 = img.nominal_zeropoint(img.band)
    info('Got nominal zeropoint for band', img.band, ':', zp0)
    skybr = zp0 - 2.5*np.log10(skymed / img.pixscale / img.pixscale / img.exptime)
    info('Sky level: %.2f count/pix' % skymed)
    info('Sky brightness: %.3f mag/arcsec^2 (assuming nominal zeropoint)' % skybr)

    # Estimate per-pixel noise via Blanton's 5-pixel MAD
    slice1 = (slice(0,-5,10),slice(0,-5,10))
    slice2 = (slice(5,None,10),slice(5,None,10))
    mad = np.median(np.abs(impix[slice1] - impix[slice2]).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    info('Sky sig1 estimate by Blanton method:', sig1)
    info('Pipeline sig1:', img.sig1 * img.exptime)

    if ps is not None:

        img.set_calib_filenames()

        goodpix = np.flatnonzero(invvar > 0)
        lo,hi = np.percentile(impix.flat[goodpix], [25,98])
        plt.clf()
        plt.imshow(impix, interpolation='nearest', origin='lower', vmin=lo, vmax=hi)
        plt.title('Image pix')
        plt.colorbar()
        ps.savefig()

        skymod = img.read_sky_model()
        skyimg = np.zeros_like(impix)
        skymod.addTo(skyimg)

        plt.clf()
        plt.imshow(skyimg, interpolation='nearest', origin='lower', vmin=lo, vmax=hi)
        plt.title('Sky model')
        plt.colorbar()
        ps.savefig()

        pix = impix - skyimg

        lo,hi = np.percentile(pix.flat[goodpix], [25,98])
        plt.clf()
        plt.imshow(pix, interpolation='nearest', origin='lower', vmin=lo, vmax=hi)
        plt.title('Image pix - sky model')
        plt.colorbar()
        ps.savefig()

        lo,hi = np.percentile(impix.flat[goodpix], [5,95])
        p16,p84 = np.percentile(impix.flat[goodpix], [16,84])
        s1 = img.sig1 * img.exptime

        plt.clf()
        plt.hist(impix.flat[goodpix], bins=50, range=(lo,hi))
        plt.axvline(skymed - s1, color='k', label='Pipeline +- 1 sigma')
        plt.axvline(skymed, color='k')
        plt.axvline(skymed + s1, color='k')
        plt.axvline(p16, color='r', label='16/84th percentiles')
        plt.axvline(p84, color='r')
        plt.legend()
        plt.xlabel('Image pixels, median +- pipeline sig1 (ADU)')
        plt.title('Image pix')
        ps.savefig()

        plt.clf()
        plt.hist(impix.flat[goodpix], bins=50, range=(lo,hi), log=True)
        plt.axvline(skymed - s1, color='k', label='Pipeline +- 1 sigma')
        plt.axvline(skymed, color='k')
        plt.axvline(skymed + s1, color='k')
        plt.axvline(p16, color='r', label='16/84th percentiles')
        plt.axvline(p84, color='r')
        plt.legend()
        plt.xlabel('Image pixels, median +- pipeline sig1 (ADU)')
        plt.title('Image pix')
        ps.savefig()

        lo,hi = np.percentile(pix.flat[goodpix], [5,95])
        p1,p2 = np.percentile(pix.flat[goodpix], [16,84])

        plt.clf()
        plt.hist(pix.flat[goodpix], bins=50, log=True, range=(lo,hi))
        plt.axvline(-s1, color='k')
        plt.axvline(0, color='k')
        plt.axvline(+s1, color='k')
        plt.axvline(p1, color='r')
        plt.axvline(p2, color='r')
        plt.xlabel('Image pix - sky model (ADU)')
        plt.title('Image pixels - sky model, w/pipeline sig1')
        ps.savefig()

        plt.clf()
        n,_,_ = plt.hist(pix.flat[goodpix] * np.sqrt(invvar.flat[goodpix]), bins=50,
                         range=(-5,+5))
        xx = np.linspace(-5, +5, 200)
        yy = np.exp(-0.5 * xx**2)
        mx = max(n)
        plt.plot(xx, mx * yy/max(yy), 'b-')
        plt.xlim(-5,+5)
        plt.xlabel('(Image pixels - sky model)* sqrt(invvar)  (sigma)')
        plt.title('Is uncertainty map correct?')
        ps.savefig()

        plt.clf()
        plt.hist((impix.flat[goodpix] - skymed) * np.sqrt(invvar.flat[goodpix]), bins=50,
                 range=(-5,+5), log=True)
        plt.plot(xx, mx * yy/max(yy), 'b-')
        plt.xlim(-5,+5)
        plt.xlabel('Image pixels * sqrt(invvar)  (sigma)')
        plt.title('Is uncertainty map correct?')
        ps.savefig()

    # Read PSF model
    print('Reading FWHM...')
    psf_fwhm = img.get_fwhm(primhdr, hdr)
    psf_sigma = psf_fwhm / 2.35
    print('Got PSF fwhm', psf_fwhm, 'and sigma', psf_sigma, 'pixels')
    h,w = img.shape
    print('Image shape:', w, 'x', h)
    print('Reading PSF model...')
    psf = img.read_psf_model(w/2, h/2, gaussPsf=False, pixPsf=True, hybridPsf=True,
                             normalizePsf=True,
                             psf_sigma=psf_sigma,
                             w=w, h=h)
    print('Got PSF model:', psf)

    if ps is not None:
        nx,ny = 5,5
        xx = np.linspace(0, w-1, nx)
        yy = np.linspace(0, h-1, ny)
        k = 1
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        for y in yy:
            for x in xx:
                plt.subplot(ny, nx, k)
                k += 1
                patch = psf.getPointSourcePatch(x, y)
                plt.imshow(patch.patch, interpolation='nearest', origin='lower')
                plt.xticks([]); plt.yticks([])
        plt.suptitle('PSF model')
        ps.savefig()

        # zoom-in half-size
        S = 5
        k = 1
        center_psf = None
        psfims = []
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        for y in yy:
            for x in xx:
                plt.subplot(ny, nx, k)
                k += 1
                patch = psf.getPointSourcePatch(x, y)
                patch = patch.patch
                ph,pw = patch.shape
                patch = patch[ph//2-S : ph//2+S+1, pw//2-S : pw//2+S+1]
                psfims.append(patch)
                if y == yy[len(yy)//2] and x == xx[len(xx)//2]:
                    center_psf = patch
                plt.imshow(patch, interpolation='nearest', origin='lower')
                plt.xticks([]); plt.yticks([])
        plt.suptitle('PSF model (zoom-in)')
        ps.savefig()

        mx = max([np.max(np.abs(p - center_psf)) for p in psfims])

        # zoom-in half-size, differential
        k = 1
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        for y in yy:
            for x in xx:
                plt.subplot(ny, nx, k)
                plt.imshow(psfims[k-1] - center_psf, interpolation='nearest', origin='lower', vmin=-mx, vmax=mx)
                plt.xticks([]); plt.yticks([])
                k += 1
        plt.suptitle('PSF model (difference from center)')
        ps.savefig()

        # Gaussian
        # ASSUME HybridPixelizedPSF
        gpsf = psf.gauss
        print('Gaussian PSF:', gpsf)
        cx,cy = xx[len(xx)//2], yy[len(yy)//2]
        gpatch = gpsf.getPointSourcePatch(cx, cy, modelMask=ModelMask(int(cx-S), int(cy-S), 1+S*2, 1+S*2))
        gpatch = gpatch.patch

        mx = max([np.max(gpatch), np.max(center_psf)])
        mn = min([np.min(gpatch), np.min(center_psf)])
        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(center_psf, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.title('Pixelized PSF (center)')
        plt.subplot(1,3,2)
        plt.imshow(gpatch, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.title('Gaussian PSF')
        plt.subplot(1,3,3)
        plt.imshow(center_psf - gpatch, interpolation='nearest', origin='lower')
        plt.title('Pix(center) - Gauss')
        ps.savefig()

        mx = max([np.max(np.abs(p - gpatch)) for p in psfims])

        k = 1
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        for y in yy:
            for x in xx:
                plt.subplot(ny, nx, k)
                plt.imshow(psfims[k-1] - gpatch, interpolation='nearest', origin='lower', vmin=-mx, vmax=mx)
                plt.xticks([]); plt.yticks([])
                k += 1
        plt.suptitle('PSF model (difference from Gaussian)')
        ps.savefig()

    zpt = img.get_zeropoint(primhdr, hdr)
    info('Does a zeropoint already exist in the image headers?  zpt=', zpt)

    phot = None
    if zpt is None:
        if opt.gaia_photom:
            info('Fetching Gaia stars inside this image...')
            gaia = GaiaCatalog().get_catalog_in_wcs(wcs)
            assert(gaia is not None)
            assert(len(gaia) > 0)
            gaia = GaiaCatalog.catalog_nantozero(gaia)
            assert(gaia is not None)
            print(len(gaia), 'Gaia stars')
            phot = gaia
        else:
            info('Fetching Pan-STARRS stars inside this image...')
            try:
                phot = ps1cat(ccdwcs=wcs).get_stars(magrange=None)
                info('Found', len(phot), 'PS1 stars in this image')
            except OSError as e:
                print('No PS1 stars found for this image -- outside the PS1 footprint, or in the Galactic plane?', e)
    phot_name = 'none'
    if phot is not None:
        if opt.gaia_photom:
            phot_name = 'gaia'
        else:
            phot_name = 'ps1'
        info('Choosing calibrator stars...')
        phot.use_for_photometry = img.get_photometric_calibrator_cuts(phot_name, phot)
        info('use for photometry:', Counter(phot.use_for_photometry))

        if len(phot) == 0 or np.sum(phot.use_for_photometry) == 0:
            phot = None
        else:
            # Convert to Legacy Survey mags
            phot.legacy_survey_mag = img.photometric_calibrator_to_observed(phot_name, phot)
            print(len(phot), 'photometric calibrator stars')

    # Check the WCS -- show image cutouts at the locations of the brightest Gaia/PS1 stars.
    print('Calling get_wcs()...')
    wcs = img.get_wcs(hdr=hdr)
    print('Got WCS model:', wcs)

    mjdobs = img.get_mjd(primhdr)
    print('MJD-obs:', mjdobs)
    from tractor.tractortime import TAITime
    import astropy.time
    mjd_tai = astropy.time.Time(mjdobs, format='mjd', scale='utc').tai.mjd
    tai = TAITime(None, mjd=mjd_tai)
    print('TAI:', tai)

    print('Calling get_tractor_wcs()...')
    twcs = img.get_tractor_wcs(wcs, 0., 0., primhdr=primhdr, imghdr=hdr, tai=tai)
    print('Got tractor WCS:', twcs)

    if (phot is not None) and (ps is not None):
        from tractor import RaDecPos

        if phot_name == 'gaia':
            from legacypipe.survey import GaiaPosition
            pos = [GaiaPosition(*a) for a in zip(phot.ra, phot.dec, phot.ref_epoch,
                                                 phot.pmra, phot.pmdec, phot.parallax)]
        else:
            pos = [RaDecPos(ra, dec) for ra,dec in zip(phot.ra, phot.dec)]
        
        xy = [twcs.positionToPixel(p) for p in pos]
        xy = np.array(xy)
        # image cutout half-size
        S = 10
        x,y = xy[:,0], xy[:,1]
        keep = phot.use_for_photometry * (x > S) * (x < w-S) * (y > S) * (y < h-S)
        kphot = phot[keep]
        x,y = x[keep], y[keep]
        I = np.argsort(kphot.legacy_survey_mag)
        R,C = 6,6
        for s in [S, S//2]:
            plt.clf()
            for i in range(R*C):
                plt.subplot(R, C, i+1)
                j = I[i]
                ix,iy = int(x[j]), int(y[j])
                x0,y0 = ix - s, iy - s
                plt.imshow(impix[y0:y0+2*s+1, x0:x0+2*s+1], interpolation='nearest', origin='lower',
                           cmap='gray')
                ax = plt.axis()
                # "imshow" puts pixel centers on integer coordinates;
                # the first pixel goes from -0.5 to +0.5.
                # If we have x == ix, the marker will be centered in the pixel.
                plt.plot(x[j] - x0, y[j] - y0, 's', mec='r', mfc='none', ms=15)
                plt.axis(ax)
                plt.xticks([]); plt.yticks([])
            plt.suptitle('%s stars, according to image WCS' % phot_name)
            ps.savefig()

    print('Running run_zeropoints from legacyzpts...')
    from legacyzpts.legacy_zeropoints import run_zeropoints
    ccds,phot = run_zeropoints(img, splinesky=True, gaia_photom=opt.gaia_photom, ps=ps)


if __name__ == '__main__':
    main()
