"""
LSLGA.galex
===========

Code to generate GALEX custom coadds / mosaics.

"""
import os, pdb
import numpy as np

import fitsio

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table

import logging
logger = logging.getLogger('legacypipe.galex')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)


def stage_galex_forced(
    survey=None,
    cat=None,
    T=None,
    targetwcs=None,
    targetrd=None,
    W=None, H=None,
    pixscale=None,
    brickname=None,
    galex_dir=None,
    brick=None,
    version_header=None,
    maskbits=None,
    mp=None,
    record_event=None,
    ps=None,
    plots=False,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the GALEX coadds.
    '''
    from legacypipe.runbrick import _add_stage_version
    from legacypipe.bits import MASKBITS
    #from legacypipe.galex import galex_phot, galex_tiles_touching_wcs
    #from legacypipe.unwise import unwise_phot, collapse_unwise_bitmask, unwise_tiles_touching_wcs
    #from legacypipe.survey import wise_apertures_arcsec
    from tractor import NanoMaggies

    record_event and record_event('stage_galex_forced: starting')
    _add_stage_version(version_header, 'GALEX', 'galex_forced')

    if not plots:
        ps = None

    # Skip if $GALEX_DIR or --galex-dir not set.
    if galex_dir is None:
        info('GALEX_DIR not set -- skipping GALEX')
        return None

    tiles = galex_tiles_touching_wcs(targetwcs, galex_dir)
    info('Cut to', len(tiles), 'GALEX tiles')

    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [targetrd[0,0], targetrd[2,0], targetrd[0,1], targetrd[2,1]]

    # Sources to photometer
    do_phot = np.ones(len(cat), bool)

    # Drop sources within the CLUSTER mask from forced photometry.
    Icluster = None
    if maskbits is not None:
        incluster = (maskbits & MASKBITS['CLUSTER'] > 0)
        if np.any(incluster):
            print('Checking for sources inside CLUSTER mask')
            ra  = np.array([src.getPosition().ra  for src in cat])
            dec = np.array([src.getPosition().dec for src in cat])
            ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx - 1).astype(int)
            yy = np.round(yy - 1).astype(int)
            I = np.flatnonzero(ok * (xx >= 0)*(xx < W) * (yy >= 0)*(yy < H))
            if len(I):
                Icluster = I[incluster[yy[I], xx[I]]]
                print('Found', len(Icluster), 'of', len(cat), 'sources inside CLUSTER mask')
                do_phot[Icluster] = False
    Nskipped = len(do_phot) - np.sum(do_phot)

    gcat = []
    for i in np.flatnonzero(do_phot):
        src = cat[i]
        src = src.copy()
        src.setBrightness(NanoMaggies(galex=1.))
        gcat.append(src)

    # use pixelized PSF model
    pixpsf = True

    # Photometer the two bands in parallel
    args = []
    for band in ['n','f']:
        btiles = tiles[tiles.get('has_%s' % band)]
        if len(btiles) == 0:
            continue
        args.append((galex_dir, gcat, btiles, band, roiradec, pixpsf, ps))
    # Run the forced photometry!
    record_event and record_event('stage_galex_forced: photometry')
    phots = mp.map(galex_phot, args)
    record_event and record_event('stage_galex_forced: results')

    # Unpack results...
    GALEX = None
    if len(phots):
        # The "phot" results for the full-depth coadds are one table per
        # band.  Merge all those columns.
        galex_models = {}
        for i,p in enumerate(phots[:len(args)]):
            if p is None:
                (_,_,tiles,band) = args[i][:4]
                print('"None" result from GALEX forced phot:', tiles, band)
                continue
            galex_models.update(p.models)
            if GALEX is None:
                GALEX = p.phot
            else:
                # remove duplicates
                #p.phot.delete_column('galex_tile')
                #p.phot.delete_column('galex_x')
                #p.phot.delete_column('galex_y')
                GALEX.add_columns_from(p.phot)

        if False:
            from legacypipe.coadds import GalexCoadd
            # Create the WCS into which we'll resample the tiles.
            # Same center as "targetwcs" but bigger pixel scale.
            gpixscale = 1.5
            gra  = np.array([src.getPosition().ra  for src in cat])
            gdec = np.array([src.getPosition().dec for src in cat])
    
            gcoadds = GalexCoadd(targetwcs, W, H, pixscale, gpixscale)
            for tile in tiles.tilename:
                gcoadds.add(tile, galex_models)
        #apphot = wcoadds.finish(survey, brickname, version_header,
        #                        apradec=(wra,wdec),
        #                        apertures=wise_apertures_arcsec/wpixscale)
        #api,apd,apr = apphot
        #for iband,band in enumerate([1,2,3,4]):
        #    WISE.set('apflux_w%i' % band, api[iband])
        #    WISE.set('apflux_resid_w%i' % band, apr[iband])
        #    d = apd[iband]
        #    iv = np.zeros_like(d)
        #    iv[d != 0.] = 1./(d[d != 0]**2)
        #    WISE.set('apflux_ivar_w%i' % band, iv)
        #    print('Setting WISE apphot')

        if Nskipped > 0:
            assert(len(GALEX) == len(wcat))
            WISE = _fill_skipped_values(GALEX, Nskipped, do_phot)
            assert(len(GALEX) == len(cat))
            assert(len(GALEX) == len(T))

    debug('Returning: GALEX', GALEX)

    #### FIXME DEBUG
    GALEX.writeto('galex-phot.fits')
    
    return dict(GALEX=GALEX,
                version_header=version_header)#,
                #wise_apertures_arcsec=wise_apertures_arcsec)

def galex_phot(X):
    '''
    one band x multiple GALEX tiles/images
    '''
    (galex_dir, cat, tiles, band, roiradec, pixelized_psf, ps) = X
    kwargs = dict(pixelized_psf=pixelized_psf, ps=ps)

    W = None
    try:
        W = galex_forcedphot(galex_dir, cat, tiles, band, roiradec, **kwargs)
    except:
        import traceback
        print('galex_forcedphot failed:')
        traceback.print_exc()
    return W

def galex_forcedphot(galex_dir, cat, tiles, band, roiradecbox,
                     pixelized_psf=False, ps=None):
    '''
    Given a list of tractor sources *cat*
    and a list of GALEX tiles *tiles* (a fits_table with RA,Dec,tilename)
    runs forced photometry, returning a FITS table the same length as *cat*.
    '''
    from tractor import NanoMaggies, PointSource, Tractor, ExpGalaxy, DevGalaxy
    from tractor.sersic import SersicGalaxy
    from astrometry.util.ttime import Time

    if False:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('wise-forced-w%i' % band)
    plots = (ps is not None)
    if plots:
        import pylab as plt

    use_ceres = True
    wantims = True
    get_models = True
    
    gband = 'galex'
    #fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix',
    #          'pronexp']

    Nsrcs = len(cat)
    phot = fits_table()
    # Filled in based on unique tile overlap
    # phot.galex_tile = np.array([''] * Nsrcs)
    # phot.galex_x = np.zeros(Nsrcs, np.float32)
    # phot.galex_y = np.zeros(Nsrcs, np.float32)
    # phot.set(wband + '_psfdepth', np.zeros(len(phot), np.float32))

    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])

    # nexp = np.zeros(Nsrcs, np.int16)
    # mjd  = np.zeros(Nsrcs, np.float64)
    # central_flux = np.zeros(Nsrcs, np.float32)

    # fitstats = {}
    tims = []

    for tile in tiles:
        info('Reading GALEX tile', tile.visitname.strip(), 'band', band)

        tim = galex_tractor_image(tile, band, galex_dir, roiradecbox, gband)
        if tim is None:
            debug('Actually, no overlap with tile', tile.tilename)
            continue

        # if plots:
        #     sig1 = tim.sig1
        #     plt.clf()
        #     plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
        #                cmap='gray', vmin=-3 * sig1, vmax=10 * sig1)
        #     plt.colorbar()
        #     tag = '%s W%i' % (tile.tilename, band)
        #     plt.title('%s: tim data' % tag)
        #     ps.savefig()

        # if pixelized_psf:
        #     from unwise_psf import unwise_psf
        #     if (band == 1) or (band == 2):
        #         # we only have updated PSFs for W1 and W2
        #         psfimg = unwise_psf.get_unwise_psf(band, tile.tilename,
        #                                            modelname='neo6_unwisecat')
        #     from tractor.psf import PixelizedPSF
        #     psfimg /= psfimg.sum()
        #     fluxrescales = {1: 1.04, 2: 1.005, 3: 1.0, 4: 1.0}
        #     psfimg *= fluxrescales[band]
        #     tim.psf = PixelizedPSF(psfimg)

        # nexp[I] = tim.nuims[y[I], x[I]]
        # if hasattr(tim, 'mjdmin') and hasattr(tim, 'mjdmax'):
        #     mjd[I] = (tim.mjdmin + tim.mjdmax) / 2.
        # phot.galex_tilename[I] = tile.tilename
        # phot.galex_x[I] = fx[I] - 1.
        # phot.galex_y[I] = fy[I] - 1.
        # central_flux[I] = tim.getImage()[y[I], x[I]]

        # # PSF norm for depth
        # psf = tim.getPsf()
        # h,w = tim.shape
        # patch = psf.getPointSourcePatch(h//2, w//2).patch
        # psfnorm = np.sqrt(np.sum(patch**2))
        # # To handle zero-depth, we return 1/nanomaggies^2 units rather than mags.
        # psfdepth = 1. / (tim.sig1 / psfnorm)**2
        # phot.get(wband + '_psfdepth')[I] = psfdepth

        tim.tile = tile
        tims.append(tim)

    # # Eddie's non-secret recipe:
    # #- central pixel <= 1000: 19x19 pix box size
    # #- central pixel in 1000 - 20000: 59x59 box size
    # #- central pixel > 20000 or saturated: 149x149 box size
    # #- object near "bright star": 299x299 box size
    # nbig = nmedium = nsmall = 0
    # for src,cflux in zip(cat, central_flux):
    #     if cflux > 20000:
    #         R = 100
    #         nbig += 1
    #     elif cflux > 1000:
    #         R = 30
    #         nmedium += 1
    #     else:
    #         R = 15
    #         nsmall += 1
    #     if isinstance(src, PointSource):
    #         src.fixedRadius = R
    #     else:
    #         ### FIXME -- sizes for galaxies..... can we set PSF size separately?
    #         galrad = 0
    #         # RexGalaxy is a subclass of ExpGalaxy
    #         if isinstance(src, (ExpGalaxy, DevGalaxy, SersicGalaxy)):
    #             galrad = src.shape.re
    #         pixscale = 2.75
    #         src.halfsize = int(np.hypot(R, galrad * 5 / pixscale))
    # debug('Set WISE source sizes:', nbig, 'big', nmedium, 'medium', nsmall, 'small')

    tractor = Tractor(tims, cat)
    if use_ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        ceres_block = 8
        tractor.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
    tractor.freezeParamsRecursive('*')
    tractor.thawPathsTo(gband)

    t0 = Time()

    R = tractor.optimize_forced_photometry(
        fitstats=True, variance=True, shared_params=False,
        wantims=wantims)
    info('GALEX forced photometry took', Time() - t0)
    #info('Result:', R)
    
    if use_ceres:
        term = R.ceres_status['termination']
        # Running out of memory can cause failure to converge and term
        # status = 2.  Fail completely in this case.
        if term != 0:
            info('Ceres termination status:', term)
            raise RuntimeError('Ceres terminated with status %i' % term)

    if wantims:
        ims1 = R.ims1
    flux_invvars = R.IV
    # if R.fitstats is not None:
    #     for k in fskeys:
    #         x = getattr(R.fitstats, k)
    #         fitstats[k] = np.array(x).astype(np.float32)

    # if plots:
    #     # Create models for just the brightest sources
    #     bright_cat = [src for src in cat
    #                   if src.getBrightness().getBand(wanyband) > 1000]
    #     debug('Bright soures:', len(bright_cat))
    #     btr = Tractor(tims, bright_cat)
    #     for tim in tims:
    #         mod = btr.getModelImage(tim)
    #         tile = tim.tile
    #         tag = '%s W%i' % (tile.tilename, band)
    #         sig1 = tim.sig1
    #         plt.clf()
    #         plt.imshow(mod, interpolation='nearest', origin='lower',
    #                    cmap='gray', vmin=-3 * sig1, vmax=25 * sig1)
    #         plt.colorbar()
    #         plt.title('%s: bright-star models' % tag)
    #         ps.savefig()

    if get_models:
        models = {}
        for i,tim in enumerate(tims):
            tile = tim.tile
            (dat, mod, ie, chi, roi) = ims1[i]
            models[(tile.visitname, band)] = (mod, dat, ie, tim.roi, tim.wcs.wcs)

    if plots:
        for i,tim in enumerate(tims):
            tile = tim.tile
            tag = '%s %s' % (tile.tilename, band)
            (dat, mod, ie, chi, roi) = ims1[i]
            sig1 = tim.sig1
            plt.clf()
            plt.imshow(dat, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3 * sig1, vmax=25 * sig1)
            plt.colorbar()
            plt.title('%s: data' % tag)
            ps.savefig()
            plt.clf()
            plt.imshow(mod, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3 * sig1, vmax=25 * sig1)
            plt.colorbar()
            plt.title('%s: model' % tag)
            ps.savefig()

            plt.clf()
            plt.imshow(chi, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-5, vmax=+5)
            plt.colorbar()
            plt.title('%s: chi' % tag)
            ps.savefig()

    nm = np.array([src.getBrightness().getBand(gband) for src in cat])
    nm_ivar = flux_invvars
    # Sources out of bounds, eg, never change from their default
    # (1-sigma or whatever) initial fluxes.  Zero them out instead.
    nm[nm_ivar == 0] = 0.

    phot.set(band + '_nanomaggies', nm.astype(np.float32))
    phot.set(band + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
    #for k in fskeys:
    #    phot.set(band + '_' + k, fitstats[k])
    #phot.set(band + '_nexp', nexp)
    #phot.set(band + '_mjd', mjd)

    rtn = gphotduck()
    rtn.phot = phot
    rtn.models = None
    if get_models:
        rtn.models = models
    return rtn

class gphotduck(object):
    pass


def _ra_ranges_overlap(ralo, rahi, ra1, ra2):
    import numpy as np
    x1 = np.cos(np.deg2rad(ralo))
    y1 = np.sin(np.deg2rad(ralo))
    x2 = np.cos(np.deg2rad(rahi))
    y2 = np.sin(np.deg2rad(rahi))
    x3 = np.cos(np.deg2rad(ra1))
    y3 = np.sin(np.deg2rad(ra1))
    x4 = np.cos(np.deg2rad(ra2))
    y4 = np.sin(np.deg2rad(ra2))
    cw32 = x2*y3 - x3*y2
    cw41 = x1*y4 - x4*y1
    return np.logical_and(cw32 <= 0, cw41 >= 0)

def _galex_rgb_dstn(imgs, **kwargs):
    nuv,fuv = imgs
    h,w = nuv.shape
    myrgb = np.zeros((h,w,3), np.float32)
    lo,hi = -0.0005, 0.01
    myrgb[:,:,0] = myrgb[:,:,1] = np.clip((nuv - lo) / (hi - lo), 0., 1.)
    lo,hi = -0.00015, 0.003
    myrgb[:,:,2] = np.clip((fuv - lo) / (hi - lo), 0., 1.)
    myrgb[:,:,1] = np.clip((myrgb[:,:,0] + myrgb[:,:,2]*0.2), 0., 1.)
    return myrgb

def _galex_rgb_official(imgs, **kwargs):
    from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
        mx = np.maximum(red, np.maximum(green, blue))
        mx = np.maximum(1., mx)
    red   /= mx
    green /= mx
    blue  /= mx
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

def _galex_rgb_moustakas(imgs, **kwargs):
    #from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    #blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 0.5 # 1.0 # 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
    mx = np.maximum(red, np.maximum(green, blue))
    mx = np.maximum(1., mx)

    lo = -0.1
    red   = (red - lo) / (mx - lo)
    green = (green - lo) / (mx - lo)
    blue  = (blue - lo) / (mx - lo)
    #red   /= mx
    #green /= mx
    #blue  /= mx
    
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

def galex_tiles_touching_wcs(targetwcs, galex_dir):
    """Find and read the overlapping GALEX FUV/NUV tiles."""

    H, W = targetwcs.shape
    
    ralo, declo = targetwcs.pixelxy2radec(W, 1)
    rahi, dechi = targetwcs.pixelxy2radec(1, H)
    #print('RA',  ralo,rahi)
    #print('Dec', declo,dechi)

    fn = os.path.join(galex_dir, 'galex-images.fits')
    #print('Reading', fn)
    # galex "bricks" (actually just GALEX tiles)
    galex_tiles = fits_table(fn)
    galex_tiles.rename('ra_cent', 'ra')
    galex_tiles.rename('dec_cent', 'dec')
    galex_tiles.rename('have_n', 'has_n')
    galex_tiles.rename('have_f', 'has_f')
    
    cosd = np.cos(np.deg2rad(galex_tiles.dec))
    galex_tiles.ra1 = galex_tiles.ra - 3840*1.5/3600./2./cosd
    galex_tiles.ra2 = galex_tiles.ra + 3840*1.5/3600./2./cosd
    galex_tiles.dec1 = galex_tiles.dec - 3840*1.5/3600./2.
    galex_tiles.dec2 = galex_tiles.dec + 3840*1.5/3600./2.
    visnames = []
    for tile, subvis in zip(galex_tiles.tilename, galex_tiles.subvis):
        if subvis == -999:
            visnames.append(tile.strip())
        else:
            visnames.append('%s_sg%02i' % (tile.strip(), subvis))
    galex_tiles.visitname = np.array(visnames)

    # bricks_touching_radec_box(self, ralo, rahi, declo, dechi, scale=None):
    I, = np.nonzero((galex_tiles.dec1 <= dechi) * (galex_tiles.dec2 >= declo))
    ok = _ra_ranges_overlap(ralo, rahi, galex_tiles.ra1[I], galex_tiles.ra2[I])
    I = I[ok]
    galex_tiles.cut(I)
    # print('-> bricks', galex_tiles.brickname, flush=True, file=log)

    return galex_tiles

def galex_tractor_image(tile, band, galex_dir, radecbox, bandname):
    from tractor import (NanoMaggies, Image, LinearPhotoCal,
                         NCircularGaussianPSF, ConstantFitsWcs, ConstantSky)

    assert(band in ['n','f'])

    #nicegbands = ['NUV', 'FUV']
    #zps = dict(n=20.08, f=18.82)
    #zp = zps[band]
    
    imfn = os.path.join(galex_dir, tile.tilename.strip(),
                        '%s-%sd-intbgsub.fits.gz' % (tile.visitname.strip(), band))
    gwcs = Tan(*[float(f) for f in
                 [tile.crval1, tile.crval2, tile.crpix1, tile.crpix2,
                  tile.cdelt1, 0., 0., tile.cdelt2, 3840., 3840.]])
    (r0,r1,d0,d1) = radecbox
    H,W = gwcs.shape
    ok,xx,yy = gwcs.radec2pixelxy([r0,r0,r1,r1], [d0,d1,d1,d0])
    #print('GALEX WCS pixel positions of RA,Dec box:', xx, yy)
    if np.any(np.logical_not(ok)):
        return None
    x0 = np.clip(np.floor(xx-1).astype(int).min(), 0, W-1)
    x1 = np.clip(np.ceil (xx-1).astype(int).max(), 0, W)
    if x1-x0 <= 1:
        return None
    y0 = np.clip(np.floor(yy-1).astype(int).min(), 0, H-1)
    y1 = np.clip(np.ceil (yy-1).astype(int).max(), 0, H)
    if y1-y0 <= 1:
        return None
    debug('Reading GALEX subimage x0,y0', x0,y0, 'size', x1-x0, y1-y0)
    gwcs = gwcs.get_subimage(x0, y0, x1 - x0, y1 - y0)
    twcs = ConstantFitsWcs(gwcs)
    roislice = (slice(y0, y1), slice(x0, x1))
    
    fitsimg = fitsio.FITS(imfn)[0]
    hdr = fitsimg.read_header()
    img = fitsimg[roislice]

    inverr = np.ones_like(img)
    inverr[img == 0.] = 0.

    zp = tile.get('%s_zpmag' % band)
    
    photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp), band=bandname)

    tsky = ConstantSky(0.)

    name = 'GALEX ' + hdr['OBJECT'] + ' ' + band

    # HACK -- circular Gaussian PSF of fixed size...
    # in arcsec
    fwhms = dict(n=6.0, f=6.0)
    # -> sigma in pixels
    sig = fwhms[band] / 2.35 / twcs.pixel_scale()
    tpsf = NCircularGaussianPSF([sig], [1.])

    tim = Image(data=img, inverr=inverr, psf=tpsf, wcs=twcs,
                sky=tsky, photocal=photocal, name=name)
    tim.roi = [x0,x1,y0,y1]
    return tim

def galex_coadds(onegal, galaxy=None, radius_mosaic=30, radius_mask=None,
                 pixscale=1.5, ref_pixscale=0.262, output_dir=None, galex_dir=None,
                 log=None, centrals=True, verbose=False):
    '''Generate custom GALEX cutouts.
    
    radius_mosaic and radius_mask in arcsec
    
    pixscale: GALEX pixel scale in arcsec/pixel.

    '''
    import fitsio
    import matplotlib.pyplot as plt

    from astrometry.libkd.spherematch import match_radec
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from tractor import (Tractor, NanoMaggies, Image, LinearPhotoCal,
                         NCircularGaussianPSF, ConstantFitsWcs, ConstantSky)

    from legacypipe.survey import imsave_jpeg
    from legacypipe.catalog import read_fits_catalog

    if galaxy is None:
        galaxy = 'galaxy'

    if galex_dir is None:
        galex_dir = os.environ.get('GALEX_DIR')

    if output_dir is None:
        output_dir = '.'

    if radius_mask is None:
        radius_mask = radius_mosaic
        radius_search = 5.0 # [arcsec]
    else:
        radius_search = radius_mask

    W = H = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]
    targetwcs = Tan(onegal['RA'], onegal['DEC'], (W+1) / 2.0, (H+1) / 2.0,
                    -pixscale / 3600.0, 0.0, 0.0, pixscale / 3600.0, float(W), float(H))

    # Read the custom Tractor catalog
    tractorfile = os.path.join(output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0

    cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(cat), tractorfile), flush=True, file=log)

    keep = np.ones(len(cat)).astype(bool)
    if centrals:
        # Find the large central galaxy and mask out (ignore) all the models
        # which are within its elliptical mask.

        # This algorithm will have to change for mosaics not centered on large
        # galaxies, e.g., in galaxy groups.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, onegal['RA'], onegal['DEC'],
                                  radius_search/3600.0, nearest=False)
        if len(m1) == 0:
            print('No central galaxies found at the central coordinates!', flush=True, file=log)
        else:
            pixfactor = ref_pixscale / pixscale # shift the optical Tractor positions
            for mm in m1:
                morphtype = cat.type[mm].strip()
                if morphtype == 'EXP' or morphtype == 'COMP':
                    e1, e2, r50 = cat.shapeexp_e1[mm], cat.shapeexp_e2[mm], cat.shapeexp_r[mm] # [arcsec]
                elif morphtype == 'DEV' or morphtype == 'COMP':
                    e1, e2, r50 = cat.shapedev_e1[mm], cat.shapedev_e2[mm], cat.shapedev_r[mm] # [arcsec]
                else:
                    r50 = None

                if r50:
                    majoraxis =  r50 * 5 / pixscale # [pixels]
                    ba, phi = LSLGA.misc.convert_tractor_e1e2(e1, e2)
                    these = LSLGA.misc.ellipse_mask(W / 2, W / 2, majoraxis, ba * majoraxis,
                                                    np.radians(phi), cat.bx*pixfactor, cat.by*pixfactor)
                    if np.sum(these) > 0:
                        #keep[these] = False
                        pass
                print('Hack!')
                keep[mm] = False

            #srcs = read_fits_catalog(cat)
            #_srcs = np.array(srcs)[~keep].tolist()
            #mod = LSLGA.misc.srcs2image(_srcs, ConstantFitsWcs(targetwcs), psf_sigma=3.0)
            #import matplotlib.pyplot as plt
            ##plt.imshow(mod, origin='lower') ; plt.savefig('junk.png')
            #plt.imshow(np.log10(mod), origin='lower') ; plt.savefig('junk.png')
            #pdb.set_trace()

    srcs = read_fits_catalog(cat)
    for src in srcs:
        src.freezeAllBut('brightness')
    #srcs_nocentral = np.array(srcs)[keep].tolist()
    
    # Find all overlapping GALEX tiles and then read the tims.
    galex_tiles = _read_galex_tiles(targetwcs, galex_dir, log=log, verbose=verbose)

    gbands = ['n','f']
    nicegbands = ['NUV', 'FUV']

    zps = dict(n=20.08, f=18.82)

    coimgs, comods, coresids, coimgs_central, comods_nocentral = [], [], [], [], []
    for niceband, band in zip(nicegbands, gbands):
        J = np.flatnonzero(galex_tiles.get('has_'+band))
        print(len(J), 'GALEX tiles have coverage in band', band)

        coimg = np.zeros((H, W), np.float32)
        comod = np.zeros((H, W), np.float32)
        cowt  = np.zeros((H, W), np.float32)

        comod_nocentral = np.zeros((H, W), np.float32)

        for src in srcs:
            src.setBrightness(NanoMaggies(**{band: 1}))

        for j in J:
            tile = galex_tiles[j]
            fn = os.path.join(galex_dir, tile.tilename.strip(),
                              '%s-%sd-intbgsub.fits.gz' % (tile.tilename, band))
            #print(fn)

            gwcs = Tan(*[float(f) for f in
                         [tile.crval1, tile.crval2, tile.crpix1, tile.crpix2,
                          tile.cdelt1, 0., 0., tile.cdelt2, 3840., 3840.]])
            img = fitsio.read(fn)
            #print('Read', img.shape)

            try:
                Yo, Xo, Yi, Xi, nil = resample_with_wcs(targetwcs, gwcs, [], 3)
            except OverlapError:
                continue

            K = np.flatnonzero(img[Yi, Xi] != 0.)
            if len(K) == 0:
                continue
            Yo, Xo, Yi, Xi = Yo[K], Xo[K], Yi[K], Xi[K]

            wt = tile.get(band + 'exptime')
            coimg[Yo, Xo] += wt * img[Yi, Xi]
            cowt [Yo, Xo] += wt

            x0, x1, y0, y1 = min(Xi), max(Xi), min(Yi), max(Yi)
            subwcs = gwcs.get_subimage(x0, y0, x1-x0+1, y1-y0+1)
            twcs = ConstantFitsWcs(subwcs)
            timg = img[y0:y1+1, x0:x1+1]

            tie = np.ones_like(timg)  ## HACK!
            #hdr = fitsio.read_header(fn)
            #zp = hdr['']
            zp = zps[band]
            photocal = LinearPhotoCal( NanoMaggies.zeropointToScale(zp), band=band)
            tsky = ConstantSky(0.0)
            
            # HACK -- circular Gaussian PSF of fixed size...
            # in arcsec
            #fwhms = dict(NUV=6.0, FUV=6.0)
            # -> sigma in pixels
            #sig = fwhms[band] / 2.35 / twcs.pixel_scale()
            sig = 6.0 / np.sqrt(8 * np.log(2)) / twcs.pixel_scale()
            tpsf = NCircularGaussianPSF([sig], [1.])

            tim = Image(data=timg, inverr=tie, psf=tpsf, wcs=twcs, sky=tsky,
                        photocal=photocal, name='GALEX ' + band + tile.tilename)

            ## Build the model image with and without the central galaxy model.
            tractor = Tractor([tim], srcs)
            mod = tractor.getModelImage(0)
            tractor.freezeParam('images')
            tractor.optimize_forced_photometry(priors=False, shared_params=False)
            mod = tractor.getModelImage(0)

            srcs_nocentral = np.array(srcs)[keep].tolist()
            #srcs_nocentral = np.array(srcs)[nocentral].tolist()
            tractor_nocentral = Tractor([tim], srcs_nocentral)
            mod_nocentral = tractor_nocentral.getModelImage(0)

            comod[Yo, Xo] += wt * mod[Yi-y0, Xi-x0]
            comod_nocentral[Yo, Xo] += wt * mod_nocentral[Yi-y0, Xi-x0]

        coimg /= np.maximum(cowt, 1e-18)
        comod /= np.maximum(cowt, 1e-18)
        comod_nocentral /= np.maximum(cowt, 1e-18)

        coresid = coimg - comod

        # Subtract the model image which excludes the central (comod_nocentral)
        # from the data (coimg) to isolate the light of the central
        # (coimg_central).
        coimg_central = coimg - comod_nocentral

        coimgs.append(coimg)
        comods.append(comod)
        coresids.append(coresid)

        comods_nocentral.append(comod_nocentral)
        coimgs_central.append(coimg_central)

        # Write out the final images with and without the central, making sure
        # to apply the zeropoint to go from counts/s to AB nanomaggies.
        # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
        for thisimg, imtype in zip( (coimg, comod, comod_nocentral),
                                ('image', 'model', 'model-nocentral') ):
            fitsfile = os.path.join(output_dir, '{}-{}-{}.fits'.format(galaxy, imtype, niceband))
            if verbose:
                print('Writing {}'.format(fitsfile))
            fitsio.write(fitsfile, thisimg * 10**(-0.4 * (zp - 22.5)), clobber=True)

    # Build a color mosaic (but note that the images here are in units of
    # background-subtracted counts/s).

    #_galex_rgb = _galex_rgb_moustakas
    #_galex_rgb = _galex_rgb_dstn
    _galex_rgb = _galex_rgb_official

    for imgs, imtype in zip( (coimgs, comods, coresids, comods_nocentral, coimgs_central),
                             ('image', 'model', 'resid', 'model-nocentral', 'image-central') ):
        rgb = _galex_rgb(imgs)
        jpgfile = os.path.join(output_dir, '{}-{}-FUVNUV.jpg'.format(galaxy, imtype))
        if verbose:
            print('Writing {}'.format(jpgfile))
        imsave_jpeg(jpgfile, rgb, origin='lower')

    return 1
