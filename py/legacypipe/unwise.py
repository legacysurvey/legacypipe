import os
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.ttime import Time

from wise.unwise import get_unwise_tractor_image

'''
This function was imported whole from the tractor repo:
wise/forcedphot.py because I figured we were doing enough
LegacySurvey-specific stuff in it that it was time to just import it
and edit it rather than build elaborate options.
'''
def unwise_forcedphot(cat, tiles, band=1, roiradecbox=None,
                      use_ceres=True, ceres_block=8,
                      save_fits=False, get_models=False, ps=None,
                      psf_broadening=None,
                      pixelized_psf=False,
                      get_masks=None,
                      move_crpix=False,
                      modelsky_dir=None):
    '''
    Given a list of tractor sources *cat*
    and a list of unWISE tiles *tiles* (a fits_table with RA,Dec,coadd_id)
    runs forced photometry, returning a FITS table the same length as *cat*.

    *get_masks*: the WCS to resample mask bits into.
    '''
    from tractor import NanoMaggies, PointSource, Tractor, ExpGalaxy, DevGalaxy, FixedCompositeGalaxy

    if not pixelized_psf and psf_broadening is None:
        # PSF broadening in post-reactivation data, by band.
        # Newer version from Aaron's email to decam-chatter, 2018-06-14.
        broadening = { 1: 1.0405, 2: 1.0346, 3: None, 4: None }
        psf_broadening = broadening[band]

    if False:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('wise-forced-w%i' % band)
    plots = (ps is not None)
    if plots:
        import pylab as plt
    
    wantims = (plots or save_fits or get_models)
    wanyband = 'w'
    if get_models:
        models = {}

    fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix',
              'pronexp']

    Nsrcs = len(cat)
    phot = fits_table()
    phot.wise_coadd_id = np.array(['        '] * Nsrcs)

    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])

    print('Photometering WISE band', band)
    wband = 'w%i' % band

    nexp = np.zeros(Nsrcs, np.int16)
    mjd  = np.zeros(Nsrcs, np.float64)
    central_flux = np.zeros(Nsrcs, np.float32)

    fitstats = {}
    tims = []

    if get_masks:
        mh,mw = get_masks.shape
        maskmap = np.zeros((mh,mw), np.uint32)
    
    for tile in tiles:
        print('Reading tile', tile.coadd_id)

        tim = get_unwise_tractor_image(tile.unwise_dir, tile.coadd_id, band,
                                       bandname=wanyband, roiradecbox=roiradecbox)
        if tim is None:
            print('Actually, no overlap with tile', tile.coadd_id)
            continue

        if plots:
            sig1 = tim.sig1
            plt.clf()
            plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3 * sig1, vmax=10 * sig1)
            plt.colorbar()
            tag = '%s W%i' % (tile.coadd_id, band)
            plt.title('%s: tim data' % tag)
            ps.savefig()

            plt.clf()
            plt.hist((tim.getImage() * tim.inverr)[tim.inverr > 0].ravel(),
                     range=(-5,10), bins=100)
            plt.xlabel('Per-pixel intensity (Sigma)')
            plt.title(tag)
            ps.savefig()

        if move_crpix and band in [1, 2]:
            realwcs = tim.wcs.wcs
            x,y = realwcs.crpix
            tile_crpix = tile.get('crpix_w%i' % band)
            dx = tile_crpix[0] - 1024.5
            dy = tile_crpix[1] - 1024.5
            realwcs.set_crpix(x+dx, y+dy)
            print('CRPIX', x,y, 'shift by', dx,dy, 'to', realwcs.crpix)

        if modelsky_dir and band in [1, 2]:
            fn = os.path.join(modelsky_dir, '%s.%i.mod.fits' % (tile.coadd_id, band))
            if not os.path.exists(fn):
                raise RuntimeError('WARNING: does not exist:', fn)
            x0,x1,y0,y1 = tim.roi
            bg = fitsio.FITS(fn)[2][y0:y1, x0:x1]
            print('Read background map:', bg.shape, bg.dtype, 'vs image', tim.shape)

            if plots:
                plt.clf()
                plt.subplot(1,2,1)
                plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
                           cmap='gray', vmin=-3 * sig1, vmax=5 * sig1)
                plt.subplot(1,2,2)
                plt.imshow(bg, interpolation='nearest', origin='lower',
                           cmap='gray', vmin=-3 * sig1, vmax=5 * sig1)
                tag = '%s W%i' % (tile.coadd_id, band)
                plt.suptitle(tag)
                ps.savefig()

                plt.clf()
                ha = dict(range=(-5,10), bins=100, histtype='step')
                plt.hist((tim.getImage() * tim.inverr)[tim.inverr > 0].ravel(),
                         color='b', label='Original', **ha)
                plt.hist(((tim.getImage()-bg) * tim.inverr)[tim.inverr > 0].ravel(),
                         color='g', label='Minus Background', **ha)
                plt.axvline(0, color='k', alpha=0.5)
                plt.xlabel('Per-pixel intensity (Sigma)')
                plt.legend()
                plt.title(tag + ': background')
                ps.savefig()

            # Actually subtract the background!
            tim.data -= bg

        # Floor the per-pixel variances
        if band in [1,2]:
            # in Vega nanomaggies per pixel
            floor_sigma = {1: 0.5, 2: 2.0}
            with np.errstate(divide='ignore'):
                new_ie = 1. / np.hypot(1./tim.inverr, floor_sigma[band])
            new_ie[tim.inverr == 0] = 0.

            if plots:
                plt.clf()
                plt.plot((1. / tim.inverr[tim.inverr>0]).ravel(), (1./new_ie[tim.inverr>0]).ravel(), 'b.')
                plt.title('unWISE per-pixel error: %s band %i' % (tile.coadd_id, band))
                plt.xlabel('original')
                plt.ylabel('floored')
                ps.savefig()

            tim.inverr = new_ie

        # Read mask file?
        if get_masks:
            from astrometry.util.resample import resample_with_wcs, OverlapError
            # unwise_dir can be a colon-separated list of paths
            tilemask = None
            for d in tile.unwise_dir.split(':'):
                fn = os.path.join(d, tile.coadd_id[:3], tile.coadd_id,
                                  'unwise-%s-msk.fits.gz' % tile.coadd_id)
                print('Looking for unWISE mask file', fn)
                if os.path.exists(fn):
                    x0,x1,y0,y1 = tim.roi
                    tilemask = fitsio.FITS(fn)[0][y0:y1,x0:x1]
                    break
            if tilemask is None:
                print('unWISE mask file for tile', tile.coadd_id, 'does not exist')
            else:
                try:
                    tanwcs = tim.wcs.wcs
                    assert(tanwcs.shape == tilemask.shape)
                    Yo,Xo,Yi,Xi,_ = resample_with_wcs(get_masks, tanwcs)
                    # Only deal with mask pixels that are set.
                    I, = np.nonzero(tilemask[Yi,Xi] > 0)
                    # Trim to unique area for this tile
                    rr,dd = get_masks.pixelxy2radec(Yo[I]+1, Xo[I]+1)
                    good = radec_in_unique_area(rr, dd, tile.ra1, tile.ra2, tile.dec1, tile.dec2)
                    I = I[good]
                    maskmap[Yo[I],Xo[I]] = tilemask[Yi[I], Xi[I]]
                except OverlapError:
                    # Shouldn't happen by this point
                    print('No overlap between WISE tile', tile.coadd_id, 'and brick')

        # The tiles have some overlap, so zero out pixels outside the
        # tile's unique area.
        th,tw = tim.shape
        xx,yy = np.meshgrid(np.arange(tw), np.arange(th))
        rr,dd = tim.wcs.wcs.pixelxy2radec(xx+1, yy+1)
        unique = radec_in_unique_area(rr, dd, tile.ra1, tile.ra2, tile.dec1, tile.dec2)
        print(np.sum(unique), 'of', (th*tw), 'pixels in this tile are unique')
        tim.inverr[unique == False] = 0.

        if plots:
            sig1 = tim.sig1
            plt.clf()
            plt.imshow(tim.getImage() * (tim.inverr > 0),
                       interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3 * sig1, vmax=10 * sig1)
            plt.colorbar()
            tag = '%s W%i' % (tile.coadd_id, band)
            plt.title('%s: tim data (unique)' % tag)
            ps.savefig()

        del xx,yy,rr,dd,unique

        wcs = tim.wcs.wcs
        ok,x,y = wcs.radec2pixelxy(ra, dec)
        x = np.round(x - 1.).astype(int)
        y = np.round(y - 1.).astype(int)
        good = (x >= 0) * (x < tw) * (y >= 0) * (y < th)
        # Which sources are in this brick's unique area?
        usrc = radec_in_unique_area(ra, dec, tile.ra1, tile.ra2, tile.dec1, tile.dec2)
        I, = np.nonzero(good * usrc)

        nexp[I] = tim.nuims[y[I], x[I]]
        if hasattr(tim, 'mjdmin') and hasattr(tim, 'mjdmax'):
            mjd[I] = (tim.mjdmin + tim.mjdmax) / 2.
        phot.wise_coadd_id[I] = tile.coadd_id
        central_flux[I] = tim.getImage()[y[I], x[I]]
        del x,y,good,usrc

        if pixelized_psf:
            import unwise_psf
            if (band == 1) or (band == 2):
                # we only have updated PSFs for W1 and W2
                psfimg = unwise_psf.get_unwise_psf(band, tile.coadd_id, 
                                                   modelname='neo4_unwisecat')
            else:
                psfimg = unwise_psf.get_unwise_psf(band, tile.coadd_id)

            if band == 4:
                # oversample (the unwise_psf models are at native W4 5.5"/pix,
                # while the unWISE coadds are made at 2.75"/pix.
                ph,pw = psfimg.shape
                subpsf = np.zeros((ph*2-1, pw*2-1), np.float32)
                from astrometry.util.util import lanczos3_interpolate
                xx,yy = np.meshgrid(np.arange(0., pw-0.51, 0.5, dtype=np.float32),
                                    np.arange(0., ph-0.51, 0.5, dtype=np.float32))
                xx = xx.ravel()
                yy = yy.ravel()
                ix = xx.astype(np.int32)
                iy = yy.astype(np.int32)
                dx = (xx - ix).astype(np.float32)
                dy = (yy - iy).astype(np.float32)
                psfimg = psfimg.astype(np.float32)
                rtn = lanczos3_interpolate(ix, iy, dx, dy, [subpsf.flat], [psfimg])
                print('Lanczo3_interpolate result:', rtn)

                if plots:
                    plt.clf()
                    plt.imshow(psfimg, interpolation='nearest', origin='lower')
                    plt.title('Original PSF model')
                    ps.savefig()
                    plt.clf()
                    plt.imshow(subpsf, interpolation='nearest', origin='lower')
                    plt.title('Subsampled PSF model')
                    ps.savefig()

                psfimg = subpsf
                del xx, yy, ix, iy, dx, dy

            from tractor.psf import PixelizedPSF
            psfimg /= psfimg.sum()
            fluxrescales = {1: 1.04, 2: 1.005, 3: 1.0, 4: 1.0}
            psfimg *= fluxrescales[band]
            tim.psf = PixelizedPSF(psfimg)

        if psf_broadening is not None and not pixelized_psf:
            # psf_broadening is a factor by which the PSF FWHMs
            # should be scaled; the PSF is a little wider
            # post-reactivation.
            psf = tim.getPsf()
            from tractor import GaussianMixturePSF
            if isinstance(psf, GaussianMixturePSF):
                #
                print('Broadening PSF: from', psf)
                p0 = psf.getParams()
                pnames = psf.getParamNames()
                p1 = [p * psf_broadening**2 if 'var' in name else p
                      for (p, name) in zip(p0, pnames)]
                psf.setParams(p1)
                print('Broadened PSF:', psf)
            else:
                print('WARNING: cannot apply psf_broadening to WISE PSF of type', type(psf))

        # PSF norm for depth
        psf = tim.getPsf()
        h,w = tim.shape
        patch = psf.getPointSourcePatch(h//2, w//2).patch
        psfnorm = np.sqrt(np.sum(patch**2))

        print('unWISE tile', tile.coadd_id, ': read image with shape', tim.shape)
        tim.tile = tile
        tims.append(tim)

    if plots:
        plt.clf()
        mn,mx = 0.1, 20000
        plt.hist(np.log10(np.clip(central_flux, mn, mx)), bins=100,
                 range=(np.log10(mn), np.log10(mx)))
        logt = np.arange(0, 5)
        plt.xticks(logt, ['%i' % i for i in 10.**logt])
        plt.title('Central fluxes (W%i)' % band)
        plt.axvline(np.log10(20000), color='k')
        plt.axvline(np.log10(1000), color='k')
        ps.savefig()

    # Eddie's non-secret recipe:
    #- central pixel <= 1000: 19x19 pix box size
    #- central pixel in 1000 - 20000: 59x59 box size
    #- central pixel > 20000 or saturated: 149x149 box size
    #- object near "bright star": 299x299 box size
    nbig = nmedium = nsmall = 0
    for src,cflux in zip(cat, central_flux):
        if cflux > 20000:
            R = 100
            nbig += 1
        elif cflux > 1000:
            R = 30
            nmedium += 1
        else:
            R = 15
            nsmall += 1
        if isinstance(src, PointSource):
            src.fixedRadius = R
        else:
            ### FIXME -- sizes for galaxies..... can we set PSF size separately?
            galrad = 0
            # RexGalaxy is a subclass of ExpGalaxy
            if isinstance(src, (ExpGalaxy, DevGalaxy)):
                galrad = src.shape.re
            elif isinstance(src, FixedCompositeGalaxy):
                galrad = max(src.shapeExp.re, src.shapeDev.re)
            pixscale = 2.75
            src.halfsize = int(np.hypot(R, galrad * 5 / pixscale))

    print('Set source sizes:', nbig, 'big', nmedium, 'medium', nsmall, 'small')

    minsb = 0.
    fitsky = False

    tractor = Tractor(tims, cat)
    if use_ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        tractor.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
    tractor.freezeParamsRecursive('*')
    tractor.thawPathsTo(wanyband)

    kwa = dict(fitstat_extras=[('pronexp', [tim.nims for tim in tims])])
    t0 = Time()

    R = tractor.optimize_forced_photometry(
        minsb=minsb, mindlnp=1., sky=fitsky, fitstats=True,
        variance=True, shared_params=False,
        wantims=wantims, **kwa)
    print('unWISE forced photometry took', Time() - t0)

    if use_ceres:
        term = R.ceres_status['termination']
        print('Ceres termination status:', term)
        # Running out of memory can cause failure to converge
        # and term status = 2.
        # Fail completely in this case.
        if term != 0:
            raise RuntimeError(
                'Ceres terminated with status %i' % term)

    if wantims:
        ims1 = R.ims1
    flux_invvars = R.IV
    if R.fitstats is not None:
        for k in fskeys:
            x = getattr(R.fitstats, k)
            fitstats[k] = np.array(x).astype(np.float32)

    if save_fits:
        for i,tim in enumerate(tims):
            tile = tim.tile
            (dat, mod, ie, chi, roi) = ims1[i]
            wcshdr = fitsio.FITSHDR()
            tim.wcs.wcs.add_to_header(wcshdr)
            tag = 'fit-%s-w%i' % (tile.coadd_id, band)
            fitsio.write('%s-data.fits' %
                         tag, dat, clobber=True, header=wcshdr)
            fitsio.write('%s-mod.fits' % tag,  mod,
                         clobber=True, header=wcshdr)
            fitsio.write('%s-chi.fits' % tag,  chi,
                         clobber=True, header=wcshdr)

    if plots:
        # Create models for just the brightest sources
        bright_cat = [src for src in cat
                      if src.getBrightness().getBand(wanyband) > 1000]
        print('Bright soures:', len(bright_cat))
        btr = Tractor(tims, bright_cat)
        for tim in tims:
            mod = btr.getModelImage(tim)
            tile = tim.tile
            tag = '%s W%i' % (tile.coadd_id, band)
            sig1 = tim.sig1
            plt.clf()
            plt.imshow(mod, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3 * sig1, vmax=25 * sig1)
            plt.colorbar()
            plt.title('%s: bright-star models' % tag)
            ps.savefig()

    if get_models:
        for i,tim in enumerate(tims):
            tile = tim.tile
            (dat, mod, ie, chi, roi) = ims1[i]
            print('unWISE get_models: ims1 roi:', roi, 'tim.roi:', tim.roi)
            models[(tile.coadd_id, band)] = (mod, dat, ie, tim.roi, tim.wcs.wcs)

    if plots:
        for i,tim in enumerate(tims):
            tile = tim.tile
            tag = '%s W%i' % (tile.coadd_id, band)
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


    nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
    nm_ivar = flux_invvars
    # Sources out of bounds, eg, never change from their default
    # (1-sigma or whatever) initial fluxes.  Zero them out instead.
    nm[nm_ivar == 0] = 0.

    phot.set(wband + '_nanomaggies', nm.astype(np.float32))
    phot.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
    dnm = np.zeros(len(nm_ivar), np.float32)
    okiv = (nm_ivar > 0)
    dnm[okiv] = (1. / np.sqrt(nm_ivar[okiv])).astype(np.float32)
    okflux = (nm > 0)
    mag = np.zeros(len(nm), np.float32)
    mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])
                   ).astype(np.float32)
    dmag = np.zeros(len(nm), np.float32)
    ok = (okiv * okflux)
    dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])
                ).astype(np.float32)
    mag[np.logical_not(okflux)] = np.nan
    dmag[np.logical_not(ok)] = np.nan

    phot.set(wband + '_mag', mag)
    phot.set(wband + '_mag_err', dmag)
    psfdepth = np.empty(len(phot), np.float32)
    psfdepth[:] = -2.5 * (np.log10(5. * tim.sig1 / psfnorm) - 9.)
    phot.set(wband + '_psfdepth', psfdepth)

    for k in fskeys:
        phot.set(wband + '_' + k, fitstats[k])
    phot.set(wband + '_nexp', nexp)
    if not np.all(mjd == 0):
        phot.set(wband + '_mjd', mjd)

    rtn = wphotduck()
    rtn.phot = phot
    rtn.models = None
    rtn.maskmap = None
    if get_models:
        rtn.models = models
    if get_masks:
        rtn.maskmap = maskmap
    return rtn

class wphotduck(object):
    pass

def radec_in_unique_area(rr, dd, ra1, ra2, dec1, dec2):
    ''' Returns a boolean array. '''
    unique = (dd >= dec1) * (dd < dec2)
    if ra1 < ra2:
        # normal RA
        unique *= (rr >= ra1) * (rr < ra2)
    else:
        # RA wrap-around
        unique[rr > 180] *= (rr[rr > 180] >= ra1)
        unique[rr < 180] *= (rr[rr < 180] <  ra2)
    return unique

def unwise_phot(X):
    '''
    This is the entry-point from runbrick.py, called via mp.map()
    '''
    (wcat, tiles, band, roiradec, wise_ceres, pixelized_psf, get_mods, get_masks, ps,
     move_crpix, modelsky_dir) = X
    kwargs = dict(roiradecbox=roiradec, band=band, pixelized_psf=pixelized_psf,
                  get_masks=get_masks, ps=ps, move_crpix=move_crpix,
                  modelsky_dir=modelsky_dir)
    if get_mods:
        kwargs.update(get_models=get_mods)

    # DEBUG
    #kwargs.update(save_fits=True)
    W = None
    try:
        W = unwise_forcedphot(wcat, tiles, use_ceres=wise_ceres, **kwargs)
    except:
        import traceback
        print('unwise_forcedphot failed:')
        traceback.print_exc()
        if wise_ceres:
            print('Trying without Ceres...')
            try:
                W = unwise_forcedphot(wcat, tiles, use_ceres=False, **kwargs)
            except:
                print('unwise_forcedphot failed (2):')
                traceback.print_exc()
    return W

def collapse_unwise_bitmask(bitmask, band):
    '''
    Converts WISE mask bits (in the unWISE data products) into the
    more compact codes reported in the tractor files as
    WISEMASK_W[12], and the "maskbits" WISE extensions.

    output bits :
    # 2^0 = bright star core and wings
    # 2^1 = PSF-based diffraction spike
    # 2^2 = optical ghost
    # 2^3 = first latent
    # 2^4 = second latent
    # 2^5 = AllWISE-like circular halo
    # 2^6 = bright star saturation
    # 2^7 = geometric diffraction spike
    '''
    assert((band == 1) or (band == 2))
    from collections import OrderedDict

    bits_w1 = OrderedDict([('core_wings', 2**0 + 2**1),
                           ('psf_spike', 2**27),
                           ('ghost', 2**25 + 2**26),
                           ('first_latent', 2**13 + 2**14),
                           ('second_latent', 2**17 + 2**18),
                           ('circular_halo', 2**23),
                           ('saturation', 2**4),
                           ('geom_spike', 2**29)])

    bits_w2 = OrderedDict([('core_wings', 2**2 + 2**3),
                           ('psf_spike', 2**28),
                           ('ghost', 2**11 + 2**12),
                           ('first_latent', 2**15 + 2**16),
                           ('second_latent', 2**19 + 2**20),
                           ('circular_halo', 2**24),
                           ('saturation', 2**5),
                           ('geom_spike', 2**30)])

    bits = (bits_w1 if (band == 1) else bits_w2)

    # hack to handle both scalar and array inputs
    result = 0*bitmask
    for i, feat in enumerate(bits.keys()):
        result += ((2**i)*(np.bitwise_and(bitmask, bits[feat]) != 0)).astype(np.uint8)
    return result.astype('uint8')

###
# This is taken directly from tractor/wise.py, replacing only the filename.
###
def unwise_tiles_touching_wcs(wcs, polygons=True):
    '''
    Returns a FITS table (with RA,Dec,coadd_id) of unWISE tiles
    '''
    from astrometry.util.miscutils import polygons_intersect
    from astrometry.util.starutil_numpy import degrees_between

    from pkg_resources import resource_filename
    atlasfn = resource_filename('legacypipe', 'data/wise-tiles.fits')

    T = fits_table(atlasfn)
    trad = wcs.radius()
    wrad = np.sqrt(2.) / 2. * 2048 * 2.75 / 3600.
    rad = trad + wrad
    r, d = wcs.radec_center()
    I, = np.nonzero(np.abs(T.dec - d) < rad)
    I = I[degrees_between(T.ra[I], T.dec[I], r, d) < rad]

    if not polygons:
        return T[I]
    # now check actual polygon intersection
    tw, th = wcs.imagew, wcs.imageh
    targetpoly = [(0.5, 0.5), (tw + 0.5, 0.5),
                  (tw + 0.5, th + 0.5), (0.5, th + 0.5)]
    cd = wcs.get_cd()
    tdet = cd[0] * cd[3] - cd[1] * cd[2]
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)
    keep = []
    for i in I:
        wwcs = unwise_tile_wcs(T.ra[i], T.dec[i])
        cd = wwcs.get_cd()
        wdet = cd[0] * cd[3] - cd[1] * cd[2]
        H, W = wwcs.shape
        poly = []
        for x, y in [(0.5, 0.5), (W + 0.5, 0.5), (W + 0.5, H + 0.5), (0.5, H + 0.5)]:
            rr, dd = wwcs.pixelxy2radec(x, y)
            ok, xx, yy = wcs.radec2pixelxy(rr, dd)
            poly.append((xx, yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    return T[I]

### Also direct from tractor/wise.py
def unwise_tile_wcs(ra, dec, W=2048, H=2048, pixscale=2.75):
    from astrometry.util.util import Tan
    '''
    Returns a Tan WCS object at the given RA,Dec center, axis aligned, with the
    given pixel W,H and pixel scale in arcsec/pixel.
    '''
    cowcs = Tan(ra, dec, (W + 1) / 2., (H + 1) / 2.,
                -pixscale / 3600., 0., 0., pixscale / 3600., W, H)
    return cowcs
