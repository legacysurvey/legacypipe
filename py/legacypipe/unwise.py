import numpy as np

from tractor import NanoMaggies, PointSource, Tractor
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
                      pixelized_psf=False):
    '''
    Given a list of tractor sources *cat*
    and a list of unWISE tiles *tiles* (a fits_table with RA,Dec,coadd_id)
    runs forced photometry, returning a FITS table the same length as *cat*.
    '''

    # PSF broadening in post-reactivation data, by band.
    # Newer version from Aaron's email to decam-chatter, 2018-06-14.
    broadening = { 1: 1.0405, 2: 1.0346, 3: None, 4: None }
    
    from astrometry.util.plotutils import PlotSequence
    ps = PlotSequence('wise-forced-w%i' % band)
    plots = (ps is not None)
    if plots:
        import pylab as plt
    
    from collections import Counter
    count_types = Counter([type(src) for src in cat])
    print('Source types:', count_types)
        
    wantims = ((ps is not None) or save_fits or get_models)
    wanyband = 'w'
    if get_models:
        models = {}

    fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix',
              'pronexp']

    Nsrcs = len(cat)
    phot = fits_table()
    phot.tile = np.array(['        '] * Nsrcs)

    ra = np.array([src.getPosition().ra for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])

    print('Photometering WISE band', band)
    wband = 'w%i' % band

    nexp = np.zeros(Nsrcs, np.int16)
    mjd = np.zeros(Nsrcs, np.float64)

    fitstats = {}
    tims = []

    central_flux = np.zeros(Nsrcs, np.float32)
    
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
            
        # The tiles have some overlap, so zero out pixels outside the
        # tile's unique area.
        th,tw = tim.shape
        xx,yy = np.meshgrid(np.arange(tw), np.arange(th))
        rr,dd = tim.wcs.wcs.pixelxy2radec(xx+1, yy+1)
        unique = (dd >= tile.dec1) * (dd < tile.dec2)
        if tile.ra1 < tile.ra2:
            # normal RA
            unique *= (rr >= tile.ra1) * (rr < tile.ra2)
        else:
            # RA wrap-around
            unique[rr > 180] *= (rr[rr > 180] >= tile.ra1)
            unique[rr < 180] *= (rr[rr < 180] <  tile.ra2)

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
        
        del xx,yy,rr,dd
        wcs = tim.wcs.wcs
        ok,x,y = wcs.radec2pixelxy(ra, dec)
        x = np.round(x - 1.).astype(int)
        y = np.round(y - 1.).astype(int)
        good = (x >= 0) * (x < tw) * (y >= 0) * (y < th)
        good[good] *= unique[y[good],x[good]]

        nexp[good] = tim.nuims[y[good], x[good]]
        if hasattr(tim, 'mjdmin') and hasattr(tim, 'mjdmax'):
            mjd[good] = (tim.mjdmin + tim.mjdmax) / 2.
        central_flux[good] = tim.getImage()[y[good],x[good]]
        del x,y,good,unique
        
        if pixelized_psf:
            import unwise_psf
            psfimg = unwise_psf.get_unwise_psf(band, tile.coadd_id)
            print('PSF postage stamp', psfimg.shape, 'sum', psfimg.sum())
            from tractor.psf import PixelizedPSF
            psfimg /= psfimg.sum()
            tim.psf = PixelizedPSF(psfimg)
            print('### HACK ### normalized PSF to 1.0')
            print('Set PSF to', tim.psf)

            if False:
                ph,pw = psfimg.shape
                px,py = np.meshgrid(np.arange(ph), np.arange(pw))
                cx = np.sum(psfimg * px)
                cy = np.sum(psfimg * py)
                print('PSF center of mass: %.2f, %.2f' % (cx, cy))
                for sz in range(1, 11):
                    middle = pw//2
                    sub = (slice(middle-sz, middle+sz+1),
                           slice(middle-sz, middle+sz+1))
                    cx = np.sum((psfimg * px)[sub]) / np.sum(psfimg[sub])
                    cy = np.sum((psfimg * py)[sub]) / np.sum(psfimg[sub])
                    print('Size', sz, ': PSF center of mass: %.2f, %.2f' % (cx, cy))
                import fitsio
                fitsio.write('psfimg-%s-w%i.fits' % (tile.coadd_id, band), psfimg,
                         clobber=True)
        
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
                #print('Params:', p0)
                pnames = psf.getParamNames()
                #print('Param names:', pnames)
                p1 = [p * psf_broadening**2 if 'var' in name else p
                      for (p, name) in zip(p0, pnames)]
                #print('Broadened:', p1)
                psf.setParams(p1)
                print('Broadened PSF:', psf)
            else:
                print(
                    'WARNING: cannot apply psf_broadening to WISE PSF of type', type(psf))

        print('unWISE tile', tile.coadd_id, ': read image with shape', tim.shape)

        print('tim ROI:', tim.roi)

        ### FIXME -- read msk file here? (for saturation)
        
        tim.tile = tile
        tims.append(tim)

    print('Central flux: max', central_flux.max(), 'median',
          np.median(central_flux))

    if plots:
        plt.clf()
        mn,mx = 0.1, 20000
        plt.hist(np.log10(np.clip(central_flux, mn, mx)), bins=100, range=(np.log10(mn), np.log10(mx)))
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
        ### FIXME -- sizes for galaxies..... can we set PSF size separately?
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
            src.halfsize = R

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
        ims0 = R.ims0
        ims1 = R.ims1
    flux_invvars = R.IV
    if R.fitstats is not None:
        for k in fskeys:
            x = getattr(R.fitstats, k)
            fitstats[k] = np.array(x).astype(np.float32)

    if save_fits:
        import fitsio
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
            models[(tile.coadd_id, band)] = (mod, tim.roi)

    if ps:
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


    # #### FIXME --
    # #if hasattr(tim, 'mjdmin') and hasattr(tim, 'mjdmax'):
    # #    mjd[srci] = (tim.mjdmin + tim.mjdmax) / 2.
    # if fs is None:
    #     continue
    # for k in fskeys:
    #     x = getattr(fs, k)
    #     # fitstats are returned only for un-frozen sources
    #     fitstats[k] = np.array(x).astype(np.float32)

    #     phot.tile[srci] = tile.coadd_id
    #     
    #     nexp[srci] = tim.nuims[np.clip(np.round(y[srci]).astype(int), 0, H - 1),
    #                            np.clip(np.round(x[srci]).astype(int), 0, W - 1)]
    # Note, this is *outside* the loop over tiles.
    # The fluxes are saved in the source objects, and will be set based on
    # the 'tiledists' logic above.
    nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
    nm_ivar = flux_invvars

    # Sources out of bounds, eg, never change from their default
    # (1-sigma or whatever) initial fluxes.  Zero them out instead.
    nm[nm_ivar == 0] = 0.

    phot.set(wband + '_nanomaggies', nm.astype(np.float32))
    phot.set(wband + '_nanomaggies_ivar', nm_ivar)
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

    #### FIXME
    for k in fskeys:
        phot.set(wband + '_' + k, fitstats[k])
    phot.set(wband + '_nexp', nexp)
    if not np.all(mjd == 0):
        phot.set(wband + '_mjd', mjd)

    if get_models:
        return phot,models
    return phot


def unwise_phot(X):
    '''
    This is the entry-point from runbrick.py, called via mp.map()
    '''
    (wcat, tiles, band, roiradec, wise_ceres, pixelized_psf, get_mods) = X
    kwargs = dict(roiradecbox=roiradec, band=band, pixelized_psf=pixelized_psf)
    if get_mods:
        kwargs.update(get_models=get_mods)

    ### FIXME
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
    from astrometry.util.fits import fits_table
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
