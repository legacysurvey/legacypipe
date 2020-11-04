import os
import numpy as np
import fitsio
from astrometry.util.fits import fits_table, merge_tables

import logging
logger = logging.getLogger('legacypipe.reference')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def get_reference_sources(survey, targetwcs, pixscale, bands,
                          tycho_stars=True,
                          gaia_stars=True,
                          large_galaxies=True,
                          star_clusters=True,
                          clean_columns=True,
                          plots=False, ps=None,
                          gaia_margin=None,
                          galaxy_margin=None):
    # If bands = None, does not create sources.

    H,W = targetwcs.shape
    H,W = int(H),int(W)

    # How big of a margin to search for bright stars and star clusters --
    # this should be based on the maximum radius they are considered to
    # affect.  In degrees.
    if gaia_margin is not None:
        ref_margin = gaia_margin
    else:
        ref_margin = mask_radius_for_mag(0.)
    mpix = int(np.ceil(ref_margin * 3600. / pixscale))
    marginwcs = targetwcs.get_subimage(-mpix, -mpix, W+2*mpix, H+2*mpix)

    # Table of reference-source properties, including a field 'sources',
    # with tractor source objects.
    refs = []

    # Tycho-2 stars
    tycho = []
    if tycho_stars:
        tycho = read_tycho2(survey, marginwcs, bands)
        if tycho and len(tycho):
            refs.append(tycho)

    # Add Gaia stars
    gaia = None
    if gaia_stars:
        from astrometry.libkd.spherematch import match_radec
        gaia = read_gaia(marginwcs, bands)
    if gaia is not None:
        # Handle sources that appear in both Gaia and Tycho-2 by
        # dropping the entry from Tycho-2.
        if len(gaia) and len(tycho):
            # Before matching, apply proper motions to bring them to
            # the same epoch.  We want to use the more-accurate Gaia
            # proper motions, so rewind Gaia positions to the Tycho
            # epoch.  (Note that in read_tycho2, we massaged the
            # epochs)
            cosdec = np.cos(np.deg2rad(gaia.dec))
            # First do a coarse matching with the approximate epoch:
            dt = 1991.5 - gaia.ref_epoch
            gra  = gaia.ra  + dt * gaia.pmra  / (3600.*1000.) / cosdec
            gdec = gaia.dec + dt * gaia.pmdec / (3600.*1000.)
            # Max Tycho-2 PM is 10"/yr, max |epoch_ra,epoch_dec - mean| = 0.5
            I,J,_ = match_radec(tycho.ra, tycho.dec, gra, gdec, 10./3600.,
                                nearest=True)
            debug('Initially matched', len(I), 'Tycho-2 stars to Gaia stars (10").')

            if plots:
                import pylab as plt
                plt.clf()
                plt.plot(gra, gdec, 'bo', label='Gaia (1991.5)')
                plt.plot(gaia.ra, gaia.dec, 'gx', label='Gaia (2015.5)')
                plt.plot([gaia.ra, gra], [gaia.dec, gdec], 'k-')
                plt.plot([tycho.ra[I], gra[J]], [tycho.dec[I], gdec[J]], 'r-')
                plt.plot(tycho.ra, tycho.dec, 'rx', label='Tycho-2')
                plt.plot(tycho.ra[I], tycho.dec[I], 'o', mec='r', ms=8, mfc='none',
                         label='Tycho-2 matched')
                plt.legend()
                r0,r1,d0,d1 = targetwcs.radec_bounds()
                plt.axis([r0,r1,d0,d1])
                plt.title('Initial (10") matching')
                ps.savefig()

            dt = tycho.ref_epoch[I] - gaia.ref_epoch[J]
            cosdec = np.cos(np.deg2rad(gaia.dec[J]))
            gra  = gaia.ra[J]  + dt * gaia.pmra[J]  / (3600.*1000.) / cosdec
            gdec = gaia.dec[J] + dt * gaia.pmdec[J] / (3600.*1000.)
            dists = np.hypot((gra - tycho.ra[I]) * cosdec, gdec - tycho.dec[I])
            K = np.flatnonzero(dists <= 1./3600.)
            if len(K)<len(I):
                debug('Unmatched Tycho-2 - Gaia stars: dists', dists[dists > 1./3600.]*3600.)
            I = I[K]
            J = J[K]
            debug('Matched', len(I), 'Tycho-2 stars to Gaia stars.')
            if len(I):
                keep = np.ones(len(tycho), bool)
                keep[I] = False
                tycho.cut(keep)
                gaia.isbright[J] = True
                gaia.istycho[J] = True
        if gaia is not None and len(gaia) > 0:
            refs.append(gaia)

    # Read the catalog of star (open and globular) clusters and add them to the
    # set of reference stars (with the isbright bit set).
    if star_clusters:
        clusters = read_star_clusters(marginwcs)
        if clusters is not None:
            debug('Found', len(clusters), 'star clusters nearby')
            refs.append(clusters)

    # Read large galaxies nearby.
    if large_galaxies:
        kw = {}
        if galaxy_margin is not None:
            kw.update(max_radius=galaxy_margin + np.hypot(H,W)/2.*pixscale/3600)
        galaxies = read_large_galaxies(survey, targetwcs, bands, clean_columns=clean_columns, **kw)
        if galaxies is not None:
            # Resolve possible Gaia-large-galaxy duplicates
            if gaia and len(gaia):
                I,J,_ = match_radec(galaxies.ra, galaxies.dec, gaia.ra, gaia.dec,
                                    2./3600., nearest=True)
                info('Matched', len(I), 'large galaxies to Gaia stars.')
                if len(I):
                    gaia.donotfit[J] = True
            # Resolve possible Tycho2-large-galaxy duplicates (with larger radius)
            if tycho and len(tycho):
                I,J,_ = match_radec(galaxies.ra, galaxies.dec, tycho.ra, tycho.dec,
                                    5./3600., nearest=True)
                info('Matched', len(I), 'large galaxies to Tycho-2 stars.')
                if len(I):
                    tycho.donotfit[J] = True
            refs.append(galaxies)

    if len(refs):
        refs = merge_tables([r for r in refs if r is not None],
                            columns='fillzero')
    if len(refs) == 0:
        return None,None

    # these x,y are in the margin-padded WCS; not useful.
    # See ibx,iby computed below instead.
    for c in ['x','y']:
        if c in refs.get_columns():
            refs.delete_column(c)

    # radius / radius_pix are used to set the MASKBITS shapes;
    # keep_radius determines which sources are kept (because we subtract
    # stellar halos out to N x their radii)
    refs.radius_pix = np.ceil(refs.radius * 3600. / pixscale).astype(int)

    debug('Increasing radius for', np.sum(refs.keep_radius > refs.radius),
          'ref sources based on keep_radius')
    keeprad = np.maximum(refs.keep_radius, refs.radius)
    # keeprad to pix
    keeprad = np.ceil(keeprad * 3600. / pixscale).astype(int)

    _,xx,yy = targetwcs.radec2pixelxy(refs.ra, refs.dec)
    # ibx = integer brick coords
    refs.ibx = np.round(xx-1.).astype(np.int32)
    refs.iby = np.round(yy-1.).astype(np.int32)

    # cut ones whose position + radius are outside the brick bounds.
    refs.cut((xx > -keeprad) * (xx < W+keeprad) *
             (yy > -keeprad) * (yy < H+keeprad))
    # mark ones that are actually inside the brick area.
    refs.in_bounds = ((refs.ibx >= 0) * (refs.ibx < W) *
                      (refs.iby >= 0) * (refs.iby < H))

    # ensure bool columns
    for col in ['isbright', 'ismedium', 'islargegalaxy', 'iscluster', 'isgaia',
                'istycho', 'donotfit', 'freezeparams']:
        if not col in refs.get_columns():
            refs.set(col, np.zeros(len(refs), bool))
    # Copy flags from the 'refs' table to the source objects themselves.
    sources = refs.sources
    refs.delete_column('sources')
    for i,(donotfit,freeze) in enumerate(zip(refs.donotfit, refs.freezeparams)):
        if donotfit:
            sources[i] = None
        if sources[i] is None:
            continue
        sources[i].is_reference_source = True
        if freeze:
            sources[i].freezeparams = True

    return refs,sources

def read_gaia(wcs, bands):
    '''
    *wcs* here should include margin
    '''
    from legacypipe.gaiacat import GaiaCatalog
    from legacypipe.survey import GaiaSource

    gaia = GaiaCatalog().get_catalog_in_wcs(wcs)
    debug('Got', len(gaia), 'Gaia stars nearby')

    gaia.G = gaia.phot_g_mean_mag
    # Sort by brightness (for reference-*.fits output table)
    gaia.cut(np.argsort(gaia.G))

    # Gaia to DECam color transformations for stars
    color = gaia.phot_bp_mean_mag - gaia.phot_rp_mean_mag
    # From Rongpu, 2020-04-12
    # no BP-RP color: use average color
    color[np.logical_not(np.isfinite(color))] = 1.4
    # clip to reasonable range for the polynomial fit
    color = np.clip(color, -0.6, 4.1)
    for b,coeffs in [
            ('g', [-0.1178631039, 0.3650113495, 0.5608615360, -0.2850687702,
                   -1.0243473939, 1.4378375491, 0.0679401731, -1.1713172509,
                   0.9107811975, -0.3374324004, 0.0683946390, -0.0073089582,
                   0.0003230170]),
            ('r', [0.1139078673, -0.2868955307, 0.0013196434, 0.1029151074,
                   0.1196710702, -0.3729031390, 0.1859874242, 0.1370162451,
                   -0.1808580848, 0.0803219195, -0.0180218196, 0.0020584707,
                   -0.0000953486]),
            ('z', [0.4811198057, -0.9990015041, 0.1403990019, 0.2150988888,
                   -0.2917655866, 0.1326831887, -0.0259205004, 0.0018548776])]:
        mag = gaia.G.copy()
        for order,c in enumerate(coeffs):
            mag += c * color**order
        gaia.set('decam_mag_%s' % b, mag)
    del color

    #  For possible future use:
    #  BASS/MzLS:
    #  coeffs = dict(
    #  g = [-0.1299895823, 0.3120393968, 0.5989482686, 0.3125882487,
    #      -1.9401592247, 1.1011670449, 2.0741304659, -3.3930306403,
    #      2.1857291197, -0.7674676232, 0.1542300648, -0.0167007725,
    #      0.0007573720],
    #  r = [0.0901464643, -0.2463711147, 0.0094963025, -0.1187138789,
    #      0.4131107392, -0.1832183301, -0.6015486252, 0.9802538471,
    #      -0.6613809948, 0.2426395251, -0.0505867727, 0.0056462458,
    #      -0.0002625789],
    #  z = [0.4862049092, -1.0278704657, 0.1220984456, 0.3000129189,
    #      -0.3770662617, 0.1696090596, -0.0331679127, 0.0023867628])

    # force this source to remain a point source?
    # Long history here, starting DJS, [decam-chatter 5486] Solved! GAIA separation
    #   of point sources from extended sources
    # Updated for Gaia DR2 by Eisenstein,
    # [decam-data 2770] Re: [desi-milkyway 639] GAIA in DECaLS DR7
    # And made far more restrictive following BGS feedback.
    gaia.pointsource = np.logical_or((gaia.G <= 18.) * (gaia.astrometric_excess_noise < 10.**0.5),
                                     (gaia.G <= 13.))

    # in our catalog files, this is in float32; in the Gaia data model it's
    # a byte, with only values 3 and 31 in DR2.
    gaia.astrometric_params_solved = gaia.astrometric_params_solved.astype(np.uint8)

    # Gaia version?
    gaiaver = int(os.getenv('GAIA_CAT_VER', '1'))
    gaia_release = 'G%i' % gaiaver
    gaia.ref_cat = np.array([gaia_release] * len(gaia))
    gaia.ref_id  = gaia.source_id
    gaia.pmra_ivar  = 1./gaia.pmra_error **2
    gaia.pmdec_ivar = 1./gaia.pmdec_error**2
    gaia.parallax_ivar = 1./gaia.parallax_error**2
    # mas -> deg
    gaia.ra_ivar  = 1./(gaia.ra_error  / 1000. / 3600.)**2
    gaia.dec_ivar = 1./(gaia.dec_error / 1000. / 3600.)**2

    for c in ['ra_error', 'dec_error', 'parallax_error',
              'pmra_error', 'pmdec_error']:
        gaia.delete_column(c)
    for c in ['pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar',
              'parallax_ivar']:
        X = gaia.get(c)
        X[np.logical_not(np.isfinite(X))] = 0.

    # uniform name w/ Tycho-2
    gaia.zguess = gaia.decam_mag_z
    gaia.mag = gaia.G
    # Take the brighter of G, z to expand masks around red stars.
    gaia.mask_mag = np.minimum(gaia.G, gaia.zguess + 1.)

    # radius to consider affected by this star, for MASKBITS
    gaia.radius = mask_radius_for_mag(gaia.mask_mag)
    # radius for keeping this source in the ref catalog
    # (eg, for halo subtraction)
    gaia.keep_radius = 4. * gaia.radius
    gaia.delete_column('G')
    gaia.isgaia = np.ones(len(gaia), bool)
    gaia.istycho = np.zeros(len(gaia), bool)
    gaia.isbright = (gaia.mask_mag < 13.)
    gaia.ismedium = (gaia.mask_mag < 16.)
    gaia.donotfit = np.zeros(len(gaia), bool)

    # NOTE, must initialize gaia.sources array this way, or else numpy
    # will try to be clever and create a 2-d array, because GaiaSource is
    # iterable.
    gaia.sources = np.empty(len(gaia), object)
    if bands is not None:
        for i,g in enumerate(gaia):
            gaia.sources[i] = GaiaSource.from_catalog(g, bands)
    return gaia

def mask_radius_for_mag(mag):
    # Returns a masking radius in degrees for a star of the given magnitude.
    # Used for Tycho-2 and Gaia stars.

    # This is in degrees, and is from Rongpu in the thread [decam-chatter 12099].
    return 1630./3600. * 1.396**(-mag)

def read_tycho2(survey, targetwcs, bands):
    from astrometry.libkd.spherematch import tree_open, tree_search_radec
    from legacypipe.survey import GaiaSource
    tycho2fn = survey.find_file('tycho2')
    radius = 1.
    ra,dec = targetwcs.radec_center()
    # John added the "isgalaxy" flag 2018-05-10, from the Metz &
    # Geffert (04) catalog.

    # Eddie added the "zguess" column 2019-03-06, by matching with
    # 2MASS and estimating z based on APASS.

    # The "tycho2.kd.fits" file read here was produced by:
    #
    # fitscopy ~schlafly/legacysurvey/tycho-isgalaxyflag-2mass.fits"[col \
    #   tyc1;tyc2;tyc3;ra;dec;sigma_ra;sigma_dec;mean_ra;mean_dec;pm_ra;pm_dec; \
    #   sigma_pm_ra;sigma_pm_dec;epoch_ra;epoch_dec;mag_bt;mag_vt;mag_hp; \
    #   isgalaxy;Jmag;Hmag;Kmag,zguess]" /tmp/tycho2-astrom.fits
    # startree -P -k -n stars -T -i /tmp/tycho2-astrom.fits \
    #  -o /global/project/projectdirs/cosmo/staging/tycho2/tycho2.kd.fits

    kd = tree_open(tycho2fn, 'stars')
    I = tree_search_radec(kd, ra, dec, radius)
    debug(len(I), 'Tycho-2 stars within', radius, 'deg of RA,Dec (%.3f, %.3f)' % (ra,dec))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    tycho = fits_table(tycho2fn, rows=I)
    del kd
    if 'isgalaxy' in tycho.get_columns():
        tycho.cut(tycho.isgalaxy == 0)
        debug('Cut to', len(tycho), 'Tycho-2 stars on isgalaxy==0')
    else:
        print('Warning: no "isgalaxy" column in Tycho-2 catalog')

    tycho.ref_cat = np.array(['T2'] * len(tycho))
    # tyc1: [1,9537], tyc2: [1,12121], tyc3: [1,3]
    tycho.ref_id = (tycho.tyc1.astype(np.int64)*1000000 +
                    tycho.tyc2.astype(np.int64)*10 +
                    tycho.tyc3.astype(np.int64))
    with np.errstate(divide='ignore'):
        # In our Tycho-2 catalog, sigma_pm_* are in *arcsec/yr*, Gaia is in mas/yr.
        tycho.pmra_ivar  = 1./(tycho.sigma_pm_ra  * 1000.)**2
        tycho.pmdec_ivar = 1./(tycho.sigma_pm_dec * 1000.)**2
        tycho.ra_ivar  = 1./tycho.sigma_ra **2
        tycho.dec_ivar = 1./tycho.sigma_dec**2
    tycho.rename('pm_ra', 'pmra')
    tycho.rename('pm_dec', 'pmdec')
    for c in ['pmra', 'pmdec', 'pmra_ivar', 'pmdec_ivar']:
        X = tycho.get(c)
        X[np.logical_not(np.isfinite(X))] = 0.
    tycho.mag = tycho.mag_vt
    # Patch missing mag values...
    tycho.mag[tycho.mag == 0] = tycho.mag_hp[tycho.mag == 0]
    tycho.mag[tycho.mag == 0] = tycho.mag_bt[tycho.mag == 0]

    # Use zguess
    tycho.mask_mag = tycho.mag
    with np.errstate(invalid='ignore'):
        I = np.flatnonzero(np.isfinite(tycho.zguess) *
                           (tycho.zguess + 1. < tycho.mag))
    tycho.mask_mag[I] = tycho.zguess[I]
    # Per discussion in issue #306 -- cut on mag < 13.
    # This drops only 13k/2.5M stars.
    tycho.cut(tycho.mask_mag < 13.)

    tycho.radius = mask_radius_for_mag(tycho.mask_mag)
    tycho.keep_radius = 2. * tycho.radius

    for c in ['tyc1', 'tyc2', 'tyc3', 'mag_bt', 'mag_vt', 'mag_hp',
              'mean_ra', 'mean_dec',
              'sigma_pm_ra', 'sigma_pm_dec', 'sigma_ra', 'sigma_dec']:
        tycho.delete_column(c)
    # add Gaia-style columns
    # No parallaxes in Tycho-2
    tycho.parallax = np.zeros(len(tycho), np.float32)
    # Tycho-2 "supplement" stars, from Hipparcos and Tycho-1 catalogs, have
    # ref_epoch = 0.  Fill in with the 1991.25 epoch of those catalogs.
    tycho.epoch_ra [tycho.epoch_ra  == 0] = 1991.25
    tycho.epoch_dec[tycho.epoch_dec == 0] = 1991.25
    # Tycho-2 has separate epoch_ra and epoch_dec.
    # Move source to the mean epoch.
    tycho.ref_epoch = (tycho.epoch_ra + tycho.epoch_dec) / 2.
    cosdec = np.cos(np.deg2rad(tycho.dec))
    tycho.ra  += (tycho.ref_epoch - tycho.epoch_ra ) * tycho.pmra  / 3600. / cosdec
    tycho.dec += (tycho.ref_epoch - tycho.epoch_dec) * tycho.pmdec / 3600.
    # Tycho-2 proper motions are in arcsec/yr; Gaia are mas/yr.
    tycho.pmra  *= 1000.
    tycho.pmdec *= 1000.
    # We already cut on John's "isgalaxy" flag
    tycho.pointsource = np.ones(len(tycho), bool)
    # phot_g_mean_mag -- for initial brightness of source
    tycho.phot_g_mean_mag = tycho.mag
    tycho.delete_column('epoch_ra')
    tycho.delete_column('epoch_dec')
    tycho.istycho = np.ones(len(tycho), bool)
    tycho.isbright = np.ones(len(tycho), bool)
    tycho.ismedium = np.ones(len(tycho), bool)
    tycho.donotfit = np.zeros(len(tycho), bool)
    tycho.sources = np.empty(len(tycho), object)
    if bands is not None:
        for i,t in enumerate(tycho):
            tycho.sources[i] = GaiaSource.from_catalog(t, bands)
    return tycho

def get_large_galaxy_version(fn):
    preburn = False
    hdr = fitsio.read_header(fn)
    try:
        v = hdr.get('SGAVER')
        if v is None: # old version
            v = hdr.get('LSLGAVER')
        if v is not None:
            v = v.strip()
            if 'ellipse' in v.lower():
                preburn = True
                v, _ = v.split('-')
            assert(len(v) == 2)
            return v, preburn
    except KeyError:
        pass
    for k in ['3.0', '2.0']:
        if k in fn:
            return 'L'+k[0], preburn
    return 'LG', preburn

def read_large_galaxies(survey, targetwcs, bands, clean_columns=True,
                        max_radius=2.):
    # Note, max_radius must include the brick radius!
    from astrometry.libkd.spherematch import tree_open, tree_search_radec
    galfn = survey.find_file('large-galaxies')
    if galfn is None:
        debug('No large-galaxies catalog file')
        return None
    radius = max_radius
    rc,dc = targetwcs.radec_center()

    debug('Reading', galfn)
    try:
        kd = tree_open(galfn, 'stars')
    except:
        kd = tree_open(galfn, 'largegals')
    I = tree_search_radec(kd, rc, dc, radius)
    debug('%i large galaxies within %.3g deg of RA,Dec (%.3f, %.3f)' %
          (len(I), radius, rc,dc))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    galaxies = fits_table(galfn, rows=I)
    del kd

    refcat, preburn = get_large_galaxy_version(galfn)
    debug('Large galaxies version: "%s", preburned?' % refcat, preburn)

    if preburn:
        # SGA ellipse catalog
        # NOTE: fields such as ref_cat, preburned, etc, already exist in the
        # "galaxies" catalog read from disk.
        # The galaxies we want to appear in MASKBITS get
        # 'islargegalaxy' set.  This includes both pre-burned
        # galaxies, and ones where the preburning failed and we want
        # to fall back to the SGA-parent ellipse for masking.
        galaxies.islargegalaxy = ((galaxies.ref_cat == refcat) *
                                  (galaxies.sga_id > -1))
        # The pre-fit galaxies whose parameters will stay fixed
        galaxies.freezeparams = (galaxies.preburned * galaxies.freeze)
    else:
        # SGA parent catalog
        galaxies.ref_cat = np.array([refcat] * len(galaxies))
        galaxies.islargegalaxy = np.ones(len(galaxies), bool)
        galaxies.freezeparams = np.zeros(len(galaxies), bool)
        galaxies.preburned = np.zeros(len(galaxies), bool)
        galaxies.rename('sga_id', 'ref_id')

    galaxies.rename('mag_leda', 'mag')
    # Pre-burned, frozen but non-SGA sources have diam=-1.
    galaxies.radius = np.maximum(0., galaxies.diam / 2. / 60.) # [degree]
    galaxies.keep_radius = 2. * galaxies.radius
    galaxies.sources = np.empty(len(galaxies), object)
    galaxies.sources[:] = None

    if bands is not None:
        galaxies.sources[:] = get_galaxy_sources(galaxies, bands)

    if clean_columns:
        keep_columns = ['ra', 'dec', 'radius', 'mag', 'ref_cat', 'ref_id', 'ba', 'pa',
                        'sources', 'islargegalaxy', 'freezeparams', 'keep_radius']
        for c in galaxies.get_columns():
            if not c in keep_columns:
                galaxies.delete_column(c)
    return galaxies

def get_galaxy_sources(galaxies, bands):
    from legacypipe.catalog import fits_reverse_typemap
    from legacypipe.survey import (LegacySersicIndex, LegacyEllipseWithPriors,
                                   LogRadius, RexGalaxy)
    from tractor import NanoMaggies, RaDecPos, PointSource
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor.galaxy import DevGalaxy, ExpGalaxy
    from tractor.sersic import SersicGalaxy

    # Factor of HyperLEDA to set the galaxy max radius
    radius_max_factor = 2.

    srcs = [None for g in galaxies]

    # If we have pre-burned galaxies, re-create the Tractor sources for them.
    I, = np.nonzero(galaxies.preburned)
    for ii,g in zip(I, galaxies[I]):
        typ = fits_reverse_typemap[g.type.strip()]
        pos = RaDecPos(g.ra, g.dec)
        fluxes = dict([(band, g.get('flux_%s' % band)) for band in bands])
        bright = NanoMaggies(order=bands, **fluxes)
        shape = None
        # put the Rex branch first, because Rex is a subclass of ExpGalaxy!
        if issubclass(typ, RexGalaxy):
            assert(np.isfinite(g.shape_r))
            logre = np.log(g.shape_r)
            shape = LogRadius(logre)
            # set prior max at 2x SGA radius
            shape.setMaxLogRadius(logre + np.log(radius_max_factor))
        elif issubclass(typ, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
            assert(np.isfinite(g.shape_r))
            assert(np.isfinite(g.shape_e1))
            assert(np.isfinite(g.shape_e2))
            shape = EllipseE(g.shape_r, g.shape_e1, g.shape_e2)
            # switch to softened ellipse (better fitting behavior)
            shape = EllipseESoft.fromEllipseE(shape)
            # and then to our custom ellipse class
            logre = shape.logre
            shape = LegacyEllipseWithPriors(logre, shape.ee1, shape.ee2)
            assert(np.all(np.isfinite(shape.getParams())))
            # set prior max at 2x SGA radius
            shape.setMaxLogRadius(logre + np.log(radius_max_factor))

        if issubclass(typ, PointSource):
            src = typ(pos, bright)
        # this catches Rex too
        elif issubclass(typ, (DevGalaxy, ExpGalaxy)):
            src = typ(pos, bright, shape)
        elif issubclass(typ, (SersicGalaxy)):
            assert(np.isfinite(g.sersic))
            sersic = LegacySersicIndex(g.sersic)
            src = typ(pos, bright, shape, sersic)
        else:
            raise RuntimeError('Unknown preburned SGA source type "%s"' % typ)
        debug('Created', src)
        assert(np.isfinite(src.getLogPrior()))
        srcs[ii] = src

    # SGA parent catalog: 'preburned' is not set
    # This also can happen in the preburned/ellipse catalog when fitting
    # fails, or no-grz, etc.
    I, = np.nonzero(np.logical_not(galaxies.preburned))
    for ii,g in zip(I, galaxies[I]):
        # Initialize each source with an exponential disk--
        fluxes = dict([(band, NanoMaggies.magToNanomaggies(g.mag))
                       for band in bands])
        assert(np.all(np.isfinite(list(fluxes.values()))))
        rr = g.radius * 3600. / 2 # factor of two accounts for R(25)-->reff [arcsec]
        assert(np.isfinite(rr))
        assert(np.isfinite(g.ba))
        assert(np.isfinite(g.pa))
        ba = g.ba
        if ba <= 0.0 or ba > 1.0:
            # Make round!
            ba = 1.0
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(rr, ba, 180-g.pa) # note the 180 rotation
        assert(np.isfinite(logr))
        assert(np.isfinite(ee1))
        assert(np.isfinite(ee2))
        shape = LegacyEllipseWithPriors(logr, ee1, ee2)
        shape.setMaxLogRadius(logr + np.log(radius_max_factor))
        src = ExpGalaxy(RaDecPos(g.ra, g.dec),
                        NanoMaggies(order=bands, **fluxes),
                        shape)
        assert(np.isfinite(src.getLogPrior()))
        src.needs_initial_flux = True
        srcs[ii] = src

    return srcs

def read_star_clusters(targetwcs):
    """The code to generate the NGC-star-clusters-fits catalog is in
    legacypipe/bin/build-cluster-catalog.py.

    """
    from pkg_resources import resource_filename
    from astrometry.util.starutil_numpy import degrees_between

    clusterfile = resource_filename('legacypipe', 'data/NGC-star-clusters.fits')
    debug('Reading {}'.format(clusterfile))
    clusters = fits_table(clusterfile, columns=['ra', 'dec', 'radius', 'type', 'ba', 'pa'])
    clusters.ref_id = np.arange(len(clusters))

    radius = 1.
    rc,dc = targetwcs.radec_center()
    d = degrees_between(rc, dc, clusters.ra, clusters.dec)
    clusters.cut(d < radius)
    if len(clusters) == 0:
        return None

    debug('Cut to {} star cluster(s) within the brick'.format(len(clusters)))
    clusters.ref_cat = np.array(['CL'] * len(clusters))

    # Radius in degrees
    clusters.radius = clusters.radius
    clusters.radius[np.logical_not(np.isfinite(clusters.radius))] = 1./60.

    # Set isbright=True
    clusters.isbright = np.zeros(len(clusters), bool)
    clusters.iscluster = np.ones(len(clusters), bool)

    clusters.sources = np.array([None] * len(clusters))

    return clusters

def get_reference_map(wcs, refs):
    from legacypipe.bits import IN_BLOB

    H,W = wcs.shape
    H = int(H)
    W = int(W)
    refmap = np.zeros((H,W), np.uint8)
    pixscale = wcs.pixel_scale()
    cd = wcs.cd
    cd = np.reshape(cd, (2,2)) / (pixscale / 3600.)

    # circular/elliptical regions:
    for col,bit,ellipse in [('isbright', 'BRIGHT', False),
                            ('ismedium', 'MEDIUM', False),
                            ('iscluster', 'CLUSTER', True),
                            ('islargegalaxy', 'GALAXY', True),]:
        isit = refs.get(col)
        if not np.any(isit):
            debug('None marked', col)
            continue
        I, = np.nonzero(isit)
        debug(len(I), 'with', col, 'set')
        if len(I) == 0:
            continue
        thisrefs = refs[I]

        radius_pix = np.ceil(thisrefs.radius * 3600. / pixscale).astype(np.int32)

        if bit == 'BRIGHT':
            # decrease the BRIGHT masking radius by a factor of two!
            debug('Scaling down BRIGHT masking radius by a factor of 2')
            radius_pix = (radius_pix + 1) // 2

        _,xx,yy = wcs.radec2pixelxy(thisrefs.ra, thisrefs.dec)
        xx -= 1.
        yy -= 1.
        for x,y,rpix,ref in zip(xx,yy,radius_pix,thisrefs):
            # Cut to bounding square
            xlo = int(np.clip(np.floor(x   - rpix), 0, W))
            xhi = int(np.clip(np.ceil (x+1 + rpix), 0, W))
            ylo = int(np.clip(np.floor(y   - rpix), 0, H))
            yhi = int(np.clip(np.ceil (y+1 + rpix), 0, H))
            if xlo == xhi or ylo == yhi:
                continue
            bitval = np.uint8(IN_BLOB[bit])
            if not ellipse:
                rr = ((np.arange(ylo,yhi)[:,np.newaxis] - y)**2 +
                      (np.arange(xlo,xhi)[np.newaxis,:] - x)**2)
                masked = (rr <= rpix**2)
            else:
                # *should* have ba and pa if we got here...
                xgrid,ygrid = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
                dx = xgrid - x
                dy = ygrid - y
                # Rotate to "intermediate world coords" via the unit-scaled CD matrix
                du = cd[0][0] * dx + cd[0][1] * dy
                dv = cd[1][0] * dx + cd[1][1] * dy
                debug('Object: PA', ref.pa, 'BA', ref.ba, 'Radius', ref.radius, 'pix', rpix)
                if not np.isfinite(ref.pa):
                    ref.pa = 0.
                ct = np.cos(np.deg2rad(90.+ref.pa))
                st = np.sin(np.deg2rad(90.+ref.pa))
                v1 = ct * du + -st * dv
                v2 = st * du +  ct * dv
                r1 = rpix
                r2 = rpix * ref.ba
                masked = (v1**2 / r1**2 + v2**2 / r2**2 < 1.)
            refmap[ylo:yhi, xlo:xhi] |= (bitval * masked)
    return refmap
