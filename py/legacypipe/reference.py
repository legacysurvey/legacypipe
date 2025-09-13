import os
import warnings
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

def get_reference_sources(survey, targetwcs, bands,
                          tycho_stars=True,
                          gaia_stars=True,
                          large_galaxies=True,
                          star_clusters=True,
                          clean_columns=True,
                          plots=False, ps=None,
                          gaia_margin=None,
                          galaxy_margin=None):
    # If bands = None, does not create sources.
    from astrometry.libkd.spherematch import match_radec
    from collections import Counter

    H,W = targetwcs.shape
    H,W = int(H),int(W)
    pixscale = targetwcs.pixel_scale()

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
            info('Found', len(tycho), 'Tycho-2 stars nearby')
            refs.append(tycho)

    # Add Gaia stars
    gaia = None
    if gaia_stars:
        gaia = read_gaia(marginwcs, bands)
    if gaia is not None:
        # Handle sources that appear in both Gaia and Tycho-2 by
        # dropping the entry from Tycho-2.
        if len(gaia) and len(tycho):
            merge_gaia_tycho(gaia, tycho, plots=plots, ps=ps,
                             targetwcs=targetwcs)
        if gaia is not None and len(gaia) > 0:
            info('Found', len(gaia), 'Gaia stars nearby')
            gaia.dup = np.zeros(len(gaia), bool)
            refs.append(gaia)

    # Read the catalog of star (open and globular) clusters and add them to the
    # set of reference stars (with the isbright bit set).
    if star_clusters:
        clusters = read_star_clusters(marginwcs)
        if clusters is not None:
            info('Found', len(clusters), 'star clusters nearby')
            refs.append(clusters)

    # Read large galaxies nearby.
    if large_galaxies:
        galaxies = read_large_galaxies(survey, targetwcs, bands, clean_columns=clean_columns,
                                       max_radius=galaxy_margin)
        if galaxies is not None:
            # Resolve possible Gaia-large-galaxy duplicates
            if gaia and len(gaia):
                I,J,_ = match_radec(galaxies.ra, galaxies.dec, gaia.ra, gaia.dec,
                                    2./3600., nearest=True)
                debug('Matched', len(I), 'large galaxies to Gaia stars.')
                if len(I):
                    gaia.ignore_source[J] = True
                    gaia.dup[J] = True
            # Resolve possible Tycho2-large-galaxy duplicates (with larger radius)
            if tycho and len(tycho):
                I,J,_ = match_radec(galaxies.ra, galaxies.dec, tycho.ra, tycho.dec,
                                    5./3600., nearest=True)
                debug('Matched', len(I), 'large galaxies to Tycho-2 stars.')
                if len(I):
                    tycho.ignore_source[J] = True
                    tycho.dup[J] = True
            info('Found', len(galaxies), 'large galaxies nearby')
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
    info('Reference sources touching this brick:', Counter([str(r) for r in refs.ref_cat]).most_common())
    info('Reference sources within this brick:', Counter([str(r) for r in refs.ref_cat[refs.in_bounds]]).most_common())

    # ensure bool columns exist
    for col in ['isbright', 'ismedium', 'islargegalaxy', 'iscluster', 'isgaia',
                'istycho', 'freezeparams', 'isresolved', 'ismcloud', 'ignore_source']:
        if not col in refs.get_columns():
            refs.set(col, np.zeros(len(refs), bool))
            debug('Adding False values for missing column "%s" in refs' % col)
    # Copy flags from the 'refs' table to the source objects themselves.
    sources = refs.sources
    refs.delete_column('sources')
    for i,(ignore,freeze) in enumerate(zip(refs.ignore_source, refs.freezeparams)):
        if sources[i] is None:
            continue
        sources[i].ignore_source = ignore
        sources[i].is_reference_source = True
        sources[i].freezeparams = freeze

    return refs,sources

def merge_gaia_tycho(gaia, tycho, plots=False, ps=None, targetwcs=None):
    from astrometry.libkd.spherematch import match_radec
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

def read_gaia(wcs, bands):
    '''
    Reads Gaia stars within the given *wcs* object.

    *wcs* should include any margin you want (eg with *wcs.get_subimage()*).

    If *bands* is not *None*, the returned *GaiaSource* objects will
    be created with slots for fluxes in the given *bands*.
    '''
    from legacypipe.gaiacat import GaiaCatalog
    from legacypipe.survey import GaiaSource

    # See also format_catalog.py
    cols = [
        'source_id', 'ra', 'dec', 'pmra', 'pmdec', 'parallax',
        'ref_epoch',
        'ra_error', 'dec_error', 'pmra_error', 'pmdec_error', 'parallax_error',
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
        'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
        'phot_rp_mean_flux_over_error',
        'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs',
        'astrometric_params_solved',
        'phot_variable_flag', 'astrometric_excess_noise', 'astrometric_excess_noise_sig',
        'astrometric_n_obs_al', 'astrometric_n_good_obs_al',
        # 'astrometric_weight_al',  <-- does not exist in our Gaia-DR3 healpix catalogs
        # 'a_g_val',
        # 'e_bp_min_rp_val',
        'duplicated_source', 'phot_bp_rp_excess_factor',
        'astrometric_sigma5d_max',
    ]

    gaia = GaiaCatalog().get_catalog_in_wcs(wcs, columns=cols)
    debug('Got', len(gaia), 'Gaia stars nearby')

    fix_gaia(gaia, bands)

    gaia.sources = np.empty(len(gaia), object)
    # np.empty seems to already initialize to "None", but just to be sure...
    gaia.sources[:] = None
    if bands is not None:
        for i,g in enumerate(gaia):
            gaia.sources[i] = GaiaSource.from_catalog(g, bands)
    return gaia

def fix_gaia(gaia, bands):
    from legacypipe.gaiacat import gaia_to_decam

    gaia.phot_g_mean_mag = gaia.phot_g_mean_mag.astype(np.float32)
    gaia.G = gaia.phot_g_mean_mag
    # Sort by brightness (for reference-*.fits output table)
    sortmag = gaia.G.copy()
    sortmag[sortmag == 0] = gaia.phot_rp_mean_mag[sortmag == 0]
    gaia.cut(np.argsort(sortmag))

    # Including DECam griz, plus the bands we're actually processing
    bb = ['g','r','i','z']
    for band in bands:
        if not band in bb:
            bb.append(band)
    mags = gaia_to_decam(gaia, bb)
    for band,mag in zip(bb, mags):
        # no color terms - skip
        if mag is None:
            continue
        gaia.set('decam_mag_%s' % band, mag)

    # force this source to remain a point source?
    # Long history here, starting DJS, [decam-chatter 5486] Solved! GAIA separation
    #   of point sources from extended sources
    # Updated for Gaia DR2 by Eisenstein,
    # [decam-data 2770] Re: [desi-milkyway 639] GAIA in DECaLS DR7
    # And made far more restrictive following BGS feedback.
    # Then, for Gaia-EDR3, Rongpu found we no longer need to look at
    # astrometric_excess_noise.
    gaia.pointsource = (gaia.G <= 18.)

    # in our catalog files, this is in float32; in the Gaia data model it's
    # a byte, with only values 3 and 31 in DR2.
    gaia.astrometric_params_solved = gaia.astrometric_params_solved.astype(np.uint8)

    # Gaia version?
    gaiaver = os.getenv('GAIA_CAT_VER', '1')
    gaia_release = 'G%s' % gaiaver
    gaia.ref_cat = np.array([gaia_release] * len(gaia))
    gaia.ref_id  = gaia.source_id
    with np.errstate(divide='ignore'):
        gaia.pmra_ivar  = 1./gaia.pmra_error **2
        gaia.pmdec_ivar = 1./gaia.pmdec_error**2
        gaia.parallax_ivar = 1./gaia.parallax_error**2
    # mas -> deg
    gaia.ra_ivar  = (1./(gaia.ra_error  / 1000. / 3600.)**2).astype(np.float32)
    gaia.dec_ivar = (1./(gaia.dec_error / 1000. / 3600.)**2).astype(np.float32)

    for c in ['ra_error', 'dec_error', 'parallax_error',
              'pmra_error', 'pmdec_error']:
        gaia.delete_column(c)
    for c in ['pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar',
              'parallax_ivar']:
        X = gaia.get(c)
        X[np.logical_not(np.isfinite(X))] = 0.

    gaia.mag = gaia.G
    # Use Gaia RP (then BP) if G is not measured
    gaia.mag[gaia.mag == 0] = gaia.phot_rp_mean_mag[gaia.mag == 0]
    gaia.mag[gaia.mag == 0] = gaia.phot_bp_mean_mag[gaia.mag == 0]
    # uniform name w/ Tycho-2
    gaia.zguess = gaia.decam_mag_z.copy()
    # no zguess -- fill with optical mag.
    gaia.zguess[gaia.zguess == 0] = gaia.mag[gaia.zguess == 0]
    # Take the brighter of optical, z to expand masks around red stars.
    gaia.mask_mag = np.minimum(gaia.mag, gaia.zguess + 1.)

    # Plug in a tiny mag for stars with no Gaia-EDR3 mag measurements
    # (eg, sourceid 3638309166294796544 has Gaia G = BP = RP = none)
    Ibad = np.flatnonzero(gaia.mag == 0.)
    gaia.mask_mag[Ibad] = 99.

    # radius to consider affected by this star, for MASKBITS
    gaia.radius = mask_radius_for_mag(gaia.mask_mag)
    gaia.radius[Ibad] = 0.
    # radius for keeping this source in the ref catalog
    # (eg, for halo subtraction)
    gaia.keep_radius = 4. * gaia.radius
    gaia.delete_column('G')
    gaia.isgaia = np.ones(len(gaia), bool)
    gaia.istycho = np.zeros(len(gaia), bool)
    gaia.isbright = (gaia.mask_mag < 13.)
    gaia.ismedium = (gaia.mask_mag < 16.)
    gaia.ignore_source = np.zeros(len(gaia), bool)

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
    # Tycho-2 catalog:
    # - John added the "isgalaxy" flag 2018-05-10, from the Metz &
    # Geffert (04) catalog.
    # - Eddie added the "zguess" column 2019-03-06, by matching with
    # 2MASS and estimating z based on APASS.
    # - Rongpu added the "ggguess" column (Gaia-G guess) 2021-09-13, by matching with 2MASS.

    # The "tycho2.kd.fits" file read here was produced by:
    # startree -P -k -n stars -T -i /global/cfs/cdirs/desi/users/rongpu/useful/tycho2-reference.fits -o tycho2.kd.fits

    kd = tree_open(tycho2fn, 'stars')
    I = tree_search_radec(kd, ra, dec, radius)
    debug(len(I), 'Tycho-2 stars within', radius, 'deg of RA,Dec (%.3f, %.3f)' % (ra,dec))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    tycho = fits_table(tycho2fn, rows=I)
    del kd
    fix_tycho(tycho)
    tycho.sources = np.empty(len(tycho), object)
    if bands is not None:
        for i,t in enumerate(tycho):
            tycho.sources[i] = GaiaSource.from_catalog(t, bands)
    return tycho

def fix_tycho(tycho):
    if 'isgalaxy' in tycho.get_columns():
        tycho.cut(tycho.isgalaxy == 0)
        debug('Cut to', len(tycho), 'Tycho-2 stars on isgalaxy==0')
    else:
        warnings.warn('No "isgalaxy" column in Tycho-2 catalog')

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
    # Use Rongpu's Gaia-G guesses mag, when available and non-NaN
    if not 'ggguess' in tycho.get_columns():
        warnings.warn('No "ggguess" column in Tycho-2 catalog')
        tycho.mag = tycho.mag_vt
    else:
        tycho.mag = tycho.ggguess.astype(np.float32)
        # Fall back to V_T
        I = np.flatnonzero(np.logical_not(np.isfinite(tycho.mag)))
        tycho.mag[I] = tycho.mag_vt[I]
    # Fall back further to MAG_HP, MAG_BT.
    tycho.mag[tycho.mag == 0] = tycho.mag_hp[tycho.mag == 0]
    tycho.mag[tycho.mag == 0] = tycho.mag_bt[tycho.mag == 0]

    # For very red stars, use the brighter of zguess+1 and the optical mag
    # for the masking radius.
    tycho.mask_mag = tycho.mag
    with np.errstate(invalid='ignore'):
        I = np.flatnonzero(np.isfinite(tycho.zguess) *
                           (tycho.zguess + 1. < tycho.mag))
    tycho.mask_mag[I] = tycho.zguess[I] + 1.
    # Per discussion in issue #306 -- cut to mag < 13.
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
    tycho.ignore_source = np.zeros(len(tycho), bool)

def get_large_galaxy_version(fn):
    ellipse = False
    hdr = fitsio.read_header(fn)
    try:
        # SGA2025
        v = hdr.get('VER')
        if v is None:
            # SGA2020
            v = hdr.get('SGAVER')
        if v is not None:
            v = v.strip()
            if 'ellipse' in v.lower():
                ellipse = True
                v, _ = v.split('-')
            assert(len(v) == 2)
            return v, ellipse
    except KeyError:
        pass
    for k in ['3.0', '2.0']:
        if k in fn:
            return 'L'+k[0], ellipse
    return 'LG', ellipse

def read_large_galaxies(survey, targetwcs, bands, clean_columns=True,
                        extra_columns=None,
                        max_radius=None):
    from astrometry.util.starutil_numpy import degrees_between
    from legacypipe.bits import SGA_FITMODE, sga_fitmode_type

    # max_radius (in deg) should be the largest radius in the SGA catalog!
    if max_radius is None:
        max_radius = 2.

    galaxy_tables = []

    rc,dc = targetwcs.radec_center()
    brick_radius = targetwcs.radius()
    max_radius += brick_radius

    # Hard-coded Magellanic clouds
    # (ra, dec, diam_arcmin, b/a, PA):
    # LMC = (80.894167, -69.756111, 645.65, 0.85, 171.)
    # SMC = (13.186667, -72.828611, 380.19, 0.64, 45.)
    mclouds = fits_table()
    mclouds.ra   = np.array([80.894167, 13.186667])
    mclouds.dec  = np.array([-69.756111, -72.828611])
    mclouds.diam = np.array([645.65, 380.19])
    mclouds.ba   = np.array([0.85, 0.64])
    mclouds.pa   = np.array([171., 45.])
    mclouds.ref_cat = np.array(['MC', 'MC'])
    mclouds.ref_id  = np.array([1, 2])
    mclouds.name    = np.array(['LMC', 'SMC'])
    mclouds.radius  = mclouds.diam / 2. / 60.
    d = degrees_between(rc, dc, mclouds.ra, mclouds.dec)
    touch = (d < brick_radius + mclouds.radius)
    if np.any(touch):
        mclouds.cut(touch)
        mclouds.fitmode = np.array([SGA_FITMODE['MCLOUDS']]*len(mclouds), sga_fitmode_type)
        mclouds.preburned = np.array([False] * len(mclouds))
        mclouds.mag = np.array([0.] * len(mclouds))
        mclouds.ignore_source = np.array([True] * len(mclouds))
        print('Overlaps Magellanic cloud:', mclouds.name)
        galaxy_tables.append(mclouds)
    del mclouds

    galaxies = read_sga(survey, rc, dc, max_radius)
    if galaxies is not None:
        galaxy_tables.append(galaxies)

    if len(galaxy_tables) == 0:
        return None
    galaxies = merge_tables(galaxy_tables, columns='fillzero')

    galaxies.isresolved = ((galaxies.fitmode & SGA_FITMODE['RESOLVED']) != 0)
    galaxies.ismcloud   = ((galaxies.fitmode & SGA_FITMODE['MCLOUDS' ]) != 0)

    galaxies.sources = np.empty(len(galaxies), object)
    galaxies.sources[:] = None
    if bands is not None:
        galaxies.sources[:] = get_galaxy_sources(galaxies, bands)

    if clean_columns:
        keep_columns = ['ra', 'dec', 'radius', 'mag', 'ref_cat', 'ref_id', 'ba', 'pa',
                        'sources', 'islargegalaxy', 'freezeparams', 'keep_radius',
                        'fitmode', 'isresolved', 'ismcloud', 'ignore_source']
        if extra_columns is not None:
            keep_columns.extend(extra_columns)
        for c in galaxies.get_columns():
            if not c in keep_columns:
                galaxies.delete_column(c)
                debug('Deleting extra column "%s" from galaxy table' % c)
    '''
    Desired behaviors.

    Parent catalog:
    - if overlaps LMC/SMC mask,
      - no new source detection (only Gaia and SGA)
      - in DR10, we did this by setting CLUSTER
      - add new MASKBITS bit: MCLOUDS (LMC/SMC), same behavior as CLUSTER
    - ignore LMC/SMC SGA sources
      - don't want to set GALAXY
    '''
    return galaxies

def read_sga(survey, rc, dc, max_radius):
    from astrometry.libkd.spherematch import tree_open, tree_search_radec
    from legacypipe.bits import SGA_FITMODE, sga_fitmode_type
    from functools import reduce

    galfn = survey.find_file('large-galaxies')
    if galfn is None:
        warnings.warn('No large-galaxies catalog (SGA) file!')
        return None

    debug('Reading', galfn)
    try:
        kd = tree_open(galfn, 'stars')
    except:
        kd = tree_open(galfn, 'largegals')
    I = tree_search_radec(kd, rc, dc, max_radius)
    debug('%i large galaxies within %.3g deg of RA,Dec (%.3f, %.3f)' %
          (len(I), max_radius, rc,dc))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    galaxies = fits_table(galfn, rows=I)
    del kd

    refcat, is_ellipse = get_large_galaxy_version(galfn)
    debug('Large galaxies version: "%s", ellipse catalog?' % refcat, is_ellipse)

    has_fitmode = ('fitmode' in galaxies.get_columns())
    # FIXME - just die?
    if not has_fitmode:
        warnings.warn('No "fitmode" column in SGA catalog!  Assuming fitmode = 0!')
        galaxies.fitmode = np.zeros(len(galaxies), sga_fitmode_type)

    galaxies.ignore_source = np.zeros(len(galaxies), bool)

    if is_ellipse:
        # SGA ellipse catalog
        if has_fitmode:
            # SGA-2025
            galaxies.islargegalaxy = (galaxies.sga_id > -1)
            galaxies.freezeparams = ((galaxies.fitmode & SGA_FITMODE['FREEZE']) != 0)

            galaxies.set_galaxy_maskbit = reduce(np.logical_or, [
                galaxies.freezeparams & galaxies.islargegalaxy,
                galaxies.fitmode == 0,
                galaxies.fitmode == SGA_FITMODE['FIXGEO'] # FIXGEO and not RESOLVED
            ])
            # FIXGEO sources should not generate Tractor sources that we fit or
            # render in any way - they're just a place to hold data so they can finally
            # appear in the output catalog.
            galaxies.ignore_source |= ((galaxies.fitmode & SGA_FITMODE['FIXGEO']) != 0)

        else:
            # SGA-2020
            # NOTE: fields such as ref_cat, preburned, etc, already exist in the
            # "galaxies" catalog read from disk.
            # The galaxies we want to appear in MASKBITS get
            # 'islargegalaxy' set.  This includes both pre-burned
            # galaxies, and ones where the preburning failed and we want
            # to fall back to the SGA-parent ellipse for masking.
            galaxies.islargegalaxy = np.logical_or(
                np.logical_not(galaxies.in_footprint_grz),
                (galaxies.ref_cat == refcat) * (galaxies.sga_id > -1))
            # The pre-fit galaxies whose parameters will stay fixed
            galaxies.freezeparams = (galaxies.preburned * galaxies.freeze)

            galaxies.set_galaxy_maskbit = galaxies.islargegalaxy
            
        # set ref_cat and ref_id for galaxies outside the footprint
        I = np.flatnonzero(np.logical_not(galaxies.in_footprint_grz))
        galaxies.ref_id[I] = galaxies.sga_id[I]
        galaxies.ref_cat[I] = np.array([refcat] * len(I))

    else:
        # SGA parent catalog.
        galaxies.ref_cat = np.array([refcat] * len(galaxies))

        if has_fitmode:
            # SGA-2025
            # The fitmode FREEZE bit is not allowed in the parent catalog
            assert(np.all((galaxies.fitmode & SGA_FITMODE['FREEZE']) == 0))
            galaxies.set_galaxy_maskbit = reduce(np.logical_or, [
                galaxies.fitmode == 0,
                galaxies.fitmode == SGA_FITMODE['FIXGEO']
            ])
        else:
            # SGA-2020
            galaxies.islargegalaxy = np.ones(len(galaxies), bool)
            galaxies.freezeparams = np.zeros(len(galaxies), bool)
            galaxies.preburned = np.zeros(len(galaxies), bool)
        if 'sga_id' in galaxies.columns():
            galaxies.rename('sga_id', 'ref_id')

    if 'mag_leda' in galaxies.columns():
        galaxies.rename('mag_leda', 'mag')

    # Pre-burned, frozen but non-SGA sources have diam=-1.
    if 'd26' in galaxies.get_columns():
        # SGA-2020
        galaxies.radius = np.maximum(0., galaxies.d26 / 2. / 60.) # [degree]
    else:
        galaxies.radius = np.maximum(0., galaxies.diam / 2. / 60.) # [degree]

    galaxies.keep_radius = 2. * galaxies.radius

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

    # DR10 -- estimated i band flux from r,z
    cols = galaxies.get_columns()
    if ('i' in bands and (not 'flux_i' in cols)
        and 'flux_r' in cols and 'flux_z' in cols):
        with np.errstate(divide='ignore', invalid='ignore'):
            mag_r = -2.5 * (np.log10(galaxies.flux_r) - 9)
            mag_z = -2.5 * (np.log10(galaxies.flux_z) - 9)
        mag_r[np.logical_not(np.isfinite(mag_r))] = 30.
        mag_z[np.logical_not(np.isfinite(mag_z))] = 30.
        color = np.clip(mag_r - mag_z, -0.5, 2.0)
        cc = [-0.13689305,
              0.80606322,
              -0.24921022,
              -0.15773003,
              0.10645930,
              -0.0050743524]
        iz = 0.
        for i,c in enumerate(cc):
            iz += c * color**i
        mag_i = mag_z + iz
        galaxies.flux_i = NanoMaggies.magToNanomaggies(mag_i)
        debug('Estimated i mags for SGA galaxies:')
        debug('r:', mag_r[:10])
        debug('z:', mag_z[:10])
        debug('i:', mag_i[:10])
        cols = galaxies.get_columns()

    # Does the SGA catalog have flux_ columns for all the bands we're working with?
    missing_band = False
    has_band = {}
    for band in bands:
        has_band[band] = True
        if not 'flux_%s' % band in cols:
            has_band[band] = False
            missing_band = True
            warnings.warn('No "flux_%s" column in SGA catalog; will have to fit for fluxes' % band)

    # If we have pre-burned galaxies, re-create the Tractor sources for them.
    I, = np.nonzero(galaxies.preburned)
    for ii,g in zip(I, galaxies[I]):
        typ = fits_reverse_typemap[g.type.strip()]
        pos = RaDecPos(g.ra, g.dec)
        fluxes = dict([(band, g.get('flux_%s' % band) if has_band[band] else 1.) for band in bands])
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
        if missing_band:
            src.needs_initial_flux = True
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

    assert(np.all(np.isfinite(clusters.radius)))

    rc,dc = targetwcs.radec_center()
    wcs_rad = targetwcs.radius()
    d = degrees_between(rc, dc, clusters.ra, clusters.dec)
    clusters.cut(d < wcs_rad + clusters.radius)
    if len(clusters) == 0:
        return None

    debug('Cut to {} star cluster(s) possibly touching the brick'.format(len(clusters)))
    clusters.ref_cat = np.array(['CL'] * len(clusters))

    # Set isbright=True
    clusters.isbright = np.zeros(len(clusters), bool)
    clusters.iscluster = np.ones(len(clusters), bool)

    clusters.sources = np.array([None] * len(clusters))

    return clusters

def get_reference_map(wcs, refs):
    from legacypipe.bits import REF_MAP_BITS

    H,W = wcs.shape
    H = int(H)
    W = int(W)
    refmap = np.zeros((H,W), np.uint8)
    pixscale = wcs.pixel_scale()
    cd = wcs.cd
    cd_pix = np.reshape(cd, (2,2)) / (pixscale / 3600.)
    #debug('Scaled CD matrix:', cd_pix)

    # circular/elliptical regions:
    for col,bit,ellipse in [('isbright',           'BRIGHT',   False),
                            ('ismedium',           'MEDIUM',   False),
                            ('iscluster',          'CLUSTER',  True),
                            ('set_galaxy_maskbit', 'GALAXY',   True),
                            ('isresolved',         'RESOLVED', True),
                            ('ismcloud',           'MCLOUDS',  True),
                            ]:
        if not col in refs.get_columns():
            debug('No "%s" column in reference table; skipping' % col)
            continue
        isit = refs.get(col)
        if not np.any(isit & (refs.radius > 0)):
            debug('None marked', col)
            continue
        I, = np.nonzero(isit & (refs.radius > 0))
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
            # debug('Reference source location: x,y (%.1f, %.1f)' % (x, y))
            # debug('xlo, xhi', xlo, xhi)
            # debug('ylo, yhi', ylo, yhi)
            if xlo == xhi or ylo == yhi:
                continue
            bitval = np.uint8(REF_MAP_BITS[bit])
            if not ellipse:
                rr = ((np.arange(ylo,yhi)[:,np.newaxis] - y)**2 +
                      (np.arange(xlo,xhi)[np.newaxis,:] - x)**2)
                masked = (rr <= rpix**2)
            else:
                # *should* have ba and pa if we got here...
                dx, dy = np.meshgrid(np.arange(xlo,xhi) - x,
                                     np.arange(ylo,yhi) - y)
                # Rotate to "intermediate world coords" via the unit-scaled CD matrix
                du = cd_pix[0][0] * dx + cd_pix[0][1] * dy
                dv = cd_pix[1][0] * dx + cd_pix[1][1] * dy
                debug('Object: PA', ref.pa, 'BA', ref.ba, 'Radius', ref.radius, 'pix', rpix)
                # debug('corners:')
                # debug('dx:', dx[0,0], dx[0,-1], dx[-1,0], dx[-1,-1])
                # debug('dy:', dy[0,0], dy[0,-1], dy[-1,0], dy[-1,-1])
                # debug('r(x,y):',
                #       np.hypot(dx[0,0], dy[0,0]), np.hypot(dx[0,-1], dy[0,-1]),
                #       np.hypot(dx[-1,0], dy[-1,0]), np.hypot(dx[-1,-1], dy[-1,-1]))
                # debug('du:', du[0,0], du[0,-1], du[-1,0], du[-1,-1])
                # debug('dv:', dv[0,0], dv[0,-1], dv[-1,0], dv[-1,-1])
                # debug('r(u,v):',
                #       np.hypot(du[0,0],   dv[0,0]), np.hypot(du[0,-1],  dv[0,-1]),
                #       np.hypot(du[-1,0],  dv[-1,0]), np.hypot(du[-1,-1], dv[-1,-1]))
                if not np.isfinite(ref.pa):
                    ref.pa = 0.
                ct = np.cos(np.deg2rad(90.+ref.pa))
                st = np.sin(np.deg2rad(90.+ref.pa))
                v1 = ct * du + -st * dv
                v2 = st * du +  ct * dv
                # debug('v1:', v1[0,0], v1[0,-1], v1[-1,0], v1[-1,-1])
                # debug('v2:', v2[0,0], v2[0,-1], v2[-1,0], v2[-1,-1])
                # debug('r(v1,v2):',
                #       np.hypot(v1[0,0],   v2[0,0]), np.hypot(v1[0,-1],  v2[0,-1]),
                #       np.hypot(v1[-1,0],  v2[-1,0]), np.hypot(v1[-1,-1], v2[-1,-1]))
                r1 = float(rpix)
                r2 = float(rpix) * ref.ba
                masked = (v1**2 / r1**2 + v2**2 / r2**2 < 1.)
                #e = (v1**2 / r1**2 + v2**2 / r2**2)
                #debug('ellipse ratio:', e[0,0], e[0,-1], e[-1,0], e[-1,-1])
                debug('Masking', np.sum(masked), 'of', len(v1.flat), 'pixels')
            refmap[ylo:yhi, xlo:xhi] |= (bitval * masked)
    return refmap
