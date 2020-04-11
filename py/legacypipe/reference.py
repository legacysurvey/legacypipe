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
                          star_clusters=True):
    H,W = targetwcs.shape
    H,W = int(H),int(W)

    # How big of a margin to search for bright stars and star clusters --
    # this should be based on the maximum radius they are considered to
    # affect.  In degrees.
    ref_margin = 0.125
    mpix = int(np.ceil(ref_margin * 3600. / pixscale))
    marginwcs = targetwcs.get_subimage(-mpix, -mpix, W+2*mpix, H+2*mpix)

    # Table of reference-source properties, including a field 'sources',
    # with tractor source objects.
    refs = []

    # Tycho-2 stars
    if tycho_stars:
        tycho = read_tycho2(survey, marginwcs, bands)
        if len(tycho):
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
            # proper motions, so rewind Gaia positions to the
            # approximate epoch of Tycho-2: 1991.5.
            cosdec = np.cos(np.deg2rad(gaia.dec))
            gra  = gaia.ra +  (1991.5 - gaia.ref_epoch) * gaia.pmra  / (3600.*1000.) / cosdec
            gdec = gaia.dec + (1991.5 - gaia.ref_epoch) * gaia.pmdec / (3600.*1000.)
            I,J,_ = match_radec(tycho.ra, tycho.dec, gra, gdec, 1./3600.,
                                nearest=True)
            debug('Matched', len(I), 'Tycho-2 stars to Gaia stars.')
            if len(I):
                keep = np.ones(len(tycho), bool)
                keep[I] = False
                tycho.cut(keep)
                gaia.isbright[J] = True
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
        galaxies = read_large_galaxies(survey, targetwcs, bands)
        if galaxies is not None:
            # Resolve possible Gaia-large-galaxy duplicates
            if gaia and len(gaia):
                I,J,_ = match_radec(galaxies.ra, galaxies.dec, gaia.ra, gaia.dec,
                                    2./3600., nearest=True)
                print('Matched', len(I), 'large galaxies to Gaia stars.')
                if len(I):
                    gaia.donotfit[J] = True
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

    keeprad = np.maximum(refs.keep_radius, refs.radius)
    # keeprad to pix
    keeprad = np.ceil(keeprad * 3600. / pixscale).astype(int)

    ok,xx,yy = targetwcs.radec2pixelxy(refs.ra, refs.dec)
    # ibx = integer brick coords
    refs.ibx = np.round(xx-1.).astype(int)
    refs.iby = np.round(yy-1.).astype(int)

    # cut ones whose position + radius are outside the brick bounds.
    refs.cut((xx > -keeprad) * (xx < W+keeprad) *
             (yy > -keeprad) * (yy < H+keeprad))
    # mark ones that are actually inside the brick area.
    refs.in_bounds = ((refs.ibx >= 0) * (refs.ibx < W) *
                      (refs.iby >= 0) * (refs.iby < H))

    for col in ['isbright', 'ismedium', 'islargegalaxy', 'iscluster', 'isgaia',
                'donotfit', 'freezeparams']:
        if not col in refs.get_columns():
            refs.set(col, np.zeros(len(refs), bool))

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

def read_gaia(targetwcs, bands):
    '''
    *targetwcs* here should include margin
    '''
    from legacypipe.gaiacat import GaiaCatalog
    from legacypipe.survey import GaiaSource

    gaia = GaiaCatalog().get_catalog_in_wcs(targetwcs)
    debug('Got', len(gaia), 'Gaia stars nearby')

    # DJS, [decam-chatter 5486] Solved! GAIA separation of point sources
    #   from extended sources
    # Updated for Gaia DR2 by Eisenstein,
    # [decam-data 2770] Re: [desi-milkyway 639] GAIA in DECaLS DR7
    # But shifted one mag to the right in G.
    gaia.G = gaia.phot_g_mean_mag

    # Gaia to DECam color transformations for stars
    color = gaia.phot_bp_mean_mag - gaia.phot_rp_mean_mag
    color[np.logical_not(np.isfinite(color))] = 0.
    color = np.clip(color, -0.5, 3.3)
    # Use Arjun's Gaia-to-DECam transformations.
    for b,coeffs in [
            ('g', [-0.11368, 0.37504, 0.17344, -0.08107, 0.28088,
                   -0.21250, 0.05773,-0.00525]),
            ('r', [ 0.10533,-0.22975, 0.06257,-0.24142, 0.24441,
                    -0.07248, 0.00676]),
            ('z', [ 0.46744,-0.95143, 0.19729,-0.08810, 0.01566])]:
        mag = gaia.G.copy()
        for order,c in enumerate(coeffs):
            mag += c * color**order
        gaia.set('decam_mag_%s' % b, mag)

    # force this source to remain a point source?
    gaia.pointsource = (gaia.G <= 18.) * (gaia.astrometric_excess_noise < 10.**0.5)

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

    # Take the brighter of G, z to expand masks around red stars.
    gaia.mask_mag = np.minimum(gaia.G, gaia.decam_mag_z + 1.)

    # radius to consider affected by this star, for MASKBITS
    gaia.radius = mask_radius_for_mag(gaia.mask_mag)
    # radius for keeping this source in the ref catalog
    # (eg, for halo subtraction)
    gaia.keep_radius = 4. * gaia.radius
    gaia.delete_column('G')
    gaia.isgaia = np.ones(len(gaia), bool)
    gaia.isbright = (gaia.phot_g_mean_mag < 13.)
    gaia.ismedium = (gaia.phot_g_mean_mag < 16.)
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

    # This is in degrees and the magic 0.262 (indeed the whole
    # relation) is from eyeballing a radius-vs-mag plot that was in
    # pixels; that is unrelated to the present targetwcs pixel scale.
    radius = np.minimum(1800., 150. * 2.5**((11. - mag)/3.)) * 0.262/3600.
    return radius

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
        tycho.pmra_ivar = 1./tycho.sigma_pm_ra**2
        tycho.pmdec_ivar = 1./tycho.sigma_pm_dec**2
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
    tycho.isbright = np.ones(len(tycho), bool)
    tycho.ismedium = np.ones(len(tycho), bool)
    tycho.sources = np.empty(len(tycho), object)
    if bands is not None:
        for i,t in enumerate(tycho):
            tycho.sources[i] = GaiaSource.from_catalog(t, bands)
    return tycho

def get_large_galaxy_version(fn):
    preburn = False
    hdr = fitsio.read_header(fn)
    try:
        v = hdr.get('LSLGAVER')
        if v is not None:
            v = v.strip()
            if 'model' in v.lower():
                preburn = True
                v, _ = v.split('-')
            assert(len(v) == 2)
            return v, preburn
    except KeyError:
        pass
    for k in ['3.0', '2.0']:
        if k in fn:
            return 'L'+k[0]
    return 'LG', preburn

def read_large_galaxies(survey, targetwcs, bands):
    from astrometry.libkd.spherematch import tree_open, tree_search_radec

    from legacypipe.catalog import fits_reverse_typemap
    from tractor import NanoMaggies, RaDecPos, PointSource
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor.galaxy import DevGalaxy, ExpGalaxy
    from tractor.sersic import SersicGalaxy
    from legacypipe.survey import LegacySersicIndex, LegacyEllipseWithPriors, LogRadius, RexGalaxy

    galfn = survey.find_file('large-galaxies')
    if galfn is None:
        debug('No large-galaxies catalog file')
        return None
    radius = 1.
    rc,dc = targetwcs.radec_center()

    kd = tree_open(galfn, 'largegals')
    I = tree_search_radec(kd, rc, dc, radius)
    debug(len(I), 'large galaxies within', radius,
          'deg of RA,Dec (%.3f, %.3f)' % (rc,dc))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    galaxies = fits_table(galfn, rows=I)
    del kd

    refcat, preburn = get_large_galaxy_version(galfn)

    if not preburn:
        # Original LSLGA
        galaxies.rename('lslga_id', 'ref_id')
        galaxies.ref_cat = np.array([refcat] * len(galaxies))
        galaxies.islargegalaxy = np.array([True] * len(galaxies))

    else:
        # Need to initialize islargegalaxy to False because we will bring in
        # pre-burned sources that we do not want to mask.
        galaxies.islargegalaxy = np.zeros(len(galaxies), bool)

    # Deal with NaN position angles & axis ratios
    galaxies.rename('pa', 'pa_orig')
    galaxies.pa = np.zeros(len(galaxies), np.float32)
    gd = np.where(np.isfinite(galaxies.pa_orig))[0]
    if len(gd) > 0:
        galaxies.pa[gd] = galaxies.pa_orig[gd]
    galaxies.rename('ba', 'ba_orig')
    galaxies.ba = np.zeros(len(galaxies), np.float32)
    gd = np.where(np.isfinite(galaxies.ba_orig))[0]
    if len(gd) > 0:
        galaxies.ba[gd] = galaxies.ba_orig[gd]
        
    galaxies.radius = galaxies.d25 / 2. / 60. # [degree]

    galaxies.freezeparams = np.zeros(len(galaxies), bool)
    galaxies.sources = np.empty(len(galaxies), object)
    galaxies.sources[:] = None

    # use the pre-burned LSLGA catalog
    if 'preburned' in galaxies.get_columns():
        preburned = np.logical_and(preburn, galaxies.preburned)
    else:
        preburned = np.zeros(len(galaxies), bool)

    I, = np.nonzero(preburned)
    # only fix the parameters of pre-burned galaxies
    for ii,g in zip(I, galaxies[I]):
        try:
            typ = fits_reverse_typemap[g.type.strip()]
            pos = RaDecPos(g.ra, g.dec)
            fluxes = dict([(band, g.get('flux_%s' % band)) for band in bands])
            bright = NanoMaggies(order=bands, **fluxes)
            shape = None
            # put the Rex branch first, because Rex is a subclass of ExpGalaxy!
            if issubclass(typ, RexGalaxy):
                assert(np.isfinite(g.shape_r))
                shape = LogRadius(np.log(g.shape_r))
            elif issubclass(typ, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
                assert(np.isfinite(g.shape_r))
                assert(np.isfinite(g.shape_e1))
                assert(np.isfinite(g.shape_e2))
                shape = EllipseE(g.shape_r, g.shape_e1, g.shape_e2)
                # switch to softened ellipse (better fitting behavior)
                shape = EllipseESoft.fromEllipseE(shape)
                # and then to our custom ellipse class
                shape = LegacyEllipseWithPriors(shape.logre, shape.ee1, shape.ee2)
                assert(np.all(np.isfinite(shape.getParams())))

            if issubclass(typ, (DevGalaxy, ExpGalaxy)):
                src = typ(pos, bright, shape)
            elif issubclass(typ, (SersicGalaxy)):
                assert(np.isfinite(g.sersic))
                sersic = LegacySersicIndex(g.sersic)
                src = typ(pos, bright, shape, sersic)
            elif issubclass(typ, PointSource):
                src = typ(pos, bright)
            else:
                print('Unknown type', typ)
            print('Created', src)

            galaxies.sources[ii] = src

            if galaxies.freeze[ii] and galaxies.ref_cat[ii] == refcat:
                galaxies.islargegalaxy[ii] = True
                ###
                # galaxies.radius[ii] = galaxies.d25_model[ii] / 2 / 60 # [degree]
                # galaxies.pa[ii] = galaxies.pa_model[ii]
                # galaxies.ba[ii] = galaxies.ba_model[ii]

            if galaxies.freeze[ii]:
                galaxies.freezeparams[ii] = True
        except:
            import traceback
            print('Failed to create Tractor source for LSLGA entry:',
                  traceback.print_exc())
            raise

    I, = np.nonzero(np.logical_not(preburned))
    for ii,g in zip(I, galaxies[I]):
        # Initialize each source with an exponential disk--
        fluxes = dict([(band, NanoMaggies.magToNanomaggies(g.mag)) for band in bands])
        assert(np.all(np.isfinite(list(fluxes.values()))))
        rr = g.radius * 3600. / 2 # factor of two accounts for R(25)-->reff [arcsec]
        assert(np.isfinite(rr))
        assert(np.isfinite(g.ba))
        assert(np.isfinite(g.pa))
        ba = g.ba
        if ba == 0.0:
            # Make round!
            ba = 1.0
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(rr, ba, 180-g.pa) # note the 180 rotation
        assert(np.isfinite(logr))
        assert(np.isfinite(ee1))
        assert(np.isfinite(ee2))
        src = ExpGalaxy(RaDecPos(g.ra, g.dec),
                        NanoMaggies(order=bands, **fluxes),
                        LegacyEllipseWithPriors(logr, ee1, ee2))
        galaxies.sources[ii] = src

    keep_columns = ['ra', 'dec', 'radius', 'mag', 'ref_cat', 'ref_id', 'ba', 'pa',
                    'sources', 'islargegalaxy', 'freezeparams']

    for c in galaxies.get_columns():
        if not c in keep_columns:
            galaxies.delete_column(c)

    return galaxies

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
