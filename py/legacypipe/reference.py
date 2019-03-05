import os
import numpy as np
from astrometry.util.fits import fits_table, merge_tables

def get_reference_sources(survey, targetwcs, pixscale, bands,
                          tycho_stars=True, gaia_stars=True, large_galaxies=True,
                          star_clusters=True):
    
    from legacypipe.survey import GaiaSource
    from legacypipe.survey import LegacyEllipseWithPriors
    from tractor import PointSource, NanoMaggies, RaDecPos
    from tractor.galaxy import ExpGalaxy
    from tractor.ellipses import EllipseESoft

    H,W = targetwcs.shape
    H,W = int(H),int(W)

    # How big of a margin to search for bright stars and star clusters --
    # this should be based on the maximum radius they are considered to
    # affect.
    ref_margin = 0.125
    mpix = int(np.ceil(ref_margin * 3600. / pixscale))
    marginwcs = targetwcs.get_subimage(-mpix, -mpix, W+2*mpix, H+2*mpix)
    
    refs = None

    if tycho_stars:
        #print('Enlarged target WCS from', targetwcs, 'to', marginwcs, 'for ref stars')
        # Read Tycho-2 stars and use as saturated sources.
        tycho = read_tycho2(survey, marginwcs)
        if len(tycho) > 0:
            refs = [tycho]
            
    # Add Gaia stars
    gaia = None
    if gaia_stars:
        from astrometry.libkd.spherematch import match_radec
        gaia = read_gaia(marginwcs)
    if gaia is not None:
        gaia.isbright = np.zeros(len(gaia), bool)
        gaia.ismedium = np.ones(len(gaia), bool)
        # Handle sources that appear in both Gaia and Tycho-2 by dropping the entry from Tycho-2.
        if len(gaia) and len(tycho):
            # Before matching, apply proper motions to bring them to
            # the same epoch.
            # We want to use the more-accurate Gaia proper motions, so
            # rewind Gaia positions to the approximate epoch of
            # Tycho-2: 1991.5.
            cosdec = np.cos(np.deg2rad(gaia.dec))
            gra  = gaia.ra +  (1991.5 - gaia.ref_epoch) * gaia.pmra  / (3600.*1000.) / cosdec
            gdec = gaia.dec + (1991.5 - gaia.ref_epoch) * gaia.pmdec / (3600.*1000.)
            I,J,d = match_radec(tycho.ra, tycho.dec, gra, gdec, 1./3600.,
                                nearest=True)
            #print('Matched', len(I), 'Tycho-2 stars to Gaia stars.')
            if len(I):
                keep = np.ones(len(tycho), bool)
                keep[I] = False
                tycho.cut(keep)
                #print('Cut to', len(tycho), 'Tycho-2 stars that do not match Gaia')
                gaia.isbright[J] = True
        if gaia is not None and len(gaia) > 0:
            if refs is None:
                refs = [gaia]
            else:
                refs.append(gaia)

    # Read the catalog of star (open and globular) clusters and add them to the
    # set of reference stars (with the isbright bit set).
    if star_clusters:
        clusters = read_star_clusters(marginwcs)
        if clusters is not None:
            print('Found', len(clusters), 'star clusters nearby')
            clusters.iscluster = np.ones(len(clusters), bool)
            if refs is None:
                refs = [clusters]
            else:
                refs.append(clusters)

    # Read large galaxies nearby.
    if large_galaxies:
        galaxies = read_large_galaxies(survey, targetwcs)
        if galaxies is not None:
            if refs is None:
                refs = [galaxies]
            else:
                refs.append(galaxies)

    refcat = None
    if refs:
        refs = merge_tables([r for r in refs if r is not None], columns='fillzero')

        refs.radius_pix = np.ceil(refs.radius * 3600. / pixscale).astype(int)

        ok,xx,yy = targetwcs.radec2pixelxy(refs.ra, refs.dec)
        # ibx = integer brick coords
        refs.ibx = np.round(xx-1.).astype(int)
        refs.iby = np.round(yy-1.).astype(int)

        refs.cut((xx > -refs.radius_pix) * (xx < W+refs.radius_pix) *
                 (yy > -refs.radius_pix) * (yy < H+refs.radius_pix))

        refs.in_bounds = ((refs.ibx >= 0) * (refs.ibx < W) *
                          (refs.iby >= 0) * (refs.iby < H))

        for col in ['isbright', 'ismedium', 'islargegalaxy', 'iscluster']:
            if not col in refs.get_columns():
                refs.set(col, np.zeros(len(refs), bool))

        ## Create Tractor sources from reference stars
        refcat = []
        for g in refs:
            if g.isbright or g.ismedium or g.iscluster:
                refcat.append(GaiaSource.from_catalog(g, bands))
            elif g.islargegalaxy:
                fluxes = dict([(band, NanoMaggies.magToNanomaggies(g.mag)) for band in bands])
                assert(np.all(np.isfinite(list(fluxes.values()))))
                rr = g.radius * 3600.
                pa = g.pa
                if not np.isfinite(pa):
                    pa = 0.
                logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(rr, g.ba, pa)
                gal = ExpGalaxy(RaDecPos(g.ra, g.dec),
                                NanoMaggies(order=bands, **fluxes),
                                LegacyEllipseWithPriors(logr, ee1, ee2))
                gal.isForcedLargeGalaxy = True
                refcat.append(gal)
            else:
                assert(False)

        for src in refcat:
            src.is_reference_source = True

    return refs, refcat


def read_gaia(targetwcs):
    '''
    *targetwcs* here should include margin
    '''
    from legacypipe.gaiacat import GaiaCatalog

    gaia = GaiaCatalog().get_catalog_in_wcs(targetwcs)
    print('Got Gaia stars:', gaia)

    # DJS, [decam-chatter 5486] Solved! GAIA separation of point sources
    #   from extended sources
    # Updated for Gaia DR2 by Eisenstein,
    # [decam-data 2770] Re: [desi-milkyway 639] GAIA in DECaLS DR7
    # But shifted one mag to the right in G.
    gaia.G = gaia.phot_g_mean_mag
    gaia.pointsource = np.logical_or(
        (gaia.G <= 19.) * (gaia.astrometric_excess_noise < 10.**0.5),
        (gaia.G >= 19.) * (gaia.astrometric_excess_noise < 10.**(0.5 + 0.2*(gaia.G - 19.))))

    # ok,xx,yy = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
    # margin = 10
    # H,W = targetwcs.shape
    # gaia.cut(ok * (xx > -margin) * (xx < W+margin) *
    #           (yy > -margin) * (yy < H+margin))
    # print('Cut to', len(gaia), 'Gaia stars within brick')
    # del ok,xx,yy

    # Gaia version?
    gaiaver = int(os.getenv('GAIA_CAT_VER', '1'))
    print('Assuming Gaia catalog Data Release', gaiaver)
    gaia_release = 'G%i' % gaiaver
    gaia.ref_cat = np.array([gaia_release] * len(gaia))
    gaia.ref_id  = gaia.source_id
    gaia.pmra_ivar  = 1./gaia.pmra_error **2
    gaia.pmdec_ivar = 1./gaia.pmdec_error**2
    gaia.parallax_ivar = 1./gaia.parallax_error**2
    # mas -> deg
    gaia.ra_ivar  = 1./(gaia.ra_error  / 1000. / 3600.)**2
    gaia.dec_ivar = 1./(gaia.dec_error / 1000. / 3600.)**2

    for c in ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']:
        gaia.delete_column(c)
    for c in ['pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar']:
        X = gaia.get(c)
        X[np.logical_not(np.isfinite(X))] = 0.

    # radius to consider affected by this star --
    # FIXME -- want something more sophisticated here!
    # (also see tycho.radius below)
    # This is in degrees and the magic 0.262 (indeed the whole
    # relation) is from eyeballing a radius-vs-mag plot that was in
    # pixels; that is unrelated to the present targetwcs pixel scale.
    gaia.radius = np.minimum(1800., 150. * 2.5**((11. - gaia.G)/3.)) * 0.262/3600.

    return gaia

def read_tycho2(survey, targetwcs):
    from astrometry.libkd.spherematch import tree_open, tree_search_radec
    tycho2fn = survey.find_file('tycho2')
    radius = 1.
    ra,dec = targetwcs.radec_center()
    # fitscopy /data2/catalogs-fits/TYCHO2/tycho2.fits"[col tyc1;tyc2;tyc3;ra;dec;sigma_ra;sigma_dec;mean_ra;mean_dec;pm_ra;pm_dec;sigma_pm_ra;sigma_pm_dec;epoch_ra;epoch_dec;mag_bt;mag_vt;mag_hp]" /tmp/tycho2-astrom.fits
    # startree -i /tmp/tycho2-astrom.fits -o ~/cosmo/work/legacysurvey/dr7/tycho2.kd.fits -P -k -n stars -T
    # John added the "isgalaxy" flag 2018-05-10, from the Metz & Geffert (04) catalog.
    kd = tree_open(tycho2fn, 'stars')
    I = tree_search_radec(kd, ra, dec, radius)
    print(len(I), 'Tycho-2 stars within', radius, 'deg of RA,Dec (%.3f, %.3f)' % (ra,dec))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    tycho = fits_table(tycho2fn, rows=I)
    del kd
    if 'isgalaxy' in tycho.get_columns():
        tycho.cut(tycho.isgalaxy == 0)
        print('Cut to', len(tycho), 'Tycho-2 stars on isgalaxy==0')
    else:
        print('Warning: no "isgalaxy" column in Tycho-2 catalog')
    # ok,xx,yy = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    # margin = 10
    # H,W = targetwcs.shape
    # tycho.cut(ok * (xx > -margin) * (xx < W+margin) *
    #           (yy > -margin) * (yy < H+margin))
    # print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    # del ok,xx,yy

    tycho.ref_cat = np.array(['T2'] * len(tycho))
    # tyc1: [1,9537], tyc2: [1,12121], tyc3: [1,3]
    tycho.ref_id = (tycho.tyc1.astype(np.int64)*1000000 +
                    tycho.tyc2.astype(np.int64)*10 +
                    tycho.tyc3.astype(np.int64))
    tycho.pmra_ivar = 1./tycho.sigma_pm_ra**2
    tycho.pmdec_ivar = 1./tycho.sigma_pm_dec**2
    tycho.ra_ivar  = 1./tycho.sigma_ra **2
    tycho.dec_ivar = 1./tycho.sigma_dec**2

    tycho.rename('pm_ra', 'pmra')
    tycho.rename('pm_dec', 'pmdec')
    tycho.mag = tycho.mag_vt
    tycho.mag[tycho.mag == 0] = tycho.mag_hp[tycho.mag == 0]

    # See note on gaia.radius above -- don't change the 0.262 to
    # targetwcs.pixel_scale()!
    tycho.radius = np.minimum(1800., 150. * 2.5**((11. - tycho.mag)/3.)) * 0.262/3600.
    
    for c in ['tyc1', 'tyc2', 'tyc3', 'mag_bt', 'mag_vt', 'mag_hp',
              'mean_ra', 'mean_dec', #'epoch_ra', 'epoch_dec',
              'sigma_pm_ra', 'sigma_pm_dec', 'sigma_ra', 'sigma_dec']:
        tycho.delete_column(c)
    for c in ['pmra', 'pmdec', 'pmra_ivar', 'pmdec_ivar']:
        X = tycho.get(c)
        X[np.logical_not(np.isfinite(X))] = 0.

    # add Gaia-style columns
    # No parallaxes in Tycho-2
    tycho.parallax = np.zeros(len(tycho), np.float32)
    # Arrgh, Tycho-2 has separate epoch_ra and epoch_dec.
    # Move source to the mean epoch.
    # FIXME -- check this!!
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

    return tycho

def read_large_galaxies(survey, targetwcs):
    from astrometry.libkd.spherematch import tree_open, tree_search_radec

    galfn = survey.find_file('large-galaxies')
    radius = 1.
    rc,dc = targetwcs.radec_center()

    kd = tree_open(galfn, 'largegals')
    I = tree_search_radec(kd, rc, dc, radius)
    print(len(I), 'large galaxies within', radius, 'deg of RA,Dec (%.3f, %.3f)' % (rc,dc))
    if len(I) == 0:
        return None
    # Read only the rows within range.
    galaxies = fits_table(galfn, rows=I, columns=['ra', 'dec', 'd25', 'mag', 'lslga_id', 'ba', 'pa'])
    del kd
    ok,xx,yy = targetwcs.radec2pixelxy(galaxies.ra, galaxies.dec)
    H,W = targetwcs.shape

    # # D25 is diameter in arcmin
    # pixsizes = gals.d25 * (60./2.) / targetwcs.pixel_scale()
    # gals.cut(ok * (xx > -pixsizes) * (xx < W+pixsizes) *
    #          (yy > -pixsizes) * (yy < H+pixsizes))
    # print('Cut to', len(gals), 'large galaxies touching brick')
    # del ok,xx,yy,pixsizes
    # if len(gals) == 0:
    #     return None,None
    galaxies.radius = galaxies.d25 / 2. / 60.
    # John told me to do this
    galaxies.radius *= 1.2
    galaxies.delete_column('d25')
    galaxies.rename('lslga_id', 'ref_id')
    galaxies.ref_cat = np.array(['L2'] * len(galaxies))
    galaxies.islargegalaxy = np.ones(len(galaxies), bool)
    return galaxies

def read_star_clusters(targetwcs):
    """
    Code to regenerate the NGC-star-clusters-fits catalog:

    wget https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/NGC.csv

    import os
    import numpy as np
    import numpy.ma as ma
    from astropy.io import ascii
    from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec
    import desimodel.io
    import desimodel.footprint
        
    names = ('name', 'type', 'ra_hms', 'dec_dms', 'const', 'majax', 'minax',
             'pa', 'bmag', 'vmag', 'jmag', 'hmag', 'kmag', 'sbrightn', 'hubble',
             'cstarumag', 'cstarbmag', 'cstarvmag', 'messier', 'ngc', 'ic',
             'cstarnames', 'identifiers', 'commonnames', 'nednotes', 'ongcnotes')
    NGC = ascii.read('NGC.csv', delimiter=';', names=names)
  
    ra, dec = [], []
    for _ra, _dec in zip(ma.getdata(NGC['ra_hms']), ma.getdata(NGC['dec_dms'])):
        ra.append(hmsstring2ra(_ra.replace('h', ':').replace('m', ':').replace('s','')))
        dec.append(dmsstring2dec(_dec.replace('d', ':').replace('m', ':').replace('s','')))
    NGC['ra'] = ra
    NGC['dec'] = dec
        
    objtype = np.char.strip(ma.getdata(NGC['type']))

    # Keep all globular clusters and planetary nebulae
    keeptype = ('PN', 'GCl')
    keep = np.zeros(len(NGC), dtype=bool)
    for otype in keeptype:
        ww = [otype == tt for tt in objtype]
        keep = np.logical_or(keep, ww)
    print(np.sum(keep))

    clusters = NGC[keep]
    clusters.write('NGC-star-clusters.fits', overwrite=True)

    # Code to help visually check all open clusters that are in the DESI footprint.
    checktype = ('OCl', 'Cl+N')
    check = np.zeros(len(NGC), dtype=bool)
    for otype in checktype:
        ww = [otype == tt for tt in objtype]
        check = np.logical_or(check, ww)
    check_clusters = NGC[check] # 845 of them
    
    tiles = desimodel.io.load_tiles(onlydesi=True)
    indesi = desimodel.footprint.is_point_in_desi(tiles, ma.getdata(clusters['ra']),
                                                  ma.getdata(clusters['dec']))
    print(np.sum(indesi))

    # Write out a catalog, load it into the viewer and look at each of them.
    check_clusters[['ra', 'dec', 'name']][indesi].write('check.fits', overwrite=True) # 25 of them
    
    """
    from pkg_resources import resource_filename
    from astrometry.util.starutil_numpy import degrees_between

    clusterfile = resource_filename('legacypipe', 'data/NGC-star-clusters.fits')
    print('Reading {}'.format(clusterfile))
    clusters = fits_table(clusterfile, columns=['ra', 'dec', 'majax', 'type'])
    clusters.ref_id = np.arange(len(clusters))

    radius = 1.
    rc,dc = targetwcs.radec_center()
    d = degrees_between(rc, dc, clusters.ra, clusters.dec)
    clusters.cut(d < radius)
    if len(clusters) == 0:
        return None
    
    print('Cut to {} star cluster(s) within the brick'.format(len(clusters)))

    # For each cluster, add a single faint star at the same coordinates, but
    # set the isbright bit so we get all the brightstarinblob logic.
    #clusters.ref_cat = clusters.name
    clusters.ref_cat = np.array(['CL'] * len(clusters))
    clusters.mag = np.array([35])

    # Radius in degrees (from "majax" in arcmin)
    clusters.radius = clusters.majax / 60.
    clusters.radius[np.logical_not(np.isfinite(clusters.radius))] = 1./60.

    # Remove unnecessary columns but then add all the Gaia-style columns we need.
    # for c in ['name', 'type', 'ra_hms', 'dec_dms', 'const', 'majax', 'minax', 'pa',
    #           'bmag', 'vmag', 'jmag', 'hmag', 'kmag', 'sbrightn', 'hubble', 'cstarumag',
    #           'cstarbmag', 'cstarvmag', 'messier', 'ngc', 'ic', 'cstarnames', 'identifiers',
    #           'commonnames', 'nednotes', 'ongcnotes']:
    #     clusters.delete_column(c)

    # Set isbright=True
    clusters.isbright = np.zeros(len(clusters), bool)
    clusters.iscluster = np.ones(len(clusters), bool)
        
    return clusters

