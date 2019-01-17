import os
import numpy as np
from astrometry.util.fits import fits_table, merge_tables

def read_gaia(targetwcs):
    '''
    *margin* in degrees
    '''
    from legacypipe.gaiacat import GaiaCatalog

    ##### FIXME! -- Need stars outside the WCS!

    gaia = GaiaCatalog().get_catalog_in_wcs(targetwcs)
    print('Got Gaia stars:', gaia)
    gaia.about()

    # DJS, [decam-chatter 5486] Solved! GAIA separation of point sources
    #   from extended sources
    # Updated for Gaia DR2 by Eisenstein,
    # [decam-data 2770] Re: [desi-milkyway 639] GAIA in DECaLS DR7
    # But shifted one mag to the right in G.
    gaia.G = gaia.phot_g_mean_mag
    gaia.pointsource = np.logical_or(
        (gaia.G <= 19.) * (gaia.astrometric_excess_noise < 10.**0.5),
        (gaia.G >= 19.) * (gaia.astrometric_excess_noise < 10.**(0.5 + 0.2*(gaia.G - 19.))))

    ok,xx,yy = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
    margin = 10
    H,W = targetwcs.shape
    gaia.cut(ok * (xx > -margin) * (xx < W+margin) *
              (yy > -margin) * (yy < H+margin))
    print('Cut to', len(gaia), 'Gaia stars within brick')
    del ok,xx,yy

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
    #gaia.radius = np.minimum(1800., 150. * 2.5**((11. - gaia.G)/4.)) * 0.262/3600.
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
        tycho = []
    # Read only the rows within range.
    tycho = fits_table(tycho2fn, rows=I)
    del kd
    if 'isgalaxy' in tycho.get_columns():
        tycho.cut(tycho.isgalaxy == 0)
        print('Cut to', len(tycho), 'Tycho-2 stars on isgalaxy==0')
    else:
        print('Warning: no "isgalaxy" column in Tycho-2 catalog')
    #print('Read', len(tycho), 'Tycho-2 stars')
    ok,xx,yy = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    margin = 10
    H,W = targetwcs.shape
    tycho.cut(ok * (xx > -margin) * (xx < W+margin) *
              (yy > -margin) * (yy < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    del ok,xx,yy

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

    ## FIXME -- want something better here!!
    #

    # See note on gaia.radius above -- don't change the 0.262 to
    # targetwcs.pixel_scale()!
    #tycho.radius = np.minimum(1800., 150. * 2.5**((11. - tycho.mag)/4.)) * 0.262/3600.
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

def read_large_galaxies(survey, targetwcs, bands):
    from legacypipe.survey import LegacyEllipseWithPriors
    from tractor.galaxy import ExpGalaxy
    from tractor import NanoMaggies, RaDecPos
    from tractor.ellipses import EllipseESoft
    from astrometry.libkd.spherematch import tree_open, tree_search_radec

    galfn = survey.find_file('large-galaxies')
    radius = 1.
    rc,dc = targetwcs.radec_center()

    kd = tree_open(galfn, 'largegals')
    I = tree_search_radec(kd, rc, dc, radius)
    print(len(I), 'large galaxies within', radius, 'deg of RA,Dec (%.3f, %.3f)' % (rc,dc))
    if len(I) == 0:
        return None,None
    # Read only the rows within range.
    gals = fits_table(galfn, rows=I, columns=['ra', 'dec', 'd25', 'mag', 'lslga_id', 'ba', 'pa'])
    del kd
    ok,xx,yy = targetwcs.radec2pixelxy(gals.ra, gals.dec)
    H,W = targetwcs.shape
    # D25 is diameter in arcmin
    pixsizes = gals.d25 * (60./2.) / targetwcs.pixel_scale()
    gals.ibx = (xx - 1.).astype(int)
    gals.iby = (yy - 1.).astype(int)
    gals.cut(ok * (xx > -pixsizes) * (xx < W+pixsizes) *
             (yy > -pixsizes) * (yy < H+pixsizes))
    print('Cut to', len(gals), 'large galaxies touching brick')
    del ok,xx,yy,pixsizes
    if len(gals) == 0:
        return None,None
        
    # Instantiate a galaxy model at the position of each object.
    largecat = []
    for g in gals:
        fluxes = dict([(band, NanoMaggies.magToNanomaggies(g.mag)) for band in bands])
        assert(np.all(np.isfinite(list(fluxes.values()))))
        ss = g.d25 * 60. / 2.
        pa = g.pa
        if not np.isfinite(pa):
            pa = 0.
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(ss, g.ba, pa)
        gal = ExpGalaxy(RaDecPos(g.ra, g.dec),
                        NanoMaggies(order=bands, **fluxes),
                        LegacyEllipseWithPriors(logr, ee1, ee2))
        gal.isForcedLargeGalaxy = True
        largecat.append(gal)
    gals.radius = gals.d25 / 2. / 60.
    gals.delete_column('d25')
    gals.rename('lslga_id', 'ref_id')
    gals.ref_cat = np.array(['L2'] * len(gals))
    gals.islargegalaxy = np.ones(len(gals), bool)
    return gals, largecat

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
  
    objtype = np.char.strip(ma.getdata(NGC['type']))
    keeptype = ('PN', 'OCl', 'GCl', 'Cl+N')
    keep = np.zeros(len(NGC), dtype=bool)
    for otype in keeptype:
        ww = [otype == tt for tt in objtype]
        keep = np.logical_or(keep, ww)

    clusters = NGC[keep]

    ra, dec = [], []
    for _ra, _dec in zip(ma.getdata(clusters['ra_hms']), ma.getdata(clusters['dec_dms'])):
        ra.append(hmsstring2ra(_ra.replace('h', ':').replace('m', ':').replace('s','')))
        dec.append(dmsstring2dec(_dec.replace('d', ':').replace('m', ':').replace('s','')))
    clusters['ra'] = ra
    clusters['dec'] = dec
        
    tiles = desimodel.io.load_tiles(onlydesi=True)
    indesi = desimodel.footprint.is_point_in_desi(tiles, ma.getdata(clusters['ra']),
                                                  ma.getdata(clusters['dec']))
    print(np.sum(indesi))
    clusters.write('NGC-star-clusters.fits', overwrite=True)

    """
    from pkg_resources import resource_filename

    clusterfile = resource_filename('legacypipe', 'data/NGC-star-clusters.fits')
    print('Reading {}'.format(clusterfile))
    clusters = fits_table(clusterfile)

    ok, xx, yy = targetwcs.radec2pixelxy(clusters.ra, clusters.dec)
    margin = 10
    H, W = targetwcs.shape
    clusters.cut( ok * (xx > -margin) * (xx < W+margin) *
                  (yy > -margin) * (yy < H+margin) )
    if len(clusters) > 0:
        print('Cut to {} star cluster(s) within the brick'.format(len(clusters)))
        del ok,xx,yy

        # For each cluster, add a single faint star at the same coordinates, but
        # set the isbright bit so we get all the brightstarinblob logic.
        clusters.ref_cat = clusters.name
        clusters.mag = np.array([35])

        # Radius in degrees (from "majax" in arcmin)
        clusters.radius = clusters.majax / 60.
        clusters.radius[np.logical_not(np.isfinite(clusters.radius))] = 1./60.

        # Remove unnecessary columns but then add all the Gaia-style columns we need.
        for c in ['name', 'type', 'ra_hms', 'dec_dms', 'const', 'majax', 'minax', 'pa',
                  'bmag', 'vmag', 'jmag', 'hmag', 'kmag', 'sbrightn', 'hubble', 'cstarumag',
                  'cstarbmag', 'cstarvmag', 'messier', 'ngc', 'ic', 'cstarnames', 'identifiers',
                  'commonnames', 'nednotes', 'ongcnotes']:
            clusters.delete_column(c)

        # Set isbright=True
        clusters.isbright = np.ones(len(clusters), bool)
        clusters.iscluster = np.ones(len(clusters), bool)
    else:
        clusters = []
        
    return clusters

