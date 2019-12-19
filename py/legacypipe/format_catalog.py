from __future__ import print_function

import numpy as np

import logging
logger = logging.getLogger('legacypipe.format_catalog')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infn',
                        help='Input intermediate tractor catalog file')
    parser.add_argument('--out', help='Output catalog filename')
    parser.add_argument('--allbands', default='ugrizY',
                        help='Full set of bands to expand arrays out to')
    parser.add_argument('--flux-prefix', default='',
                        help='Prefix on FLUX etc columns (eg, "decam_" to match DR3) for output file')
    parser.add_argument('--in-flux-prefix', default='',
                        help='Prefix on FLUX etc columns (eg, "decam_" to match DR3) for input file')
    parser.add_argument('--release', type=int, default=8000)

    opt = parser.parse_args(args=args)

    import fitsio
    from astrometry.util.fits import fits_table

    T = fits_table(opt.infn)
    hdr = T.get_header()
    primhdr = fitsio.read_header(opt.infn)
    allbands = opt.allbands

    format_catalog(T, hdr, primhdr, allbands, opt.out, opt.release,
                   in_flux_prefix=opt.in_flux_prefix,
                   flux_prefix=opt.flux_prefix)
    T.writeto(opt.out)
    print('Wrote', opt.out)

def format_catalog(T, hdr, primhdr, allbands, outfn, release,
                   in_flux_prefix='', flux_prefix='',
                   write_kwargs=None, N_wise_epochs=None,
                   motions=True, gaia_tagalong=False):
    if write_kwargs is None:
        write_kwargs = {}
    # Retrieve the bands in this catalog.
    bands = []
    for i in range(10):
        b = primhdr.get('BAND%i' % i)
        if b is None:
            break
        b = b.strip()
        bands.append(b)
    debug('Bands in this catalog:', bands)

    has_wise =    'flux_w1'    in T.columns()
    has_wise_lc = 'lc_flux_w1' in T.columns()
    has_ap =      'apflux'     in T.columns()

    # Nans,Infs
    # Ivar -> 0
    ivar_nans= ['ra_ivar','dec_ivar',
                'shape_r_ivar','shape_e1_ivar','shape_e2_ivar']
    for key in ivar_nans:
        ind= np.isfinite(T.get(key)) == False
        if np.any(ind):
            T.get(key)[ind]= 0.
    # Other --> NaN (PostgreSQL can work with NaNs)
    other_nans= ['dchisq','rchisq','mjd_min','mjd_max']
    for key in other_nans:
        ind= np.isfinite(T.get(key)) == False
        if np.any(ind):
            T.get(key)[ind]= np.nan

    # Expand out FLUX and related fields from grz arrays to 'allbands'
    # (eg, ugrizY) arrays.
    B = np.array([allbands.index(band) for band in bands])
    keys = ['flux', 'flux_ivar', 'rchisq', 'fracflux', 'fracmasked', 'fracin',
            'nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth',
            'fiberflux', 'fibertotflux']
    if has_ap:
        keys.extend(['apflux', 'apflux_resid', 'apflux_ivar'])

    for k in keys:
        incol = '%s%s' % (in_flux_prefix, k)
        X = T.get(incol)
        # Handle array columns (eg, apflux)
        sh = X.shape
        if len(sh) == 3:
            nt,nb,N = sh
            A = np.zeros((len(T), len(allbands), N), X.dtype)
            A[:,B,:] = X
        else:
            A = np.zeros((len(T), len(allbands)), X.dtype)
            # If there is only one band, these can show up as scalar arrays.
            if len(sh) == 1:
                A[:,B] = X[:,np.newaxis]
            else:
                A[:,B] = X
        T.delete_column(incol)

        # FLUX_b for each band, rather than array columns.
        for i,b in enumerate(allbands):
            T.set('%s%s_%s' % (flux_prefix, k, b), A[:,i])

    from tractor.sfd import SFDMap
    info('Reading SFD maps...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, T.ra, T.dec, get_ebv=True)
    T.ebv = ebv.astype(np.float32)
    ext = ext.astype(np.float32)
    decam_ext = ext[:,:len(allbands)]
    if has_wise:
        wise_ext  = ext[:,len(allbands):]

    wbands = ['w1','w2','w3','w4']

    trans_cols_opt  = []
    trans_cols_wise = []

    # No MW_TRANSMISSION_* columns at all
    for i,b in enumerate(allbands):
        col = 'mw_transmission_%s' % b
        T.set(col, 10.**(-decam_ext[:,i] / 2.5))
        trans_cols_opt.append(col)
    if has_wise:
        for i,b in enumerate(wbands):
            col = 'mw_transmission_%s' % b
            T.set(col, 10.**(-wise_ext[:,i] / 2.5))
            trans_cols_wise.append(col)

    T.release = np.zeros(len(T), np.int16) + release

    # Column ordering...
    cols = ['release', 'brickid', 'brickname', 'objid', 'brick_primary',
            'brightblob', 'maskbits', 'iterative',
            'type', 'ra', 'dec', 'ra_ivar', 'dec_ivar',
            'bx', 'by', 'dchisq', 'ebv', 'mjd_min', 'mjd_max',
            'ref_cat', 'ref_id']
    if motions:
        cols.extend(['pmra', 'pmdec', 'parallax',
            'pmra_ivar', 'pmdec_ivar', 'parallax_ivar', 'ref_epoch',])
        # Add zero columns if these are missing (eg, if there are no
        # Tycho-2 or Gaia stars in the brick).
        tcols = T.get_columns()
        for c in cols:
            if not c in tcols:
                T.set(c, np.zeros(len(T), np.float32))

    if gaia_tagalong:
        gaia_cols = [#('source_id', np.int64),  # already have this in ref_id
            ('pointsource', bool),  # did we force it to be a point source?
            ('phot_g_mean_mag', np.float32),
            ('phot_g_mean_flux_over_error', np.float32),
            ('phot_g_n_obs', np.int16),
            ('phot_bp_mean_mag', np.float32),
            ('phot_bp_mean_flux_over_error', np.float32),
            ('phot_bp_n_obs', np.int16),
            ('phot_rp_mean_mag', np.float32),
            ('phot_rp_mean_flux_over_error', np.float32),
            ('phot_rp_n_obs', np.int16),
            ('phot_variable_flag', bool),
            ('astrometric_excess_noise', np.float32),
            ('astrometric_excess_noise_sig', np.float32),
            ('astrometric_n_obs_al', np.int16),
            ('astrometric_n_good_obs_al', np.int16),
            ('astrometric_weight_al', np.float32),
            ('duplicated_source', bool),
            ('a_g_val', np.float32),
            ('e_bp_min_rp_val', np.float32),
            ('phot_bp_rp_excess_factor', np.float32),
            ('astrometric_sigma5d_max', np.float32),
            ('astrometric_params_solved', np.uint8),
        ]
        tcols = T.get_columns()
        for c,t in gaia_cols:
            gc = 'gaia_' + c
            if not c in tcols:
                T.set(gc, np.zeros(len(T), t))
            else:
                T.rename(c, gc)
            cols.append(gc)

    def add_fluxlike(c):
        for b in allbands:
            cols.append('%s%s_%s' % (flux_prefix, c, b))
    def add_wiselike(c, bands=None):
        if bands is None:
            bands = wbands
        for b in bands:
            cols.append('%s_%s' % (c, b))

    add_fluxlike('flux')
    if has_wise:
        add_wiselike('flux')
    add_fluxlike('flux_ivar')
    if has_wise:
        add_wiselike('flux_ivar')
    add_fluxlike('fiberflux')
    add_fluxlike('fibertotflux')
    if has_ap:
        for c in ['apflux', 'apflux_resid','apflux_ivar']:
            add_fluxlike(c)

    cols.extend(trans_cols_opt)
    cols.extend(trans_cols_wise)

    for c in ['nobs', 'rchisq', 'fracflux']:
        add_fluxlike(c)
        if has_wise:
            add_wiselike(c)
    for c in ['fracmasked', 'fracin', 'anymask', 'allmask']:
        add_fluxlike(c)
    if has_wise:
        for i,b in enumerate(wbands[:2]):
            col = 'wisemask_%s' % (b)
            T.set(col, T.wise_mask[:,i])
            cols.append(col)
    for c in ['psfsize', 'psfdepth', 'galdepth']:
        add_fluxlike(c)
    if has_wise:
        add_wiselike('psfdepth')

    if has_wise:
        cols.append('wise_coadd_id')
    if has_wise_lc:
        lc_cols = ['lc_flux', 'lc_flux_ivar', 'lc_nobs', 'lc_fracflux',
                   'lc_rchisq','lc_mjd']
        for c in lc_cols:
            add_wiselike(c, bands=['w1','w2'])
        # Cut down to a fixed number of WISE time-resolved epochs?
        if N_wise_epochs is not None:

            # mapping from (nobs, mjd) to indices to keep
            keep_epochs = {}
            newvals = {}

            for band in ['w1', 'w2']:
                # initialize new (cut) arrays
                for col in lc_cols:
                    colname = col + '_' + band
                    oldval = T.get(colname)
                    n,ne = oldval.shape
                    newval = np.zeros((n, N_wise_epochs), oldval.dtype)
                    newvals[colname] = newval

                lc_nobs = T.get('lc_nobs_%s' % band)
                lc_mjd = T.get('lc_mjd_%s' % band)
                # Check each row (source) individually, since coverage isn't
                # uniform across a brick
                for row,(nobs,mjd) in enumerate(zip(lc_nobs, lc_mjd)):
                    key = tuple(nobs) + tuple(mjd)
                    if not key in keep_epochs:
                        # required by one_lightcurve_bitmask!
                        assert(N_wise_epochs == 13)
                        I = one_lightcurve_bitmask(nobs, mjd)
                        # convert to integer index list
                        I = np.flatnonzero(I)
                        keep_epochs[key] = I
                    else:
                        I = keep_epochs[key]

                    for col in lc_cols:
                        colname = col + '_' + band
                        oldval = T.get(colname)
                        newval = newvals[colname]
                        newval[row,:len(I)] = oldval[row,I]

            for k,v in newvals.items():
                T.set(k, v)
            del keep_epochs

    cols.extend([
        'sersic',  'sersic_ivar',
        'shape_r', 'shape_r_ivar',
        'shape_e1', 'shape_e1_ivar',
        'shape_e2', 'shape_e2_ivar'])

    debug('Columns:', cols)
    debug('T columns:', T.columns())

    # match case to T.
    cc = T.get_columns()
    cclower = [c.lower() for c in cc]
    for i,c in enumerate(cols):
        if (not c in cc) and c in cclower:
            j = cclower.index(c)
            cols[i] = cc[j]

    # Units
    deg = 'deg'
    degiv = '1/deg^2'
    arcsec = 'arcsec'
    flux = 'nanomaggy'
    fluxiv = '1/nanomaggy^2'
    units = dict(
        ra=deg, dec=deg, ra_ivar=degiv, dec_ivar=degiv, ebv='mag',
        shape_r=arcsec, shape_r_ivar='1/arcsec^2')
    # WISE fields
    wunits = dict(flux=flux, flux_ivar=fluxiv,
                  lc_flux=flux, lc_flux_ivar=fluxiv)
    # Fields that take prefixes (and have bands)
    funits = dict(
        flux=flux, flux_ivar=fluxiv,
        apflux=flux, apflux_ivar=fluxiv, apflux_resid=flux,
        psfdepth=fluxiv, galdepth=fluxiv, psfsize=arcsec,
        fiberflux=flux, fibertotflux=flux)
    # add prefixes
    units.update([('%s%s' % (flux_prefix, k), v) for k,v in funits.items()])
    # add bands
    for b in allbands:
        units.update([('%s%s_%s' % (flux_prefix, k, b), v)
                      for k,v in funits.items()])
    # add WISE bands
    for b in wbands:
        units.update([('%s_%s' % (k, b), v)
                      for k,v in wunits.items()])

    # Create a list of units aligned with 'cols'
    units = [units.get(c, '') for c in cols]

    T.writeto(outfn, columns=cols, header=hdr, primheader=primhdr, units=units,
              **write_kwargs)

def one_lightcurve_bitmask(lc_nobs, lc_mjd):
    # row is a single tractor-i catalog row
    # the only columns used from within this row are:
    #     LC_NOBS_W[1-2]
    #     LC_MJD_W[1-2]
    # band should be an integer, either 1 (W1) or 2 (W2)
    #
    # return value is a 1D boolean numpy array with the same
    # length as the input row's WISE lightcurve

    #assert(band in [1, 2])
    #row['LC_NOBS_W' + str(band)]
    #row['LC_MJD_W' + str(band)]

    nobs = lc_nobs
    mjd = lc_mjd

    n_epochs = len(nobs)

    # could imagine making this a keyword arg rather than hardcoded
    # but don't want to have to test that it works for other
    # values...
    n_final = 13 # NEO5

    # if the number of epochs fits within the number available then
    # retain all epochs
    if n_epochs <= n_final:
        return np.ones(n_epochs, dtype=bool)

    # ceiling value for minimum required LC_NOBS_w[1-2]
    nobs_min_ceil = 12
    keep = np.zeros(n_epochs, dtype=bool)

    sind = np.argsort(-1.0*nobs)

    nobs_min = min(nobs[sind[n_final-1]], nobs_min_ceil)

    is_cand = (nobs >= nobs_min)

    if np.sum(is_cand) == n_final:
        keep[sind[0:n_final]] = True
        return keep

    indmin = np.argmin(mjd + np.logical_not(is_cand)*(1.0e6))
    keep[indmin] = True
    is_cand[indmin] = False
    t0 = mjd[indmin]

    for j in range(1, n_final):
        t_target = t0 + j*(365.25/2.0)
        if (t0  < 56000) and (t_target > 55593):
            t_target += (1046.0 - 365.25/2.0)

        indmin = np.argmin(np.abs(mjd - t_target) + np.logical_not(is_cand)*1e6)
        keep[indmin] = True
        is_cand[indmin] = False

    return keep

if __name__ == '__main__':
    main()
