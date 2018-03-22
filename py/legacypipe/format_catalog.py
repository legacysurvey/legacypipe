from __future__ import print_function
import sys
import os

import numpy as np

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
    
    opt = parser.parse_args(args=args)

    import fitsio
    from astrometry.util.fits import fits_table

    T = fits_table(opt.infn)
    hdr = T.get_header()
    primhdr = fitsio.read_header(opt.infn)
    allbands = opt.allbands

    format_catalog(T, hdr, primhdr, allbands, opt.out,
                   in_flux_prefix=opt.in_flux_prefix,
                   flux_prefix=opt.flux_prefix)
    T.writeto(opt.out)
    print('Wrote', opt.out)

def format_catalog(T, hdr, primhdr, allbands, outfn,
                   in_flux_prefix='', flux_prefix='',
                   write_kwargs={}, N_wise_epochs=None):
    # Retrieve the bands in this catalog.
    bands = []
    for i in range(10):
        b = primhdr.get('BAND%i' % i)
        if b is None:
            break
        b = b.strip()
        bands.append(b)
    print('Bands in this catalog:', bands)

    #allbands = ['g','r','z']

    has_wise =    'flux_w1'    in T.columns()
    has_wise_lc = 'lc_flux_w1' in T.columns()
    has_ap =      'apflux'     in T.columns()

    # Nans,Infs
    # Ivar -> 0
    ivar_nans= ['ra_ivar','dec_ivar',
                'shapeexp_r_ivar','shapeexp_e1_ivar','shapeexp_e2_ivar'] 
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
            'nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth']
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
    print('Reading SFD maps...')
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

    from legacypipe.survey import release_number
    T.release = np.zeros(len(T), np.int16) + release_number
        
    # Column ordering...
    cols = ['release', 'brickid', 'brickname', 'objid', 'brick_primary', 
            'type', 'ra', 'dec', 'ra_ivar', 'dec_ivar',
            'bx', 'by', 'dchisq', 'ebv', 'mjd_min', 'mjd_max',
            'ref_cat', 'ref_id', 'pmra', 'pmdec', 'parallax',
            'pmra_ivar', 'pmdec_ivar', 'parallax_ivar', 'ref_epoch', ]
    def add_fluxlike(c):
        for b in allbands:
            cols.append('%s%s_%s' % (flux_prefix, c, b))
    def add_wiselike(c, bands=wbands):
        for b in bands:
            cols.append('%s_%s' % (c, b))
            
    add_fluxlike('flux')
    if has_wise:
        add_wiselike('flux')
    add_fluxlike('flux_ivar')
    if has_wise:
        add_wiselike('flux_ivar')
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
        cols.append('wise_coadd_id')
    if has_wise_lc:
        lc_cols = ['lc_flux', 'lc_flux_ivar', 'lc_nobs', 'lc_fracflux',
                   'lc_rchisq','lc_mjd']
        for c in lc_cols:
            add_wiselike(c, bands=['w1','w2'])
        # Cut down to a fixed number of WISE time-resolved epochs?
        if N_wise_epochs is not None:
            for col in lc_cols:
                for band in ['w1','w2']:
                    colname = col + '_' + band
                    # Cut or pad old value to have N_wise_epochs-length arrays
                    oldval = T.get(colname)
                    n,ne = oldval.shape
                    newval = np.zeros((n,N_wise_epochs), oldval.dtype)
                    ncopy = min(N_wise_epochs,ne)
                    newval[:, :ncopy] = oldval[:,:ncopy]
                    T.set(colname, newval)
    cols.extend([
        'fracdev', 'fracdev_ivar',
        'shapeexp_r', 'shapeexp_r_ivar',
        'shapeexp_e1', 'shapeexp_e1_ivar',
        'shapeexp_e2', 'shapeexp_e2_ivar',
        'shapedev_r',  'shapedev_r_ivar',
        'shapedev_e1', 'shapedev_e1_ivar',
        'shapedev_e2', 'shapedev_e2_ivar',])

    print('Columns:', cols)
    print('T columns:', T.columns())
    
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
        shapeexp_r=arcsec, shapeexp_r_ivar='1/arcsec^2',
        shapedev_r=arcsec, shapedev_r_ivar='1/arcsec^2')
    # WISE fields
    wunits = dict(flux=flux, flux_ivar=fluxiv,
                  lc_flux=flux, lc_flux_ivar=fluxiv)
    # Fields that take prefixes (and have bands)
    funits = dict(
        flux=flux, flux_ivar=fluxiv,
        apflux=flux, apflux_ivar=fluxiv, apflux_resid=flux,
        psfdepth=fluxiv, galdepth=fluxiv, psfsize=arcsec)
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
        
if __name__ == '__main__':
    main()
    
