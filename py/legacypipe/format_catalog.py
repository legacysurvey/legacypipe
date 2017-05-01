from __future__ import print_function
import sys
import os

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables

from catalog import prepare_fits_catalog

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
    parser.add_argument('--dr4', default=False, action='store_true',
                        help='MzLS+BASS DR4 format')
    
    opt = parser.parse_args(args=args)

    T = fits_table(opt.infn)
    hdr = T.get_header()
    primhdr = fitsio.read_header(opt.infn)
    allbands = opt.allbands

    format_catalog(T, hdr, primhdr, allbands, opt.out,
                   in_flux_prefix=opt.in_flux_prefix,
                   flux_prefix=opt.flux_prefix,
                   dr4=opt.dr4)
    #T.writeto(opt.out)
    print('Wrote', opt.out)

def format_catalog(T, hdr, primhdr, allbands, outfn,
                   in_flux_prefix='', flux_prefix='',
                   dr4=False,
                   write_kwargs={}):
    # Retrieve the bands in this catalog.
    bands = []
    for i in range(10):
        b = primhdr.get('BAND%i' % i)
        if b is None:
            break
        b = b.strip()
        bands.append(b)
    print('Bands in this catalog:', bands)

    primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                            comment='Band order in array values'))

    has_wise =    'wise_flux'    in T.columns()
    has_wise_lc = 'wise_lc_flux' in T.columns()
    has_ap =      'apflux'       in T.columns()

    # Expand out FLUX and related fields from grz arrays to 'allbands'
    # (eg, ugrizY) arrays.
    B = np.array([allbands.index(band) for band in bands])
    keys = ['flux', 'flux_ivar', 'rchi2', 'fracflux', 'fracmasked', 'fracin',
            'nobs', 'anymask', 'allmask', 'psfsize', 'depth', 'galdepth']
    if has_ap:
        keys.extend(['apflux', 'apflux_resid', 'apflux_ivar'])
    for k in keys:
        incol = '%s%s' % (in_flux_prefix, k)
        X = T.get(incol)
        # print('Column', k, 'has shape', X.shape)

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

        if dr4:
            # FLUX_b for each band, rather than array columns.
            for i,b in enumerate(allbands):
                T.set('%s%s_%s' % (flux_prefix, k, b), A[:,i])
        else:
            T.set('%s%s' % (flux_prefix, k), A)

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
    if dr4:
        # # No MW_TRANSMISSION_* columns at all
        # for i,b in enumerate(allbands):
        #         col = 'mw_transmission_%s' % b
        #         T.set(col, 10.**(-decam_ext[:,i] / 2.5))
        #         trans_cols_opt.append(col)
        #     if has_wise:
        #         for i,b in enumerate(wbands):
        #             col = 'mw_transmission_%s' % b
        #             T.set(col, 10.**(-wise_ext[:,i] / 2.5))
        #             trans_cols_wise.append(col)
        pass
    else:
        T.decam_mw_transmission = 10.**(-decam_ext / 2.5)
        trans_cols_opt.append('decam_mw_transmission')
        if has_wise:
            T.wise_mw_transmission  = 10.**(-wise_ext / 2.5)
            trans_cols_wise.append('wise_mw_transmission')


    # Column ordering...
    cols = []
    if dr4:
        cols.append('release')
        #T.release = np.array(['MzLS+BASS-DR4'] * len(T))
        T.release = np.zeros(len(T), np.int16) + 4000
        
    cols.extend([
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'ninblob',
        'tycho2inblob', 'type', 'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by'])
    if not dr4:
        cols.extend(['bx0', 'by0', 'left_blob', 'out_of_bounds',])
    cols.extend(['dchisq', 'ebv',])
    if not dr4:
        cols.extend(['cpu_source', 'cpu_blob',
                     'blob_width', 'blob_height', 'blob_npix', 'blob_nimages',
                     'blob_totalpix',])
    cols.extend(['mjd_min', 'mjd_max'])

    if dr4:
        cambits = { 'decam': 0x1,
                    'mosaic': 0x2,
                    '90prime': 0x4,
                    }
        for b in allbands:
            cams = 0
            camstring = primhdr.get('CAMS_%s' % b.upper(), '')
            camstring.strip()
            camnames = camstring.split(' ')
            camnames = [c for c in camnames if len(c)]
            print('Camera names for', b, '=', camnames)
            for c in camnames:
                cams += cambits[c]
            col = 'camera_%s' % b
            T.set(col, np.zeros(len(T), np.int32) + cams)
            cols.append(col)
    if dr4:
        cc = ['flux', 'flux_ivar']
        if has_ap:
            cc.extend(['apflux', 'apflux_resid','apflux_ivar'])
        for c in cc:
            for b in allbands:
                cols.append('%s%s_%s' % (flux_prefix, c, b))
    else:
        cols.extend([flux_prefix + c for c in ['flux', 'flux_ivar']])
        if has_ap:
            cols.extend([flux_prefix + c for c in
                         ['apflux', 'apflux_resid','apflux_ivar']])

    cols.extend(trans_cols_opt)

    if dr4:
        cc = ['nobs', 'rchi2', 'fracflux', 'fracmasked', 'fracin', 'anymask',
              'allmask', 'psfsize', 'depth', 'galdepth']
        for c in cc:
            for b in allbands:
                cols.append('%s%s_%s' % (flux_prefix, c, b))
    else:
        cols.extend([flux_prefix + c for c in [
            'nobs', 'rchi2', 'fracflux', 'fracmasked', 'fracin', 'anymask',
            'allmask', 'psfsize', 'depth', 'galdepth']])

    if has_wise:
        cols.append('wise_coadd_id')
        if dr4:
            cc = ['wise_flux', 'wise_flux_ivar', 'wise_mask',
                  'wise_nobs', 'wise_fracflux','wise_rchi2']
            for c in cc:
                cbare = c.replace('wise_','')
                thiswbands = wbands
                if cbare == 'mask':
                    cbare = 'wisemask'
                    thiswbands = wbands[:2]
                X = T.get(c)
                for i,b in enumerate(thiswbands):
                    col = '%s_%s' % (cbare, b)
                    T.set(col, X[:,i])
                    cols.append(col)
            cols.extend(trans_cols_wise)
            
        else:
            cols.extend(['wise_flux', 'wise_flux_ivar', 'wise_mask'])
            cols.extend(trans_cols_wise)
            cols.extend(['wise_nobs', 'wise_fracflux','wise_rchi2'])

    if has_wise_lc:
        if dr4:
            cc = ['wise_lc_flux', 'wise_lc_flux_ivar', 'wise_lc_nobs',
                  'wise_lc_fracflux', 'wise_lc_rchi2','wise_lc_mjd']
            for c in cc:
                cbare = c.replace('wise_','')
                X = T.get(c)
                for i,b in enumerate(wbands[:2]):
                    col = '%s_%s' % (cbare, b)
                    T.set(col, X[:,i,:])
                    cols.append(col)
        else:
            cols.extend([
                'wise_lc_flux', 'wise_lc_flux_ivar',
                'wise_lc_nobs', 'wise_lc_fracflux', 'wise_lc_rchi2','wise_lc_mjd'])

    cols.extend([
        'fracdev', 'fracdev_ivar', 'shapeexp_r', 'shapeexp_r_ivar',
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
    deg='deg'
    degiv='1/deg^2'
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
        depth=fluxiv, galdepth=fluxiv, psfsize=arcsec)
    # add prefixes
    units.update([('%s%s' % (flux_prefix, k), v) for k,v in funits.items()])
    # add bands
    for b in allbands:
        units.update([('%s%s_%s' % (flux_prefix, k, b), v)
                      for k,v in funits.items()])
    # add WISE bands
    for b in wbands:
        if dr4:
            units.update([('%s_%s' % (k, b), v)
                          for k,v in wunits.items()])
        else:
            units.update([('%s%s' % ('wise_', k), v)
                          for k,v in wunits.items()])
    
    # Create a list of units aligned with 'cols'
    units = [units.get(c, '') for c in cols]

    T.writeto(outfn, columns=cols, header=hdr, primheader=primhdr, units=units,
              **write_kwargs)
        
if __name__ == '__main__':
    main()
    
