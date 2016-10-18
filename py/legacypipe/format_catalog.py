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

    opt = parser.parse_args(args=args)

    T = fits_table(opt.infn)
    hdr = T.get_header()
    primhdr = fitsio.read_header(opt.infn)
    allbands = opt.allbands

    format_catalog(T, hdr, primhdr, allbands, opt.out,
                   in_flux_prefix=opt.in_flux_prefix,
                   flux_prefix=opt.flux_prefix)
    #T.writeto(opt.out)
    print('Wrote', opt.out)
    
def format_catalog(T, hdr, primhdr, allbands, outfn,
                   in_flux_prefix='', flux_prefix=''):
    
    # Retrieve the bands in this catalog.
    bands = []
    for i in range(10):
        b = hdr.get('BAND%i' % i)
        if b is None:
            break
        bands.append(b)

    # Expand out FLUX and related fields.

    B = np.array([allbands.index(band) for band in bands])

    for k in ['rchi2', 'fracflux', 'fracmasked', 'fracin', 'nobs',
              'anymask', 'allmask', 'psfsize', 'depth', 'galdepth']:
        incol = '%s%s' % (in_flux_prefix, k)
        X = T.get(incol)
        A = np.zeros((len(T), len(allbands)), X.dtype)
        A[:,B] = X
        T.delete_column(incol)
        T.set('%s%s' % (flux_prefix, k), A)

    primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                            comment='Band order in array values'))

    has_wise = 'wise_flux' in T.columns()
    has_wise_lc = 'wise_lc_flux' in T.columns()
    has_ap = 'decam_apflux' in T.columns()
    
    from tractor.sfd import SFDMap
    print('Reading SFD maps...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, T.ra, T.dec, get_ebv=True)
    T.ebv = ebv.astype(np.float32)
    ext = ext.astype(np.float32)
    decam_ext = ext[:,:len(allbands)]
    T.decam_mw_transmission = 10.**(-decam_ext / 2.5)
    if has_wise:
        wise_ext  = ext[:,len(allbands):]
        T.wise_mw_transmission  = 10.**(-wise_ext / 2.5)

    # Column ordering...
    cols = [
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'ninblob',
        'tycho2inblob', 'type', 'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by', 'bx0', 'by0', 'left_blob', 'out_of_bounds',
        'dchisq', 'ebv', 
        'cpu_source', 'cpu_blob',
        'blob_width', 'blob_height', 'blob_npix', 'blob_nimages',
        'blob_totalpix',
        'decam_flux', 'decam_flux_ivar',
        ]

    if has_ap:
        cols.extend(['decam_apflux', 'decam_apflux_resid','decam_apflux_ivar'])

    cols.extend(['decam_mw_transmission', 'decam_nobs',
        'decam_rchi2', 'decam_fracflux', 'decam_fracmasked', 'decam_fracin',
        'decam_anymask', 'decam_allmask', 'decam_psfsize',
        'decam_depth', 'decam_galdepth' ])

    if has_wise:
        cols.extend([
            'wise_flux', 'wise_flux_ivar',
            'wise_mw_transmission', 'wise_nobs', 'wise_fracflux','wise_rchi2'])

    if has_wise_lc:
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
    flux = 'nanomaggy'
    fluxiv = '1/nanomaggy^2'
    units = dict(
        ra=deg, dec=deg, ra_ivar=degiv, dec_ivar=degiv, ebv='mag',
        wise_flux=flux, wise_flux_ivar=fluxiv,
        wise_lc_flux=flux, wise_lc_flux_ivar=fluxiv,
        shapeexp_r='arcsec', shapeexp_r_ivar='1/arcsec^2',
        shapedev_r='arcsec', shapedev_r_ivar='1/arcsec^2')
    # Fields that take prefixes
    funits = dict(
        flux=flux, flux_ivar=fluxiv,
        apflux=flux, apflux_ivar=fluxiv, apflux_resid=flux,
        depth=fluxiv, galdepth=fluxiv)
    units.update([('%s%s' % (flux_prefix, k), v) for k,v in funits.items()])

    # Reformat as list aligned with cols
    units = [units.get(c, '') for c in cols]
    
    T.writeto(outfn, columns=cols, header=hdr, primheader=primhdr, units=units)
        
if __name__ == '__main__':
    main()
    
