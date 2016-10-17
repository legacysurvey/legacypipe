from __future__ import print_function
import sys
import os

from astrometry.util.fits import fits_table, merge_tables

from tractor.sfd import SFDMap

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

    allbands = opt.allbands
    
    # Retrieve the bands in this catalog.
    bands = []
    for i in range(10):
        b = hdr.get('BAND%i' % i)
        if b is None:
            break
        bands.append(b)

    # Expand out FLUX and related fields.

    B = np.array([allbands.index(band) for band in bands])

    atypes = dict(nobs=np.uint8, anymask=TT.anymask.dtype,
                  allmask=TT.allmask.dtype)

    for k in ['rchi2', 'fracflux', 'fracmasked', 'fracin', 'nobs',
              'anymask', 'allmask', 'psfsize', 'depth', 'galdepth']:
        incol = '%s%s' % (opt.in_flux_prefix, k)
        X = TT.get(incol)
        A = np.zeros((len(TT), len(allbands)), X.dtype)
        A[:,B] = X
        TT.delete_column(incol)
        TT.set('%s%s' % (opt.flux_prefix, k), A)
        
    
if __name__ == '__main__':
    main()
    
