import numpy as np
import os
from glob import glob
from astrometry.util.fits import fits_table, merge_tables

'''
For ODIN, pull in the grz forced photometry and summarize it.
'''

def summarize_forced_phot(tractorfns, forced_dir, bands):
    fns = tractorfns
    fns.sort()

    TT = []
    for fn in fns:
        T = fits_table(fn)
        for band in bands:
            T.set('forced_mean_flux_'+band, np.zeros(len(T), np.float32))
            T.set('forced_mean_flux_ivar_'+band, np.zeros(len(T), np.float32))
            T.set('forced_mean_flux_chi2_'+band, np.zeros(len(T), np.float32))
            T.set('forced_nobs_'+band, np.zeros(len(T), np.int32))

        # tractor BRI dir & filename (148/tractor-1484p023.fits)
        tfn = os.path.basename(fn)
        bridir = os.path.basename(os.path.dirname(fn))
        # forced fn:
        ffn = os.path.join(forced_dir, bridir, tfn.replace('tractor-', 'forced-'))
        if not os.path.exists(ffn):
            print('Skipping:', ffn)
            continue
        F = fits_table(ffn)
        FF = fits_table(ffn, hdu=2)
        print(fn, 'read', len(T), len(F), len(FF))
        assert(len(F) == len(T))
        I = np.flatnonzero(T.brick_primary)
        T.cut(I)
        F.cut(I)

        for i,f in enumerate(F):
            for band in bands:
                i0 = f.get('index_'+band)
                n = f.get('nobs_'+band)
                i1 = i0 + n
                flux = FF.flux[i0:i1]
                fiv  = FF.flux_ivar[i0:i1]
                if np.sum(fiv) == 0:
                    continue
                # Only keep measurements with iv>0
                if np.any(fiv == 0):
                    I = np.flatnonzero(fiv > 0)
                    flux = flux[I]
                    fiv  = fiv [I]
                meanflux = np.sum(flux * fiv) / np.sum(fiv)
                T.get('forced_nobs_'+band)[i] = len(fiv)
                T.get('forced_mean_flux_'+band)[i] = meanflux
                T.get('forced_mean_flux_ivar_'+band)[i] = np.sum(fiv)
                T.get('forced_mean_flux_chi2_'+band)[i] = np.sum((flux - meanflux)**2 * fiv)
        for band in bands:
            print('Mean nobs', band, ':', np.mean(T.get('forced_nobs_'+band)))
        TT.append(T)

    T = merge_tables(TT)
    return T

if __name__ == '__main__':

    #tfns = glob('odin/2band/tractor/*/tractor-*.fits')
    #forced_dir = 'odin/2band/forced-deep/forced-brick'
    #bands = ['g','r','z']
    #outfn = 'odin/2band/odin-2band+deep-cosmos.fits'

    #tfns = glob('odin/2band/tractor/*/tractor-*.fits')
    #forced_dir = 'odin/2band/forced-u/forced-brick'
    #bands = ['u']
    #outfn = 'odin/2band/odin-2band+u.fits'

    fdir = '/global/cfs/cdirs/cosmo/work/users/dstn/dr9.1.1b/forced/forced-brick'
    tdir = '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9.1.1/tractor'
    ffns = glob(fdir + '/*/forced-*.fits')
    print(len(ffns), 'forced-phot bricks')
    tfns = []
    for fn in ffns:
        tfn = fn.replace(fdir, tdir)
        tfn = os.path.join(os.path.dirname(tfn), os.path.basename(tfn).replace('forced-','tractor-'))
        if not os.path.exists(tfn):
            print('Does not exist:', tfn)
            sys.exit(-1)
        tfns.append(tfn)
    forced_dir = fdir
    bands = ['u']
    outfn = 'dr9.1.1+decam-u.fits'

    T = summarize_forced_phot(tfns, forced_dir, bands)
    T.writeto(outfn)
