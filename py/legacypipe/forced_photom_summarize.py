import pylab as plt
import numpy as np
import os
from glob import glob
from astrometry.util.fits import fits_table, merge_tables

'''
For ODIN, pull in the grz forced photometry and summarize it.
'''

def main():
    fns = glob('odin/2band/tractor/*/tractor-*.fits')
    fns.sort()

    TT = []
    for fn in fns:
        T = fits_table(fn)
        T.cut(T.brick_primary)
        T.forced_mean_flux_g = np.zeros(len(T), np.float32)
        T.forced_mean_flux_r = np.zeros(len(T), np.float32)
        T.forced_mean_flux_z = np.zeros(len(T), np.float32)
        T.forced_mean_flux_ivar_g = np.zeros(len(T), np.float32)
        T.forced_mean_flux_ivar_r = np.zeros(len(T), np.float32)
        T.forced_mean_flux_ivar_z = np.zeros(len(T), np.float32)
        T.forced_mean_flux_chi2_g = np.zeros(len(T), np.float32)
        T.forced_mean_flux_chi2_r = np.zeros(len(T), np.float32)
        T.forced_mean_flux_chi2_z = np.zeros(len(T), np.float32)
        T.forced_nobs_g = np.zeros(len(T), np.float32)
        T.forced_nobs_r = np.zeros(len(T), np.float32)
        T.forced_nobs_z = np.zeros(len(T), np.float32)
    
        ffn = fn.replace('/tractor/', '/forced-deep/forced-brick/').replace('tractor', 'forced')
        if not os.path.exists(ffn):
            print('Skipping:', ffn)
            continue
        F = fits_table(ffn)
        FF = fits_table(ffn, hdu=2)
        print(fn, 'read', len(T), len(F), len(FF))
        assert(len(F) == len(T))
    
        for i,f in enumerate(F):
            for band in 'grz':
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
        for band in 'grz':
            print('Mean nobs', band, ':', np.mean(T.get('forced_nobs_'+band)))
        TT.append(T)

        T = merge_tables(TT)
        T.writeto('odin/2band/odin-2band+deep-cosmos.fits')

if __name__ == '__main__':
    main()

