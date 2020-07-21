from glob import glob
import os
import fitsio
import numpy as np
from scipy.ndimage.filters import median_filter
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.multiproc import multiproc

dirprefix = '/global/cfs/cdirs/cosmo/staging/'

def one_file(fn):
    T = fits_table()
    T.filename = []
    T.ext = []
    T.ccdname = []
    T.expnum = []
    T.obsid = []
    T.acqnam = []
    T.filter = []
    T.wcs_ok = []
    
    T.oow_min = []
    T.oow_max = []
    T.oow_median = []
    T.oow_percentiles = []

    T.oow_unmasked_min = []
    T.oow_unmasked_max = []
    T.oow_unmasked_median = []
    T.oow_unmasked_percentiles = []

    T.oow_m3_min = []
    T.oow_m3_max = []
    T.oow_m3_median = []
    T.oow_m3_percentiles = []

    T.oow_m5_min = []
    T.oow_m5_max = []
    T.oow_m5_median = []
    T.oow_m5_percentiles = []

    print(fn)
    F = fitsio.FITS(fn)
    phdr = F[0].read_header()
    D = fitsio.FITS(fn.replace('_oow_', '_ood_'))

    wcs_ok = (phdr.get('WCSCAL', '').strip().lower().startswith('success') or
              phdr.get('SCAMPFLG', -1) == 0)
    
    #print(len(F), 'extensions')
    for ext in range(1, len(F)):
        oow = F[ext].read()
        hdr = F[ext].read_header()
        ood = D[ext].read()
        pct = np.arange(101)
        
        T.filename.append(fn.replace(dirprefix, ''))
        T.wcs_ok.append(wcs_ok)
        T.ext.append(ext)
        T.ccdname.append(hdr['EXTNAME'])
        expnum = phdr.get('EXPNUM', 0)
        print(fn, ext, expnum)
        T.expnum.append(phdr.get('EXPNUM', 0))
        T.obsid.append(phdr.get('OBSID', ''))
        T.acqnam.append(phdr.get('DTACQNAM', ''))
        T.filter.append(phdr.get('FILTER'))
        T.oow_min.append(oow.min())
        T.oow_max.append(oow.max())
        T.oow_median.append(np.median(oow))
        T.oow_percentiles.append(np.percentile(oow, pct, interpolation='nearest').astype(np.float32))
        uw = oow[ood == 0]
        if len(uw) == 0:
            med = np.median(oow)
            T.oow_unmasked_min.append(0.)
            T.oow_unmasked_max.append(0.)
            T.oow_unmasked_median.append(0.)
            T.oow_unmasked_percentiles.append(np.zeros(len(pct), np.float32))
        else:
            med = np.median(uw)
            T.oow_unmasked_min.append(uw.min())
            T.oow_unmasked_max.append(uw.max())
            T.oow_unmasked_median.append(np.median(uw))
            T.oow_unmasked_percentiles.append(np.percentile(uw, pct, interpolation='nearest').astype(np.float32))

        # Fill masked OOW pixels with the median value.
        oow[ood > 0] = med
        # Median filter
        m3 = median_filter(oow, 3, mode='constant', cval=med)
        m5 = median_filter(oow, 5, mode='constant', cval=med)
        
        T.oow_m3_min.append(m3.min())
        T.oow_m3_max.append(m3.max())
        T.oow_m3_median.append(np.median(m3))
        T.oow_m3_percentiles.append(np.percentile(m3.flat, pct, interpolation='nearest').astype(np.float32))

        T.oow_m5_min.append(m5.min())
        T.oow_m5_max.append(m5.max())
        T.oow_m5_median.append(np.median(m5))
        T.oow_m5_percentiles.append(np.percentile(m5.flat, pct, interpolation='nearest').astype(np.float32))
        
    T.to_np_arrays()
    return T

def main():
    #dirs = glob(dirprefix + '90prime/CP/V2.3/CP*')
    #dirs = glob(dirprefix + 'mosaic/CP/V4.3/CP*')
    dirs = glob(dirprefix + 'decam/CP/V4.8.2a/CP*')
    dirs.sort()
    dirs = list(reversed(dirs))

    mp = multiproc(16)
    #mp = multiproc(1)
    
    for dirnm in dirs:
        print('Dir', dirnm)
        outfn = 'oow-stats-' + '-'.join(dirnm.split('/')[-4:]) + '.fits'
        print('looking for', outfn)
        if os.path.exists(outfn):
            print('skipping', outfn)
            continue
        pat =  os.path.join(dirnm, '*_oow_*.fits.fz')
        print('Pattern', pat)
        fns = glob(pat)
        fns.sort()
        print(len(fns), 'oow files')

        TT = mp.map(one_file, fns)
        T = merge_tables(TT)
        T.writeto(outfn)
        print('Wrote', outfn)


if __name__ == '__main__':
    main()
    
