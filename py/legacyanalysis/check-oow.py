from glob import glob
import os
import fitsio
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_dilation
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
    T.n_masked = []
    T.n_dilated_masked = []
    
    T.oow_min = []
    T.oow_max = []
    T.oow_median = []
    T.oow_percentiles = []

    T.oow_unmasked_min = []
    T.oow_unmasked_max = []
    T.oow_unmasked_median = []
    T.oow_unmasked_percentiles = []

    T.oow_dilated_min = []
    T.oow_dilated_max = []
    T.oow_dilated_median = []
    T.oow_dilated_percentiles = []

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
        # /global/cfs/cdirs/cosmo/staging/mosaic/CP/V4.3/CP20160124/k4m_160125_104535_oow_zd_ls9.fits.fz
        # has EXPNUM blank -> becomes None
        if expnum is None:
            expnum = 0
        print(fn, ext, expnum)
        T.expnum.append(expnum)
        T.obsid.append(phdr.get('OBSID', ''))
        T.acqnam.append(phdr.get('DTACQNAM', ''))
        T.filter.append(phdr.get('FILTER'))

        bad = (ood > 0)
        T.n_masked.append(np.sum(bad))
        dbad = binary_dilation(bad, structure=np.ones((3,3),bool))
        T.n_dilated_masked.append(np.sum(dbad))
        uw = oow[np.logical_not(dbad)]
        if len(uw) == 0:
            med = np.median(oow)
            T.oow_dilated_min.append(0.)
            T.oow_dilated_max.append(0.)
            T.oow_dilated_median.append(0.)
            T.oow_dilated_percentiles.append(np.zeros(len(pct), np.float32))
        else:
            med = np.median(uw)
            T.oow_dilated_min.append(uw.min())
            T.oow_dilated_max.append(uw.max())
            T.oow_dilated_median.append(np.median(uw))
            T.oow_dilated_percentiles.append(np.percentile(uw, pct, interpolation='nearest').astype(np.float32))

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

    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_102549_oow_r_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_100244_oow_g_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_100433_oow_g_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_102401_oow_g_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_100622_oow_r_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_100802_oow_r_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_102727_oow_r_v1.fits.fz')
    # one_file('/global/cfs/cdirs/cosmo/staging/decam/CP/V4.8.2a/CP20190719/c4d_190720_102212_oow_g_v1.fits.fz')
    # return

    if True:
        #dirs = glob(dirprefix + '90prime/CP/V2.3/CP*')
        dirs = glob(dirprefix + 'mosaic/CP/V4.3/CP*')
        #dirs = glob(dirprefix + 'mosaic/CP/V4.3/CP2017*')
        #dirs = (glob(dirprefix + 'mosaic/CP/V4.3/CP2015*') +
        #        glob(dirprefix + 'mosaic/CP/V4.3/CP2016*'))
        #dirs = glob(dirprefix + 'decam/CP/V4.8.2a/CP*')
        dirs.sort()
        dirs = list(reversed(dirs))
    
        keepdirs = []
        for dirnm in dirs:
            outfn = 'oow-stats-' + '-'.join(dirnm.split('/')[-4:]) + '.fits'
            if os.path.exists(outfn):
                print('skipping', outfn)
                continue
            keepdirs.append(dirnm)
        dirs = keepdirs
    
        print('Directories to run:')
        for dirnm in dirs:
            print('  ', dirnm)

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dirs', nargs='+',
    #                     help='Directories to process')
    # args = parser.parse_args()
    # 
    # dirs = args.dirs
    # print('Dirs:', dirs)
    
    mp = multiproc(32)
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
    
