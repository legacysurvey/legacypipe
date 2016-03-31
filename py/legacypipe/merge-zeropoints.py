from __future__ import print_function
import numpy as np
from glob import glob
import os
from astrometry.util.fits import fits_table, merge_tables


def decals_dr2():
    decals_dir = os.environ['DECALS_DIR']
    cam = 'decam'
    image_basedir = os.path.join(decals_dir, 'images')

    TT = []

    #zpdir = '/project/projectdirs/cosmo/work/decam/cats/ZeroPoints'
    for fn,dirnms in [
        ('/scratch1/scratchdirs/desiproc/ZeroPoints/decals-zpt-all-2015oct30.fits',
         ['CP20140810_?_v2', 'CP20141227', 'CP20150108', 'CP20150326',
          'CP20150407', 'NonDECaLS/*','COSMOS', 'CPDES82']),
        # (os.path.join(zpdir, 'decals-zpt-20140810.fits'), ['CP20140810_?_v2']),
        # (os.path.join(zpdir, 'decals-zpt-20141227.fits'), ['CP20141227']),
        # (os.path.join(zpdir, 'decals-zpt-20150108.fits'), ['CP20150108']),
        # (os.path.join(zpdir, 'decals-zpt-20150326.fits'), ['CP20150326']),
        # (os.path.join(zpdir, 'decals-zpt-20150407.fits'), ['CP20150407']),
        # (os.path.join(zpdir, 'decals-zpt-nondecals.fits'), ['NonDECaLS/*','COSMOS', 'CPDES82']),
        ]:
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'zp.fits'
    T.writeto(outfn)
    print('Wrote', outfn)

    
def normalize_zeropoints(fn, dirnms, image_basedir, cam, T=None):
    if T is None:
        print('Reading', fn)
        T = fits_table(fn)
    T.camera = np.array([cam] * len(T))
    T.expid = np.array(['%08i-%s' % (expnum,extname.strip())
                        for expnum,extname in zip(T.expnum, T.ccdname)])
    cols = T.columns()
    if not 'naxis1' in cols:
        T.naxis1 = np.zeros(len(T), np.int16) + 2046
    if not 'naxis2' in cols:
        T.naxis2 = np.zeros(len(T), np.int16) + 4094

    fns = []
    fnmap = {}
    for fn,filt in zip(T.filename, T.filter):
        if fn in fnmap:
            fns.append(fnmap[fn])
            continue
        orig_fn = fn
        fn = fn.strip()
        fnlist = []
        for dirnm in dirnms:
            pattern = os.path.join(image_basedir, cam, dirnm, fn + '*')
            matched = glob(pattern)
            #print('Glob pattern', pattern, 'matched', matched)
            fnlist.extend(matched)

        pattern_string = os.path.join(image_basedir, cam, dirnm, fn + '*')
        if len(dirnms) > 1:
            pattern_string = os.path.join(
                image_basedir, cam, '{' + ','.join(dirnms) + '}', fn + '*')

        # If multiple versions are available, take the one with greatest
        # PLVER community pipeline version.
        if len(fnlist) > 1:
            import fitsio
            from distutils.version import StrictVersion
            print('WARNING', pattern_string, '->')
            for fn in fnlist:
                print('  ', fn)
            hdrs = [fitsio.read_header(fn) for fn in fnlist]
            assert(len(fnlist) == 2)
            vers = [hdr['PLVER'].strip().replace('V','') for hdr in hdrs]
            print('Versions', vers)
            ilast, lastver = None,None
            for i,ver in enumerate(vers):
                if lastver is None or StrictVersion(ver) > StrictVersion(lastver):
                    ilast = i
                    lastver = ver
            print('Latest version:', lastver, 'in file', fnlist[ilast])
            fnlist = [fnlist[ilast]]
            
        if len(fnlist) == 0:
            print('WARNING**', pattern_string, '->', fnlist)
            assert(False)

        fn = fnlist[0].replace(os.path.join(image_basedir, ''), '')
        fns.append(fn)
        fnmap[orig_fn] = fn
        assert(os.path.exists(os.path.join(image_basedir, fn)))
    T.filename = np.array(fns)

    T.rename('ccdhdunum', 'image_hdu')
    T.rename('filename', 'image_filename')
    T.rename('naxis1', 'width')
    T.rename('naxis2', 'height')
    T.rename('ra',  'ra_bore')
    T.rename('dec', 'dec_bore')
    T.rename('ccdra',  'ra')
    T.rename('ccddec', 'dec')

    T.width  = T.width.astype(np.int16)
    T.height = T.height.astype(np.int16)
    T.ccdnum = T.ccdnum.astype(np.int16)
    T.cd1_1 = T.cd1_1.astype(np.float32)
    T.cd1_2 = T.cd1_2.astype(np.float32)
    T.cd2_1 = T.cd2_1.astype(np.float32)
    T.cd2_2 = T.cd2_2.astype(np.float32)
    
    return T



if __name__ == '__main__':
    import sys

    #decals_dr2()
    #sys.exit(0)

    # Mosaicz tests
    from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec
    import fitsio
    cam = 'mosaic'

    TT = []
    #zpdir = '/project/projectdirs/cosmo/staging/mosaicz/Test'
    zpdir = '/project/projectdirs/desi/imaging/data/mosaic/cosmos' #/global/cscratch1/sd/arjundey/mosaicz'
    imgdir = '/project/projectdirs/desi/imaging/data/mosaic/cosmos' #'/project/projectdirs/cosmo/staging/mosaicz/Test'
    #/global/cscratch1/sd/arjundey/mosaicz/zeropoint-mzls_test1.fits
    for fn,dirnms in [
        #(os.path.join(zpdir, 'ZP-MOS3-20151213.fits'),
        # #[os.path.join(imgdir, 'MOS151213_8a516af')]),
        # [os.path.join(imgdir, 'MOS151213_8a7fcee')]),
        (os.path.join(zpdir, 'zeropoint-mosaic-arjunzp-cosmos.fits.fits'), #zeropoint-mzls_oki_test1_v1.fits'),
         [os.path.join(imgdir, './')]), #'MOS151213_8a7fcee')]),
        ]:
        print('Reading', fn)
        T = fits_table(fn)

        # forgot to include EXPTIME in zeropoint, thus TRANSPARENCY is way off
        tmags = 2.5 * np.log10(T.exptime)
        T.ccdphoff += tmags
        T.ccdtransp = 10.**(T.ccdphoff / -2.5)

        # Fill in BOGUS values; update from header below
        T.ccdhdunum = np.zeros(len(T), np.int32)
        T.ccdname = np.array(['ccd%i' % ccdnum for ccdnum in T.ccdnum])

        T = normalize_zeropoints(fn, dirnms, imgdir, cam, T=T)

        # HDU number wasn't recorded in zeropoint file -- search for EXTNAME
        fns = np.unique(T.image_filename)
        for fn in fns:
            print('Filename', fn)
            pth = os.path.join(imgdir, fn)
            F = fitsio.FITS(pth)
            print('File', fn, 'exts:', len(F))
            for ext in range(1, len(F)):
                print('extension:', ext)
                hdr = F[ext].read_header()
                extname = hdr['EXTNAME'].strip()
                print('EXTNAME "%s"' % extname)
                I = np.flatnonzero((T.image_filename == fn) *
                                   (T.ccdname == extname))
                print(len(I), 'rows match')
                assert(len(I) == 1)
                T.image_hdu[I] = ext
        T.fwhm = T.seeing
        TT.append(T)
    T = merge_tables(TT)
    T.image_filename = np.array([cam + '/' + fn for fn in T.image_filename])
    outfn = 'mosaicz-ccds.fits'
    T.writeto(outfn)
    print('Wrote', outfn)
    
    print('exiting after making MZLS ccds.fits file')
    sys.exit(0)

    # Bok tests
    cam = '90prime'
    TT = []
    zpdir = '/scratch1/scratchdirs/arjundey/Bok'

    for fn,dirnms in [
        (os.path.join(zpdir, 'g/zeropoint-BOK20150413_g.fits'),
         [os.path.join(zpdir, 'g')]),
        #(os.path.join(zpdir, 'r/zeropoint-BOK20150413_g.fits'),
        # [os.path.join(zpdir, 'g')]),
        ]:
        image_basedir = '.'
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        # fake up the exposure number
        T.expnum = (T.mjd_obs * 100000.).astype(int)
        # compute extension name
        T.ccdname = np.array(['ccd%i' % n for n in T.ccdnum])
        # compute FWHM from Seeing
        pixscale = 0.45
        T.fwhm = T.seeing / pixscale

        T.expid = np.array(['%10i-%s' % (expnum,extname.strip())
                            for expnum,extname in zip(T.expnum, T.ccdname)])

        TT.append(T)
    T = merge_tables(TT)
    outfn = 'bok-zp.fits'
    T.writeto(outfn)
    print('Wrote', outfn)


    
