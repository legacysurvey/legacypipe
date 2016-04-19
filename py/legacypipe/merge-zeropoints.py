from __future__ import print_function
import numpy as np
from glob import glob
import os
from astrometry.util.fits import fits_table, merge_tables

def decals_dr3_dedup():
    SN = fits_table('survey-ccds-nondecals.fits.gz')
    SD = fits_table('survey-ccds-decals.fits.gz')
    sne = np.unique(SN.expnum)
    sde = np.unique(SD.expnum)
    isec = set(sne).intersection(sde)
    print(len(isec), 'exposures in common between "decals" and "nondecals"')
    # These are two versions of CP reductions in CPDES82 and CPHETDEX.
    I = np.flatnonzero(np.array([e in isec for e in SD.expnum]))
    print(len(I), 'rows in "decals"')

    I = np.flatnonzero(np.array([e not in isec for e in SD.expnum]))
    SD.cut(I)
    print(len(SD), 'rows remaining')

    # Now, also move "decals" filenames containing "CPDES82" to "nondecals".
    I = np.flatnonzero(np.array(['CPDES82' in fn for fn in SD.image_filename]))
    keep = np.ones(len(SD), bool)
    keep[I] = False

    print('Merging:')
    SN.about()
    SD[I].about()
    
    SN2 = merge_tables((SN, SD[I]), columns='fillzero')
    SD2 = SD[keep]

    print('Moved CPDES82: now', len(SN2), 'non-decals and', len(SD2), 'decals')
    
    SN2.writeto('survey-ccds-nondecals.fits')
    SD2.writeto('survey-ccds-decals.fits')

    SE = fits_table('survey-ccds-extra.fits.gz')
    SN = fits_table('survey-ccds-nondecals.fits')
    SD = fits_table('survey-ccds-decals.fits')

    sne = np.unique(SN.expnum)
    sde = np.unique(SD.expnum)
    see = np.unique(SE.expnum)

    i1 = set(sne).intersection(sde)
    i2 = set(sne).intersection(see)
    i3 = set(sde).intersection(see)
    print('Intersections:', len(i1), len(i2), len(i3))

    

def decals_dr3_extra():
    # /global/homes/a/arjundey/ZeroPoints/decals-zpt-dr3-all.fits
    T = fits_table('/global/cscratch1/sd/desiproc/zeropoints/decals-zpt-dr3-all.fits')

    S1 = fits_table('survey-ccds-nondecals.fits.gz')
    S2 = fits_table('survey-ccds-decals.fits.gz')
    S = merge_tables([S1,S2])

    gotchips = set(zip(S.expnum, S.ccdname))

    got = np.array([(e,c) in gotchips for e,c in zip(T.expnum, T.ccdname)])
    print('Found', sum(got), 'of', len(T), 'dr3-all CCDs in existing surveys tables of size', len(S))

    I = np.flatnonzero(np.logical_not(got))
    T.cut(I)
    print(len(T), 'remaining')
    #print('Directories:', np.unique([os.path.basename(os.path.dirname(fn.strip())) for fn in T.filename]))
    print('Filenames:', np.unique(T.filename))

    T.writeto('extras.fits')

    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    TT = []

    for fn,dirnms in [
        ('extras.fits',
         ['CP20140810_?_v2',
          'CP20141227', 'CP20150108', 'CP20150326',
          'CP20150407', 'CP20151010', 'CP20151028', 'CP20151126',
          'CP20151226', 'CP20160107', 'CP20160225',
          'COSMOS', 'CPDES82',
          'NonDECaLS/*',
         ]),
        ]:
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-extra.fits'
    T.writeto(outfn)
    print('Wrote', outfn)
    

    
    
def decals_dr3():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')

    TT = []

    #zpdir = '/project/projectdirs/cosmo/work/decam/cats/ZeroPoints'
    for fn,dirnms in [
        ('/global/cscratch1/sd/desiproc/zeropoints/decals-zpt-dr3pr1233b.fits',
         ['CP20140810_?_v2',
          'CP20141227', 'CP20150108', 'CP20150326',
          'CP20150407', 'CP20151010', 'CP20151028', 'CP20151126',
          'CP20151226', 'CP20160107', 'CP20160225',
          'COSMOS', 'CPDES82',
          'NonDECaLS/*',
         ]),
        ]:
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'zp.fits'
    T.writeto(outfn)
    print('Wrote', outfn)

    nd = np.array(['NonDECaLS' in fn for fn in T.image_filename])
    I = np.flatnonzero(nd)
    T[I].writeto('survey-ccds-nondecals.fits')

    I = np.flatnonzero(np.logical_not(nd))
    T[I].writeto('survey-ccds-decals.fits')

    for fn in ['survey-ccds-nondecals.fits', 'survey-ccds-decals.fits']:
        os.system('gzip --best ' + fn)

    
def normalize_zeropoints(fn, dirnms, image_basedir, cam, T=None):
    if T is None:
        print('Reading', fn)
        T = fits_table(fn)
        print('Read', len(T), 'rows')
    T.camera = np.array([cam] * len(T))
    T.expid = np.array(['%08i-%s' % (expnum,extname.strip())
                        for expnum,extname in zip(T.expnum, T.ccdname)])
    cols = T.columns()
    if not 'naxis1' in cols:
        T.naxis1 = np.zeros(len(T), np.int16) + 2046
    if not 'naxis2' in cols:
        T.naxis2 = np.zeros(len(T), np.int16) + 4094

    # Expand wildcarded directory names
    realdirs = []
    for dirnm in dirnms:
        pattern = os.path.join(image_basedir, cam, dirnm)
        matched = glob(pattern)
        print('Pattern', pattern, '->', matched)
        if len(matched) == 0:
            continue
        realdirs.extend(matched)
    dirnms = realdirs

    # Search all given directory names
    allfiles = {}
    for dirnm in dirnms:
        pattern = os.path.join(image_basedir, cam, dirnm, '*.fits*')
        matched = glob(pattern)
        allfiles[dirnm] = matched
        print('Pattern', pattern, '->', len(matched))
        
    fns = []
    fnmap = {}
    for ifile,(fn,filt) in enumerate(zip(T.filename, T.filter)):
        print('CCD', ifile, 'of', len(T), ':', fn)
        if fn in fnmap:
            fns.append(fnmap[fn])
            continue
        orig_fn = fn
        fn = fn.strip()
        fnlist = []

        for dirnm in dirnms:
            pattern = os.path.join(image_basedir, cam, dirnm, fn)
            for afn in allfiles[dirnm]:
                # check for prefix
                if pattern in afn:
                    fnlist.append(afn)
                    print('File', fn, 'matched', afn)
                    
        # for dirnm in dirnms:
        #     pattern = os.path.join(image_basedir, cam, dirnm, fn + '*')
        #     matched = glob(pattern)
        #     print('Glob pattern', pattern, 'matched', len(matched))
        #     fnlist.extend(matched)

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

    #decals_dr3()
    #decals_dr3_extra()
    decals_dr3_dedup()
    sys.exit(0)

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


    
