from __future__ import print_function
import numpy as np
from glob import glob
import os
from collections import Counter
from astrometry.util.fits import fits_table, merge_tables

def mzls_dr4(v2=True): 
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'mosaic'
    image_basedir = os.path.join(basedir, 'images')
    if v2:
        #zpname='arjuns-ccds-mzls-v2thruMarch19.fits'
        #savefn='survey-ccds-mzls-v2thruMarch19.fits.gz'
        #cpdirsfn= 'mzls_cpdirs_thruMarch19.txt'
        zpname='/global/homes/a/arjundey/ZeroPoints/mzls-v2-zpt-all-2016dec06.fits'
        savefn='NEW-survey-ccds-dr4-mzlsv2.fits.gz'
        # for dr in `find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*v2 -maxdepth 0 -type d`;do echo $(basename $dr) >> mzlsv2_cpdirs.txt ;done
        cpdirsfn= os.path.join(basedir,'mzlsv2_cpdirs.txt')
    else:
        #zpname='arjuns-ccds-mzls-v3.fits'
        #savefn='survey-ccds-mzls-v3.fits.gz'
        #zpname='/global/homes/a/arjundey/ZeroPoints/mzls-zpt-2016feb-v3.fits'
        #savefn='survey-ccds-dr4-mzlsv3-2016feb.fits.gz'
        zpname='/global/homes/a/arjundey/ZeroPoints/mzls-v3-zpt-2016dec19.fits'
        savefn='survey-ccds-dr4-mzlsv3.fits.gz'
        cpdirsfn= os.path.join(basedir,'mzlsv3_cpdirs.txt')

    TT = []
    # mzls_cpdirs_...txt is list of all possible CP directories
    for fn,dirnms in [
        (zpname,
         list(np.loadtxt(cpdirsfn,dtype=str))),
        ]:
        T = fits_table(fn)
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)

    I = np.flatnonzero(T.fwhm == 0)
    if len(I):
        T.fwhm[I] = T.seeing[I] / 0.262

    T.writeto(savefn)
    print('Wrote', savefn)


def bok_dr4():
    basedir = os.environ['LEGACY_SURVEY_DIR'] 
    cam = '90prime'
    image_basedir = os.path.join(basedir, 'images')

    zpname='/global/homes/a/arjundey/ZeroPoints/bass-zpt-all-2016dec06.fits'
    cpdirsfn= os.path.join(basedir,'bok_cpdirs.txt')

    #/scratch1/scratchdirs/desiproc/DRs/cp-images/bootes/project/projectdirs/cosmo/staging/bok/BOK_CP/CP20160703
    #/scratch2/scratchdirs/arjundey/ForKaylan/zeropoint-ksb_160704_052458_ooi_g_v1.fits
    TT = []
    for fn,dirnms in [
        (zpname,
         list(np.loadtxt(cpdirsfn,dtype=str))),
        #(os.path.join(zpdir, 'r/zeropoint-BOK20150413_g.fits'),
        # [os.path.join(zpdir, 'g')]),
        ]:
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        pixscale = 0.455
        T.fwhm = T.seeing / pixscale
        # fake up the exposure number
        #T.expnum = (T.mjd_obs * 100000.).astype(int)
        # compute extension name
        #T.ccdname = np.array(['ccd%i' % n for n in T.ccdnum])
        # compute FWHM from Seeing
        #T.expid = np.array(['%10i-%s' % (expnum,extname.strip())
        #                    for expnum,extname in zip(T.expnum, T.ccdname)])

        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-dr4-90prime.fits.gz'
    T.writeto(outfn)
    print('Wrote', outfn)



def mzls_to_20160315():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'mosaic'
    image_basedir = os.path.join(basedir, 'images')
    TT = []

    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/mzls-zpt-all.fits',
         ['CP20160202','CP20160203','CP20160204','CP20160205','CP20160206','CP20160208',
          'CP20160209','CP20160210','CP20160211','CP20160212','CP20160213','CP20160214',
          'CP20160215','CP20160216','CP20160217','CP20160219','CP20160224','CP20160225',
          'CP20160226','CP20160227','CP20160228','CP20160229','CP20160301','CP20160302',
          'CP20160303','CP20160304','CP20160305','CP20160306','CP20160308','CP20160309',
          'CP20160310','CP20160311','CP20160312','CP20160313','CP20160314','CP20160315',
          'CP20160316','CP20160317','CP20160318','CP20160319','CP20160320','CP20160325',
          'CP20160326','CP20160327','CP20160328','CP20160330','CP20160331','CP20160401',
          'CP20160402','CP20160403','CP20160404','CP20160408',]),
        ]:
        T = fits_table(fn)
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)

    I = np.flatnonzero(T.fwhm == 0)
    if len(I):
        T.fwhm[I] = T.seeing[I] / 0.262

    outfn = 'survey-ccds-mzls-to-20160315.fits'
    T.writeto(outfn)
    print('Wrote', outfn)

    for fn in [outfn]:
        os.system('gzip --best ' + fn)

# Runs 25+26
def decals_run25():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    TT = []
    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20170228-0316.fits',
         ['CP20170228',
          'CP20170301',
          'CP20170302',
          'CP20170303',
          'CP20170304',
          'CP20170305',
          'CP20170315',
          'CP20170316',]),
        ]:
        T = fits_table(fn)
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-run25.fits'
    T.writeto(outfn)
    print('Wrote', outfn)
    for fn in [outfn]:
        os.system('gzip --best ' + fn)


# Runs 21, 22+23 and 24, for adjusting obstatus file.
def decals_run21():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    TT = []
    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20161113.fits',
         ['CP20161113']),
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20161220-28.fits',
         ['CP20161220', 'CP20161227', 'CP20161228']),
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20170108-10.fits',
         ['CP20170108', 'CP20170109', 'CP20170110']),
        ]:
        T = fits_table(fn)
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-run21.fits'
    T.writeto(outfn)
    print('Wrote', outfn)
    for fn in [outfn]:
        os.system('gzip --best ' + fn)


# Runs 19 and 20, for adjusting obstatus file...
def decals_run19():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    TT = []
    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20160801_20161011.fits',
         ['CP20160801', 'CP20161011']),
        ]:
        T = fits_table(fn)
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-run19.fits'
    T.writeto(outfn)
    print('Wrote', outfn)
    for fn in [outfn]:
        os.system('gzip --best ' + fn)

# Runs 16 and 17, for adjusting obstatus file...
def decals_run16():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    TT = []

    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20160709_20.fits',
         ['CP20160709', 'CP20160720']),
        ]:
        T = fits_table(fn)
        #T.cut(np.nonzero([e not in expnums for e in T.expnum])[0])
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)
    T = merge_tables(TT)
    outfn = 'survey-ccds-run16.fits'
    T.writeto(outfn)
    print('Wrote', outfn)

    for fn in [outfn]:
        os.system('gzip --best ' + fn)


# Runs 13 and 15, for adjusting obstatus file...
def decals_dr3_plus():
    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')

    TT = []

    expnums = [547257, 535154, 535106]
    for fn,dirnms in [
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20160407.fits',
         ['CP20160407',]),
        ('/global/homes/a/arjundey/ZeroPoints/decals-zpt-20160606.fits',
         ['CP20160606',]),
        ]:
        T = fits_table(fn)
        T.cut(np.nonzero([e not in expnums for e in T.expnum])[0])
        normalize_zeropoints(fn, dirnms, image_basedir, cam, T=T)
        TT.append(T)

    dirnms = ['CP20160407','CP20160606']
    for fn in [
        '/global/cscratch1/sd/arjundey/ZeroPoints/zeropoint-c4d_160408_023844_oki_r_v1.fits',
        '/global/cscratch1/sd/arjundey/ZeroPoints/zeropoint-c4d_160408_001343_oki_r_v1.fits',
        '/global/cscratch1/sd/arjundey/ZeroPoints/zeropoint-c4d_160607_023641_oki_g_v1.fits',
        ]:
        T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
        TT.append(T)

    T = merge_tables(TT)
    outfn = 'survey-ccds-dr3plus.fits'
    T.writeto(outfn)
    print('Wrote', outfn)

    for fn in [outfn]:
        os.system('gzip --best ' + fn)


def decals_dr3_check_wcsfailed():
    import fitsio
    basedir = os.environ['LEGACY_SURVEY_DIR']
    image_basedir = os.path.join(basedir, 'images')
    for fn in [#'survey-ccds-decals.fits.gz',
               'survey-ccds-nondecals.fits.gz',
               'survey-ccds-extra.fits.gz',
               ]:
        T = fits_table(fn)
        T.wcscal = np.zeros(len(T), bool)
        fns = np.unique(np.array([f.strip() for f in T.image_filename]))

        H = fits_table()
        H.image_filename = fns
        H.primary_header = []
        
        for ifn,f in enumerate(fns):
            imgfn = os.path.join(image_basedir, f)
            print('Reading', imgfn)
            
            ff = open(imgfn, 'r')
            h = ff.read(32768)
            ff.close()
            hdr = fitsio.FITSHDR()
            hdrstring = ''
            while True:
                line = h[:80]
                h = h[80:]
                # fitsio apparently can't handle CONTINUE
                if line[:8] != 'CONTINUE':
                    hdr.add_record(line)
                hdrstring += line
                if line == ('END' + ' '*77):
                    break
            H.primary_header.append(hdrstring)
                
            #hdr = fitsio.read_header(imgfn)
            expnum = hdr['EXPNUM']
            wcscal = hdr['WCSCAL']
            wcsok = (wcscal == 'Successful')
            print('File', f, 'expnum', expnum, 'WCS cal', wcscal, 'ok', wcsok)
            I = np.flatnonzero(T.expnum == expnum)
            T.wcscal[I] = wcsok

        T.wcsok = T.wcscal.astype(np.uint8)
        T.writeto('new-' + fn)

        H.primary_header = np.array(H.primary_header)
        H.writeto('headers-' + fn)
        
def decals_dr3_fix392400():
    T = fits_table('survey-ccds-decals.fits.gz')
    print('Started with', len(T), 'CCDs')
    T.cut(T.expnum != 392400)
    print('Removed bad CCDs:', len(T))

    basedir = os.environ['LEGACY_SURVEY_DIR']
    cam = 'decam'
    image_basedir = os.path.join(basedir, 'images')
    fn = '/global/cscratch1/sd/desiproc/zeropoints/zeropoint-c4d_141228_060426_ooi_g_v1.fits'
    dirnms = ['CP20141227']
    Tnew = normalize_zeropoints(fn, dirnms, image_basedir, cam)
    print('Replacement CCDs:', len(Tnew))
    T = merge_tables([T, Tnew])
    print('Merged:', len(T))
    T.writeto('new-survey-ccds-decals.fits.gz')
    

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

#
#   basedir = os.environ['LEGACY_SURVEY_DIR']
#    cam = 'mosaic'
#    image_basedir = os.path.join(basedir, 'images')
#    TT = []
#
#    for fn,dirnms in [
#        ('/global/homes/a/arjundey/ZeroPoints/mzls-zpt-all.fits',
#         ['CP20160202','CP20160203','CP20160204','CP20160205','CP20160206','CP20160208',
#          'CP20160209','CP20160210','CP20160211','CP20160212','CP20160213','CP20160214',
#          'CP20160215','CP20160216','CP20160217','CP20160219','CP20160224','CP20160225',
#          'CP20160226','CP20160227','CP20160228','CP20160229','CP20160301','CP20160302',
#          'CP20160303','CP20160304','CP20160305','CP20160306','CP20160308','CP20160309',
#          'CP20160310','CP20160311','CP20160312','CP20160313','CP20160314','CP20160315',
#          'CP20160316','CP20160317','CP20160318','CP20160319','CP20160320','CP20160325',
#          'CP20160326','CP20160327','CP20160328','CP20160330','CP20160331','CP20160401',
#          'CP20160402','CP20160403','CP20160404','CP20160408',]),
#        ]:
#        T = fits_table(fn)
#  


def normalize_zeropoints(fn, dirnms, image_basedir, cam, T=None):
    if T is None:
        print('Reading', fn)
        T = fits_table(fn)
        print('Read', len(T), 'rows')
    # Remove empty filename rows if exist (90Prime issue)
    keep= np.char.strip(T.filename) != ''
    rem= keep == False
    if np.where(rem)[0].size > 0:
        print('WARNING: %d rows have empty filename, removing:\n' % np.where(rem)[0].size,np.where(rem)[0])
        T.cut(keep)
    
    if cam == '90prime':
        # EXPNUM is all 0s, change to some unique number
        T.expnum = (T.mjd_obs * 100000.).astype(int)
        # EXTNAME doesn't exist, it is ccdname instead
        T.set('extname',T.ccdname)
    T.camera = np.array([cam] * len(T))
    T.expid = np.array(['%08i-%s' % (expnum,extname.strip())
                        for expnum,extname in zip(T.expnum, T.ccdname)])
    c = Counter(T.expid)
    bad = False
    for k,v in c.most_common():
        if v == 1:
            break
        print('Warning: repeated EXPNUM/CCDNAME:', k, 'appears', v, 'times')
        bad = True
    #assert(not bad)

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
        pattern = os.path.join(dirnm, '*.fits*')
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
            if not os.path.exists(dirnm):
                print('WARNING, creating empty dir: %s' % dirnm)
                os.makedirs(dirnm)
            pattern = os.path.join(dirnm, fn)
            for afn in allfiles[dirnm]:
                # check for prefix
                if pattern in afn:
                    fnlist.append(afn)
                    print('File', fn, 'matched', afn)
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
            
        print('LENGTH len(fnlist)=%d' % len(fnlist))       
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


def gather_arjuns_zpts(cpimage_list='bootes-90prime-abspath.txt',
                       name='arjuns-ccds-90prime.fits'):
    zpdir='/scratch2/scratchdirs/arjundey/ForKaylan'
    # Get list of cpimages want zeropoints files for
    cps=np.loadtxt(cpimage_list,dtype=str)
    cats=[]
    for i,cp in enumerate(cps):
        print('reading %d/%d' % (i+1,len(cps)))
        if '90prime' in name:
            zpname= os.path.basename(cp).replace('ksb','zeropoint-ksb').replace('.fits.fz','.fits')
        elif 'mzls' in name:
            zpname= os.path.basename(cp).replace('k4m','zeropoint-k4m').replace('.fits.fz','.fits')
        else:
            raise ValueError('%s' % name)
        zpname=os.path.join(zpdir,zpname)
        try:
            cats.append( fits_table(zpname) )
        except IOError:
            print('WARNING: Cannot read this file: %s' % zpname)
    bigcat=merge_tables(cats, columns='fillzero')
    bigcat.writeto(name)
        

if __name__ == '__main__':
    import sys

    #decals_dr3()
    #decals_dr3_extra()
    #decals_dr3_dedup()
    #decals_dr3_fix392400()
    #decals_dr3_check_wcsfailed()
    #decals_dr3_plus()
    #decals_run16()
    #mzls_to_20160315()
    #decals_run19()
    #decals_run21()

    decals_run25()

    #dr4_bootes=False
    #if dr4_bootes:
    #    for name in ['90prime','mzls-v2thruMarch19','mzls-v3']:
    #        gather_arjuns_zpts(cpimage_list='bootes-%s-abspath.txt' % name,\
    #                           name='arjuns-ccds-%s.fits' % name)
    #bok_dr4()
    #mzls_dr4(v2=True)
    #mzls_dr4(v2=False)
    sys.exit(0)
    
    
    # cam = 'mosaic'
    # image_basedir = os.path.join(basedir, 'images')
    # TT = []
    # for fn,dirnms in [
    #         (os.path.join(basedir, 'zeropoint-arjun_zpts.fits'),
    #          ['CP20151213',]),
    #     ]:
    #     T = normalize_zeropoints(fn, dirnms, image_basedir, cam)
    #     TT.append(T)
    # T = merge_tables(TT)
    # T.fwhm = T.seeing / 0.262
    # T.ccdname = np.array([n.replace('LBL-0', 'ccd') for n in T.ccdname])
    # outfn = 'zp.fits'
    # T.writeto(outfn)
    # print('Wrote', outfn)

    sys.exit(0)

    
    # MzLS DEEP2 inventory
    import fitsio
    fns = glob('/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP20151213/*_oki_*')
    print(len(fns), 'files')
    T = fits_table()
    T.image_filename = np.array(fns)
    T.expnum = np.zeros(len(T), np.int32)
    T.exptime = np.zeros(len(T), np.float32)
    T.ra = np.zeros(len(T), np.float64)
    T.dec = np.zeros(len(T), np.float64)
    T.band = np.zeros(len(T), str)
    for i,fn in enumerate(fns):
        print('Reading', fn)
        hdr = fitsio.read_header(fn)
        T.expnum [i] = hdr['EXPNUM']
        T.exptime[i] = hdr['EXPTIME']
        T.ra     [i] = hdr['CENTRA']
        T.dec    [i] = hdr['CENTDEC']
        T.band   [i] = hdr['FILTER'][0]
    T.writeto('mzls-20151213.fits')

    import pylab as plt
    plt.clf()
    plt.plot(T.ra, T.dec, 'mo')
    plt.savefig('mzls-1.png')

    T.cut(np.hypot(T.ra - 350, T.dec - 0) < 5)
    T.writeto('mzls-d2f3.fits')
    
    plt.clf()
    plt.plot(T.ra, T.dec, 'mo')
    plt.savefig('mzls-2.png')
    
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

    
    
