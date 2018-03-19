from astrometry.util.fits import fits_table
from glob import glob
import os
import numpy as np

'''After DR5 we moved a bunch of CP input files that were in a
NonDECaLS-DR5 directory into NonDECaLS, and also rationalized some
duplicate files, including deleting some "v1" CP files in favor of
"v2" versions.  This script patches up the CCDs tables to point to the
new files.

Before running this script, create a symlink to the 'decam' directory:
 ln -s ~/cosmo/staging/decam/ .

'''


def update_paths(T):
    replacements = dict()
    
    for i,fn in enumerate(T.image_filename):
        #fn = t.image_filename.strip()
        #fn = os.path.join(basedir, fn)
        #print(fn)
        fn = fn.strip()
        rfn = replacements.get(fn)
        if rfn is not None:
            T.image_filename[i] = rfn
            continue
        if os.path.exists(fn):
            print('Found', fn)
            replacements[fn] = fn
            continue
        #print('Not found:', fn)
        dirnm,filename = os.path.split(fn)
        dirnm,cpdir = os.path.split(dirnm)
        dirnm,nddir = os.path.split(dirnm)
        #print('components', dirnm, nddir, cpdir, filename)
        if nddir == 'NonDECaLS-DR5':
            nddir = 'NonDECaLS'
        pat = os.path.join(dirnm, nddir, '*', filename)
        #print('Pattern', pat)
        fns = glob(pat)
        #print('-> ', fns)
        if len(fns) == 1:
            rfn = fns[0]
            T.image_filename[i] = rfn
            replacements[fn] = rfn
            #print('Found', len(fns))
            print('Nondecals', fn, '->', rfn)
            continue
    
        assert(len(fns) == 0)
    
        # v1 -> v2, etc
        components = filename.split('.')
        fn2 = '_'.join(components[0].split('_')[:-1]) + '*'
        #c4d_140626_015021_ooi_g_v2.fits.fz
        pat = os.path.join(dirnm, nddir, '*', fn2)
        #print('Pattern', pat)
        fns = glob(pat)
        #print('-> ', fns)
        if len(fns) == 1:
            rfn = fns[0]
            T.image_filename[i] = rfn
            replacements[fn] = rfn
            #print('Found', len(fns))
            print('Version', fn, '->', rfn)
            continue
            #print('Found', len(fns))
            #break
        assert(False)

    print('Maximum length of replacement filenames:', max([len(r) for r in replacements.values()]))


if True:
    T = fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/ccds-annotated-dr5.fits.gz')
    update_paths(T)
    # convert from 32A to float
    T.temp = np.array([float(t) for t in T.temp])
    # gratuitous
    T.delete_column('expid')
    T.writeto('ccds-annotated-dr5-patched.fits.gz')

if False:
    T = fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/survey-ccds-dr5.kd.fits')
    update_paths(T)
    T.writeto('survey-ccds-dr5-patched.fits')
