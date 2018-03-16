from astrometry.util.fits import fits_table
from glob import glob
import os
import numpy as np

T = fits_table('/project/projectdirs/cosmo/data/legacysurvey/dr5/ccds-annotated-dr5.fits.gz')
basedir = '/project/projectdirs/cosmo/staging/'

# fns,I = np.unique(T.image_filename, return_index=True)
# T.cut(I)
# print('Cut to', len(T), 'unique filenames')
# 
# I, = np.nonzero(['NonDECaLS' in fn for fn in T.image_filename])
# print('Cut to', len(I), 'non-decals')
# T.cut(I)

replacements = dict()

for i,t in enumerate(T):
    fn = t.image_filename.strip()
    fn = os.path.join(basedir, fn)
    #print(fn)
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

# convert from 32A to float
T.temp = np.array([float(t) for t in T.temp])
# gratuitous
T.delete_column('expid')
T.writeto('ccds-annotated-dr5-patched.fits.gz')
