import numpy as np
from glob import glob
import os
from astrometry.util.fits import fits_table, merge_tables

decals_dir = os.environ['DECALS_DIR']

cam = 'decam'
imagedir = os.path.join(decals_dir, 'images', cam)

# decals-zpt-20140810.fits - /project/projecdirs/cosmo/work/decam/cats/CP20140810_*_v2/
# decals-zpt-20141227.fits - /project/projecdirs/cosmo/work/decam/cats/CP20141227/
# decals-zpt-20150108.fits - /project/projecdirs/cosmo/work/decam/cats/CP20150108/
# decals-zpt-20150326.fits - /global/scratch2/sd/arjundey/CP20150326/
# decals-zpt-20150407.fits - /global/scratch2/sd/arjundey/CP20150407/

TT = []

T = fits_table('/project/projectdirs/cosmo/work/decam/cats/ZeroPoints/ZeroPoints_ALL_2015apr07.fits')
T.camera = np.array([cam] * len(T))
T.expid = np.array(['%08i-%s' % (expnum,extname.strip())
                    for expnum,extname in zip(T.expnum, T.ccdname)])
cols = T.columns()
if not 'naxis1' in cols:
    T.naxis1 = np.zeros(len(T), np.int16) + 2046
if not 'naxis2' in cols:
    T.naxis2 = np.zeros(len(T), np.int16) + 4094

T.filename = np.array(['%s/CP20150407/%s.fz' % (cam, fn) for fn in T.filename])

T.rename('ccdhdunum', 'image_hdu')
T.rename('filename', 'image_filename')
T.rename('naxis1', 'width')
T.rename('naxis2', 'height')
T.rename('ra',  'ra_bore')
T.rename('dec', 'dec_bore')
T.rename('ccdra',  'ra')
T.rename('ccddec', 'dec')

outfn = 'zp.fits'
T.writeto(outfn)
print 'Wrote', outfn
