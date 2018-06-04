'''
After running forced_photom.py on a set of CCDs, this script merges
the results back into a catalog.
'''
from astrometry.util.fits import *
import numpy as np
from glob import glob
from collections import Counter

from legacypipe.survey import LegacySurveyData

fns = glob('forced/*/*/forced-*.fits')
F = merge_tables([fits_table(fn) for fn in fns])

dr6 = LegacySurveyData('/project/projectdirs/cosmo/data/legacysurvey/dr6')
B = dr6.get_bricks_readonly()

I = np.flatnonzero((B.ra1 < F.ra.max()) * (B.ra2 > F.ra.min()) * (B.dec1 < F.dec.max()) * (B.dec2 > F.dec.min()))
print(len(I), 'bricks')
T = merge_tables([fits_table(dr6.find_file('tractor', brick=B.brickname[i])) for i in I])
print(len(T), 'sources')
T.cut(T.brick_primary)
print(len(T), 'primary')

# map from F to T index
imap = dict([((b,o),i) for i,(b,o) in enumerate(zip(T.brickid, T.objid))])
F.tindex = np.array([imap[(b,o)] for b,o in zip(F.brickid, F.objid)])
assert(np.all(T.brickid[F.tindex] == F.brickid))
assert(np.all(T.objid[F.tindex] == F.objid))

fcols = 'apflux apflux_ivar camera expnum ccdname exptime flux flux_ivar fracflux mask mjd rchi2 x y brickid objid'.split()

bands = np.unique(F.filter)
for band in bands:
    Fb = F[F.filter == band]
    print(len(Fb), 'in band', band)
    c = Counter(zip(Fb.brickid, Fb.objid))
    NB = c.most_common()[0][1]
    print('Maximum of', NB, 'exposures per object')

    # we use uint8 below...
    assert(NB < 256)
    
    sourcearrays = []
    sourcearrays2 = []
    destarrays = []
    destarrays2 = []
    for c in fcols:
        src = Fb.get(c)
        if len(src.shape) == 2:
            narray = src.shape[1]
            dest = np.zeros((len(T), Nb, narray), src.dtype)
            T.set('forced_%s_%s' % (band, c), dest)
            sourcearrays2.append(src)
            destarrays2.append(dest)
        else:
            dest = np.zeros((len(T), Nb), src.dtype)
            T.set('forced_%s_%s' % (band, c), dest)
            sourcearrays.append(src)
            destarrays.append(dest)
    nf = np.zeros(len(T), np.uint8)
    T.set('forced_%s_n' % band, nf)
    for i,ti in enumerate(Fb.tindex):
        k = nf[ti]
        for src,dest in zip(sourcearrays, destarrays):
            dest[ti,k] = src[i]
        for src,dest in zip(sourcearrays2, destarrays2):
            dest[ti,k,:] = src[i,:]
        nf[ti] += 1

for band in bands:
    flux = T.get('forced_%s_flux' % band)
    ivar = T.get('forced_%s_flux_ivar' % band)
    miv = np.sum(ivar, axis=1)
    T.set('forced_%s_mean_flux' % band, np.sum(flux * ivar, axis=1) / np.maximum(1e-16, miv))
    T.set('forced_%s_mean_flux_ivar' % band, miv)

#K = np.flatnonzero(np.logical_or(T.forced_mean_u_flux_ivar > 0, T.forced_mean_r_flux_ivar > 0))
#T[K].writeto('forced/forced-cfis-deep2f2.fits')

T.writeto('forced-merged.fits')

