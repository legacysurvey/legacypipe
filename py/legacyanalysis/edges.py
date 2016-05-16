from __future__ import print_function
import pylab as plt
import numpy as np
from glob import glob
from legacypipe.cpimage import CP_DQ_BITS
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *

ps = PlotSequence('edges')
#fns = glob('dr2-tractor/*/*.fits')
#fns = glob('dr2-tractor/24[12]/tractor-*p0[56]*.fits')

plotrange = ((240.9, 242.1), (4.8, 5.9))

fns = glob('dr2-tractor/241/tractor-*p05*.fits')
fns.sort()
TT2,TT3 = [],[]
for fn in fns:
    fn3 = fn.replace('dr2-tractor', 'dr3-tractor')
    if not os.path.exists(fn3):
        print('Does not exist:', fn3)
        continue
    T2 = fits_table(fn)
    T3 = fits_table(fn3)
    print('Reading', fn, '->', len(T2), 'DR2,', len(T3), 'DR3')
    TT2.append(T2)
    TT3.append(T3)
T2 = merge_tables(TT2)
T3 = merge_tables(TT3)

plt.clf()
plothist(T2.ra, T2.dec, nbins=200, range=plotrange)
plt.title('DR2')
ps.savefig()

plt.clf()
plothist(T3.ra, T3.dec, nbins=200, range=plotrange)
plt.title('DR3')
ps.savefig()

plt.clf()
I = np.flatnonzero(T2.brick_primary)
print('DR2:', len(I), 'brick_primary')
H2,xe,ye = plothist(T2.ra[I], T2.dec[I], nbins=200, range=plotrange)
mx = H2.max()
plt.title('DR2 brick_primary')
ps.savefig()

plt.clf()
I = np.flatnonzero(T3.brick_primary)
print('DR3:', len(I), 'brick_primary')
H3,xe,ye = plothist(T3.ra[I], T3.dec[I], nbins=200, imshowargs=dict(vmax=mx), range=plotrange)
plt.title('DR3 brick_primary')
ps.savefig()

plt.clf()
plt.imshow((H3 - H2).T, interpolation='nearest', origin='lower', cmap='hot',
           extent=(min(xe), max(xe), min(ye), max(ye)),
           aspect='auto')
plt.colorbar()
plt.title('DR3 - DR2 brick_primary')
ps.savefig()


for name,bit in CP_DQ_BITS.items():
    plt.clf()
    anymask = reduce(np.bitwise_or, (T3.decam_anymask[:,1],
                                     T3.decam_anymask[:,2],
                                     T3.decam_anymask[:,4]))
    I = np.flatnonzero((anymask & bit) > 0)
    #plt.subplot(1,2,1)
    plothist(T3.ra[I], T3.dec[I], nbins=200, imshowargs=dict(vmax=mx),
             doclf=False, range=plotrange)
    plt.title('DR3, %s any' % name)
    #plt.suptitle('DR3')
    ps.savefig()



I,J,d = match_radec(T2.ra, T2.dec, T2.ra, T2.dec, 1./3600., notself=True)
K = (I < J)
I = I[K]
J = J[K]
d1 = d[K]

K = np.flatnonzero(T2.brickname[I] != T2.brickname[J])
print(len(K), 'matches from different bricks (all objs, DR2)')

P2 = T2[T2.brick_primary]
I,J,d = match_radec(P2.ra, P2.dec, P2.ra, P2.dec, 1./3600., notself=True)
K = (I < J)
I = I[K]
J = J[K]
d2 = d[K]

K = np.flatnonzero(P2.brickname[I] != P2.brickname[J])
print(len(K), 'matches from different bricks (primary, DR2)')

plt.clf()
n,b,p1 = plt.hist(d1 * 3600., bins=50, range=(0,1), histtype='step', color='r',
              log=True)
n,b,p2 = plt.hist(d2 * 3600., bins=50, range=(0,1), histtype='step', color='b',
              log=True)
plt.legend((p1[0],p2[0]), ('All', 'Primary'))
plt.title('DR2 matches')
plt.xlabel('Match distance (arcsec)')
ps.savefig()



I,J,d = match_radec(T3.ra, T3.dec, T3.ra, T3.dec, 1./3600., notself=True)
K = (I < J)
I = I[K]
J = J[K]
d1 = d[K]

K = np.flatnonzero(T3.brickname[I] != T3.brickname[J])
print(len(K), 'matches from different bricks (all objs, DR3)')

P3 = T3[T3.brick_primary]
I,J,d = match_radec(P3.ra, P3.dec, P3.ra, P3.dec, 1./3600., notself=True)
print(len(I), 'primary matches, DR3, dups')
K = (I < J)
I = I[K]
J = J[K]
print(len(I), 'primary matches, DR3')
d2 = d[K]

K = np.flatnonzero(P3.brickname[I] != P3.brickname[J])
print(len(K), 'matches from different bricks (primary, DR3)')
matches = P3[np.array(zip(I[K], J[K])).flat]
matches.writeto('dbrick-dr3.fits')

#  PSF  2412p050  -> BRICKQ 0
#  SIMP 2415p050  -> BRICKQ 1   ~ 1% of the flux

plt.clf()
n,b,p1 = plt.hist(d1 * 3600., bins=50, range=(0,1), histtype='step',
                  color='r', log=True)
n,b,p2 = plt.hist(d2 * 3600., bins=50, range=(0,1), histtype='step',
                  color='b', log=True)
plt.legend((p1[0],p2[0]), ('All', 'Primary'))
plt.title('DR3 matches')
plt.xlabel('Match distance (arcsec)')
ps.savefig()


i = np.argmin(d2)
nearest = P3[np.array([I[i],J[i]])]
nearest.writeto('nearest.fits')


