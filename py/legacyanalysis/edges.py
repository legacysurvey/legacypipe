from __future__ import print_function
from collections import Counter
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


def edr_dr2_vs_dr3():
    #plotrange = ((240.9, 242.1), (4.8, 5.9))
    plotrange = ((240, 245), (4.5, 12.0))
    
    #fns = glob('dr2-tractor/241/tractor-*p05*.fits')
    
    fns = glob('dr2-tractor/24[01234]/tractor-*.fits')
    fns.sort()
    TT2,TT3 = [],[]
    for fn in fns:
        fn3 = fn.replace('dr2-tractor', 'dr3-tractor')
        if not os.path.exists(fn3):
            print('Does not exist:', fn3)
            continue
        cols = 'ra dec brick_primary brickname decam_anymask'.split()
        T2 = fits_table(fn, columns=cols)
        T3 = fits_table(fn3, columns=cols)
        print('Reading', fn, '->', len(T2), 'DR2,', len(T3), 'DR3')
        TT2.append(T2)
        TT3.append(T3)
    T2 = merge_tables(TT2, columns='fillzero')
    T3 = merge_tables(TT3, columns='fillzero')
    del TT2, TT3
    
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


# Sweeps have only brick_primary objects
#fns = glob('/project/projectdirs/cosmo/data/legacysurvey/dr2/sweep/2.0/sweep-00*.fits')
#fns = glob('/project/projectdirs/cosmo/data/legacysurvey/dr2/tractor/00?/tractor-*.fits')
fns = glob('/project/projectdirs/cosmo/data/legacysurvey/dr2/tractor/000/tractor-*.fits')
fns.sort()
T2 = []
for fn in fns:
    T = fits_table(fn, columns=['ra','dec','brick_primary','brickname',
                                'decam_flux', 'type'])
    print(len(T), 'from', fn, 'brick_primary', np.unique(T.brick_primary))
    T2.append(T)
T2 = merge_tables(T2)
print('Merged', len(T2))

I = match_radec(T2.ra, T2.dec, T2.ra, T2.dec, 1./3600, notself=True, indexlist=True)
II = np.array([i  for i,jj in enumerate(I) if jj is not None])
JJ = np.array([jj for i,jj in enumerate(I) if jj is not None])
print('matched', len(II), 'sources')
print('matched to total of', sum([len(jj) for jj in JJ]))
print('number of matches:', Counter([len(jj) for jj in JJ]).most_common())

II = cluster_radec(T2.ra, T2.dec, 1./3600.)
print(len(II), 'clusters')
print('Cluster sizes:', Counter([len(ii) for ii in II]))

#samebrick = 0
nbricks = []
nprim = []

badii = []
badj = []

for igroup,ii in enumerate(II):
    ii = np.array(ii)
    # Are the matches within the same brick?
    brick = T2.brickname[ii]
    ubricks = np.unique(brick)
    nbricks.append(len(ubricks))
    if len(ubricks) == 1:
        #samebrick += 1
        continue
    
    prim = T2.brick_primary[ii]
    nprim.append(sum(prim))
    if sum(prim) != 1:
        badii.append(ii)
        badj.append([igroup] * len(ii))

ii = np.hstack(badii)
Tbad = T2[ii]
Tbad.igroup = np.hstack(badj)
Tbad.writeto('bad.fits')
        
print('Number of bricks in cluster:', Counter(nbricks).most_common())
print('Number of primary objects in cluster:', Counter(nprim).most_common())


if False:
    I,J,d = match_radec(T2.ra, T2.dec, T2.ra, T2.dec, 1./3600., notself=True)
    K = (I < J)
    I = I[K]
    J = J[K]
    d1 = d[K]
    print('Matched', len(I))
    
    K = np.flatnonzero(T2.brickname[I] != T2.brickname[J])
    print(len(K), 'matches from different bricks (all objs, DR2)')
    
    IK = I[K]
    JK = J[K]
    primaryI = T2.brick_primary[IK]
    primaryJ = T2.brick_primary[JK]
    print(sum(np.logical_xor(primaryI, primaryJ)), 'are primary in only one brick')
    print(sum(np.logical_and(primaryI, primaryJ)), 'are primary in both bricks')
    print(sum(np.logical_and(np.logical_not(primaryI), np.logical_not(primaryJ))), 'are primary in neither brick')
    
    Kboth = np.flatnonzero(np.logical_and(primaryI, primaryJ))
    Kneither = np.flatnonzero(np.logical_and(np.logical_not(primaryI), np.logical_not(primaryJ)))
    
    Kb = np.hstack(zip(IK[Kboth], JK[Kboth]))
    T2[Kb].writeto('both.fits')
    
    Kn = np.hstack(zip(IK[Kneither], JK[Kneither]))
    T2[Kn].writeto('neither.fits')


# P2 = T2[T2.brick_primary]
# I,J,d = match_radec(P2.ra, P2.dec, P2.ra, P2.dec, 1./3600., notself=True)
# K = (I < J)
# I = I[K]
# J = J[K]
# d2 = d[K]
# 
# K = np.flatnonzero(P2.brickname[I] != P2.brickname[J])
# print(len(K), 'matches from different bricks (primary, DR2)')
# 
# plt.clf()
# n,b,p1 = plt.hist(d1 * 3600., bins=50, range=(0,1), histtype='step', color='r',
#               log=True)
# n,b,p2 = plt.hist(d2 * 3600., bins=50, range=(0,1), histtype='step', color='b',
#               log=True)
# plt.legend((p1[0],p2[0]), ('All', 'Primary'))
# plt.title('DR2 matches')
# plt.xlabel('Match distance (arcsec)')
# ps.savefig()
# 
# 
# 
# I,J,d = match_radec(T3.ra, T3.dec, T3.ra, T3.dec, 1./3600., notself=True)
# K = (I < J)
# I = I[K]
# J = J[K]
# d1 = d[K]
# 
# K = np.flatnonzero(T3.brickname[I] != T3.brickname[J])
# print(len(K), 'matches from different bricks (all objs, DR3)')
# 
# P3 = T3[T3.brick_primary]
# I,J,d = match_radec(P3.ra, P3.dec, P3.ra, P3.dec, 1./3600., notself=True)
# print(len(I), 'primary matches, DR3, dups')
# K = (I < J)
# I = I[K]
# J = J[K]
# print(len(I), 'primary matches, DR3')
# d2 = d[K]
# 
# K = np.flatnonzero(P3.brickname[I] != P3.brickname[J])
# print(len(K), 'matches from different bricks (primary, DR3)')
# matches = P3[np.array(zip(I[K], J[K])).flat]
# matches.writeto('dbrick-dr3.fits')
# 
# #  PSF  2412p050  -> BRICKQ 0
# #  SIMP 2415p050  -> BRICKQ 1   ~ 1% of the flux
# 
# plt.clf()
# n,b,p1 = plt.hist(d1 * 3600., bins=50, range=(0,1), histtype='step',
#                   color='r', log=True)
# n,b,p2 = plt.hist(d2 * 3600., bins=50, range=(0,1), histtype='step',
#                   color='b', log=True)
# plt.legend((p1[0],p2[0]), ('All', 'Primary'))
# plt.title('DR3 matches')
# plt.xlabel('Match distance (arcsec)')
# ps.savefig()
# 
# 
# i = np.argmin(d2)
# nearest = P3[np.array([I[i],J[i]])]
# nearest.writeto('nearest.fits')


