from __future__ import print_function
from astrometry.util.fits import *
from glob import glob
import os
import pylab as plt
import numpy as np
from astrometry.util.plotutils import *

ps = PlotSequence('rexpsf')

dirnm = 'cosmos-52-rex2'
#dirnm = 'cosmos-50-rex2'
#allfns = glob(os.path.join(dirnm, 'metrics', '*', 'all-models-*.fits'))
#allfns.sort()
#for fn in allfns:
#    T = fits_table(fn)
#    print(fn, '->', len(T))
TT = []
for brick in ['1498p017', '1498p020', '1498p022', '1498p025', '1501p017', '1501p020', '1501p022', '1501p025', '1503p017', '1503p020', '1503p022', '1503p025', '1506p017', '1506p020', '1506p022', '1506p025']:
    fn = os.path.join(dirnm, 'metrics', brick[:3], 'all-models-%s.fits' % brick)
    T = fits_table(fn)
    print(fn, '->', len(T))

    fn = os.path.join(dirnm, 'tractor', brick[:3], 'tractor-%s.fits' % brick)
    T2 = fits_table(fn)
    print(fn, '->', len(T2))
    T.bx = T2.bx
    T.by = T2.by

    fn = os.path.join(dirnm, 'coadd', brick[:3], brick, 'legacysurvey-%s-image.jpg' % brick)
    jpg = plt.imread(fn)

    jpg = np.flipud(jpg)
    T.ix = np.round(T.bx).astype(int)
    T.iy = np.round(T.by).astype(int)
    H,W,d = jpg.shape
    S = 15
    T.cut((T.ix >= S) * (T.iy >= S) * (T.ix < (W-S)) * (T.iy < (H-S)))
    stamps = []
    for i in range(len(T)):
        stamps.append((jpg[T.iy[i] - S : T.iy[i] + S + 1,
                           T.ix[i] - S : T.ix[i] + S + 1, :]))
    T.stamp = stamps
    
    TT.append(T)
T = merge_tables(TT)
print('Total', len(T))
T.ispsf = (T.type == 'PSF ')
T.isrex = (T.type == 'REX ')
T.cut(np.logical_or(T.ispsf, T.isrex))
print(len(T), 'PSF/REX')

plt.clf()
ha = dict(bins=100, range=(0, 1), histtype='step')
I = T.isrex
plt.hist(T.rex_shapeexp_r[I], color='r', label='REX', **ha)
I = T.ispsf
plt.hist(T.rex_shapeexp_r[I], color='b', label='PSF', **ha)
plt.ylim(0, 2000)
plt.xlabel('REX radius (arcsec)')
plt.axvline(0.02, color='k', alpha=0.25)
plt.axvline(0.08, color='k', alpha=0.25)
plt.legend()
ps.savefig()

plt.clf()
T.dchisq_psf = T.dchisq[:,0]
T.dchisq_rex = T.dchisq[:,1]
I = T.isrex
plt.plot(T.dchisq_psf[I], T.dchisq_rex[I], 'r.', label='REX')
I = T.ispsf
plt.plot(T.dchisq_psf[I], T.dchisq_rex[I], 'b.', label='PSF')
plt.xlabel('dchisq(PSF)')
plt.ylabel('dchisq(REX)')
plt.legend()
ps.savefig()


# for I,name in [(T.isrex, 'REX'), (T.ispsf, 'PSF')]:
#     plt.clf()
#     # plt.scatter(T.dchisq_psf[I], T.dchisq_rex[I], c=T.rex_shapeexp_r[I],
#     #             vmin=0, vmax=0.3, cmap='jet', s=5)
#     # plt.axis([1e2, 1e7]*2)
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt.scatter(T.dchisq_psf[I], T.dchisq_rex[I] - T.dchisq_psf[I],
#                 c=T.rex_shapeexp_r[I],
#                 vmin=0, vmax=0.2, cmap='jet', s=5)
#     plt.axis([1e1, 1e7, -1e3, 1e5])
#     plt.xscale('log')
#     plt.yscale('symlog')
#     plt.xlabel('dchisq(PSF)')
#     plt.ylabel('dchisq(REX) - dchisq(PSF)')
#     plt.title(name + ': color = REX radius')
#     plt.colorbar()
#     ps.savefig()



plt.clf()
plt.scatter(T.dchisq_psf, T.dchisq_rex - T.dchisq_psf,
            c=T.rex_shapeexp_r,
            vmin=0, vmax=0.2, cmap='jet', s=5, alpha=0.5)
xx = np.logspace(1, 7, 200)
plt.plot(xx, xx*0.01, 'k--')
plt.plot(xx, xx*0.005, 'k:')
plt.plot(xx, xx*0.0025, 'k-.')
plt.axvline(1e6, color='k', linestyle='--')
plt.axis([1e1, 1e7, -1e2, 1e6])
plt.xscale('log')
plt.yscale('symlog')
plt.xlabel('dchisq(PSF)')
plt.ylabel('dchisq(REX) - dchisq(PSF)')
plt.axhline(1., color='k')
plt.title('color: REX radius')
plt.colorbar()
ps.savefig()
    
T.fcut = np.logical_or((T.dchisq_rex - T.dchisq_psf) < (T.dchisq_psf * 0.01),
                       T.dchisq_psf > 1e6)

plt.clf()
ha = dict(bins=50, range=(0, 0.8), histtype='step')
I = (T.isrex * (T.fcut == False))
plt.hist(T.rex_shapeexp_r[I], color='r', label='REX - cut', **ha)

cut_rex = (T.isrex * (T.fcut == True))
I = cut_rex
plt.hist(T.rex_shapeexp_r[I], color='g', label='REX (cut)', **ha)

I = T.ispsf
n,b,p = plt.hist(T.rex_shapeexp_r[I], color='b', label='PSF', **ha)

I = np.logical_or(T.ispsf, cut_rex)
plt.hist(T.rex_shapeexp_r[I], color='m', label='PSF + REX(cut)', **ha)

plt.xlabel('REX radius (arcsec)')
#plt.ylim(0, np.sort(n)[-2])
plt.ylim(0, 2000)
#plt.axvline(0.02, color='k', alpha=0.25)
#plt.axvline(0.08, color='k', alpha=0.25)
plt.legend()
plt.title('1 % cut')
ps.savefig()



T.fcut2 = np.logical_or((T.dchisq_rex - T.dchisq_psf) < (T.dchisq_psf * 0.005),
                        T.dchisq_psf > 1e6)
                        

plt.clf()
I = (T.isrex * (T.fcut2 == False))
plt.hist(T.rex_shapeexp_r[I], color='r', label='REX - cut2', **ha)
cut_rex = (T.isrex * (T.fcut2 == True))
I = cut_rex
plt.hist(T.rex_shapeexp_r[I], color='g', label='REX (cut2)', **ha)
I = T.ispsf
n,b,p = plt.hist(T.rex_shapeexp_r[I], color='b', label='PSF', **ha)
I = np.logical_or(T.ispsf, cut_rex)
plt.hist(T.rex_shapeexp_r[I], color='m', label='PSF + REX(cut2)', **ha)
plt.xlabel('REX radius (arcsec)')
plt.ylim(0, 2000)
plt.legend()
plt.title('1/2 % cut')
ps.savefig()



T.fcut3 = np.logical_or((T.dchisq_rex - T.dchisq_psf) < (T.dchisq_psf * 0.0025),
                        T.dchisq_psf > 1e6)

plt.clf()
I = (T.isrex * (T.fcut3 == False))
plt.hist(T.rex_shapeexp_r[I], color='r', label='REX - cut3', **ha)
cut_rex = (T.isrex * (T.fcut3 == True))
I = cut_rex
plt.hist(T.rex_shapeexp_r[I], color='g', label='REX (cut3)', **ha)
I = T.ispsf
n,b,p = plt.hist(T.rex_shapeexp_r[I], color='b', label='PSF', **ha)
I = np.logical_or(T.ispsf, cut_rex)
plt.hist(T.rex_shapeexp_r[I], color='m', label='PSF + REX(cut3)', **ha)
plt.xlabel('REX radius (arcsec)')
plt.ylim(0, 2000)
plt.legend()
plt.title('1/4 % cut')
ps.savefig()


# Stamps for some objects caught by the 1% cut.

for I,name in [(np.flatnonzero(T.isrex * (T.fcut == False)), '1% cut'),
               (np.flatnonzero(T.isrex * (T.fcut2 == False)), '1/2 % cut'),
               (np.flatnonzero(T.isrex * (T.fcut3 == False)), '1/4 % cut'),
               (np.flatnonzero(T.isrex * (T.fcut == True) * (T.fcut2 == False)), '1/2 % cut but not 1 %'),
               (np.flatnonzero(T.isrex * (T.fcut2 == True) * (T.fcut3 == False)), '1/4 % cut but not 1/2 %'),
               ]:
    k = 0
    img = []
    for r in range(10):
        imrow = []
        for c in range(15):
            imrow.append(T.stamp[I[k]])
            k += 1
        imrow = np.concatenate(imrow, axis=1)
        img.append(imrow)
    img = np.concatenate(img, axis=0)
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower')
    plt.xticks([]); plt.yticks([])
    plt.title('Cut by %s' % name)
    ps.savefig()



plt.clf()
plt.scatter(T.dchisq_psf, (T.dchisq_rex - T.dchisq_psf) / T.dchisq_psf,
            c=T.rex_shapeexp_r,
            vmin=0, vmax=0.2, cmap='jet', s=5, alpha=0.5)
#xx = np.logspace(1, 7, 200)
#plt.plot(xx, xx*0.01, 'k--')
#plt.plot(xx, xx*0.005, 'k:')
#plt.axvline(1e6, color='k', linestyle='--')
#plt.axis([1e1, 1e7, -1e2, 1e6])
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('dchisq(PSF)')
plt.ylabel('(dchisq(REX) - dchisq(PSF)) / dchisq(PSF)')
#plt.axhline(1., color='k')
#plt.axis([1e1, 1e8, 0.5, 5])
plt.axis([1e1, 1e8, -0.05, 0.05])
plt.title('color: REX radius')
plt.colorbar()
ps.savefig()


plt.axis([1e1, 1e8, -0.01, 0.01])
ps.savefig()


T.fdiff = (T.dchisq_rex - T.dchisq_psf) / T.dchisq_psf

plt.clf()
ha = dict(range=(-0.01, 0.02), bins=100, histtype='step')
I = (T.rex_shapeexp_r > 0.2)
plt.hist(T.fdiff[I], color='r', label='Radius > 0.2', **ha)
I = ((T.rex_shapeexp_r >= 0.02) * (T.rex_shapeexp_r <= 0.2))
plt.hist(T.fdiff[I], color='g', label='Radius between 0.02 and 0.2', **ha)
I = (T.rex_shapeexp_r < 0.02)
plt.hist(T.fdiff[I], color='b', label='Radius < 0.02', **ha)
plt.xlabel('(dchisq(REX) - dchisq(PSF)) / dchisq(PSF)')
plt.legend()
plt.xlim(-0.01, 0.02)
ps.savefig()
