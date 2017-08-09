from __future__ import print_function
from astrometry.util.fits import *
import pylab as plt
import numpy as np
from glob import glob
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *
from astrometry.util.resample import *
from astrometry.util.util import *

ps = PlotSequence('cosmos')

baseA = 'cosmos-dr5-60/'
baseB = 'cosmos-dr5-67/'

Atxt = '60'
Btxt = '67'

TA = merge_tables([fits_table(fn) for fn in glob(baseA + 'tractor/*/tractor-*.fits')])
print('Total of', len(TA), 'sources in 60')
TA.cut(TA.brick_primary)
print(len(TA), 'brick primary')

TB = merge_tables([fits_table(fn) for fn in glob(baseB + 'tractor/*/tractor-*.fits')])
print('Total of', len(TB), 'sources in 67')
TB.cut(TB.brick_primary)
print(len(TB), 'brick primary')

ramin  = min(TA.ra.min(),  TB.ra.min())
ramax  = max(TA.ra.max(),  TB.ra.max())
decmin = min(TA.dec.min(), TB.dec.min())
decmax = max(TA.dec.max(), TB.dec.max())

# Create low-res depth maps
pixsc = 10. * 0.262/3600.
rc,dc = (ramin+ramax)/2., (decmin+decmax)/2.
w = int((ramax - ramin) * np.cos(np.deg2rad(dc)) / pixsc)
h = int((decmax - decmin) / pixsc)
wcs = Tan(rc, dc, w/2., h/2., -pixsc, 0., 0., pixsc, float(w), float(h))
#print('WCS:', wcs)

#for band in ['g','r','z']:
for band in ['g']:
    psfdepthA = np.zeros(wcs.shape, np.float32)
    psfdepthB = np.zeros(wcs.shape, np.float32)
    for fn in glob(baseA + 'coadd/*/*/legacysurvey-*-depth-%s.fits*' % band):
        print('Reading', fn)
        iwcs = Tan(fn, 1)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, iwcs)
        dmap = fitsio.read(fn)
        #I = np.flatnonzero(np.isfinite(dmap) * (dmap > 0))
        #print(len(I), 'finite & positive values')
        psfdepthA[Yo,Xo] = dmap[Yi,Xi]
    for fn in glob(baseB + 'coadd/*/*/legacysurvey-*-depth-%s.fits*' % band):
        print('Reading', fn)
        iwcs = Tan(fn, 1)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, iwcs)
        dmap = fitsio.read(fn)
        #I = np.flatnonzero(np.isfinite(dmap) * (dmap > 0))
        #print(len(I), 'finite & positive values')
        psfdepthB[Yo,Xo] = dmap[Yi,Xi]

    galdepthA = np.zeros(wcs.shape, np.float32)
    galdepthB = np.zeros(wcs.shape, np.float32)
    for fn in glob(baseA + 'coadd/*/*/legacysurvey-*-galdepth-%s.fits*' % band):
        print('Reading', fn)
        iwcs = Tan(fn, 1)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, iwcs)
        dmap = fitsio.read(fn)
        #I = np.flatnonzero(np.isfinite(dmap) * (dmap > 0))
        #print(len(I), 'finite & positive values')
        galdepthA[Yo,Xo] = dmap[Yi,Xi]
    for fn in glob(baseB + 'coadd/*/*/legacysurvey-*-galdepth-%s.fits*' % band):
        print('Reading', fn)
        iwcs = Tan(fn, 1)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, iwcs)
        dmap = fitsio.read(fn)
        #I = np.flatnonzero(np.isfinite(dmap) * (dmap > 0))
        #print(len(I), 'finite & positive values')
        galdepthB[Yo,Xo] = dmap[Yi,Xi]
    
    print('PsfdepthA (iv)', psfdepthA.min(), psfdepthA.max())
    print('PsfdepthB (iv)', psfdepthB.min(), psfdepthB.max())
    psfdepthA = -2.5 * (np.log10(5./np.sqrt(psfdepthA)) - 9)
    psfdepthB = -2.5 * (np.log10(5./np.sqrt(psfdepthB)) - 9)
    print('PsfdepthA', psfdepthA.min(), psfdepthA.max())
    print('PsfdepthB', psfdepthB.min(), psfdepthB.max())
    galdepthA = -2.5 * (np.log10(5./np.sqrt(galdepthA)) - 9)
    galdepthB = -2.5 * (np.log10(5./np.sqrt(galdepthB)) - 9)
    print('GaldepthA', galdepthA.min(), galdepthA.max())
    print('GaldepthB', galdepthB.min(), galdepthB.max())

    ima = dict(interpolation='nearest', origin='lower',
               extent=[ramax,ramin,decmin,decmax], vmin=20.0, vmax=24.5)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(psfdepthA, **ima)
    plt.title(Atxt)
    plt.subplot(1,2,2)
    plt.imshow(psfdepthB, **ima)
    plt.title(Btxt)
    plt.suptitle('PSF Depth maps (%s)' % band)
    ps.savefig()

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(galdepthA, **ima)
    plt.title(Atxt)
    plt.subplot(1,2,2)
    plt.imshow(galdepthB, **ima)
    plt.title(Btxt)
    plt.suptitle('Galaxy Depth maps (%s)' % band)
    ps.savefig()


# dd = np.append(galdepthA.ravel(), galdepthB.ravel())
# dd = dd[np.isfinite(dd)]
# thresh = np.percentile(dd, 10)
# print('Depth threshold:', thresh)
thresh = 24.0

hh,ww = wcs.shape
ok,xx,yy = wcs.radec2pixelxy(TA.ra, TA.dec)
xx = np.clip((np.round(xx) - 1), 0, ww-1).astype(int)
yy = np.clip((np.round(yy) - 1), 0, hh-1).astype(int)
I = np.flatnonzero((galdepthA[yy,xx] > thresh) * (galdepthB[yy,xx] > thresh))
print(len(I), 'of', len(TA), 'sources in A are in good-depth regions')
TA.cut(I)

ok,xx,yy = wcs.radec2pixelxy(TB.ra, TB.dec)
xx = np.clip((np.round(xx) - 1), 0, ww-1).astype(int)
yy = np.clip((np.round(yy) - 1), 0, hh-1).astype(int)
I = np.flatnonzero((galdepthA[yy,xx] > thresh) * (galdepthB[yy,xx] > thresh))
print(len(I), 'of', len(TB), 'sources in B are in good-depth regions')
TB.cut(I)


ha = dict(range=(18,27), bins=50, histtype='stepfilled', alpha=0.1)
hb = dict(range=(18,27), bins=50, histtype='stepfilled', alpha=0.1)

plt.clf()
plt.hist(np.maximum(psfdepthA.ravel(), 18), color='b', label=Atxt, **ha)
plt.hist(np.maximum(psfdepthB.ravel(), 18), color='r', label=Btxt, **hb)
plt.xlim(18,27)
plt.legend()
plt.title('PSF depth map values (g mag)')
ps.savefig()

plt.clf()
plt.hist(np.maximum(galdepthA.ravel(), 18), color='b', label=Atxt, **ha)
plt.hist(np.maximum(galdepthB.ravel(), 18), color='r', label=Btxt, **hb)
plt.xlim(18,27)
plt.legend()
plt.title('Galaxy depth map values (g mag)')
ps.savefig()


TA.mag_g = -2.5 * (np.log10(TA.flux_g) - 9)
TB.mag_g = -2.5 * (np.log10(TB.flux_g) - 9)

TA.psfdepth_mag_g = -2.5 * (np.log10(5./np.sqrt(TA.psfdepth_g)) - 9)
TB.psfdepth_mag_g = -2.5 * (np.log10(5./np.sqrt(TB.psfdepth_g)) - 9)
TA.galdepth_mag_g = -2.5 * (np.log10(5./np.sqrt(TA.galdepth_g)) - 9)
TB.galdepth_mag_g = -2.5 * (np.log10(5./np.sqrt(TB.galdepth_g)) - 9)

ha = dict(range=(18,27), bins=50, histtype='stepfilled', alpha=0.1)
hb = dict(range=(18,27), bins=50, histtype='stepfilled', alpha=0.1)
ha2 = dict(range=(18,27), bins=50, histtype='step', alpha=0.5)
hb2 = dict(range=(18,27), bins=50, histtype='step', alpha=0.5)

plt.clf()
plt.hist(TA.mag_g, color='b', label=Atxt, **ha)
plt.hist(TA.mag_g, color='b', **ha2)
plt.hist(TB.mag_g, color='r', label=Btxt, **hb)
plt.hist(TB.mag_g, color='r', **hb2)
plt.xlim(18,27)
plt.legend()
plt.xlabel('All sources: g mag')
ps.savefig()

ha = dict(range=(23,25), bins=50, histtype='stepfilled', alpha=0.1)
hb = dict(range=(23,25), bins=50, histtype='stepfilled', alpha=0.1)

plt.clf()
plt.hist(TA.psfdepth_mag_g, color='b', label=Atxt, **ha)
plt.hist(TB.psfdepth_mag_g, color='r', label=Btxt, **hb)
plt.xlim(23,25)
plt.legend()
plt.title('PSF depth for sources (g mag)')
ps.savefig()

plt.clf()
plt.hist(TA.galdepth_mag_g, color='b', label=Atxt, **ha)
plt.hist(TB.galdepth_mag_g, color='r', label=Btxt, **hb)
plt.xlim(23,25)
plt.legend()
plt.title('Gal depth for sources (g mag)')
ps.savefig()

ha = dict(range=((ramin,ramax),(decmin,decmax)), doclf=False,
          docolorbar=False, imshowargs=dict(vmin=0, vmax=14))

plt.clf()
plt.subplot(1,2,1)
plothist(TA.ra, TA.dec, 200, **ha)
plt.title(Atxt)
plt.subplot(1,2,2)
plothist(TB.ra, TB.dec, 200, **ha)
plt.title(Btxt)
plt.suptitle('All sources')
ps.savefig()

I,J,d = match_radec(TA.ra, TA.dec, TB.ra, TB.dec, 1./3600.)

unmatchedA = np.ones(len(TA), bool)
unmatchedB = np.ones(len(TB), bool)
unmatchedA[I] = False
unmatchedB[J] = False

ha = dict(range=((ramin,ramax),(decmin,decmax)), doclf=False,
          docolorbar=False, imshowargs=dict(vmin=0, vmax=5))

plt.clf()
plt.subplot(1,2,1)
plothist(TA.ra[unmatchedA], TA.dec[unmatchedA], 200, **ha)
plt.title(Atxt)
plt.subplot(1,2,2)
plothist(TB.ra[unmatchedB], TB.dec[unmatchedB], 200, **ha)
plt.title(Btxt)
plt.suptitle('Un-matched sources')
ps.savefig()

