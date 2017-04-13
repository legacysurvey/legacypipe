from __future__ import print_function
from __future__ import division
from tractor import *
from legacypipe.survey import RexGalaxy, LogRadius
import pylab as plt
import numpy as np
from astrometry.util.plotutils import *

ps = PlotSequence('rex')
#h,w = 100,100
#h,w = 50,50
h,w = 25,25

#import tractor.galaxy
#tractor.galaxy.debug_ps = PlotSequence('gal')

psfh,psfw = 29,29
#psf_sigma = 2.35
psf_sigma = 3.
#psfh,psfw = 31,31
xx,yy = np.meshgrid(np.arange(psfw), np.arange(psfh))
psfimg = np.exp((-0.5 * ((xx-psfw//2)**2 + (yy-psfh//2)**2) / psf_sigma**2))
psfimg /= np.sum(psfimg)

row = psfimg[0,:]
print('Bottom row of PSF img: absmax', row[np.argmax(np.abs(row))], row)

pixpsf = PixelizedPSF(psfimg)
psf = HybridPixelizedPSF(pixpsf)

tim = Image(data=np.zeros((h,w)), invvar=np.ones((h,w)), psf=psf)

pos = PixPos(h//2, w//2)
flux = 100.
bright = Flux(flux)
rex = RexGalaxy(pos, bright, LogRadius(0.))
psf = PointSource(pos, bright)

tractor = Tractor([tim], [rex])
cat = tractor.getCatalog()

mask = ModelMask(0, 0, w, h)
mm = [{rex:mask, psf:mask}]
tractor.setModelMasks(mm)


plt.clf()
row = psfimg[:,psfw//2] * flux
row = np.hstack((row, np.zeros(5)))
plt.plot(row, 'k.-')

from astrometry.util.miscutils import lanczos_filter
from scipy.ndimage.filters import correlate1d
for mux in [0.1, 0.25, 0.5]:
    L = 3
    Lx = lanczos_filter(L, np.arange(-L, L+1) + mux)
    Lx /= Lx.sum()
    cx = correlate1d(row,  Lx, mode='constant')
    plt.plot(cx, '.-')
plt.yscale('symlog', linthreshy=1e-10)
ps.savefig()

#for re in [10., 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-10]:
#for re in [2e-3, 1.05e-3, 1e-3, 0.95e-3, 5e-4]:
for re in np.linspace(0.9e-3, 1.1e-3, 25):
    rex.pos.x = psf.pos.x = 12.
    rex.pos.y = psf.pos.y = 16.
    rex.shape.logre = np.log(re)
    print('Rex:', rex)
    cat[0] = rex
    rexmod = tractor.getModelImage(0)
    cat[0] = psf
    psfmod = tractor.getModelImage(0)

    rex.pos.x = psf.pos.x = 12.5
    rex.pos.y = psf.pos.y = 15.75

    cat[0] = rex
    rexmod2 = tractor.getModelImage(0)
    cat[0] = psf
    psfmod2 = tractor.getModelImage(0)

    mx = psfmod.max()
    dmx = np.abs(rexmod - psfmod).max()

    dmx2 = np.abs(rexmod2 - psfmod2).max()

    doff = max(dmx, dmx2)
    
    plt.clf()
    #ima = dict(vmin=-0.1*mx, vmax=mx, ticks=False)
    ima = dict(vmin=-6+np.log10(mx), vmax=np.log10(mx), ticks=False)
    plt.subplot(2,3,1)
    #dimshow(rexmod, **ima)
    dimshow(np.log10(rexmod + doff), **ima)
    plt.title('REX (centered)')
    plt.subplot(2,3,2)
    #dimshow(psfmod, **ima)
    dimshow(np.log10(psfmod + doff), **ima)
    plt.title('PSF (centered)')
    plt.subplot(2,3,3)
    dimshow(rexmod - psfmod, vmin=-dmx, vmax=dmx, cmap='RdBu', ticks=False)
    plt.title('diff: %.3g' % dmx)

    row = rexmod2[2,:]
    print('REX (shifted) 3rd-bottom:', row[np.argmax(np.abs(row))], row)
    row = psfmod2[2,:]
    print('PSF (shifted) 3rd-bottom:', row[np.argmax(np.abs(row))], row)
    
    row = rexmod2[1,:]
    print('REX (shifted) 2nd-bottom:', row[np.argmax(np.abs(row))], row)
    row = psfmod2[1,:]
    print('PSF (shifted) 2nd-bottom:', row[np.argmax(np.abs(row))], row)

    row = rexmod2[0,:]
    print('REX (shifted) bottom:', row[np.argmax(np.abs(row))], row)
    row = psfmod2[0,:]
    print('PSF (shifted) bottom:', row[np.argmax(np.abs(row))], row)

    plt.subplot(2,3,4)
    #dimshow(rexmod2, **ima)
    dimshow(np.log10(rexmod2 + doff), **ima)
    plt.title('REX (shifted)')
    plt.subplot(2,3,5)
    #dimshow(psfmod2, **ima)
    dimshow(np.log10(psfmod2 + doff), **ima)
    plt.title('PSF (shifted)')
    plt.subplot(2,3,6)
    dimshow(rexmod2 - psfmod2, vmin=-dmx, vmax=dmx, cmap='RdBu', ticks=False)
    plt.title('diff: %.3g' % dmx2)

    plt.suptitle('R_e = %g' % re)
    ps.savefig()

    print('Centered diff:', dmx, 'vs shifted diff:', dmx2)
