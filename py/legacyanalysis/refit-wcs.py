import pylab as plt
import numpy as np
from astrometry.util.plotutils import *

from legacyanalysis.ps1cat import ps1cat
from legacypipe.common import LegacySurveyData

from tractor import Image, PointSource, PixPos, NanoMaggies, Tractor

ps = PlotSequence('rewcs')

expnum, ccdname = 431109, 'N14'
cat = ps1cat(expnum=expnum, ccdname=ccdname)
stars = cat.get_stars()
print len(stars), 'stars'

survey = LegacySurveyData()
ccd = survey.find_ccds(expnum=expnum,ccdname=ccdname)[0]
im = survey.get_image_object(ccd)
wcs = im.get_wcs()
tim = im.get_tractor_image(pixPsf=True, splinesky=True)

margin = 15
ok,stars.xx,stars.yy = wcs.radec2pixelxy(stars.ra, stars.dec) 
stars.xx -= 1.
stars.yy -= 1.
W,H = wcs.get_width(), wcs.get_height()
stars.ix = np.round(stars.xx).astype(int)
stars.iy = np.round(stars.yy).astype(int)
stars.cut((stars.ix >= margin) * (stars.ix < (W-margin)) *
          (stars.iy >= margin) * (stars.iy < (H-margin)))

plt.clf()
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                    hspace=0, wspace=0)

# transpose image
dimshow(tim.getImage().T, vmin=-2.*tim.sig1, vmax=10.*tim.sig1)
plt.title('DECaLS ' + tim.name)
ps.savefig()
ax = plt.axis()
plt.plot(stars.iy, stars.ix, 'r.')
plt.axis(ax)
plt.title('Pan-STARRS stars in DECaLS ' + tim.name)
ps.savefig()

iband = ps1cat.ps1band[tim.band]

stars.mag = stars.median[:,iband]
stars.flux = NanoMaggies.magToNanomaggies(stars.mag)

stars.cut(stars.flux > 1.)
print len(stars), 'brighter than 22.5'

stars.cut(np.argsort(stars.mag))



ima = dict(vmin=-2.*tim.sig1, vmax=10.*tim.sig1, ticks=False)

plt.clf()

#R,C = 10,14
#R,C = 9,13
R,C = 10,13

for i,s in enumerate(stars):
    plt.subplot(R, C, i+1)
    dimshow(tim.getImage()[s.iy - margin : s.iy + margin+1,
                           s.ix - margin : s.ix + margin+1], **ima)
plt.suptitle('DECaLS images of Pan-STARRS stars: ' + tim.name)
ps.savefig()

resids = []
chis = []

tractors = []

stars.x0 = stars.ix - margin
stars.y0 = stars.iy - margin

plt.clf()
for i,s in enumerate(stars):

    #x0,y0 = s.ix - margin, s.iy - margin
    x0,y0 = s.x0, s.y0
    slc = (slice(y0, s.iy + margin+1),
           slice(x0, s.ix + margin+1))
    subpsf = tim.psf.constantPsfAt(s.ix, s.iy)
    
    subtim = Image(data=tim.data[slc],
                   inverr=tim.inverr[slc],
                   psf=subpsf, photocal=tim.photocal,
                   name=tim.name + '/star %i' % i)

    flux = NanoMaggies.magToNanomaggies(s.mag)
    print 'Flux:', flux
    flux = max(flux, 1.)
    src = PointSource(PixPos(s.xx - x0, s.yy - y0),
                      NanoMaggies(**{tim.band: flux}))

    tr = Tractor([subtim], [src])
    mod = tr.getModelImage(0)
    #mod = src.getModelImage(tsubtim)
    tractors.append(tr)
    
    resids.append((subtim.data - mod) * (subtim.inverr > 0))
    chis.append((subtim.data - mod) * subtim.inverr)
    
    plt.subplot(R, C, i+1)
    dimshow(mod, **ima)
plt.suptitle('Models for Pan-STARRS stars: ' + tim.name)
ps.savefig()

resa = dict(vmin=-5.*tim.sig1, vmax=5.*tim.sig1, ticks=False)
chia = dict(vmin=-5., vmax=5., ticks=False)

plt.clf()
for i,resid in enumerate(resids):
    plt.subplot(R, C, i+1)
    dimshow(resid, **resa)
plt.suptitle('Residuals: ' + tim.name)
ps.savefig()

plt.clf()
for i,chi in enumerate(chis):
    plt.subplot(R, C, i+1)
    dimshow(chi, **chia)
plt.suptitle('Chis: ' + tim.name)
ps.savefig()


mods2 = []
resids2 = []
chis2 = []

stars.xfit = stars.xx.copy()
stars.yfit = stars.yy.copy()

alphas = [0.1, 0.3, 1.0]
for i,tr in enumerate(tractors):
    print tr
    src = tr.catalog[0]
    print 'Initial position:', src.pos
    x,y = src.pos.x, src.pos.y
    tr.freezeParam('images')
    tr.printThawedParams()
    for step in range(50):
        dlnp,X,alpha = tr.optimize(priors=False, shared_params=False,
                                   alphas=alphas)
        print 'dlnp', dlnp
        print 'pos', src.pos.x, src.pos.y
        print 'Delta position:', src.pos.x - x, src.pos.y - y
        #if dlnp < 0.1:
        if dlnp == 0.:
            break
    print 'Final position:', src.pos

    pos = src.getPosition()
    stars.xfit[i] = stars.x0[i] + pos.x
    stars.yfit[i] = stars.y0[i] + pos.y
        
    mod = tr.getModelImage(0)
    mods2.append(mod)
    subtim = tr.images[0]
    resids2.append((subtim.data - mod) * (subtim.inverr > 0))
    chis2.append((subtim.data - mod) * subtim.inverr)

plt.clf()
for i,mod in enumerate(mods2):
    plt.subplot(R, C, i+1)
    dimshow(mod, **ima)
plt.suptitle('Fit Models: ' + tim.name)
ps.savefig()

plt.clf()
for i,resid in enumerate(resids2):
    plt.subplot(R, C, i+1)
    dimshow(resid, **resa)
plt.suptitle('Fit Residuals: ' + tim.name)
ps.savefig()
    
plt.clf()
for i,chi in enumerate(chis2):
    plt.subplot(R, C, i+1)
    dimshow(chi, **chia)
    ax = plt.axis()
    plt.text(0, 0, '%.0f' % np.sum(chi**2),
             ha='left', va='bottom', color='r', fontsize=8)
plt.suptitle('Fit Chis: ' + tim.name)
ps.savefig()

# Vector plot
dx = stars.xfit - stars.xx
dy = stars.yfit - stars.yy

print 'dx,dy:', dx,dy

print 'dx, dy:'
for x,y in zip(dx,dy):
    print ' ', x, y


plt.clf()
dimshow(tim.getImage().T, vmin=-2.*tim.sig1, vmax=10.*tim.sig1)
ax = plt.axis()
plt.plot(stars.iy, stars.ix, 'r.')
plt.plot(np.vstack((stars.yy, stars.yy + dy*100)),
         np.vstack((stars.xx, stars.xx + dx*100)), 'r-')
plt.axis(ax)
plt.title('Delta-positions of Pan-STARRS stars: ' + tim.name)
ps.savefig()

