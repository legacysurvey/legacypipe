from legacypipe.survey import LegacySurveyData, wcs_for_brick, SimpleGalaxy
from tractor import NanoMaggies
import numpy as np
import pylab as plt

survey = LegacySurveyData()

brick = survey.get_brick_by_name('0006m062')
print('Brick', brick)
targetwcs = wcs_for_brick(brick)
H,W = targetwcs.shape
targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                     [(1,1),(W,1),(W,H),(1,H),(1,1)]])
print('Target Ra,Dec:', targetrd)

ccds = survey.ccds_touching_wcs(targetwcs)

#ccds = survey.find_ccds(expnum=239662, ccdname='S18')
ccds = ccds[(ccds.expnum == 239662) * (ccds.ccdname == 'S18')]
assert(len(ccds) == 1)
ccd = ccds[0]
im = survey.get_image_object(ccd)
tim = im.get_tractor_image(pixPsf=True, hybridPsf=True, splinesky=True,
                           radecpoly=targetrd)

fullpsf = tim.psf
th,tw = tim.shape
tim.psf = fullpsf.constantPsfAt(tw//2, th//2)

pnorm = im.psf_norm(tim)
print('PSF norm:', pnorm)

gnorm = im.galaxy_norm(tim)
print('Galaxy norm:', gnorm)

x = y = None

h,w = tim.shape
if x is None:
    x = w//2
if y is None:
    y = h//2
psfmod = tim.psf.getPointSourcePatch(x, y).patch

from tractor.galaxy import ExpGalaxy
from tractor.ellipses import EllipseE
from tractor.patch import ModelMask
band = tim.band
if x is None:
    x = w//2
if y is None:
    y = h//2
pos = tim.wcs.pixelToPosition(x, y)
gal = SimpleGalaxy(pos, NanoMaggies(**{band:1.}))
S = 32
mm = ModelMask(int(x-S), int(y-S), 2*S+1, 2*S+1)
galmod = gal.getModelPatch(tim, modelMask=mm).patch
print('Galaxy model shape', galmod.shape)

mx = max(psfmod.max(), galmod.max())
mn = min(psfmod.min(), galmod.min())
ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)

print('Range', mn,mx)

plt.clf()
plt.subplot(1,2,1)
plt.imshow(psfmod, **ima)
plt.title('PSF model')
plt.subplot(1,2,2)
plt.imshow(galmod, **ima)
plt.title('Galaxy model')
plt.savefig('norms1.png')

ima.update(vmin=-0.002, vmax=0.002)

plt.clf()
plt.subplot(1,2,1)
plt.imshow(psfmod, **ima)
plt.title('PSF model')
plt.subplot(1,2,2)
plt.imshow(galmod, **ima)
plt.title('Galaxy model')
plt.savefig('norms2.png')


plt.clf()
pnorms = []
gnorms = []
ph,pw = psfmod.shape
gh,gw = galmod.shape
rads = np.arange(1, 30)
for r in rads:
    p = psfmod[ph//2 - r : ph//2 + r+1, pw//2 - r : pw//2 +r+1]
    p = np.maximum(0, p)
    p /= p.sum()
    pnorms.append(np.sqrt(np.sum(p**2)))

    p = galmod[gh//2 - r : gh//2 + r+1, gw//2 - r : gw//2 +r+1]
    p = np.maximum(0, p)
    p /= p.sum()
    gnorms.append(np.sqrt(np.sum(p**2)))

plt.plot(rads, pnorms, color='b', label='PSF')
plt.plot(rads, gnorms, color='r', label='Gal')
plt.legend()
plt.xlabel('Patch radius')
plt.ylabel('Norm')
plt.savefig('norms3.png')

