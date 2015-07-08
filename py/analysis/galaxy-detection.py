import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from collections import Counter

from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.util import *
from astrometry.util.plotutils import *

from tractor import *

from legacypipe.runbrick import *

'''
Investigate the detectability of galaxies as a function of radius,
seeing, etc., using the DECaLS pipeline.  Also check out what types we
think they are.

'''

class FakeDecals(object):
    def __init__(self):
        self.decals_dir = ''
        self.ccds = None
        self.tims = None

    def ccds_touching_wcs(self, targetwcs):
        return self.ccds

    def get_image_object(self, t):
        return FakeImage(self, t)

class FakeImage(object):
    def __init__(self, decals, t):
        print 'FakeImage:', t
        self.tim = decals.tims[t.index]

    def run_calibs(self, ra, dec, pixscale, mock_psf, W=2048, H=4096,
                   pvastrom=True, psfex=True, sky=True,
                   se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False):
        pass

    def get_tractor_image(self, slc=None, radecpoly=None,
                          mock_psf=False, const2psf=False,
                          nanomaggies=True, subsky=True, tiny=5):
        return self.tim
        
    
W,H = 40,40
ra,dec = 0.,0.
band = 'r'
bands = [band]

decals = FakeDecals()
ccds = fits_table()
ccds.filter = [band]
ccds.to_np_arrays()
ccds.index = np.arange(len(ccds))
decals.ccds = ccds

pixscale = 0.262 / 3600.
wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
          -pixscale, 0., 0., pixscale, float(W), float(H))

data = np.zeros((H,W), np.float32)

psfsig = 2.

var1 = psfsig**2
var2 = (2.*psfsig)**2
psf = GaussianMixturePSF(0.9, 0.1, 0., 0., 0., 0.,
                         var1, var1, 0., var2, var2, 0.)

twcs = ConstantFitsWcs(wcs)
photocal = LinearPhotoCal(1., band=band)
sig1 = 1.

tim = Image(data=data, invvar=np.ones_like(data) / sig1**2, psf=psf, wcs=twcs,
            photocal=photocal)
tim.subwcs = wcs
tim.band = band
tim.psf_sigma = psfsig
tim.sig1 = sig1
decals.tims = [ tim ]


# Render synthetic source into image
flux = 300.
re = 0.45
gal = ExpGalaxy(RaDecPos(ra, dec), NanoMaggies(**{ band: flux }),
                EllipseESoft(np.log(re), 0., 0.))

tr = Tractor([tim], [gal])
mod = tr.getModelImage(0)

mp = multiproc()
ps = PlotSequence('galdet')

allcats = []

for i in range(100):

    np.random.seed(10000 + i)

    tim.data = mod + sig1 * np.random.normal(size=data.shape)

    if i == 0:
        plt.clf()
        dimshow(tim.data, vmin=-2.*sig1, vmax=5.*sig1)
        plt.title('Simulated canonical DESI ELG source (re = 0.45")')
        ps.savefig()

    kwa = dict(W=W, H=H, ra=ra, dec=dec, bands=''.join(bands), decals=decals,
               mp=mp)

    R = stage_tims(**kwa)
    kwa.update(R)
    R = stage_srcs(no_sdss=True, **kwa)
    kwa.update(R)
    R = stage_fitblobs(**kwa)
    kwa.update(R)
    R = stage_fitblobs_finish(write_metrics=False, **kwa)
    kwa.update(R)
    #T.anymask = np.zeros((len(T), len(bands)), np.float32)
    #T.allmask = np.zeros((len(T), len(bands)), np.float32)
    #T.nobs = np.ones((len(T), len(bands)), int)
    #T.oob = np.zeros(len(T), bool)
    #R = stage_writecat(write_catalog=False, **kwa)
    #kwa.update(R)
    #T = kwa['T2']
    #print len(T), 'sources'
    #T.about()
    #for t in T:
    #    print 'Type', t.type, 'flux', t.decam_flux[iband]
    #allbands = 'ugrizY'
    #iband = allbands.index(band)

    cat = kwa['cat']
    print len(cat), 'catalog objects'

    print 'Catalog:'
    for src in cat:
        print '  ', src
        print '  type', type(src), 'flux', src.getBrightness().getFlux(band)

    allcats.append(cat)
    
    
plt.clf()
nsrcs = [len(c) for c in allcats]
plt.hist(nsrcs, bins=np.arange(max(nsrcs)+2)-0.5)
plt.xlabel('Number of detected sources')
ps.savefig()

count = Counter(nsrcs)
print 'Histogram of number of detections:'
for k,n in count.most_common():
    print k, ':', n

types = Counter([type(cat[0]) for cat in allcats if len(cat) == 1])
print 'Histogram of returned object types:'
for k,n in types.most_common():
    print k, ':', n
    
cleancats = [cat for cat in allcats if len(cat) == 1]

typemap = { ExpGalaxy: 'E',
            DevGalaxy: 'D',
            FixedCompositeGalaxy: 'C',
            PointSource: 'P' }

TT = fits_table()
TT.type = np.array([typemap[type(c[0])] for c in cleancats])
TT.flux = np.array([c[0].getBrightness().getFlux(band)
                    for c in cleancats])
RE = []
for c in cleancats:
    src = c[0]
    if isinstance(src, (ExpGalaxy, DevGalaxy)):
        RE.append(src.shape.re)
    else:
        RE.append(0)
TT.re = np.array(RE)
        
ccmap = dict(E='r', D='b', C='m', P='g')

plt.clf()
lp = []
lt = []
maxn = 1
for t in typemap.values():
    I = np.flatnonzero(TT.type == t)
    if len(I) == 0:
        continue
    n,b,p = plt.hist(TT.flux[I], bins=25,
                  range=(0.5*flux, 1.5*flux), histtype='step',
                  color=ccmap[t])
    maxn = max(maxn, max(n))
    lp.append(p[0])
    lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(allcats)))))
plt.legend(lp, lt)
plt.xlabel('Measured flux')
plt.axvline(flux, color='k', ls='--')
plt.ylim(0, 1.1*maxn)
plt.title('Canonical DESI ELG source -- DR1 Type & Flux measurements')
ps.savefig()


plt.clf()
lp = []
lt = []
for t in ['E','D','P']:
    I = np.flatnonzero(TT.type == t)
    if len(I) == 0:
        continue
    n,b,p = plt.hist(TT.re[I], bins=25,
                  range=(0., 2.*re), histtype='step',
                  color=ccmap[t])
    lp.append(p[0])
    lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(allcats)))))
plt.legend(lp, lt)
plt.xlabel('Measured effective radius (arcsec)')
plt.axvline(re, color='k', ls='--')
plt.title('Canonical DESI ELG source -- DR1 Type & Radius measurements')
ps.savefig()
    


plt.clf()
lp = []
lt = []
for t in ['E','D']:
    I = np.flatnonzero(TT.type == t)
    if len(I) == 0:
        continue
    p = plt.plot(TT.re[I], TT.flux[I], '.', color=ccmap[t], ms=10, alpha=0.5)
    lp.append(p[0])
    lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(allcats)))))
plt.legend(lp, lt)
plt.xlabel('Measured effective radius (arcsec)')
plt.axvline(re, color='k', ls='--')
plt.ylabel('Measured flux')
plt.axhline(flux, color='k', ls='--')
plt.title('Canonical DESI ELG source -- DR1 Type & Flux vs Radius measurements')
plt.axis([0., 2.*re, 0.5*flux, 1.5*flux])
ps.savefig()

