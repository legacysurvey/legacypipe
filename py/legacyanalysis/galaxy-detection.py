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

    def ccds_touching_wcs(self, targetwcs, **kwargs):
        return self.ccds

    def photometric_ccds(self, CCDs):
        return np.arange(len(CCDs))
    
    def get_image_object(self, t):
        return FakeImage(self, t)

class FakeImage(object):
    def __init__(self, decals, t):
        print 'FakeImage:', t
        self.tim = decals.tims[t.index]

    def run_calibs(self, *args, **kwargs):
        pass

    def get_tractor_image(self, **kwargs):
        return self.tim

class SimResult(object):
    pass
    
def run_sim(tims, cat, N, mods=None, **kwargs):
    result = SimResult()
    
    if mods is None:
        tr = Tractor([], cat)
        mods = [tr.getModelImage(tim) for tim in tims]

    decals = FakeDecals()
    ccds = fits_table()
    ccds.filter = [tim.band for tim in tims]
    ccds.to_np_arrays()
    ccds.index = np.arange(len(ccds))
    decals.ccds = ccds
    decals.tims = tims

    allcats = []
    allivs = []
    allTs = []
    allallmods = []
    
    for i in range(N):
        np.random.seed(10000 + i)

        for tim,mod in zip(tims, mods):
            tim.data = mod + tim.sig1 * np.random.normal(size=tim.shape)
    
        kwa = kwargs.copy()
        kwa.update(decals=decals)
        R = stage_tims(do_calibs=False, **kwa)
        kwa.update(R)
        R = stage_srcs(no_sdss=True, **kwa)
        ### At this point, R['cat'] has sources whose fluxes are based on the
        ### detection maps
        kwa.update(R)
        R = stage_fitblobs(**kwa)
        kwa.update(R)
        R = stage_fitblobs_finish(write_metrics=False, get_all_models=True,
                                  **kwa)
        kwa.update(R)
    
        cat = kwa['cat']
        print len(cat), 'catalog objects'
    
        print 'Catalog:'
        for src in cat:
            print '  ', src
            #print '  type', type(src), 'flux', src.getBrightness().getFlux(band)
    
        allcats.append(cat)
    
        iv = kwa['invvars']
        allivs.append(iv)

        allTs.append(kwa['T'])

        allallmods.append(kwa['all_models'])
        
        bands = kwa['bands']
        
    result.cats = allcats
    result.ivs = allivs
    result.bands = bands
    
    T = fits_table()
    T.nsrcs = [len(c) for c in allcats]

    typemap = { ExpGalaxy: 'E',
                DevGalaxy: 'D',
                FixedCompositeGalaxy: 'C',
                PointSource: 'P',
                SimpleGalaxy: 'S', }

    T.type = []
    T.re = []
    fluxes = [[] for b in bands]
    fluxivs = [[] for b in bands]

    TT = []
    
    for c,iv,Ti,allmods in zip(allcats, allivs, allTs, allallmods):
        print 'len(c)', len(c)
        if len(c) == 1:
            src = c[0]
            T.type.append(typemap[type(src)])
            for band,f in zip(bands, fluxes):
                f.append(src.getBrightness().getFlux(band))

            if isinstance(src, (ExpGalaxy, DevGalaxy)):
                T.re.append(src.shape.re)
            else:
                T.re.append(0.)
                
            # trickiness: extract flux inverse-variances using params interface
            params = src.getParams()
            src.setParams(iv)
            for band,fiv in zip(bands, fluxivs):
                fiv.append(src.getBrightness().getFlux(band))
            src.setParams(params)

            print 'Ti:'
            Ti.about()
            print 'adding allmods:'
            allmods.about()
            Ti.add_columns_from(allmods)
            TT.append(Ti)
        else:
            if len(c) == 0:
                T.type.append('-')
            else:
                T.type.append('+')
            for band,f,fiv in zip(bands, fluxes, fluxivs):
                f.append(0)
                fiv.append(0)
            T.re.append(0)

            TT.append(fits_table())
            

    for band,f,iv in zip(bands, fluxes, fluxivs):
        T.set('flux_%s' % band, np.array(f))
        T.set('fluxiv_%s' % band, np.array(iv))
        
    T.to_np_arrays()
    result.T = T

    result.TT = merge_tables(TT)

    print 'Returning table:'
    result.TT.about()
    
    return result




W,H = 40,40
ra,dec = 0.,0.
band = 'r'

pixscale = 0.262
ps = pixscale / 3600.
wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
          -ps, 0., 0., ps, float(W), float(H))

psf = None
twcs = ConstantFitsWcs(wcs)
photocal = LinearPhotoCal(1., band=band)
sig1 = 1.
data = np.zeros((H,W), np.float32)

tim = Image(data=data, invvar=np.ones_like(data) / sig1**2, psf=psf, wcs=twcs,
            photocal=photocal)
tim.subwcs = wcs
tim.band = band
tim.sig1 = sig1
tim.skyver = (0,0)
tim.wcsver = (0,0)
tim.psfver = (0,0)
tim.plver = 0
tim.dq = np.zeros(tim.shape, np.int16)
tim.dq_bits = dict(satur=1)

# Render synthetic source into image
re = 0.45

mp = multiproc()
ps = PlotSequence('galdet')

sourcetypes = [ 'E','D','S','C','P','-','+' ]

ccmap = dict(E='r', D='b', S='c', C='m', P='g',
             exp='r', dev='b', simp='c', comp='m', psf='g')
ccmap['+'] = 'k'
ccmap['-'] = '0.5'

namemap = { 'E': 'Exp', 'D': 'deVauc', 'C': 'composite', 'P':'PSF', 'S': 'simple', '+':'>1 src', '-':'No detection' }

for flux in [ 300. ]:#, 150. ]:

    S = fits_table()
    S.psffwhm = []
    for t in sourcetypes:
        S.set('frac_%s' % t, [])
    
    #flux = 300.
    gal = ExpGalaxy(RaDecPos(ra, dec), NanoMaggies(**{ band: flux }),
                    EllipseESoft(np.log(re), 0., 0.))

    #for psfsig in [ 1.6, 1.8, 2.0, 2.2, 2.4 ]:
    #for psfsig in [ 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4 ]:
    #for psfsig in [ 1.2, 1.8, 2.4, 3.0, 3.6 ]:

    for psfsig in [2.0]:
        #psfsig = 2.
        var1 = psfsig**2
        psf = GaussianMixturePSF(1.0, 0., 0., var1, var1, 0.)
        #var2 = (2.*psfsig)**2
        #psf = GaussianMixturePSF(0.9, 0.1, 0., 0., 0., 0.,
        #                         var1, var1, 0., var2, var2, 0.)
    
        tim.psf = psf
        tim.psf_sigma = psfsig
        tr = Tractor([tim], [gal])
        mod = tr.getModelImage(0)

        N = 100
        res = run_sim([tim], [gal], N, mods=[mod], 
                      W=W, H=H, ra=ra, dec=dec, mp=mp, bands=[band])
    
        T = res.T
        T.flux = T.flux_r
        T.fluxiv = T.fluxiv_r

        catalog = res.TT
        
        S.psffwhm.append(psfsig * 2.35 * pixscale)
        for t in sourcetypes:
            S.get('frac_%s' % t).append(
                100. * np.count_nonzero(T.type == t) / float(len(T)))
    
        if True:
            plt.clf()
            plt.hist(T.nsrcs, bins=np.arange(max(T.nsrcs)+2)-0.5)
            plt.xlabel('Number of detected sources')
            ps.savefig()

            types = Counter(T.type)
            print 'Histogram of returned object types:'
            for k,n in types.most_common():
                print k, ':', n
                
            print 'Flux S/N:', np.median(T.flux * np.sqrt(T.fluxiv))
            print 'Flux S/N:', T.flux * np.sqrt(T.fluxiv)

            plt.subplots_adjust(hspace=0)
            plt.clf()
            # dchisq array:
            # ptsrc, simple, dev, exp, comp  relative to 'none'
            #modelindex = dict(P=0, S=1, D=2, E=3, C=4)
            models = 'PSDEC'
            
            dchisq = catalog.dchisq
            lo,hi = dchisq.min(), dchisq.max()
            lp,lt = [],[]
            maxn = 1
            for i,t in enumerate('PSDEC'):
                plt.subplot(5,1,i+1)
                n,b,p = plt.hist(dchisq[:, i], bins=25, range=(lo,hi),
                                 histtype='step', color=ccmap[t])
                maxn = max(maxn, max(n))
                lp.append(p[0])
                lt.append('Type = ' + t)
                if i != 4:
                    plt.xticks([])
            for i in range(1,5):
                plt.subplot(5,1,i)
                plt.ylim(0, maxn*1.1)
            plt.xlabel('chisq improvement (vs no source)')
            plt.figlegend(lp, lt, 'upper right')
            plt.suptitle('Canonical DESI ELG -- Model selection delta-chi-squareds')
            ps.savefig()

            plt.clf()
            lp,lt = [],[]
            d = (hi-lo)*0.05
            plt.plot([lo-d,hi+d], [lo-d,hi+d], 'k-', alpha=0.5)
            for i,t in enumerate(models):
                if t == 'E':
                    continue
                p = plt.plot(dchisq[:,2], dchisq[:,i], '.', ms=10, alpha=0.5,
                             color=ccmap[t])
                lp.append(p[0])
                lt.append('Type = ' + t)
            plt.xlabel('dchisq')
            plt.legend(lp, lt, loc='upper left')
            plt.axis([lo-d,hi+d,lo-d,hi+d])
            plt.title('Delta-chisq vs canonical DESI ELG')
            plt.xlabel('dchisq for EXP model')
            plt.ylabel('dchisq for other models')
            ps.savefig()
            
            
            plt.clf()
            lp,lt = [],[]
            maxn = 1
            for t in models:
                I = np.flatnonzero(T.type == t)
                if len(I) == 0:
                    continue
                n,b,p = plt.hist(T.flux[I], bins=25,
                              range=(0.5*flux, 1.5*flux), histtype='step',
                              color=ccmap[t])
                maxn = max(maxn, max(n))
                lp.append(p[0])
                lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(T)))))
            plt.legend(lp, lt)
            plt.xlabel('Measured flux')
            plt.axvline(flux, color='k', ls='--')
            plt.ylim(0, 1.1*maxn)
            plt.title('Canonical DESI ELG source -- DR2 Type & Flux measurements')
            ps.savefig()
            
            plt.clf()
            lp = []
            lt = []
            for t in models:
                I = np.flatnonzero(T.type == t)
                if len(I) == 0:
                    continue
                n,b,p = plt.hist(T.re[I], bins=25,
                              range=(0., 2.*re), histtype='step',
                              color=ccmap[t])
                lp.append(p[0])
                lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(T)))))
            plt.legend(lp, lt)
            plt.xlabel('Measured effective radius (arcsec)')
            plt.axvline(re, color='k', ls='--')
            plt.title('Canonical DESI ELG source -- DR2 Type & Radius measurements')
            ps.savefig()
            
            plt.clf()
            lp = []
            lt = []
            for t in ['E','D']:
                I = np.flatnonzero(T.type == t)
                if len(I) == 0:
                    continue
                p = plt.plot(T.re[I], T.flux[I], '.', color=ccmap[t], ms=10, alpha=0.5)
                lp.append(p[0])
                lt.append('Type = ' + t + ' (%i %%)' % (int(np.round(100. * len(I) / len(T)))))
            plt.legend(lp, lt)
            plt.xlabel('Measured effective radius (arcsec)')
            plt.axvline(re, color='k', ls='--')
            plt.ylabel('Measured flux')
            plt.axhline(flux, color='k', ls='--')
            plt.title('Canonical DESI ELG source -- DR2 Type & Flux vs Radius measurements')
            plt.axis([0., 2.*re, 0.5*flux, 1.5*flux])
            ps.savefig()




            # ALL MODELS
            cat = catalog
            
            plt.clf()
            lp = []
            lt = []
            mx = 0
            for t in ['exp','dev']:
                n,b,p = plt.hist(cat.get('%s_shape_r' % t), bins=25,
                              range=(0., 2.*re), histtype='step',
                              color=ccmap[t])
                mx = max(mx, max(n))
                lp.append(p[0])
                lt.append(t)
            plt.legend(lp, lt)
            plt.xlabel('Measured effective radius (arcsec)')
            plt.axvline(re, color='k', ls='--')
            plt.title('All models -- Radius measurements')
            plt.ylim(0, mx*1.1)
            ps.savefig()

            plt.clf()
            lp = []
            lt = []
            mx = 0
            for t in ['exp','dev']:
                n,b,p = plt.hist(cat.get('%s_decam_flux' % t)[:,2], bins=25,
                              range=(0.5*flux, 1.5*flux), histtype='step',
                              color=ccmap[t])
                mx = max(mx, max(n))
                lp.append(p[0])
                lt.append(t)
            plt.legend(lp, lt)
            plt.xlabel('Measured flux')
            plt.axvline(flux, color='k', ls='--')
            plt.title('All models -- Flux measurements')
            plt.ylim(0, mx*1.1)
            ps.savefig()

            plt.clf()
            lp = []
            lt = []
            for t in ['exp','dev']:
                p = plt.plot(cat.get('%s_shape_r' % t), cat.get('%s_decam_flux' % t)[:,2], '.', color=ccmap[t], ms=10, alpha=0.5)
                lp.append(p[0])
                lt.append(t)
            plt.legend(lp, lt)
            plt.xlabel('Measured effective radius (arcsec)')
            plt.axvline(re, color='k', ls='--')
            plt.ylabel('Measured flux')
            plt.axhline(flux, color='k', ls='--')
            plt.title('All models -- Flux vs Radius')
            plt.axis([0., 2.*re, 0.5*flux, 1.5*flux])
            ps.savefig()


            
    
    
    S.to_np_arrays()
    
    plt.clf()
    lp = []
    lt = []
    for t in sourcetypes:
        frac = S.get('frac_%s' % t)
        if np.max(frac) == 0:
            continue
        p = plt.plot(S.psffwhm, frac, ccmap[t])
        lp.append(p[0])
        lt.append('Type ' + namemap[t])
    plt.legend(lp, lt, loc='center left')
    plt.xlabel('PSF FWHM (arcsec)')
    plt.ylabel('Fraction of each source type')
    plt.title('Canonical DESI ELG source -- Type vs Seeing')
    ps.savefig()


