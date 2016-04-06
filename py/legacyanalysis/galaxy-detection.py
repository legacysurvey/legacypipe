import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from collections import Counter

from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.stages import *

from tractor import *

from legacypipe.runbrick import *

'''
Investigate the detectability of galaxies as a function of radius,
seeing, etc., using the DECaLS pipeline.  Also check out what types we
think they are.

'''

class FakeSurvey(object):
    def __init__(self):
        self.survey_dir = ''
        self.ccds = None
        self.tims = None

    def ccds_touching_wcs(self, targetwcs, **kwargs):
        return self.ccds

    def photometric_ccds(self, CCDs):
        return np.arange(len(CCDs))
    
    def get_image_object(self, t):
        return FakeImage(self, t)

class FakeImage(object):
    def __init__(self, survey, t):
        # print 'FakeImage:', t
        self.tim = survey.tims[t.index]

    def run_calibs(self, *args, **kwargs):
        pass

    def get_tractor_image(self, **kwargs):
        return self.tim

class SimResult(object):
    pass
    
def run_sim(tims, cat, N, mods=None, samenoise=True, **kwargs):
    result = SimResult()
    
    if mods is None:
        tr = Tractor([], cat)
        mods = [tr.getModelImage(tim) for tim in tims]

    survey = FakeSurvey()
    ccds = fits_table()
    ccds.filter = [tim.band for tim in tims]
    ccds.to_np_arrays()
    ccds.index = np.arange(len(ccds))
    survey.ccds = ccds
    survey.tims = tims

    allcats = []
    allivs = []
    allTs = []
    allallmods = []
    
    for i in range(N):
        if samenoise:
            np.random.seed(10000 + i)

        for tim,mod in zip(tims, mods):
            tim.data = mod + tim.sig1 * np.random.normal(size=tim.shape)
    
        kwa = kwargs.copy()
        kwa.update(survey=survey)
        R = stage_tims(do_calibs=False, pipe=True, **kwa)
        kwa.update(R)
        R = stage_srcs(**kwa)
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
        #print 'len(c)', len(c)
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

            # print 'Ti:'
            # Ti.about()
            # print 'adding allmods:'
            # allmods.about()
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

    result.TT = merge_tables(TT, columns='fillzero')

    print 'Returning table:'
    result.TT.about()
    
    return result




def stage_sims(**kwargs):

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
    
    np.random.seed(1000042)
    
    # survey target limiting mag: r=23.6, let's say 5-sig point source
    limit = 23.6
    
    data = np.zeros((H,W), np.float32)
    tim = Image(data=data, inverr=np.ones_like(data), psf=psf, wcs=twcs,
                photocal=photocal)
    tim.subwcs = wcs
    tim.band = band
    tim.skyver = (0,0)
    tim.wcsver = (0,0)
    tim.psfver = (0,0)
    tim.plver = 0
    tim.dq = np.zeros(tim.shape, np.int16)
    tim.dq_bits = dict(satur=1)
    
    mp = multiproc()
    
    sourcetypes = [ 'E','D','S','C','P','-','+' ]
    
    ccmap = dict(E='r', D='b', S='c', C='m', P='g',
                 exp='r', dev='b', simp='c', comp='m', psf='g')
    ccmap['+'] = 'k'
    ccmap['-'] = '0.5'
    
    namemap = { 'E': 'Exp', 'D': 'deVauc', 'C': 'composite', 'P':'PSF', 'S': 'simple', '+':'>1 src', '-':'No detection' }
    
    
    #
    
    #mag = 23.0
    #re = 0.45
    
    mag = np.arange(19.0, 23.5 + 1e-3, 0.5)
    #mag = np.array([19.])
    
    re  = np.arange(0.1, 0.9 + 1e-3, 0.1)
    #re  = np.array([0.9])
    
    psfsig = 2.0
    N = 20
    #N = 1
    
    S = fits_table()
    for t in sourcetypes:
        S.set('frac_%s' % t, [])
    S.frac_U = []
    
    S.mag = []
    S.re = []
    S.imag = []
    S.ire = []
    S.psffwhm = []
    S.mods = []
    S.sig1 = []
    S.dchisq = []
    S.exp_re = []
    
    simk = 0
    
    for imag,mag_i in enumerate(mag):
        for ire,re_i in enumerate(re):
    
            #simk += 1
            #print 'simk', simk
            simk = 9
            np.random.seed(1000 + simk)
            
            S.mag.append(mag_i)
            S.re.append(re_i)
            S.imag.append(imag)
            S.ire.append(ire)
            
            flux = NanoMaggies.magToNanomaggies(mag_i)
    
            gal = ExpGalaxy(RaDecPos(ra, dec), NanoMaggies(**{ band: flux }),
                            EllipseESoft(np.log(re_i), 0., 0.))
    
            var1 = psfsig**2
            psf = GaussianMixturePSF(1.0, 0., 0., var1, var1, 0.)
            tim.psf = psf
            tim.psf_sigma = psfsig
    
            # Set the per-pixel noise based on the PSF size!?!
            limitflux = NanoMaggies.magToNanomaggies(limit)
            psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
            sig1 = limitflux / 5. * psfnorm
            tim.sig1 = sig1
            tim.inverr[:,:] = 1./sig1
    
            print
            print 'Running mag=', mag_i, 're=', re_i
            print 'psf norm', psfnorm
            print 'limiting flux', limitflux
            print 'sig1:', sig1
            print 'flux:', flux
    
            tr = Tractor([tim], [gal])
            mod = tr.getModelImage(0)
    
            S.mods.append(mod + sig1 * np.random.normal(size=mod.shape))
            S.sig1.append(sig1)
            
            res = run_sim([tim], [gal], N, mods=[mod], 
                          W=W, H=H, ra=ra, dec=dec, mp=mp, bands=[band])
    
            T = res.T
            T.flux = T.flux_r
            T.fluxiv = T.fluxiv_r
            #catalog = res.TT
    
            print 'T.nsrcs:', T.nsrcs
            
            S.psffwhm.append(psfsig * 2.35 * pixscale)
            for t in sourcetypes:
                S.get('frac_%s' % t).append(
                    100. * np.count_nonzero(T.type == t) / float(len(T)))
            S.frac_U.append(100. * np.count_nonzero(T.nsrcs == 0) / float(len(T)))
    
            TT = res.TT
            
            if 'dchisq' in TT.columns():
                dchisq = TT.dchisq
                S.dchisq.append(np.mean(dchisq, axis=0))
            else:
                S.dchisq.append(np.zeros(5))
    
            I = []
            if 'exp_shape_r' in TT.columns():
                I = np.flatnonzero(TT.exp_shape_r > 0)
            if len(I):
                S.exp_re.append(np.mean(TT.exp_shape_r[I]))
            else:
                S.exp_re.append(0.)
    
    S.to_np_arrays()
    print 'S:', len(S)

    rtn = {}
    for k in ['S', 'mag', 're', 'psfsig']:
        rtn[k] = locals()[k]
    return rtn
    


def label_cells(fmap, xx, yy, fmt, **kwa):
    args = dict(color='k', fontsize=8, va='center', ha='center')
    args.update(kwa)
    for j,y in enumerate(yy):
        for i,x in enumerate(xx):
            plt.text(x, y, fmt % fmap[j,i], **args)


    
def stage_plots(S=None, mag=None, re=None, psfsig=None,
                **kwargs):
    ps = PlotSequence('galdet')
    
    # Plot mags on x axis = cols
    # Plot re   on y axis = rows
    
    #S.imag = np.array([np.argmin(np.abs(m - mag)) for m in S.mag])
    #S.ire  = np.array([np.argmin(np.abs(r - re )) for r in S.re])
    
    cols = 1 + max(S.imag)
    rows = 1 + max(S.ire)
    
    S.row = S.ire
    S.col = S.imag
    
    # Plot the sources
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.clf()
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        si = np.flatnonzero((S.row == (rows-1 - i/cols)) * (S.col == i % cols))[0]
        sig1 = S.sig1[si]
        dimshow(S.mods[si], ticks=False, vmin=-2*sig1, vmax=5*sig1)
    ps.savefig()
    
    dm = mag[1]-mag[0]
    dr = re [1]-re [0]
    extent = [mag.min()-dm/2, mag.max()+dm/2, re.min()-dr/2, re.max()+dr/2]
    ima = dict(vmin=0, vmax=100, extent=extent, cmap='jet')
    
    for name in ['UNDETECTED', 'PSF', 'SIMPLE', 'EXP', 'DEV', 'COMP']:
    
        fmap = np.zeros((rows,cols))
        fmap[S.row, S.col] = S.get('frac_%s' % name[0])
    
        print 'Fraction classified as', name
        print fmap
        
        plt.clf()
        dimshow(fmap, aspect='auto', **ima)
        plt.xlabel('Mag')
        plt.ylabel('radius r_e (arcsec)')
        plt.colorbar()
        plt.title('Fraction of sources classified as %s' % name)
        ps.savefig()
    
    
    
    ima = dict(vmin=0, extent=extent, cmap='jet')
    
    for i,name in enumerate(['PSF', 'SIMPLE', 'DEV', 'EXP', 'COMP']):
    
        fmap = np.zeros((rows,cols))
        fmap[S.row, S.col] = np.sqrt(S.dchisq[:,i])
    
        plt.clf()
        dimshow(fmap, aspect='auto', **ima)
        plt.xlabel('Mag')
        plt.ylabel('radius r_e (arcsec)')
        plt.colorbar()
        plt.title('sqrt(dchisq) for %s' % name)
        ps.savefig()
    
    
    ipsf = 0
    isimple = 1
    idev = 2
    iexp = 3
    icomp = 4
    
    ima = dict(extent=extent, cmap='jet')
    
    for i,name in enumerate(['SIMPLE', 'DEV', 'EXP', 'COMP']):
    
        fmap = np.zeros((rows,cols))
        fmap[S.row, S.col] = S.dchisq[:,i+1] - S.dchisq[:,ipsf]
    
        plt.clf()
        dimshow(fmap, aspect='auto', **ima)
        plt.xlabel('Mag')
        plt.ylabel('radius r_e (arcsec)')
        plt.colorbar()
        plt.title('dchisq_%s - dchisq_PSF' % name)
        ps.savefig()

        plt.clim(-50, 50)
        ps.savefig()
        
        if name != 'SIMPLE':
            fmap[S.row, S.col] = ( (S.dchisq[:,i+1] - S.dchisq[:,ipsf]) / 
                                   S.dchisq[:,ipsf] )
    
            plt.clf()
            dimshow(fmap, aspect='auto', vmin=-0.1, vmax=0.1, **ima)
            plt.xlabel('Mag')
            plt.ylabel('radius r_e (arcsec)')
            plt.colorbar()
            plt.title('(dchisq_%s - dchisq_PSF) / dchisq_PSF' % name)
            ps.savefig()

            label_cells(fmap, mag, re, '%.2g', color='k')
            ps.savefig()
            
            
    
    fmap = np.zeros((rows,cols))
    fmap[S.row, S.col] = S.dchisq[:,iexp] - S.dchisq[:,isimple]
    
    plt.clf()
    dimshow(fmap, aspect='auto', **ima)
    plt.xlabel('Mag')
    plt.ylabel('radius r_e (arcsec)')
    plt.colorbar()
    plt.title('dchisq_EXP - dchisq_SIMPLE')
    ps.savefig()

    plt.clim(-50, 50)
    ps.savefig()

    fmap[S.row, S.col] = ( (S.dchisq[:,iexp] - S.dchisq[:,isimple]) / 
                           S.dchisq[:,ipsf] )
    
    plt.clf()
    dimshow(fmap, aspect='auto', vmin=-0.1, vmax=0.1, **ima)
    plt.xlabel('Mag')
    plt.ylabel('radius r_e (arcsec)')
    plt.colorbar()
    plt.title('(dchisq_EXP - dchisq_SIMPLE) / dchisq_PSF')
    ps.savefig()
    
    label_cells(fmap, mag, re, '%.2g', color='k')
    ps.savefig()
    
    
    fmap = np.zeros((rows,cols))
    fmap[S.row, S.col] = S.exp_re
    
    plt.clf()
    dimshow(fmap, aspect='auto', **ima)
    plt.xlabel('Mag')
    plt.ylabel('radius r_e (arcsec)')
    plt.colorbar()
    plt.title('EXP fit r_e (arcsec)')
    ps.savefig()
    
    



stage = 'plots'

prereqs = { 'plots': 'sims',
            'sims': None }
initargs = {}
kwargs = dict()

stagefunc = CallGlobalTime('stage_%s', globals())
picklePattern = 'galdet-%s.pickle'

runstage(stage, picklePattern, stagefunc, prereqs=prereqs,
         initial_args=initargs, force=[stage], **kwargs)


    

    
sys.exit(0)
        


# Render synthetic source into image

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


