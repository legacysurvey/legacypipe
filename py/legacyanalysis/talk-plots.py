import matplotlib
matplotlib.use('Agg')
import pylab as plt

#from astrometry.util.fits import *
from legacypipe.runbrick import *
from legacypipe.runbrick_plots import *

from legacypipe.runbrick import _coadds, _get_mod

from astrometry.util.stages import *
from astrometry.util.starutil_numpy import *
from astrometry.sdss.dr9 import DR9
from astrometry.sdss import AsTransWrapper
from astrometry.libkd.spherematch import *

from scipy.ndimage.morphology import binary_dilation
from legacypipe.utils import MyMultiproc

def stage_plots(targetwcs=None, bands=None, W=None, H=None,
                coimgs=None, cons=None, tims=None, blobs=None,
                cat=None, T2=None, decals=None, **kwargs):

    plt.figure(figsize=(8,4))
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.99)

    ll = np.linspace(0, 360, 1000)
    bb = np.zeros_like(ll)
    rg,dg = lbtoradec(ll, bb)
    rg2,dg2 = lbtoradec(ll, bb + 10)
    rg3,dg3 = lbtoradec(ll, bb - 10)
    
    dall = Decals(decals_dir='decals')
    ccds = dall.get_ccds()
    bricks = dall.get_bricks_readonly()
    brick_coverage = dict()
    
    for band in bands:
        I = np.flatnonzero(ccds.filter == band)
        plt.clf()
        ccmap = dict(g='g', r='r', z='m')
        plt.plot(ccds.ra[I], ccds.dec[I], '.', color=ccmap[band], alpha=0.1)
        plt.plot(rg, dg, 'k-')
        plt.plot(rg2, dg2, 'k-', alpha=0.5)
        plt.plot(rg3, dg3, 'k-', alpha=0.5)
        plt.axis([360,0,-12,36])
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.savefig('dr2-%s.png' % band)

        II,J,d = match_radec(ccds.ra[I], ccds.dec[I], bricks.ra, bricks.dec,
                            np.hypot(0.25/2, 0.17))
        J = np.unique(J)
        hasband = np.zeros(len(bricks), bool)
        hasband[J] = True
        brick_coverage[band] = hasband

        print 'Number of bricks with', band, 'coverage:', len(J)

    print 'Bricks with grz coverage:', sum(reduce(np.logical_and, brick_coverage.values()))
        
    T = T2
    #mp = MyMultiproc(None, pool=pool)
    mp = MyMultiproc(init=runbrick_global_init, initargs=[])
    
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if True:
        # SDSS coadd
        sdsscoimgs,nil = sdss_coadd(targetwcs, bands)
        plt.clf()
        dimshow(get_rgb(sdsscoimgs, bands, **rgbkwargs), ticks=False)
        plt.savefig('sdss.png')

    C = _coadds(tims, bands, targetwcs)
    coimgs = C.coimgs

    rgb = get_rgb(coimgs, bands, **rgbkwargs)
    
    plt.clf()
    dimshow(rgb, ticks=False)
    plt.savefig('img.png')

    ax = plt.axis()
    plt.plot(T.bx0, T.by0+1, 'o', mec=(0,1,0), mfc='none', ms=12)
    plt.savefig('srcs.png')

    plt.clf()
    print 'Blobs:', blobs.dtype, blobs.min(), blobs.max()
    #dimshow(rgb, ticks=False)

    b0 = (blobs >= 0)
    b1 = binary_dilation(b0, np.ones((3,3)))
    bout = np.logical_and(b1, np.logical_not(b0))

    blobrgb = rgb.copy()
    # # set green
    blobrgb[:,:,0][bout] = 0.
    blobrgb[:,:,1][bout] = 1.
    blobrgb[:,:,2][bout] = 0.
    plt.clf()
    dimshow(blobrgb, ticks=False)
    plt.savefig('blobs.png')

    plt.clf()
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])
    comods,nil = compute_coadds(tims, bands, targetwcs, images=mods)
    dimshow(get_rgb(comods, bands, **rgbkwargs))
    plt.savefig('mod.png')

    nmods = []
    resids = []
    
    for tim,mod in zip(tims,mods):
        noise = np.random.normal(size=tim.shape)
        ie = tim.getInvError()
        print 'ie min', ie[ie>0].min(), 'median', np.median(ie)
        noise[ie > 0] *= (1. / ie[ie>0])
        noise[ie == 0] = 0
        nmods.append(mod + noise)
        res = tim.getImage() - mod
        res[ie == 0] = 0.
        resids.append(res)
        
    comods2,nil = compute_coadds(tims, bands, targetwcs, images=nmods)
    dimshow(get_rgb(comods2, bands, **rgbkwargs))
    plt.savefig('noisymod.png')

    res,nil = compute_coadds(tims, bands, targetwcs, images=resids)
    dimshow(get_rgb(res, bands, **rgbkwargs_resid))
    plt.savefig('resids.png')
    
    return dict(sdsscoimgs=sdsscoimgs, coimgs=coimgs,
                comods=comods, comods2=comods2, resids=res)


def stage_plots2(sdsscoimgs=None, coimgs=None,
                 comods=None, comods2=None, resids=None,
                 bands=None,
                 **kwargs):

    for band,co in zip(bands, sdsscoimgs):
        print 'co', co.shape
        plt.clf()
        plt.hist(co.ravel(), range=(-0.1, 0.1), bins=100)
        plt.title('SDSS %s band' % band)
        plt.savefig('sdss-%s.png' % band)

        print band, 'band 16th and 84th pcts:', np.percentile(co.ravel(), [16,84])
        
    kwa = dict(mnmx=(-2,10), scales=dict(g=(2,0.02), r=(1,0.03),
    z=(0,0.1)))
    #z=(0,0.22)))
    

    plt.clf()
    dimshow(get_rgb(sdsscoimgs, bands, **kwa), ticks=False)
    plt.savefig('sdss2.png')

    plt.clf()
    dimshow(get_rgb(coimgs, bands, **kwa), ticks=False)
    plt.savefig('img2.png')

    
    
brick = '1498p017'

stages = ['plots2']
picklePattern = ('pickles/runbrick-z7-%(brick)s-%%(stage)s.pickle' %
                 dict(brick=brick))
stagefunc = CallGlobalTime('stage_%s', globals())
initargs = {}
kwargs = {}

prereqs = { 'plots': 'writecat',
            'plots2': 'plots'}

for stage in stages:
    runstage(stage, picklePattern, stagefunc, prereqs=prereqs, force=stages,
             initial_args=initargs, write=True, **kwargs)
