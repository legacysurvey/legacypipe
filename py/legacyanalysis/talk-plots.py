import matplotlib
matplotlib.use('Agg')
import pylab as plt

#from astrometry.util.fits import *
from legacypipe.runbrick import *
from legacypipe.runbrick_plots import *

from legacypipe.runbrick import _coadds, _get_mod

from astrometry.util.stages import *
from astrometry.sdss.dr9 import DR9

from scipy.ndimage.morphology import binary_dilation
from legacypipe.utils import MyMultiproc

def stage_plots(targetwcs=None, bands=None, W=None, H=None,
                coimgs=None, cons=None, tims=None, blobs=None,
                cat=None, T2=None, **kwargs):

    T = T2
    #mp = MyMultiproc(None, pool=pool)
    mp = MyMultiproc(init=runbrick_global_init, initargs=[])
    
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if False:
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
    
    

brick = '1498p017'

stages = ['plots']
picklePattern = ('pickles/runbrick-z7-%(brick)s-%%(stage)s.pickle' %
                 dict(brick=brick))
stagefunc = CallGlobalTime('stage_%s', globals())
initargs = {}
kwargs = {}

prereqs = { 'plots': 'writecat' }

for stage in stages:
    runstage(stage, picklePattern, stagefunc, prereqs=prereqs,
             initial_args=initargs, write=False, **kwargs)
