import os
import sys
from time import time

import pylab as plt

import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from tractor import Tractor
from legacypipe.survey import LegacySurveyData
from legacypipe.catalog import read_fits_catalog
from legacypipe.outliers import read_outlier_mask_file
from legacypipe.runbrick import run_brick
import logging

def main():

    ra,dec = 20.4685, -16.5842
    W = H = 50
    hemi = 'south'

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    #logging.getLogger('tractor.engine').setLevel(lvl + 10)

    survey = LegacySurveyData()
    brick = None
    R = run_brick(brick, survey, radec=(ra,dec), width=W, height=H,
                  cache_outliers=True, wise=False, do_calibs=False,
                  stages=['halos'])
    print('Results:', R.keys())
    tims = R['tims']

    survey_cat = LegacySurveyData(survey_dir=os.path.join(survey.survey_dir, hemi))

    #targetrd = R['targetrd']
    #print('Target RD:', targetrd)

    targetwcs = R['targetwcs']

    bands = ['g','r','z']
    
    # from legacypipe.survey import bricks_touching_wcs
    # 
    # bricks = bricks_touching_wcs(targetwcs, survey=survey)
    # #survey.get_bricks_readonly()
    # #I = np.flatnonzero(bricks.ra1 > ra)
    # print('Bricks touching:', bricks.brickname)
    # 
    # for brick in bricks:
    #     ### HACK -- resolve??
    #     tfn = survey.find_file(hemi=hemi, brick=brick.brickname)
    #     print('Reading tractor file', tfn)
    #     T = fits_table(tfn)

    from legacypipe.forced_photom import get_catalog_in_wcs

    cat = get_catalog_in_wcs(targetwcs, survey, survey_cat,
                             extra_columns=['flux_%s' % b for b in bands])
    print('Found', len(cat), 'catalog sources touching WCS')

    cat.about()
    ok,xx,yy = targetwcs.radec2pixelxy(cat.ra, cat.dec)
    print('Pixel positions:', xx, yy)

    # FIXMEs
    # - DECam halo subtraction
    # - LS outlier masks?
    # - Gaia star proper motions
    # - subtract SGA galaxies outside the region

    from legacypipe.catalog import read_fits_catalog
    
    tcat = read_fits_catalog(cat)
    print('Tractor catalog:')
    for src in tcat:
        print(' ', src)

    tr = Tractor(tims, tcat)
    tr.freezeParam('images')

    from legacypipe.coadds import quick_coadds
    
    mods = list(tr.getModelImages())
    coimg,_ = quick_coadds(tims, bands, targetwcs)
    comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)

    from legacypipe.survey import get_rgb
    # from legacypipe.survey import sdss_rgb
    # def get_rgb(img, bands, resids=False):
    #     if resids:
    #         from legacypipe.survey import get_rgb as real_get_rgb
    #         return real_get_rgb(img, bands, resids=True)
    #     return sdss_rgb(img, bands, mnmx=(0, 6))
    
    ima = dict(interpolation='nearest', origin='lower')
    plt.clf()
    plt.imshow(get_rgb(coimg, bands), **ima)
    plt.savefig('img.png')
    
    plt.clf()
    plt.imshow(get_rgb(comod, bands), **ima)
    plt.savefig('mod.png')

    inbounds, = np.nonzero((xx >= 1) * (yy >= 1) * (xx <= W) * (yy <= H))
    print(len(inbounds), 'sources are within image')

    print('Subtracting off sources outside the image bounds')
    outbounds, = np.nonzero(np.logical_not((xx >= 1) * (yy >= 1) * (xx <= W) * (yy <= H)))
    if len(outbounds):
        # Subtract off the sources outside the image.
        otr = Tractor(tims, [tcat[i] for i in outbounds])
        mods = list(otr.getModelImages())
        for tim,mod in zip(tims, mods):
            if mod is not None:
                tim.data -= mod

    tcat = [tcat[i] for i in inbounds]
    tr = Tractor(tims, tcat)
    tr.freezeParam('images')
    tcat = tr.getCatalog()

    alphas = [0.1, 0.3, 1.0]
    optargs = dict(alphas=alphas, priors=False, shared_params=False)

    mods = list(tr.getModelImages())
    coimg,_ = quick_coadds(tims, bands, targetwcs)
    comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    coresid,_ = quick_coadds(tims, bands, targetwcs,
                             images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])

    plt.clf()
    plt.imshow(get_rgb(coimg, bands), **ima)
    plt.savefig('img2.png')
    
    plt.clf()
    plt.imshow(get_rgb(comod, bands), **ima)
    plt.savefig('mod2.png')

    reskw = dict(resids=True)
    
    plt.clf()
    plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    plt.savefig('res2.png')

    print('In-bounds tractor catalog:')
    for src in tcat:
        print(' ', src)

    from tractor import DevGalaxy, ExpGalaxy, EllipseESoft
    from tractor.sersic import SersicGalaxy

    print('Swapping ellipse classes...')
    for src in tcat:
        if isinstance(src, (ExpGalaxy, DevGalaxy, SersicGalaxy)):
            src.shape = EllipseESoft.fromEllipseE(src.shape)

    tr.printThawedParams()
        
    print('Optimization loop...')
    t0 = time()
    tr.optimize_loop(**optargs)
    print('Opt took', time()-t0)

    # from tractor.patch import ModelMask
    # mm = []
    # for tim in tims:
    #     mh,mw = tim.shape
    #     mm.append(dict([(src, ModelMask(0, 0, mw, mh)) for src in tcat]))
    # tr.setModelMasks(mm)
    # print('Optimization loop...')
    # t0 = time()
    # tr.optimize_loop()
    # print('Opt took', time()-t0)

    print('Post fitting:')
    for src in tcat:
        print(' ', src)

    mods = list(tr.getModelImages())
    comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    coresid,_ = quick_coadds(tims, bands, targetwcs,
                             images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])

    plt.clf()
    plt.imshow(get_rgb(comod, bands), **ima)
    plt.savefig('mod3.png')

    plt.clf()
    plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    plt.savefig('res3.png')

    # Grab the central source.
    ok,xx,yy = targetwcs.radec2pixelxy([s.getPosition().ra  for s in tcat],
                                       [s.getPosition().dec for s in tcat])
    icent = np.argmin(np.hypot(xx - W/2., yy - H/2.))
    central = tcat[icent]

    # Replace with SersicCoreGalaxy
    from tractor.sercore import SersicCoreGalaxy

    bright = central.brightness.copy()
    bright.setParams(np.zeros(len(central.brightness.getParams())))
    sercore = SersicCoreGalaxy(central.pos.copy(), central.brightness.copy(),
                               central.shape.copy(), central.sersicindex.copy(), bright)
    tcat[icent] = sercore

    print('Optimization loop...')
    t0 = time()
    tr.optimize_loop(**optargs)
    print('Opt took', time()-t0)

    mods = list(tr.getModelImages())
    comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    coresid,_ = quick_coadds(tims, bands, targetwcs,
                             images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])
    plt.clf()
    plt.imshow(get_rgb(comod, bands), **ima)
    plt.savefig('mod4.png')
    plt.clf()
    plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    ax = plt.axis()
    ok,x,y = targetwcs.radec2pixelxy(sercore.pos.ra, sercore.pos.dec)
    plt.plot(x-1, y-1, 'r+', ms=10)
    plt.axis(ax)
    plt.savefig('res4.png')

    print('Post fitting:')
    for src in tcat:
        print(' ', src)

    # for k in range(3):
    #     for i in range(len(tcat)):
    #         print('Round', k, 'fitting param', i, ':', tcat[i])
    #         tcat.freezeAllBut(i)
    #         t0 = time()
    #         tr.optimize_loop(**optargs)
    #         print('Opt took', time()-t0)
    #         print('Result:', tcat[i])
    # 
    # mods = list(tr.getModelImages())
    # comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    # coresid,_ = quick_coadds(tims, bands, targetwcs,
    #                          images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])
    # plt.clf()
    # plt.imshow(get_rgb(comod, bands), **ima)
    # plt.savefig('mod5.png')
    # plt.clf()
    # plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    # ax = plt.axis()
    # ok,x,y = targetwcs.radec2pixelxy(sercore.pos.ra, sercore.pos.dec)
    # plt.plot(x-1, y-1, 'r+', ms=10)
    # plt.axis(ax)
    # plt.savefig('res5.png')

    # dlnp,X,alpha = tr.optimize(**optargs)
    # print('dlnp', dlnp)
    # print('alpha', alpha)
    # print('X', X)
    # p0 = tr.getParams()
    # tr.setParams(np.array(p0) + np.array(X))
    # mods = list(tr.getModelImages())
    # comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    # coresid,_ = quick_coadds(tims, bands, targetwcs,
    #                          images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])
    # plt.clf()
    # plt.imshow(get_rgb(comod, bands), **ima)
    # plt.title('Proposed lsqr param update')
    # plt.savefig('mod5.png')
    # plt.clf()
    # plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    # ax = plt.axis()
    # ok,x,y = targetwcs.radec2pixelxy(sercore.pos.ra, sercore.pos.dec)
    # plt.plot(x-1, y-1, 'r+', ms=10)
    # plt.axis(ax)
    # plt.title('Proposed lsqr param update')
    # plt.savefig('res5.png')
    # tr.setParams(p0)
    
    # derivs = tr.getDerivs()
    # for j,(pderivs,pname) in enumerate(zip(derivs, tr.getParamNames())):
    #     dtims = []
    #     dimgs = []
    #     for dp,tim in pderivs:
    #         dtims.append(tim)
    #         dimg = np.zeros(tim.shape, np.float32)
    #         dp.addTo(dimg)
    #         dimgs.append(dimg)
    #     coderiv,_ = quick_coadds(dtims, bands, targetwcs, images=dimgs)
    # 
    #     mx = np.max([np.max(np.abs(d)) for d in coderiv])
    #     dg,dr,dz = coderiv
    #     drgb = np.dstack((dz,dr,dg))
    #     plt.clf()
    #     mn,mx = -mx,mx
    #     plt.imshow(np.clip((drgb - mn) / (mx - mn), 0., 1.), **ima)
    #     plt.title('deriv ' + pname)
    #     plt.savefig('deriv-%02i.png' % j)

    from tractor.ceres_optimizer import CeresOptimizer
    copt = CeresOptimizer()
    tr.optimizer = copt

    print('Optimizing with Ceres...')
    R = tr.optimize_loop()
    print('Result:', R)

    mods = list(tr.getModelImages())
    comod,_ = quick_coadds(tims, bands, targetwcs, images=mods)
    coresid,_ = quick_coadds(tims, bands, targetwcs,
                             images=[tim.getImage()-mod for tim,mod in zip(tims, mods)])
    plt.clf()
    plt.imshow(get_rgb(comod, bands), **ima)
    plt.savefig('mod6.png')
    plt.clf()
    plt.imshow(get_rgb(coresid, bands, **reskw), **ima)
    ax = plt.axis()
    ok,x,y = targetwcs.radec2pixelxy(sercore.pos.ra, sercore.pos.dec)
    plt.plot(x-1, y-1, 'r+', ms=10)
    plt.axis(ax)
    plt.savefig('res6.png')

    
if __name__ == '__main__':
    main()
