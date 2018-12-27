from __future__ import print_function

import numpy as np
import pylab as plt
import time

from astrometry.util.ttime import Time, CpuMeas
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import dimshow

from tractor import Tractor, PointSource, Image, NanoMaggies, Catalog, Patch
import tractor
print(tractor.__file__)
from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy, PSFandExpGalaxy_diffcentres, SoftenedFracDev, FracDev, disable_galaxy_cache, enable_galaxy_cache
from tractor.patch import ModelMask

from legacypipe.survey import (SimpleGalaxy, RexGalaxy, GaiaSource,
                               LegacyEllipseWithPriors, get_rgb)
from legacypipe.runbrick import rgbkwargs, rgbkwargs_resid
from legacypipe.coadds import quick_coadds
from legacypipe.runbrick_plots import _plot_mods

def one_blob(X):
    '''
    Fits sources contained within a "blob" of pixels.
    '''
    if X is None:
        return None
    (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
     srcs, bands, plots, ps, simul_opt, use_ceres, hasbright, rex) = X

    print('Fitting blob number', nblob, 'val', iblob, ':', len(Isrcs),
          'sources, size', blobw, 'x', blobh, len(timargs), 'images')

    if len(timargs) == 0:
        return None

    if plots:
        plt.figure(2, figsize=(3,3))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.figure(1)

    t0 = time.clock()
    # A local WCS for this blob
    blobwcs = brickwcs.get_subimage(bx0, by0, blobw, blobh)

    # Per-source measurements for this blob
    B = fits_table()
    B.sources = srcs
    B.Isrcs = Isrcs

    # Did sources start within the blob?
    ok,x0,y0 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in srcs]),
        np.array([src.getPosition().dec for src in srcs]))

    B.started_in_blob = blobmask[
        np.clip(np.round(y0-1).astype(int), 0,blobh-1),
        np.clip(np.round(x0-1).astype(int), 0,blobw-1)]

    B.cpu_source = np.zeros(len(B), np.float32)

    B.blob_width  = np.zeros(len(B), np.int16) + blobw
    B.blob_height = np.zeros(len(B), np.int16) + blobh
    B.blob_npix   = np.zeros(len(B), np.int32) + np.sum(blobmask)
    B.blob_nimages= np.zeros(len(B), np.int16) + len(timargs)
    
    ob = OneBlob('%i'%(nblob+1), blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, simul_opt, use_ceres, hasbright, rex)
    ob.run(B)

    B.blob_totalpix = np.zeros(len(B), np.int32) + ob.total_pix
    
    ok,x1,y1 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in B.sources]),
        np.array([src.getPosition().dec for src in B.sources]))
    B.finished_in_blob = blobmask[
        np.clip(np.round(y1-1).astype(int), 0, blobh-1),
        np.clip(np.round(x1-1).astype(int), 0, blobw-1)]
    assert(len(B.finished_in_blob) == len(B))
    assert(len(B.finished_in_blob) == len(B.started_in_blob))

    B.brightstarinblob = np.zeros(len(B), bool)
    if hasbright:
        B.brightstarinblob[:] = True

    B.cpu_blob = np.zeros(len(B), np.float32)
    t1 = time.clock()
    B.cpu_blob[:] = t1 - t0
        
    B.iblob = iblob
    return B

class OneBlob(object):
    def __init__(self, name, blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, simul_opt, use_ceres, hasbright, rex):
        self.name = name
        self.rex = rex
        self.blobwcs = blobwcs
        self.blobmask = blobmask
        self.srcs = srcs
        self.bands = bands
        self.plots = plots

        self.plots_per_source = False
        # blob-1-data.png, etc
        self.plots_single = False

        self.ps = ps
        self.simul_opt = simul_opt
        self.use_ceres = use_ceres
        self.hasbright = hasbright
        self.deblend = False
        self.tims = self.create_tims(timargs)
        self.total_pix = sum([np.sum(t.getInvError() > 0) for t in self.tims])
        self.plots2 = False
        alphas = [0.1, 0.3, 1.0]
        self.optargs = dict(priors=True, shared_params=False, alphas=alphas,
                            print_progress=True)
        self.blobh,self.blobw = blobmask.shape
        self.bigblob = (self.blobw * self.blobh) > 100*100
        if self.bigblob:
            print('Big blob:', name)
        self.trargs = dict()
    
        if use_ceres:
            from tractor.ceres_optimizer import CeresOptimizer
            ceres_optimizer = CeresOptimizer()
            self.optargs.update(scale_columns=False,
                                scaled=False,
                                dynamic_scale=False)
            self.trargs.update(optimizer=ceres_optimizer)
        else:
            self.optargs.update(dchisq = 0.1)

    def run(self, B):
        # Not quite so many plots...
        self.plots1 = self.plots
        cat = Catalog(*self.srcs)

        tlast = Time()
        if self.plots:
            self._initial_plots()

        if self.deblend:
            ### Test bogus deblending
            self.run_deblend(cat)

        self._fit_fluxes(cat, self.tims, self.bands)
        tr = self.tractor(self.tims, cat)

        if self.plots:
            self._plots(tr, 'Initial models')

        # Optimize individual sources, in order of flux.
        # First, choose the ordering...
        Ibright = _argsort_by_brightness(cat, self.bands)

        if len(cat) > 1:
            self._optimize_individual_sources_subtract(
                cat, Ibright, B.cpu_source)
        else:
            self._optimize_individual_sources(tr, cat, Ibright, B.cpu_source)

        # Optimize all at once?
        if len(cat) > 1 and len(cat) <= 10:
            #tfit = Time()
            cat.thawAllParams()
            tr.optimize_loop(**self.optargs)

        if self.plots:
            self._plots(tr, 'After source fitting')

            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('After source fitting')
            self.ps.savefig()

            if self.plots_single:
                plt.figure(2)
                mods = list(tr.getModelImages())
                coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs,self.bands), ticks=False)
                plt.savefig('blob-%s-initmodel.png' % (self.name))
                res = [(tim.getImage() - mod) for tim,mod in zip(self.tims, mods)]
                coresids,nil = quick_coadds(self.tims, self.bands, self.blobwcs, images=res)
                dimshow(get_rgb(coresids, self.bands, **rgbkwargs_resid), ticks=False)
                plt.savefig('blob-%s-initresid.png' % (self.name))
                dimshow(get_rgb(coresids, self.bands), ticks=False)
                plt.savefig('blob-%s-initsub.png' % (self.name))
                plt.figure(1)


        print('Blob', self.name, 'finished initial fitting:', Time()-tlast)
        tlast = Time()

        # Next, model selections: point source vs dev/exp vs composite.
        self.run_model_selection(cat, Ibright, B)

        print('Blob', self.name, 'finished model selection:', Time()-tlast)
        tlast = Time()

        if self.plots:
            self._plots(tr, 'After model selection')

        if self.plots_single:
            plt.figure(2)
            mods = list(tr.getModelImages())
            coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                       fill_holes=False)
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.savefig('blob-%s-model.png' % (self.name))
            res = [(tim.getImage() - mod) for tim,mod in zip(self.tims, mods)]
            coresids,nil = quick_coadds(self.tims, self.bands, self.blobwcs, images=res)
            dimshow(get_rgb(coresids, self.bands, **rgbkwargs_resid), ticks=False)
            plt.savefig('blob-%s-resid.png' % (self.name))
            plt.figure(1)

        # Cut down to just the kept sources
        I = np.array([i for i,s in enumerate(cat) if s is not None])
        B.cut(I)
        cat = Catalog(*B.sources)
        tr.catalog = cat

        # Do another quick round of flux-only fitting?
        # This does horribly -- fluffy galaxies go out of control because
        # they're only constrained by pixels within this blob.
        #_fit_fluxes(cat, tims, bands, use_ceres, alphas)

        # ### Simultaneous re-opt?
        # if simul_opt and len(cat) > 1 and len(cat) <= 10:
        #     #tfit = Time()
        #     cat.thawAllParams()
        #     #print('Optimizing:', tr)
        #     #tr.printThawedParams()
        #     max_cpu = 300.
        #     cpu0 = time.clock()
        #     for step in range(50):
        #         dlnp,X,alpha = tr.optimize(**optargs)
        #         cpu = time.clock()
        #         if cpu-cpu0 > max_cpu:
        #             print('Warning: Exceeded maximum CPU time for source')
        #             break
        #         if dlnp < 0.1:
        #             break
        #     #print('Simultaneous fit took:', Time()-tfit)

        # Compute variances on all parameters for the kept model
        B.srcinvvars = [None for i in range(len(B))]
        cat.thawAllRecursive()
        cat.freezeAllParams()
        for isub in range(len(B.sources)):
            cat.thawParam(isub)
            src = cat[isub]
            if src is None:
                cat.freezeParam(isub)
                continue
            # Convert to "vanilla" ellipse parameterization
            nsrcparams = src.numberOfParams()
            _convert_ellipses(src)
            assert(src.numberOfParams() == nsrcparams)
            # print('Computing variances for source', src, ': N params:', nsrcparams)
            # print('Source params:')
            # src.printThawedParams()
            # For Gaia sources, temporarily convert the GaiaPosition to a
            # RaDecPos in order to compute the invvar it would have in our
            # imaging?  Or just plug in the Gaia-measured uncertainties??
            # (going to implement the latter)
            # Compute inverse-variances
            allderivs = tr.getDerivs()
            ivars = _compute_invvars(allderivs)
            assert(len(ivars) == nsrcparams)
            #print('Inverse-variances:', ivars)
            B.srcinvvars[isub] = ivars
            assert(len(B.srcinvvars[isub]) == cat[isub].numberOfParams())
            cat.freezeParam(isub)

        # Check for sources with zero inverse-variance -- I think these
        # can be generated during the "Simultaneous re-opt" stage above --
        # sources can get scattered outside the blob.

        I, = np.nonzero([np.sum(iv) > 0 for iv in B.srcinvvars])
        if len(I) < len(B):
            print('Keeping', len(I), 'of', len(B),'sources with non-zero ivar')
            B.cut(I)
            cat = Catalog(*B.sources)
            tr.catalog = cat

        M = _compute_source_metrics(B.sources, self.tims, self.bands, tr)
        for k,v in M.items():
            B.set(k, v)
        print('Blob', self.name, 'finished:', Time()-tlast)
        
    def run_model_selection(self, cat, Ibright, B):
        # We compute & subtract initial models for the other sources while
        # fitting each source:
        # -Remember the original images
        # -Compute initial models for each source (in each tim)
        # -Subtract initial models from images
        # -During fitting, for each source:
        #   -add back in the source's initial model (to each tim)
        #   -fit, with Catalog([src])
        #   -subtract final model (from each tim)
        # -Replace original images
    
        print('gets here')
        models = SourceModels()
        # Remember original tim images
        models.save_images(self.tims)
        # Create initial models for each tim x each source
        # tt = Time()
        models.create(self.tims, cat, subtract=True)
        # print('Subtracting initial models:', Time()-tt)

        N = len(cat)
        B.dchisq = np.zeros((N, 6), np.float32)
        B.all_models    = np.array([{} for i in range(N)])
        B.all_model_ivs = np.array([{} for i in range(N)])
        B.all_model_cpu = np.array([{} for i in range(N)])
         
        # Model selection for sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
    
            src = cat[srci]
            print('Model selection for source %i of %i in blob %s' %
                  (numi+1, len(Ibright), self.name))
            # if isinstance(src, GaiaSource):
            #     print('Gaia source', src)
            #     if src.isForcedPointSource():
            #         print('Gaia source is forced to be a point source -- not doing model selection.')
            #         continue
            cpu0 = time.clock()
    
            # Add this source's initial model back in.
            models.add(srci, self.tims)

            if self.plots_single:
                plt.figure(2)
                tr = self.tractor(self.tims, cat)
                coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                           fill_holes=False)
                # not using rgbkwargs_resid
                #dimshow(get_rgb(coimgs,self.bands), ticks=False)
                #plt.savefig('blob-%s-%i-bdata.png' % (self.name, srci))
                rgb = get_rgb(coimgs,self.bands)
                plt.imsave('blob-%s-%s-bdata.png' % (self.name, srci), rgb,
                           origin='lower')
                plt.figure(1)

    
            if self.bigblob:
                # if self.plots:
                #     plt.clf()
                #     for j,tim in enumerate(self.tims):
                #         plt.subplot(len(self.tims), 2, j+1)
                #         dimshow(tim.getImage(), vmin=-2*tim.sig1, vmax=5*tim.sig1)
                #         ax = plt.axis()
                #         x,y = tim.wcs.positionToPixel(src.getPosition())
                #         plt.plot(x, y, 'r.')
                #     self.ps.savefig()
                mods = [mod[srci] for mod in models.models]
                srctims,modelMasks = _get_subimages(self.tims, mods, src)
                # if self.plots:
                #     for j,tim in enumerate(srctims):
                #         plt.subplot(len(srctims), 2, len(srctims)+j+1)
                #         dimshow(tim.getImage(), vmin=-2*tim.sig1, vmax=5*tim.sig1)
                #         ax = plt.axis()
                #         x,y = tim.wcs.positionToPixel(src.getPosition())
                #         plt.plot(x, y, 'r.')
                #     self.ps.savefig()

                # Create a little local WCS subregion for this source, by
                # resampling non-zero inverrs from the srctims into blobwcs
                insrc = np.zeros((self.blobh,self.blobw), bool)
                for tim in srctims:
                    try:
                        Yo,Xo,Yi,Xi,nil = resample_with_wcs(
                            self.blobwcs, tim.subwcs, [],2)
                    except:
                        continue
                    insrc[Yo,Xo] |= (tim.inverr[Yi,Xi] > 0)
    
                if np.sum(insrc) == 0:
                    # No source pixels touching blob... this can
                    # happen when a source scatters outside the blob
                    # in the fitting stage.  Drop the source here.
                    B.sources[srci] = cat[srci] = None
                    continue
                yin = np.max(insrc, axis=1)
                xin = np.max(insrc, axis=0)
                yl,yh = np.flatnonzero(yin)[np.array([0,-1])]
                xl,xh = np.flatnonzero(xin)[np.array([0,-1])]
                srcwcs = self.blobwcs.get_subimage(xl, yl, 1+xh-xl, 1+yh-yl)
                srcbounds = [xl, xh, yl, yh]
                # A mask for which pixels in the 'srcwcs' square are occupied.
                srcpix = insrc[yl:yh+1, xl:xh+1]
            else:
                modelMasks = models.model_masks(srci, src)
                srctims = self.tims
                srcwcs = self.blobwcs
                srcpix = None

            srctractor = self.tractor(srctims, [src])
            srctractor.setModelMasks(modelMasks)
            enable_galaxy_cache()

            if self.plots_per_source:
                # This is a handy blob-coordinates plot of the data
                # going into the fit.
                plt.clf()
                coimgs,cons = quick_coadds(srctims, self.bands, self.blobwcs,
                                             fill_holes=False)
                dimshow(get_rgb(coimgs, self.bands))
                plt.title('Model selection: stage1 data')
                self.ps.savefig()

                # # plot the modelmasks for each tim.
                # plt.clf()
                # R = int(np.floor(np.sqrt(len(srctims))))
                # C = int(np.ceil(len(srctims) / float(R)))
                # for i,tim in enumerate(srctims):
                #     plt.subplot(R, C, i+1)
                #     msk = modelMasks[i][src].mask
                #     print('Mask:', msk)
                #     if msk is None:
                #         continue
                #     plt.imshow(msk, interpolation='nearest', origin='lower', vmin=0, vmax=1)
                #     plt.title(tim.name)
                # plt.suptitle('Model Masks')
                # self.ps.savefig()
                
            if self.bigblob and self.plots_per_source:
                # This is a local source-WCS plot of the data going into the
                # fit.
                plt.clf()
                coimgs,cons = quick_coadds(srctims, self.bands, srcwcs,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs, self.bands))
                plt.title('Model selection: stage1 data (srcwcs)')
                self.ps.savefig()

                self._plots(srctractor, 'Model selection init')
                # srch,srcw = srcwcs.shape
                # _plot_mods(srctims, [list(srctractor.getModelImages())], srcwcs,
                #            ['Model selection init'], self.bands, None,None,
                #            None, srcw,srch, self.ps, chi_plots=False)

            if self.deblend:
                # Create tims with the deblending-weighted pixels.
                debtims = [Image(data=np.zeros_like(tim.data),
                               inverr=tim.getInvError(),
                               wcs=tim.wcs, psf=tim.psf, photocal=tim.photocal,
                               sky=tim.sky, name=tim.name) for tim in srctims]
                for dtim,tim in zip(debtims,srctims):
                    dtim.band = tim.band
                    dtim.subwcs = tim.subwcs
                    # Resample the deb weights from blob space to tim space
                    try:
                        Yo,Xo,Yi,Xi,nil = resample_with_wcs(
                            tim.subwcs, self.blobwcs, [], 2)
                    except:
                        continue
                    dpatch = self.deb_profiles[srci]
                    ph,pw = dpatch.shape
                    K = np.flatnonzero((Yi >= dpatch.y0) * (Xi >= dpatch.x0) *
                                       (Yi < (dpatch.y0+ph)) *
                                       (Xi < (dpatch.x0+pw)))
                    dtim.data[Yo[K],Xo[K]] = (tim.data[Yo[K], Xo[K]] *
                        dpatch.patch[Yi[K]-dpatch.y0, Xi[K]-dpatch.x0] /
                        np.maximum(self.deb_prosum[Yi[K], Xi[K]], 1e-16))
                debtractor = self.tractor(debtims, srctractor.catalog)

            if self.bigblob and self.plots and self.deblend:
                plt.clf()
                coimgs,cons = quick_coadds(debtims, self.bands, srcwcs,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs, self.bands))
                plt.title('Deblend-weighted data')
                self.ps.savefig()
                    
            srccat = srctractor.getCatalog()

            # Compute the log-likehood without a source here.
            srccat[0] = None
            chisqs_none = _per_band_chisqs(srctractor, self.bands)
    
            nparams = dict(ptsrc=2, simple=2, rex=3, exp=5, dev=5, comp=9,psfexp=9)
            # This is our "upgrade" threshold: how much better a galaxy
            # fit has to be versus ptsrc, and comp versus galaxy.
            galaxy_margin = 3.**2 + (nparams['exp'] - nparams['ptsrc'])
    
            # *chisqs* is actually chi-squared improvement vs no source;
            # larger is a better fit.
            chisqs = dict(none=0)
    
            oldmodel, ptsrc, simple, dev, exp, comp, psfexp = _initialize_models(
                src, self.rex)
    
            if self.rex:
                simname = 'rex'
                rex = simple
            else:
                simname = 'simple'
                
            trymodels = [('ptsrc', ptsrc)]

            if oldmodel == 'ptsrc':
                forced = False
                if isinstance(src, GaiaSource):
                    print('Gaia source', src)
                    if src.isForcedPointSource():
                        forced = True
                if forced:
                    print('Gaia source is forced to be a point source -- not trying other models')
                elif self.hasbright:
                    print('Not computing galaxy models: bright star in blob')
                else:
                    trymodels.append((simname, simple))
                    # Try galaxy models if simple > ptsrc, or if bright.
                    # The 'gals' model is just a marker
                    trymodels.append(('gals', None))
            else:
                trymodels.extend([('dev', dev), ('exp', exp), ('psfexp',psfexp), ('comp',comp)])

            cputimes = {}
            for name,newsrc in trymodels:
                cpum0 = time.clock()
                
                if name == 'gals':
                    # If 'simple' was better than 'ptsrc', or the source is
                    # bright, try the galaxy models.
                    if ((chisqs[simname] > chisqs['ptsrc']) or
                        (chisqs['ptsrc'] > 400)):
                        trymodels.extend([
                            ('dev', dev), ('exp', exp), ('psfexp',psfexp), ('comp',comp)])
                    continue
                print(chisqs.keys(),chisqs)
                if name == 'psfexp' and newsrc is None:
                    if (max(chisqs['dev'], chisqs['exp']) <
                        (chisqs['ptsrc'] + galaxy_margin)):
                        print('dev/exp not much better than ptsrc not computing psfexp model.')
                        continue

                    
                    
                    # Compute the comp model if exp or dev would be accepted
                    #if (chisqs['comp'] <
                    #    (chisqs['ptsrc'] + galaxy_margin)):
                    #    print('comp not much better than ptsrc, not computing psfexp model.')
                    #    continue
                    print(src)
                    newsrc = psfexp = PSFandExpGalaxy_diffcentres(
                        src.getPosition(), src.getBrightness(), exp.getShape(),
                        src.getPosition(),src.getBrightness()).copy()

                

                if name == 'comp' and newsrc is None:
                    # Compute the comp model if exp or dev would be accepted
                    if (max(chisqs['dev'], chisqs['exp']) <
                        (chisqs['ptsrc'] + galaxy_margin)):
                        #print('dev/exp not much better than ptsrc;
                        #not computing comp model.')
                        continue
                    newsrc = comp = FixedCompositeGalaxy(
                        src.getPosition(), src.getBrightness(),
                        SoftenedFracDev(0.5), exp.getShape(),
                        dev.getShape()).copy()
                
                srccat[0] = newsrc
                
                #print('Starting optimization for', name)
                
                # Use the same modelMask shapes as the original source ('src').
                # Need to create newsrc->mask mappings though:
                mm = remap_modelmask(modelMasks, src, newsrc)
                srctractor.setModelMasks(mm)
                enable_galaxy_cache()
    
                # Save these modelMasks for later...
                newsrc_mm = mm
    
                #lnp = srctractor.getLogProb()
                #print('Initial log-prob:', lnp)
                #print('vs original src: ', lnp - lnp0)
                # if self.plots and False:
                #     # Grid of derivatives.
                #     _plot_derivs(tims, newsrc, srctractor, ps)
                # if self.plots:
                #     mods = list(srctractor.getModelImages())
                #     plt.clf()
                #     coimgs,cons = quick_coadds(srctims, bands, srcwcs,
                #                               images=mods, fill_holes=False)
                #     dimshow(get_rgb(coimgs, bands))
                #     plt.title('Initial: ' + name)
                #     self.ps.savefig()
    
                if self.deblend:
                    debtractor.setModelMasks(mm)
                    enable_galaxy_cache()
                    debtractor.optimize_loop(**self.optargs)

                if self.deblend and self.plots1:
                    plt.clf()
                    modimgs = list(debtractor.getModelImages())
                    comods,nil = quick_coadds(
                        debtims, self.bands, srcwcs, images=modimgs)
                    dimshow(get_rgb(comods, self.bands))
                    plt.title('Deblended opt: ' + name)
                    self.ps.savefig()
                    
                # First-round optimization (during model selection)
                #print('Optimizing: first round for', name, ':', len(srctims))
                #print(newsrc)
                cpustep0 = time.clock()
                srctractor.optimize_loop(**self.optargs)
                #print('Optimizing first round', name, 'took',
                #      time.clock()-cpustep0)
                #print(newsrc)
                # print('Mod', name, 'round1 opt', Time()-t0)
                #print('Mod selection: after first-round opt:', newsrc)

                disable_galaxy_cache()
    
                # if self.plots1:
                #     print('Computing first-round plot for', name)
                #     for tim in srctims:
                #         if hasattr(tim, 'resamp'):
                #             del tim.resamp
                #     plt.clf()
                #     coimgs,cons = quick_coadds(srctims, self.bands, srcwcs, images=list(srctractor.getModelImages()), fill_holes=False)
                #     dimshow(get_rgb(coimgs,self.bands))
                #     plt.title('first round: %s' % name)
                #     self.ps.savefig()

                srctractor.setModelMasks(None)
    
                # Recompute modelMasks in the original tims

                # Limit sizes of huge models
                if len(self.tims) > 0:
                    _limit_galaxy_stamp_size(newsrc, self.tims[0])
    
                if self.hasbright:
                    modtims = None
                elif self.bigblob:
                    mods = []
                    for tim in self.tims:
                        mod = newsrc.getModelPatch(tim)
                        if mod is not None:
                            h,w = tim.shape
                            mod.clipTo(w,h)
                            if mod.patch is None:
                                mod = None
                        mods.append(mod)
                    modtims,mm = _get_subimages(self.tims, mods, newsrc)

                else:
                    mm = []
                    modtims = []
                    for tim in self.tims:
                        d = dict()
                        mod = newsrc.getModelPatch(tim)
                        if mod is None:
                            continue
                        #print('After first-round fit: model is', mod.shape)
                        mod = _clip_model_to_blob(mod, tim.shape,
                                                  tim.getInvError())
                        if mod is None:
                            continue
                        #d[newsrc] = ModelMask(mod.x0, mod.y0, mod.patch != 0)
                        mh,mw = mod.shape
                        d[newsrc] = ModelMask(mod.x0, mod.y0, mw, mh)
                        modtims.append(tim)
                        mm.append(d)

                #print('Mod selection: after second-round opt:', newsrc)

                if modtims is not None:
                    modtractor = self.tractor(modtims, [newsrc])
                    modtractor.setModelMasks(mm)
                    enable_galaxy_cache()
    
                    modtractor.optimize_loop(maxcpu=60., **self.optargs)
                    #print('Mod selection: after second-round opt:', newsrc)

                    if self.plots_per_source:
                        plt.clf()
                        modimgs = list(modtractor.getModelImages())
                        comods,nil = quick_coadds(modtims, self.bands, srcwcs,
                                                    images=modimgs)
                        dimshow(get_rgb(comods, self.bands))
                        plt.title('After second-round opt: ' + name)
                        self.ps.savefig()

                        # self._plot_coadd(modtims, self.blobwcs,resid=modtractor)
                        # plt.title('second round: resid %s' % name)
                        # self.ps.savefig()

                else:
                    # Bright star; set modtractor = srctractor for the ivars
                    srctractor.setModelMasks(newsrc_mm)
                    modtractor = srctractor

                # Compute inverse-variances for each source.
                # Convert to "vanilla" ellipse parameterization
                # (but save old shapes first)
                # we do this (rather than making a copy) because we want to
                # use the same modelMask maps.
                if isinstance(newsrc, (DevGalaxy, ExpGalaxy)):
                    oldshape = newsrc.shape
                elif isinstance(newsrc, FixedCompositeGalaxy):
                    oldshape = (newsrc.shapeExp, newsrc.shapeDev,newsrc.fracDev)

                nsrcparams = newsrc.numberOfParams()
                _convert_ellipses(newsrc)
                assert(newsrc.numberOfParams() == nsrcparams)
                # Compute inverse-variances
                # This uses the second-round modelMasks.
                allderivs = modtractor.getDerivs()
                ivars = _compute_invvars(allderivs)
                assert(len(ivars) == nsrcparams)
                B.all_model_ivs[srci][name] = np.array(ivars).astype(np.float32)
                B.all_models[srci][name] = newsrc.copy()
                assert(B.all_models[srci][name].numberOfParams() == nsrcparams)

                # print('Model', name)
                # for nm,p,piv in zip(newsrc.getParamNames(), newsrc.getParams(),
                #                     ivars):
                #     print('  S/N of', nm, '=', p * np.sqrt(piv))
                
                # Now revert the ellipses!
                if isinstance(newsrc, (DevGalaxy, ExpGalaxy)):
                    newsrc.shape = oldshape
                elif isinstance(newsrc, FixedCompositeGalaxy):
                    (newsrc.shapeExp, newsrc.shapeDev,newsrc.fracDev) = oldshape

                # Use the original 'srctractor' here so that the different
                # models are evaluated on the same pixels.
                # ---> AND with the same modelMasks as the original source...
                #
                # FIXME -- it is not clear that this is what we want!!!
                #
                srctractor.setModelMasks(newsrc_mm)
                ch = _per_band_chisqs(srctractor, self.bands)
                    
                chisqs[name] = _chisq_improvement(newsrc, ch, chisqs_none)
                cpum1 = time.clock()
                B.all_model_cpu[srci][name] = cpum1 - cpum0
                cputimes[name] = cpum1 - cpum0
                #print('Fitting', name, 'took', cputimes[name])

                # if self.plots:
                #     print('Plotting model selection evaluated for', name)
                #     print('Tim PSFs:', [str(tim.psf) for tim in srctims])
                #     self._plot_coadd(srctims, self.blobwcs, model=srctractor)
                #     plt.title('model selection evaluated for %s' % name)
                #     self.ps.savefig()
                
            # Actually select which model to keep.  This "modnames"
            # array determines the order of the elements in the DCHISQ
            # column of the catalog.
            modnames = ['ptsrc', simname, 'dev', 'exp', 'comp', 'psfexp']
            keepmod = _select_model(chisqs, nparams, galaxy_margin, self.rex)
            keepsrc = {'none':None, 'ptsrc':ptsrc, simname:simple,
                       'dev':dev, 'exp':exp, 'comp':comp, 'psfexp':psfexp}[keepmod]
            bestchi = chisqs.get(keepmod, 0.)

            if keepsrc is not None and bestchi == 0.:
                # Weird edge case, or where some best-fit fluxes go
                # negative. eg
                # https://github.com/legacysurvey/legacypipe/issues/174
                print('Best dchisq is 0 -- dropping source')
                keepsrc = None

            # for nm in modnames:
            #     c = chisqs.get(nm, 0.)
            #     more = ''
            #     if nm == 'rex':
            #         more = '     re = %.3g' % simple.shape.re
            #     print('  % -5s' % nm, 'dchisq %12.2f' % c,
            #           'delta %12.2f' % (c-bestchi), more)
            # print('->', keepmod)

            # # This makes a set of images, one per tim, showing ptsrc vs simp/rex
            # if self.plots:
            #     # DEBUGGING ptsrc vs rex
            #     modlist = [('ptsrc',ptsrc),(simname,simple)]
            #     modimgs = []
            #     for modname,mod in modlist:
            #         srccat[0] = mod
            #         #srctractor.setModelMasks(None)
            #         mm = remap_modelmask(modelMasks, src, mod)
            #         srctractor.setModelMasks(None)
            #         modimgs.append(list(srctractor.getModelImages()))
            # 
            #     nimgs = len(srctractor.getImages())
            #     for itim in range(nimgs):
            #         chi2vals = []
            #         mod0 = None
            #         plt.clf()
            #         for imod,(modname,mod) in enumerate(modlist):
            #             modimg = modimgs[imod][itim]
            #             tim = srctractor.getImages()[itim]
            #             plt.subplot(len(modlist), 4, 4*imod+1)
            #             ima = dict(interpolation='nearest', origin='lower',
            #                        vmin=-2.*tim.sig1, vmax=5.*tim.sig1)
            #             tt = modname
            #             if modname == 'rex':
            #                 tt = tt + ' re=%.2g' % mod.shape.re
            #             plt.title(tt)
            #             plt.imshow(modimg, **ima)
            #             if imod == 0:
            #                 mod0 = modimg
            #             else:
            #                 plt.subplot(len(modlist), 4, 4*imod+2)
            #                 plt.imshow(modimg - mod0, interpolation='nearest', origin='lower',
            #                            vmin=-1.*tim.sig1, vmax=1.*tim.sig1, cmap='RdBu')
            #                 plt.title('mod diff')
            #             plt.subplot(len(modlist), 4, 4*imod+3)
            #             plt.imshow(tim.getImage(), **ima)
            #             plt.subplot(len(modlist), 4, 4*imod+4)
            #             plt.imshow((tim.getImage() - modimg) * tim.getInvError(),
            #                        interpolation='nearest', origin='lower',
            #                        vmin=-3., vmax=+3.)
            #             chi2 = np.sum((tim.getImage() - modimg)**2 * tim.getInvvar())
            #             plt.title('chisq: %.2f' % chi2)
            #             chi2vals.append(chi2)
            #         plt.suptitle('Tim %s; dchisq %.2f' % (tim.name, (chi2vals[0]-chi2vals[1])))
            #         self.ps.savefig()
                        
            # This is the model-selection plot
            if self.plots_per_source:
                from collections import OrderedDict
                subplots = []
                plt.clf()
                rows,cols = 3, 6
                mods = OrderedDict([
                    ('none',None), ('ptsrc',ptsrc), (simname,simple),
                    ('dev',dev), ('exp',exp), ('comp',comp),('psfexp',psfexp)])
                for imod,modname in enumerate(mods.keys()):
                    if modname != 'none' and not modname in chisqs:
                        continue
                    srccat[0] = mods[modname]
                    srctractor.setModelMasks(None)
                    plt.subplot(rows, cols, imod+1)
                    if modname == 'none':
                        # In the first panel, we show a coadd of the data
                        coimgs, cons = quick_coadds(srctims, self.bands,srcwcs)
                        rgbims = coimgs
                        rgb = get_rgb(coimgs, self.bands)
                        dimshow(rgb, ticks=False)
                        subplots.append(('data', rgb))
                        ax = plt.axis()
                        ok,x,y = srcwcs.radec2pixelxy(
                            src.getPosition().ra, src.getPosition().dec)
                        plt.plot(x-1, y-1, 'r+')
                        plt.axis(ax)
                        tt = 'Image'
                        chis = [((tim.getImage()) * tim.getInvError())**2
                                  for tim in srctims]
                        res = [tim.getImage() for tim in srctims]
                    else:
                        modimgs = list(srctractor.getModelImages())
                        comods,nil = quick_coadds(srctims, self.bands, srcwcs,
                                                    images=modimgs)
                        rgbims = comods
                        rgb = get_rgb(comods, self.bands)
                        dimshow(rgb, ticks=False)
                        subplots.append(('mod'+modname, rgb))
                        tt = modname #+ '\n(%.0f s)' % cputimes[modname]
                        chis = [((tim.getImage() - mod) * tim.getInvError())**2
                                for tim,mod in zip(srctims, modimgs)]
                        res = [(tim.getImage() - mod) for tim,mod in
                               zip(srctims, modimgs)]

                    # Second row: same rgb image with arcsinh stretch
                    plt.subplot(rows, cols, imod+1+cols)
                    dimshow(get_rgb(rgbims, self.bands, **rgbkwargs), ticks=False)
                    plt.title(tt)

                    # residuals
                    coresids,nil = quick_coadds(srctims, self.bands, srcwcs,
                                                  images=res)
                    plt.subplot(rows, cols, imod+1+2*cols)
                    rgb = get_rgb(coresids, self.bands, **rgbkwargs_resid)
                    dimshow(rgb, ticks=False)
                    subplots.append(('res'+modname, rgb))
                    plt.title('chisq %.0f' % chisqs[modname], fontsize=8)
                plt.suptitle('Blob %s, source %i: keeping %s\nwas: %s' %
                             (self.name, srci, keepmod, str(src)), fontsize=10)
                self.ps.savefig()

                # Save individual plots (for the paper)
                if self.plots_single:
                    for name,rgb in subplots:
                        plt.figure(2)
                        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
                        dimshow(rgb, ticks=False)
                        fn = 'blob-%s-%i-%s.png' % (self.name, srci, name)
                        plt.savefig(fn)
                        print('Wrote', fn)
                        plt.figure(1)

            B.dchisq[srci, :] = np.array([chisqs.get(k,0) for k in modnames])
            B.sources[srci] = keepsrc
            cat[srci] = keepsrc
    
            # Re-remove the final fit model for this source.
            models.update_and_subtract(srci, keepsrc, self.tims)


            if self.plots_single:
                plt.figure(2)
                tr = self.tractor(self.tims, cat)
                #mods = list(tr.getModelImages())
                coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs, #images=mods,
                                           fill_holes=False)
                # not using rgbkwargs_resid
                dimshow(get_rgb(coimgs,self.bands), ticks=False)
                plt.savefig('blob-%s-%i-sub.png' % (self.name, srci))
                plt.figure(1)

            #print('Keeping model:', keepmod)
            #print('Keeping source:', keepsrc)
            cpu1 = time.clock()
            B.cpu_source[srci] += (cpu1 - cpu0)

        models.restore_images(self.tims)
        del models
            
    def _optimize_individual_sources(self, tr, cat, Ibright, cputime):
        # Single source (though this is coded to handle multiple sources)
        # Fit sources one at a time, but don't subtract other models
        cat.freezeAllParams()

        models = SourceModels()
        models.create(self.tims, cat)
        enable_galaxy_cache()

        for numi,i in enumerate(Ibright):
            cpu0 = time.clock()
            #print('Fitting source', i, '(%i of %i in blob)' %
            #  (numi, len(Ibright)))
            cat.freezeAllBut(i)
            modelMasks = models.model_masks(0, cat[i])
            tr.setModelMasks(modelMasks)
            tr.optimize_loop(**self.optargs)
            #print('Fitting source took', Time()-tsrc)
            # print(cat[i])
            cpu1 = time.clock()
            cputime[i] += (cpu1 - cpu0)
            
        tr.setModelMasks(None)
        disable_galaxy_cache()
        
    def tractor(self, tims, cat):
        tr = Tractor(tims, cat, **self.trargs)
        tr.freezeParams('images')
        return tr

    def _optimize_individual_sources_subtract(self, cat, Ibright,
                                              cputime):
        # -Remember the original images
        # -Compute initial models for each source (in each tim)
        # -Subtract initial models from images
        # -During fitting, for each source:
        #   -add back in the source's initial model (to each tim)
        #   -fit, with Catalog([src])
        #   -subtract final model (from each tim)
        # -Replace original images
    
        models = SourceModels()
        # Remember original tim images
        models.save_images(self.tims)
        # Create & subtract initial models for each tim x each source
        models.create(self.tims, cat, subtract=True)

        # For sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
            cpu0 = time.clock()
            print('Fitting source', srci, '(%i of %i in blob %s)' %
                  (numi+1, len(Ibright), self.name))
            src = cat[srci]
            # Add this source's initial model back in.
            models.add(srci, self.tims)
    
            if self.bigblob:
                # Create super-local sub-sub-tims around this source
    
                # Make the subimages the same size as the modelMasks.
                #tbb0 = Time()
                mods = [mod[srci] for mod in models.models]
                srctims,modelMasks = _get_subimages(self.tims, mods, src)
                #print('Creating srctims:', Time()-tbb0)
    
                # We plots only the first & last three sources
                if self.plots_per_source and (numi < 3 or numi >= len(Ibright)-3):
                    plt.clf()
                    # Recompute coadds because of the subtract-all-and-readd shuffle
                    coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                                 fill_holes=False)
                    rgb = get_rgb(coimgs, self.bands)
                    dimshow(rgb)
                    #dimshow(self.rgb)
                    ax = plt.axis()
                    for tim in srctims:
                        h,w = tim.shape
                        tx,ty = [0,0,w,w,0], [0,h,h,0,0]
                        rd = [tim.getWcs().pixelToPosition(xi,yi)
                              for xi,yi in zip(tx,ty)]
                        ra  = [p.ra  for p in rd]
                        dec = [p.dec for p in rd]
                        ok,x,y = self.blobwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'b-')
                        ra,dec = tim.subwcs.pixelxy2radec(tx, ty)
                        ok,x,y = self.blobwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'c-')
                    plt.title('source %i of %i' % (numi, len(Ibright)))
                    plt.axis(ax)
                    self.ps.savefig()
    
            else:
                srctims = self.tims
                modelMasks = models.model_masks(srci, src)


            srctractor = self.tractor(srctims, [src])
            #print('Setting modelMasks:', modelMasks)
            srctractor.setModelMasks(modelMasks)
            
            # if plots and False:
            #     spmods,spnames = [],[]
            #     spallmods,spallnames = [],[]
            #     if numi == 0:
            #         spallmods.append(list(tr.getModelImages()))
            #         spallnames.append('Initial (all)')
            #     spmods.append(list(srctractor.getModelImages()))
            #     spnames.append('Initial')
    
            # First-round optimization
            #print('First-round initial log-prob:', srctractor.getLogProb())
            srctractor.optimize_loop(**self.optargs)
            #print('First-round final log-prob:', srctractor.getLogProb())
    
            # if plots and False:
            #     spmods.append(list(srctractor.getModelImages()))
            #     spnames.append('Fit')
            #     spallmods.append(list(tr.getModelImages()))
            #     spallnames.append('Fit (all)')
            # 
            # if plots and False:
            #     plt.figure(1, figsize=(8,6))
            #     plt.subplots_adjust(left=0.01, right=0.99, top=0.95,
            #                         bottom=0.01, hspace=0.1, wspace=0.05)
            #     #plt.figure(2, figsize=(3,3))
            #     #plt.subplots_adjust(left=0.005, right=0.995,
            #     #                    top=0.995,bottom=0.005)
            #     #_plot_mods(tims, spmods, spnames, bands, None, None, bslc,
            #     #           blobw, blobh, ps, chi_plots=plots2)
            #     plt.figure(2, figsize=(3,3.5))
            #     plt.subplots_adjust(left=0.005, right=0.995,
            #                         top=0.88, bottom=0.005)
            #     plt.suptitle('Blob %i' % iblob)
            #     tempims = [tim.getImage() for tim in tims]
            # 
            #     _plot_mods(list(srctractor.getImages()), spmods, spnames,
            #                bands, None, None, bslc, blobw, blobh, ps,
            #                chi_plots=plots2, rgb_plots=True, main_plot=False,
            #                rgb_format=('spmods Blob %i, src %i: %%s' %
            #                            (iblob, i)))
            #     _plot_mods(tims, spallmods, spallnames, bands, None, None,
            #                bslc, blobw, blobh, ps,
            #                chi_plots=plots2, rgb_plots=True, main_plot=False,
            #                rgb_format=('spallmods Blob %i, src %i: %%s' %
            #                            (iblob, i)))
            # 
            #     models.restore_images(tims)
            #     _plot_mods(tims, spallmods, spallnames, bands, None, None,
            #                bslc, blobw, blobh, ps,
            #                chi_plots=plots2, rgb_plots=True, main_plot=False,
            #                rgb_format='Blob %i, src %i: %%s' % (iblob, i))
            #     for tim,im in zip(tims, tempims):
            #         tim.data = im
    
            # Re-remove the final fit model for this source
            models.update_and_subtract(srci, src, self.tims)
    
            srctractor.setModelMasks(None)
            disable_galaxy_cache()
    
            #print('Fitting source took', Time()-tsrc)
            #print(src)
            cpu1 = time.clock()
            cputime[srci] += (cpu1 - cpu0)
            
        models.restore_images(self.tims)
        del models
    
    def _fit_fluxes(self, cat, tims, bands):
        cat.thawAllRecursive()
        for src in cat:
            src.freezeAllBut('brightness')
        for b in bands:
            for src in cat:
                src.getBrightness().freezeAllBut(b)
            # Images for this band
            btims = [tim for tim in tims if tim.band == b]
    
            btr = self.tractor(btims, cat)
            btr.optimize_forced_photometry(shared_params=False, wantims=False)
        cat.thawAllRecursive()

    def _plots(self, tr, title):
        plotmods = []
        plotmodnames = []
        plotmods.append(list(tr.getModelImages()))
        plotmodnames.append(title)
        for tim in tr.images:
            if hasattr(tim, 'resamp'):
                del tim.resamp
        _plot_mods(tr.images, plotmods, self.blobwcs, plotmodnames, self.bands,
                   None, None, None,
                   self.blobw, self.blobh, self.ps, chi_plots=False)
        for tim in tr.images:
            if hasattr(tim, 'resamp'):
                del tim.resamp

    def _plot_coadd(self, tims, wcs, model=None, resid=None):
        if resid is not None:
            mods = list(resid.getChiImages())
            coimgs,cons = quick_coadds(tims, self.bands, wcs, images=mods,
                                       fill_holes=False)
            dimshow(get_rgb(coimgs,self.bands, **rgbkwargs_resid))
            return
            
        mods = None
        if model is not None:
            mods = list(model.getModelImages())
        coimgs,cons = quick_coadds(tims, self.bands, wcs, images=mods,
                                   fill_holes=False)
        dimshow(get_rgb(coimgs,self.bands))
        
    def _initial_plots(self):
        print('Plotting blob image for blob', self.name)
        coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                     fill_holes=False)
        self.rgb = get_rgb(coimgs, self.bands)
        plt.clf()
        dimshow(self.rgb)
        plt.title('Blob: %s' % self.name)
        self.ps.savefig()

        if self.plots_single:
            plt.figure(2)
            dimshow(self.rgb, ticks=False)
            plt.savefig('blob-%s-data.png' % (self.name))
            plt.figure(1)

        ok,x0,y0 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in self.srcs]),
            np.array([src.getPosition().dec for src in self.srcs]))

        ax = plt.axis()
        plt.plot(x0-1, y0-1, 'r.')
        plt.axis(ax)
        plt.title('initial sources')
        self.ps.savefig()

        # plt.clf()
        # ccmap = dict(g='g', r='r', z='m')
        # for tim in tims:
        #     chi = (tim.data * tim.inverr)[tim.inverr > 0]
        #     plt.hist(chi.ravel(), range=(-5,10), bins=100, histtype='step',
        #              color=ccmap[tim.band])
        # plt.xlabel('signal/noise per pixel')
        # self.ps.savefig()
        
    def create_tims(self, timargs):
        # In order to make multiprocessing easier, the one_blob method
        # is passed all the ingredients to make local tractor Images
        # rather than the Images themselves.  Here we build the
        # 'tims'.
        tims = []
        for (img, inverr, twcs, wcs, pcal, sky, psf, name, sx0, sx1, sy0, sy1,
             band, sig1, modelMinval, imobj) in timargs:
            # Mask out inverr for pixels that are not within the blob.
            subwcs = wcs.get_subimage(int(sx0), int(sy0),
                                      int(sx1-sx0), int(sy1-sy0))
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(subwcs, self.blobwcs,
                                                     [], 2)
            except OverlapError:
                continue
            if len(Yo) == 0:
                continue
            inverr2 = np.zeros_like(inverr)
            I = np.flatnonzero(self.blobmask[Yi,Xi])
            inverr2[Yo[I],Xo[I]] = inverr[Yo[I],Xo[I]]
            inverr = inverr2

            # If the subimage (blob) is small enough, instantiate a
            # constant PSF model in the center.
            if sy1-sy0 < 400 and sx1-sx0 < 400:
                subpsf = psf.constantPsfAt((sx0+sx1)/2., (sy0+sy1)/2.)
            else:
                # Otherwise, instantiate a (shifted) spatially-varying
                # PsfEx model.
                subpsf = psf.getShifted(sx0, sy0)

            tim = Image(data=img, inverr=inverr, wcs=twcs,
                        psf=subpsf, photocal=pcal, sky=sky, name=name)
            tim.band = band
            tim.sig1 = sig1
            tim.modelMinval = modelMinval
            tim.subwcs = subwcs
            tim.meta = imobj
            tims.append(tim)
        return tims

    def run_deblend(self, cat):
        ras  = np.array([src.pos.ra  for src in cat])
        decs = np.array([src.pos.dec for src in cat])
        ok,x,y = self.blobwcs.radec2pixelxy(ras, decs)
        x -= 1
        y -= 1
        Xi = np.round(x).astype(int)
        Yi = np.round(y).astype(int)

        # Combine the bands to make a single deblending profile...
        # What weighting to use though?  Median S/N?
        # Straight per-pixel weight?  (That's like flat-spectrum assumptn)
        # (this is like [sed-matched] detection...)
        coimgs,cons,cowimgs,wimgs = quick_coadds(
            self.tims, self.bands, self.blobwcs, get_cow=True,
            fill_holes=False)
        #wts = [np.median(wt[wt > 0]) for wt in wimgs]
        #print('Median weights:', wts)
        bimg = np.zeros_like(cowimgs[0])
        bwt  = np.zeros_like(cowimgs[0])
        for im,wt in zip(cowimgs,wimgs):
            bimg += wt * im
            bwt  += wt
        bimg /= np.maximum(1e-16, bwt)
        sig1 = 1. / np.sqrt(np.median(wt[wt > 0]))

        if self.plots:
            plt.clf()
            ima = dict(vmin=-2.*sig1, vmax=5.*sig1)
            dimshow(bimg, **ima)
            plt.title('Deblend -- merged bands')
            self.ps.savefig()
            ax = plt.axis()
            plt.plot(Xi, Yi, 'r.')
            plt.axis(ax)
            self.ps.savefig()
            self.deb_ima = ima

        # size of region to use as postage stamp for deblending
        sz = 32
        h,w = bimg.shape

        profiles = []
        allprofiles = np.zeros_like(bimg)
        for isrc,(xi,yi) in enumerate(zip(Xi,Yi)):
            dx = min(sz, min(xi, w-1-xi))
            dy = min(sz, min(yi, h-1-yi))
            x0,y0 = xi - dx, yi - dy
            slc = slice(y0, yi + dy+1), slice(x0, xi + dx+1)
            subimg = bimg[slc]
            subwt =   bwt[slc]
            flipped = np.fliplr(np.flipud(subimg))
            flipwt  = np.fliplr(np.flipud(subwt))
            minimg = subimg.copy()
            # Mirror the blob boundaries
            submask = self.blobmask[slc]
            flipmask = np.fliplr(np.flipud(submask))

            I = np.flatnonzero((flipwt > 0) * (flipped < subimg))
            minimg.flat[I] = flipped.flat[I]
            minimg[flipmask == False] = 0

            # Correct for min() bias for two Gaussians.  This isn't
            # really the right way to do this
            minimg[subwt > 0] += 0.545 * np.sqrt(1. / subwt[subwt > 0])
            # And this is *really* a hack
            minimg = np.maximum(0, minimg)

            patch = Patch(x0, y0, minimg)
            profiles.append(patch)
            patch.addTo(allprofiles)

            # if self.plots:
            #     plt.clf()
            #     plt.subplot(2,3,1)
            #     dimshow(subimg, **ima)
            #     plt.subplot(2,3,2)
            #     dimshow(flipped, **ima)
            #     #plt.subplot(2,3,3)
            #     #dimshow(goodpix, vmin=0, vmax=1)
            #     plt.subplot(2,3,4)
            #     dimshow(minimg, **ima)
            #     plt.subplot(2,3,5)
            #     dimshow(allprofiles, **ima)
            #     self.ps.savefig()

        if self.plots:
            plt.clf()
            dimshow(allprofiles, **ima)
            plt.title('Deblend -- sum of profiles')
            self.ps.savefig()

        self.deb_profiles = profiles
        self.deb_prosum = allprofiles

    # if plots:
    #         try:
    #             Yo,Xo,Yi,Xi,rims = resample_with_wcs(blobwcs, subwcs,[],2)
    #         except OverlapError:
    #             continue
    #         tim.resamp = (Yo, Xo, Yi, Xi)
    #         if False:
    #             plt.clf()
    #             plt.subplot(1,2,1)
    #             dimshow(img, vmin=-2.*sig1, vmax=5.*sig1)
    #             plt.subplot(1,2,2)
    #             dimshow(inverr, vmin=0, vmax=1.1/sig1)
    #             plt.suptitle('Subimage: ' + name)
    #             self.ps.savefig()
    # 
    # if plots and False:
    #     plotmods.append(list(tr.getModelImages()))
    #     plotmodnames.append('Per Source')

    # if plots and False:
    #     plotmods.append(list(tr.getModelImages()))
    #     plotmodnames.append('All Sources')
    # if plots and False:
    #     _plot_mods(tims, plotmods, plotmodnames, bands, None, None,
    #            bslc, blobw, blobh, ps)

    # if plots:
    #     plt.clf()
    #     dimshow(get_rgb(coimgs, bands))
    #     ok,sx,sy = blobwcs.radec2pixelxy(
    #         np.array([src.getPosition().ra  for src in srcs]),
    #         np.array([src.getPosition().dec for src in srcs]))
    #     plt.plot(sx, sy, 'r.')
    #     plt.title('after source fitting')
    #     self.ps.savefig()

    # FIXME -- render initial models and find significant flux overlap
    # (product)??  (Could use the same logic above!)  This would give
    # families of sources to fit simultaneously.  (The
    # not-friends-of-friends version of blobs!)


    # if plots:
    #     plotmods, plotmodnames = [],[]
    #     plotmods.append(list(tr.getModelImages()))
    #     plotmodnames.append('All model selection')
    #     _plot_mods(tims, plotmods, plotmodnames, bands, None, None, bslc,
    #                blobw, blobh, ps)


    # print('After cutting sources:')
    # for src,dchi in zip(B.sources, B.dchisqs):
    #     print('  source', src, 'max dchisq', max(dchi), 'dchisqs', dchi)

    #print('Blob variances:', Time()-tlast)
    #tlast = Time()

def _convert_ellipses(src):
    if isinstance(src, (DevGalaxy, ExpGalaxy)):
        #print('Converting ellipse for source', src)
        src.shape = src.shape.toEllipseE()
        #print('--->', src.shape)
        if isinstance(src, RexGalaxy):
            src.shape.freezeParams('e1', 'e2')
    elif isinstance(src, FixedCompositeGalaxy):
        src.shapeExp = src.shapeExp.toEllipseE()
        src.shapeDev = src.shapeDev.toEllipseE()
        src.fracDev = FracDev(src.fracDev.clipped())

def _compute_invvars(allderivs):
    ivs = []
    for iparam,derivs in enumerate(allderivs):
        chisq = 0
        for deriv,tim in derivs:
            h,w = tim.shape
            deriv.clipTo(w,h)
            ie = tim.getInvError()
            slc = deriv.getSlice(ie)
            chi = deriv.patch * ie[slc]
            chisq += (chi**2).sum()
        ivs.append(chisq)
    return ivs

def _argsort_by_brightness(cat, bands):
    fluxes = []
    for src in cat:
        # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
        br = src.getBrightness()
        flux = sum([br.getFlux(band) for band in bands])
        fluxes.append(flux)
    Ibright = np.argsort(-np.array(fluxes))
    return Ibright

def _compute_source_metrics(srcs, tims, bands, tr):
    # rchi2 quality-of-fit metric
    rchi2_num    = np.zeros((len(srcs),len(bands)), np.float32)
    rchi2_den    = np.zeros((len(srcs),len(bands)), np.float32)

    # fracflux degree-of-blending metric
    fracflux_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracflux_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracin flux-inside-blob metric
    fracin_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracin_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracmasked: fraction of masked pixels metric
    fracmasked_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracmasked_den = np.zeros((len(srcs),len(bands)), np.float32)

    for iband,band in enumerate(bands):
        for tim in tims:
            if tim.band != band:
                continue
            mod = np.zeros(tim.getModelShape(), tr.modtype)
            srcmods = [None for src in srcs]
            counts = np.zeros(len(srcs))
            pcal = tim.getPhotoCal()

            # For each source, compute its model and record its flux
            # in this image.  Also compute the full model *mod*.
            for isrc,src in enumerate(srcs):
                patch = tr.getModelPatch(tim, src, minsb=tim.modelMinval)
                if patch is None or patch.patch is None:
                    continue
                counts[isrc] = np.sum([np.abs(pcal.brightnessToCounts(b))
                                              for b in src.getBrightnesses()])
                if counts[isrc] == 0:
                    continue
                H,W = mod.shape
                patch.clipTo(W,H)
                srcmods[isrc] = patch
                patch.addTo(mod)

            # Now compute metrics for each source
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                if patch.patch is None:
                    continue
                if counts[isrc] == 0:
                    continue
                if np.sum(patch.patch**2) == 0:
                    continue
                slc = patch.getSlice(mod)
                patch = patch.patch

                # print('fracflux: band', band, 'isrc', isrc, 'tim', tim.name)
                # print('src:', srcs[isrc])
                # print('patch sum', np.sum(patch),'abs',np.sum(np.abs(patch)))
                # print('counts:', counts[isrc])
                # print('mod slice sum', np.sum(mod[slc]))
                # print('mod[slc] - patch:', np.sum(mod[slc] - patch))

                # (mod - patch) is flux from others
                # (mod - patch) / counts is normalized flux from others
                # We take that and weight it by this source's profile;
                #  patch / counts is unit profile
                # But this takes the dot product between the profiles,
                # so we have to normalize appropriately, ie by
                # (patch**2)/counts**2; counts**2 drops out of the
                # denom.  If you have an identical source with twice the flux,
                # this results in fracflux being 2.0

                # fraction of this source's flux that is inside this patch.
                # This can be < 1 when the source is near an edge, or if the
                # source is a huge diffuse galaxy in a small patch.
                fin = np.abs(np.sum(patch) / counts[isrc])

                # print('fin:', fin)
                # print('fracflux_num: fin *',
                #      np.sum((mod[slc] - patch) * np.abs(patch)) /
                #      np.sum(patch**2))

                fracflux_num[isrc,iband] += (fin *
                    np.sum((mod[slc] - patch) * np.abs(patch)) /
                    np.sum(patch**2))
                fracflux_den[isrc,iband] += fin
                
                fracmasked_num[isrc,iband] += (
                    np.sum((tim.getInvError()[slc] == 0) * np.abs(patch)) /
                    np.abs(counts[isrc]))
                    
                fracmasked_den[isrc,iband] += fin

                fracin_num[isrc,iband] += np.abs(np.sum(patch))
                fracin_den[isrc,iband] += np.abs(counts[isrc])

            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2

            for isrc,patch in enumerate(srcmods):
                if patch is None or patch.patch is None:
                    continue
                if counts[isrc] == 0:
                    continue
                slc = patch.getSlice(mod)
                # We compute numerator and denom separately to handle
                # edge objects, where sum(patch.patch) < counts.
                # Also, to normalize by the number of images.  (Being
                # on the edge of an image is like being in half an
                # image.)
                rchi2_num[isrc,iband] += (np.sum(chisq[slc] * patch.patch) / 
                                          counts[isrc])
                # If the source is not near an image edge,
                # sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.sum(patch.patch) / counts[isrc]

    #print('Fracflux_num:', fracflux_num)
    #print('Fracflux_den:', fracflux_den)
                
    fracflux   = fracflux_num   / fracflux_den
    rchi2      = rchi2_num      / rchi2_den
    fracmasked = fracmasked_num / fracmasked_den

    # Eliminate NaNs (these happen when, eg, we have no coverage in one band but
    # sources detected in another band, hence denominator is zero)
    fracflux  [  fracflux_den == 0] = 0.
    rchi2     [     rchi2_den == 0] = 0.
    fracmasked[fracmasked_den == 0] = 0.

    # fracin_{num,den} are in flux * nimages units
    tinyflux = 1e-9
    fracin     = fracin_num     / np.maximum(tinyflux, fracin_den)

    return dict(fracin=fracin, fracflux=fracflux, rchisq=rchi2,
                fracmasked=fracmasked)

def _initialize_models(src, rex):
    if isinstance(src, PointSource):
        ptsrc = src.copy()
        if rex:
            from legacypipe.survey import LogRadius
            simple = RexGalaxy(src.getPosition(), src.getBrightness(),
                               LogRadius(-1.)).copy()
            #print('Created Rex:', simple)
        else:
            simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        # logr, ee1, ee2
        shape = LegacyEllipseWithPriors(-1., 0., 0.)
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        psfexp= None  
        comp = None
        oldmodel = 'ptsrc'

    elif isinstance(src, DevGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        dev = src.copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        comp = None
        psfexp = None
        oldmodel = 'dev'

    elif isinstance(src, ExpGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        dev = DevGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        exp = src.copy()
        comp = None
        psfexp = None
        oldmodel = 'exp'

    elif isinstance(src, FixedCompositeGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        frac = src.fracDev.clipped()
        if frac > 0:
            shape = src.shapeDev
        else:
            shape = src.shapeExp
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        if frac < 1:
            shape = src.shapeExp
        else:
            shape = src.shapeDev
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        comp = src.copy()
        psfexp = PSFandExpGalaxy_diffcentres(src.getPosition(),src.getBrightnesses()[0],src.shapeExp,src.getPosition(),src.getBrightnesses()[1])
        oldmodel = 'comp'
    elif isinstance(src, PSFandExpGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        
        shape = src.shapeExp
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()

        dev = DevGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        psfexp = src.copy()
        comp = FixedCompositeGalaxy(src.getPosition(), src.BrightnessExp,src.shapeExp,src.BrightnessDev,src.shapeDev) 
        oldmodel = 'psfexp'


    return oldmodel, ptsrc, simple, dev, exp, comp, psfexp

def _get_subimages(tims, mods, src):
    subtims = []
    modelMasks = []
    #print('Big blob: trimming:')
    for tim,mod in zip(tims, mods):
        if mod is None:
            continue
        mh,mw = mod.shape
        print('MOD SHAPE',mh,mw)
        if mh == 0 or mw == 0:
            continue
        # for modelMasks
        d = { src: ModelMask(0, 0, mw, mh) } #mod.patch != 0) }
        modelMasks.append(d)

        x0,y0 = mod.x0 , mod.y0
        x1,y1 = x0 + mw, y0 + mh
        slc = slice(y0,y1), slice(x0, x1)

        subimg = tim.getImage()[slc]
        if subimg.shape != (mh,mw):
            print('Subimage shape:', subimg.shape, 'image shape',
                  tim.getImage().shape, 'slice y', y0,y1, 'x', x0,x1,
                  'mod shape', mh,mw)
        # print('  subtim: x0,y0', x0,y0, 'shape', (y1-y0,x1-x0))
        subpsf = tim.psf.constantPsfAt((x0+x1)/2., (y0+y1)/2.)
        subtim = Image(data=subimg,
                       inverr=tim.getInvError()[slc],
                       wcs=tim.wcs.shifted(x0, y0),
                       psf=subpsf,
                       photocal=tim.getPhotoCal(),
                       sky=tim.sky.shifted(x0, y0),
                       name=tim.name)
        sh,sw = subtim.shape
        subtim.subwcs = tim.subwcs.get_subimage(x0, y0, sw, sh)
        subtim.band = tim.band
        subtim.sig1 = tim.sig1
        subtim.modelMinval = tim.modelMinval
        subtim.x0 = x0
        subtim.y0 = y0
        subtim.meta = tim.meta
        subtims.append(subtim)
        #print('  ', tim.shape, 'to', subtim.shape)
    return subtims, modelMasks

class SourceModels(object):
    '''
    This class maintains a list of the model patches for a set of sources
    in a set of images.
    '''
    def __init__(self):
        self.filledModelMasks = True
    
    def save_images(self, tims):
        self.orig_images = [tim.getImage() for tim in tims]
        for tim,img in zip(tims, self.orig_images):
            tim.data = img.copy()

    def restore_images(self, tims):
        for tim,img in zip(tims, self.orig_images):
            tim.data = img

    def create(self, tims, srcs, subtract=False):
        '''
        Note that this modifies the *tims* if subtract=True.
        '''
        self.models = []
        for tim in tims:
            mods = []
            sh = tim.shape
            ie = tim.getInvError()
            for src in srcs:
                mod = src.getModelPatch(tim)
                if mod is not None and mod.patch is not None:
                    if not np.all(np.isfinite(mod.patch)):
                        print('Non-finite mod patch')
                        print('source:', src)
                        print('tim:', tim)
                        print('PSF:', tim.getPsf())
                    assert(np.all(np.isfinite(mod.patch)))
                    mod = _clip_model_to_blob(mod, sh, ie)
                    if subtract and mod is not None:
                        mod.addTo(tim.getImage(), scale=-1)
                mods.append(mod)
            self.models.append(mods)

    def add(self, i, tims):
        '''
        Adds the models for source *i* back into the tims.
        '''
        for tim,mods in zip(tims, self.models):
            mod = mods[i]
            if mod is not None:
                mod.addTo(tim.getImage())

    def update_and_subtract(self, i, src, tims):
        for tim,mods in zip(tims, self.models):
            #mod = srctractor.getModelPatch(tim, src)
            if src is None:
                mod = None
            else:
                mod = src.getModelPatch(tim)
            if mod is not None:
                mod.addTo(tim.getImage(), scale=-1)
            mods[i] = mod

    def model_masks(self, i, src):
        modelMasks = []
        for mods in self.models:
            d = dict()
            modelMasks.append(d)
            mod = mods[i]
            if mod is not None:
                if self.filledModelMasks:
                    mh,mw = mod.shape
                    d[src] = ModelMask(mod.x0, mod.y0, mw, mh)
                else:
                    d[src] = ModelMask(mod.x0, mod.y0, mod.patch != 0)
        return modelMasks

def remap_modelmask(modelMasks, oldsrc, newsrc):
    mm = []
    for mim in modelMasks:
        d = dict()
        mm.append(d)
        try:
            d[newsrc] = mim[oldsrc]
        except KeyError:
            pass
    return mm

def _clip_model_to_blob(mod, sh, ie):
    '''
    mod: Patch
    sh: tim shape
    ie: tim invError
    Returns: new Patch
    '''
    mslc,islc = mod.getSlices(sh)
    sy,sx = mslc
    patch = mod.patch[mslc] * (ie[islc]>0)
    if patch.shape == (0,0):
        return None
    mod = Patch(mod.x0 + sx.start, mod.y0 + sy.start, patch)

    # Check
    mh,mw = mod.shape
    assert(mod.x0 >= 0)
    assert(mod.y0 >= 0)
    ph,pw = sh
    assert(mod.x0 + mw <= pw)
    assert(mod.y0 + mh <= ph)

    return mod

def _select_model(chisqs, nparams, galaxy_margin, rex):
    '''
    Returns keepmod
    '''
    keepmod = 'none'

    # This is our "detection threshold": 5-sigma in
    # *parameter-penalized* units; ie, ~5.2-sigma for point sources
    cut = 5.**2
    # Take the best of all models computed
    diff = max([chisqs[name] - nparams[name] for name in chisqs.keys()
                if name != 'none'])

    if diff < cut:
        return keepmod
    # We're going to keep this source!

    if rex:
        simname = 'rex'
    else:
        simname = 'simple'

    if not simname in chisqs:
        # bright stars / reference stars: we don't test the simple model.
        return 'ptsrc'
    # Now choose between point source and simple model (SIMP/REX)
    if chisqs['ptsrc'] - nparams['ptsrc'] > chisqs[simname] - nparams[simname]:
        #print('Keeping source; PTSRC is better than SIMPLE')
        keepmod = 'ptsrc'
    else:
        #print('Keeping source; SIMPLE is better than PTSRC')
        #print('REX is better fit.  Radius', simplemod.shape.re)
        keepmod = simname
        # For REX, we also demand a fractionally better fit
        if simname == 'rex':
            dchisq_psf = chisqs['ptsrc']
            dchisq_rex = chisqs['rex']
            if (dchisq_rex - dchisq_psf) < (0.01 * dchisq_psf):
                keepmod = 'ptsrc'

    if not 'exp' in chisqs:
        return keepmod

    # This is our "upgrade" threshold: how much better a galaxy
    # fit has to be versus ptsrc, and comp versus galaxy.
    cut = galaxy_margin

    # This is the "fractional" upgrade threshold for ptsrc/simple->dev/exp:
    # 1% of ptsrc vs nothing
    cut = 0.01 * chisqs['ptsrc']
    
    #print('Cut: max of', cut, 'and', fcut, ' (fraction of chisq_psf=%.1f)'
    # % chisqs['ptsrc'])
    #cut = max(cut, fcut)

    expdiff = chisqs['exp'] - chisqs[keepmod]
    devdiff = chisqs['dev'] - chisqs[keepmod]
    
    #print('EXP vs', keepmod, ':', expdiff)
    #print('DEV vs', keepmod, ':', devdiff)

    if not (expdiff > cut or devdiff > cut):
        #print('Keeping', keepmod)
        return keepmod

    if expdiff > devdiff:
        #print('Upgrading from PTSRC to EXP: diff', expdiff)
        keepmod = 'exp'
    else:
        #print('Upgrading from PTSRC to DEV: diff', expdiff)
        keepmod = 'dev'

    if not 'comp' in chisqs:
        return keepmod

    diff = chisqs['comp'] - chisqs[keepmod]
    
    print('Got to exppsfdiff calculation') 
    exppsfdiff = chisqs['psfexp'] - chisqs[keepmod]
    if exppsfdiff > cut:
        keepmod = 'psfexp'
        print('Upgrading from composite to exppsfdiff')
        return keepmod


    print('Comparing', keepmod, 'to comp.  cut:', cut, 'comp:', diff)
    if diff > cut:
        keepmod = 'comp'
        return keepmod

    if not 'psfexp' in chisqs:
        return keepmod
        #print('Upgrading from dev/exp to composite.')
  
    return keepmod


def _chisq_improvement(src, chisqs, chisqs_none):
    '''
    chisqs, chisqs_none: dict of band->chisq
    '''
    bright = src.getBrightness()
    bands = chisqs.keys()
    fluxes = dict([(b, bright.getFlux(b)) for b in bands])
    dchisq = 0.
    for b in bands:
        flux = fluxes[b]
        if flux == 0:
            continue
        # this will be positive for an improved model
        d = chisqs_none[b] - chisqs[b]
        if flux > 0:
            dchisq += d
        else:
            dchisq -= np.abs(d)
    return dchisq

def _per_band_chisqs(tractor, bands):
    chisqs = dict([(b,0) for b in bands])
    for i,img in enumerate(tractor.images):
        chi = tractor.getChiImage(img=img)
        chisqs[img.band] = chisqs[img.band] + (chi ** 2).sum()
    return chisqs

def _limit_galaxy_stamp_size(src, tim, maxhalf=128):
    from tractor.galaxy import ProfileGalaxy
    if isinstance(src, ProfileGalaxy):
        px,py = tim.wcs.positionToPixel(src.getPosition())
        h = src._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
        if h > maxhalf:
            #print('halfsize', h, 'for', src, '-> setting to', maxhalf)
            src.halfsize = maxhalf

