from __future__ import print_function

import numpy as np
import time

from astrometry.util.ttime import Time
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import dimshow

from tractor import Tractor, PointSource, Image, Catalog, Patch
from tractor.galaxy import (DevGalaxy, ExpGalaxy,
                            disable_galaxy_cache, enable_galaxy_cache)
from tractor.patch import ModelMask
from tractor.sersic import SersicGalaxy

from legacypipe.survey import (RexGalaxy, GaiaSource,
                               LegacyEllipseWithPriors, LegacySersicIndex, get_rgb)
from legacypipe.bits import IN_BLOB
from legacypipe.coadds import quick_coadds
from legacypipe.runbrick_plots import _plot_mods

rgbkwargs_resid = dict(resids=True)

import logging
logger = logging.getLogger('legacypipe.oneblob')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

# singleton
cpu_arch = None
def get_cpu_arch():
    global cpu_arch
    import os
    if cpu_arch is not None:
        return cpu_arch
    family = None
    model = None
    modelname = None
    if os.path.exists('/proc/cpuinfo'):
        for line in open('/proc/cpuinfo').readlines():
            words = [w.strip() for w in line.strip().split(':')]
            if words[0] == 'cpu family' and family is None:
                family = int(words[1])
                #print('Set CPU family', family)
            if words[0] == 'model' and model is None:
                model = int(words[1])
                #print('Set CPU model', model)
            if words[0] == 'model name' and modelname is None:
                modelname = words[1]
                #print('CPU model', modelname)
    codenames = {
        # NERSC Cori machines
        (6, 63): 'has',
        (6, 87): 'knl',
    }
    cpu_arch = codenames.get((family, model), '')
    return cpu_arch

def one_blob(X):
    '''
    Fits sources contained within a "blob" of pixels.
    '''
    if X is None:
        return None
    (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
     srcs, bands, plots, ps, reoptimize, iterative, use_ceres, refmap,
     large_galaxies_force_pointsource) = X

    debug('Fitting blob number %i: blobid %i, nsources %i, size %i x %i, %i images' %
          (nblob, iblob, len(Isrcs), blobw, blobh, len(timargs)))

    if len(timargs) == 0:
        return None

    LegacySersicIndex.stepsize = 0.001
    
    if plots:
        import pylab as plt
        plt.figure(2, figsize=(3,3))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.figure(1)

    t0 = time.process_time()
    # A local WCS for this blob
    blobwcs = brickwcs.get_subimage(bx0, by0, blobw, blobh)

    # Per-source measurements for this blob
    B = fits_table()
    B.sources = srcs
    B.Isrcs = Isrcs
    B.iblob = iblob
    B.blob_x0 = np.zeros(len(B), np.int16) + bx0
    B.blob_y0 = np.zeros(len(B), np.int16) + by0
    # Did sources start within the blob?
    ok,x0,y0 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in srcs]),
        np.array([src.getPosition().dec for src in srcs]))
    safe_x0 = np.clip(np.round(x0-1).astype(int), 0,blobw-1)
    safe_y0 = np.clip(np.round(y0-1).astype(int), 0,blobh-1)
    B.init_x = safe_x0
    B.init_y = safe_y0
    B.started_in_blob = blobmask[safe_y0, safe_x0]
    # This uses 'initial' pixel positions, because that's what determines
    # the fitting behaviors.
    B.brightblob = refmap[safe_y0, safe_x0].astype(np.int16)
    B.cpu_source = np.zeros(len(B), np.float32)
    B.blob_width  = np.zeros(len(B), np.int16) + blobw
    B.blob_height = np.zeros(len(B), np.int16) + blobh
    B.blob_npix   = np.zeros(len(B), np.int32) + np.sum(blobmask)
    B.blob_nimages= np.zeros(len(B), np.int16) + len(timargs)
    B.blob_symm_width   = np.zeros(len(B), np.int16)
    B.blob_symm_height  = np.zeros(len(B), np.int16)
    B.blob_symm_npix    = np.zeros(len(B), np.int32)
    B.blob_symm_nimages = np.zeros(len(B), np.int16)
    B.hit_limit = np.zeros(len(B), bool)

    ob = OneBlob('%i'%(nblob+1), blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, use_ceres, refmap,
                 large_galaxies_force_pointsource)
    B = ob.run(B, reoptimize=reoptimize, iterative_detection=iterative)

    B.blob_totalpix = np.zeros(len(B), np.int32) + ob.total_pix

    ok,x1,y1 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in B.sources]),
        np.array([src.getPosition().dec for src in B.sources]))
    B.finished_in_blob = blobmask[
        np.clip(np.round(y1-1).astype(int), 0, blobh-1),
        np.clip(np.round(x1-1).astype(int), 0, blobw-1)]
    assert(len(B.finished_in_blob) == len(B))
    assert(len(B.finished_in_blob) == len(B.started_in_blob))

    # Setting values here (after .run() has completed) means that iterative sources
    # (which get merged with the original table B) get values also.
    B.cpu_arch = np.zeros(len(B), dtype='U3')
    B.cpu_arch[:] = get_cpu_arch()
    B.cpu_blob = np.empty(len(B), np.float32)
    t1 = time.process_time()
    B.cpu_blob[:] = t1 - t0
    B.blob = np.empty(len(B), np.int32)
    B.blob[:] = iblob
    return B

class OneBlob(object):
    def __init__(self, name, blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, use_ceres, refmap,
                 large_galaxies_force_pointsource):
        self.name = name
        self.blobwcs = blobwcs
        self.pixscale = self.blobwcs.pixel_scale()
        self.blobmask = blobmask
        self.srcs = srcs
        self.bands = bands
        self.plots = plots
        self.refmap = refmap
        #self.plots_per_source = False
        self.plots_per_source = plots
        self.plots_per_model = False
        # blob-1-data.png, etc
        self.plots_single = False
        self.ps = ps
        self.use_ceres = use_ceres
        self.deblend = False
        self.large_galaxies_force_pointsource = large_galaxies_force_pointsource
        self.tims = self.create_tims(timargs)
        self.total_pix = sum([np.sum(t.getInvError() > 0) for t in self.tims])
        self.plots2 = False
        alphas = [0.1, 0.3, 1.0]
        self.optargs = dict(priors=True, shared_params=False, alphas=alphas,
                            print_progress=True)
        self.blobh,self.blobw = blobmask.shape
        self.bigblob = (self.blobw * self.blobh) > 100*100
        if self.bigblob:
            debug('Big blob:', name)
        self.trargs = dict()

        # if use_ceres:
        #     from tractor.ceres_optimizer import CeresOptimizer
        #     ceres_optimizer = CeresOptimizer()
        #     self.optargs.update(scale_columns=False,
        #                         scaled=False,
        #                         dynamic_scale=False)
        #     self.trargs.update(optimizer=ceres_optimizer)
        # else:
        #     self.optargs.update(dchisq = 0.1)

        #from tractor.constrained_optimizer import ConstrainedOptimizer
        #self.trargs.update(optimizer=ConstrainedOptimizer())
        from tractor.dense_optimizer import ConstrainedDenseOptimizer
        self.trargs.update(optimizer=ConstrainedDenseOptimizer())

        self.optargs.update(dchisq = 0.1)

    def run(self, B, reoptimize=False, iterative_detection=True,
            compute_metrics=True):
        trun = tlast = Time()
        # Not quite so many plots...
        self.plots1 = self.plots
        cat = Catalog(*self.srcs)

        if self.plots:
            import pylab as plt
            self._initial_plots()
            from legacypipe.detection import plot_boundary_map
            plt.clf()
            dimshow(self.rgb)
            ax = plt.axis()
            bitset = ((self.refmap & IN_BLOB['MEDIUM']) != 0)
            plot_boundary_map(bitset, rgb=(255,0,0), iterations=2)
            bitset = ((self.refmap & IN_BLOB['BRIGHT']) != 0)
            plot_boundary_map(bitset, rgb=(200,200,0), iterations=2)
            bitset = ((self.refmap & IN_BLOB['GALAXY']) != 0)
            plot_boundary_map(bitset, rgb=(0,255,0), iterations=2)
            plt.axis(ax)
            plt.title('Reference-source Masks')
            self.ps.savefig()

        if not self.bigblob:
            debug('Fitting just fluxes using initial models...')
            self._fit_fluxes(cat, self.tims, self.bands)
        tr = self.tractor(self.tims, cat)

        if self.plots:
            self._plots(tr, 'Initial models')

        # Optimize individual sources, in order of flux.
        # First, choose the ordering...
        Ibright = _argsort_by_brightness(cat, self.bands, ref_first=True)

        # The sizes of the model patches fit here are determined by the
        # sources themselves, ie by the size of the mod patch returned by
        #  src.getModelPatch(tim)
        if len(cat) > 1:
            self._optimize_individual_sources_subtract(
                cat, Ibright, B.cpu_source)
        else:
            self._optimize_individual_sources(tr, cat, Ibright, B.cpu_source)

        # Optimize all at once?
        if len(cat) > 1 and len(cat) <= 10:
            cat.thawAllParams()
            for i,src in enumerate(cat):
                if getattr(src, 'freezeparams', False):
                    debug('Frozen source', src, '-- keeping as-is!')
                cat.freezeParam(i)
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
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs,self.bands), ticks=False)
                plt.savefig('blob-%s-initmodel.png' % (self.name))
                res = [(tim.getImage() - mod) for tim,mod in zip(self.tims, mods)]
                coresids,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=res)
                dimshow(get_rgb(coresids, self.bands, resids=True), ticks=False)
                plt.savefig('blob-%s-initresid.png' % (self.name))
                dimshow(get_rgb(coresids, self.bands), ticks=False)
                plt.savefig('blob-%s-initsub.png' % (self.name))
                plt.figure(1)

        debug('Blob', self.name, 'finished initial fitting:', Time()-tlast)
        tlast = Time()

        self.compute_segmentation_map()

        # Next, model selections: point source vs dev/exp vs composite.
        B = self.run_model_selection(cat, Ibright, B,
                                     iterative_detection=iterative_detection)

        debug('Blob', self.name, 'finished model selection:', Time()-tlast)
        tlast = Time()

        # Cut down to just the kept sources
        cat = B.sources
        I = np.array([i for i,s in enumerate(cat) if s is not None])
        B.cut(I)
        cat = Catalog(*B.sources)
        tr.catalog = cat

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
            dimshow(get_rgb(coresids, self.bands, resids=True), ticks=False)
            plt.savefig('blob-%s-resid.png' % (self.name))
            plt.figure(1)

        # Do another quick round of flux-only fitting?
        # This does horribly -- fluffy galaxies go out of control because
        # they're only constrained by pixels within this blob.
        #_fit_fluxes(cat, tims, bands, use_ceres, alphas)

        # A final optimization round?
        if reoptimize:
            if self.plots:
                import pylab as plt
                modimgs = list(tr.getModelImages())
                co,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                    images=modimgs)
                plt.clf()
                dimshow(get_rgb(co, self.bands))
                plt.title('Before final opt')
                self.ps.savefig()

            Ibright = _argsort_by_brightness(cat, self.bands, ref_first=True)
            if len(cat) > 1:
                self._optimize_individual_sources_subtract(
                    cat, Ibright, B.cpu_source)
            else:
                self._optimize_individual_sources(tr, cat, Ibright, B.cpu_source)

            if self.plots:
                import pylab as plt
                modimgs = list(tr.getModelImages())
                co,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                    images=modimgs)
                plt.clf()
                dimshow(get_rgb(co, self.bands))
                plt.title('After final opt')
                self.ps.savefig()

        if compute_metrics:
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
                # Compute inverse-variances
                allderivs = tr.getDerivs()
                ivars = _compute_invvars(allderivs)
                assert(len(ivars) == nsrcparams)
                B.srcinvvars[isub] = ivars
                assert(len(B.srcinvvars[isub]) == cat[isub].numberOfParams())
                cat.freezeParam(isub)

            # Check for sources with zero inverse-variance -- I think these
            # can be generated during the "Simultaneous re-opt" stage above --
            # sources can get scattered outside the blob.
            I, = np.nonzero([np.sum(iv) > 0 for iv in B.srcinvvars])
            if len(I) < len(B):
                debug('Keeping', len(I), 'of', len(B),'sources with non-zero ivar')
                B.cut(I)
                cat = Catalog(*B.sources)
                tr.catalog = cat

            M = _compute_source_metrics(B.sources, self.tims, self.bands, tr)
            for k,v in M.items():
                B.set(k, v)

        info('Blob', self.name, 'finished, total:', Time()-trun)
        return B

    def compute_segmentation_map(self):
        # Use ~ saddle criterion to segment the blob / mask other sources
        from functools import reduce
        from legacypipe.detection import detection_maps
        from astrometry.util.multiproc import multiproc
        from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
        from scipy.ndimage.measurements import label

        # Compute per-band detection maps
        mp = multiproc()
        detmaps,detivs,satmaps = detection_maps(
            self.tims, self.blobwcs, self.bands, mp)

        # same as in runbrick.py
        saturated_pix = reduce(np.logical_or,
                               [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps])
        del satmaps

        maxsn = 0
        for i,(detmap,detiv) in enumerate(zip(detmaps,detivs)):
            sn = detmap * np.sqrt(detiv)

            if self.plots:
                import pylab as plt
                plt.clf()
                plt.subplot(2,2,1)
                plt.imshow(detmap, interpolation='nearest', origin='lower')
                plt.title('detmap %s' % self.bands[i])
                plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(detiv, interpolation='nearest', origin='lower')
                plt.title('detiv %s' % self.bands[i])
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(sn, interpolation='nearest', origin='lower')
                plt.title('detsn %s' % self.bands[i])
                plt.colorbar()
                self.ps.savefig()

            # HACK - no SEDs...
            maxsn = np.maximum(maxsn, sn)

        segmap = np.empty((self.blobh, self.blobw), int)
        segmap[:,:] = -1

        _,ix,iy = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in self.srcs]),
            np.array([src.getPosition().dec for src in self.srcs]))
        ix = np.clip(np.round(ix)-1, 0, self.blobw-1).astype(int)
        iy = np.clip(np.round(iy)-1, 0, self.blobh-1).astype(int)

        # Do not compute segmentation map for sources in the CLUSTER mask
        Iseg, = np.nonzero((self.refmap[iy, ix] & IN_BLOB['CLUSTER']) == 0)
        # Zero out the S/N in CLUSTER mask
        maxsn[(self.refmap & IN_BLOB['CLUSTER']) > 0] = 0.

        # (also zero out the satmap)
        saturated_pix[(self.refmap & IN_BLOB['CLUSTER']) > 0] = False

        Ibright = _argsort_by_brightness([self.srcs[i] for i in Iseg], self.bands)
        rank = np.empty(len(Iseg), int)
        rank[Ibright] = np.arange(len(Iseg), dtype=int)
        rankmap = dict([(Iseg[i],r) for r,i in enumerate(Ibright)])
        del Ibright

        todo = set(Iseg)
        thresholds = list(range(3, int(np.ceil(maxsn.max()))))
        for thresh in thresholds:
            #print('S/N', thresh, ':', len(todo), 'sources to find still')
            if len(todo) == 0:
                break
            ####
            hot = np.logical_or(maxsn >= thresh, saturated_pix)
            hot = binary_fill_holes(hot)
            blobs,_ = label(hot)
            srcblobs = blobs[iy[Iseg], ix[Iseg]]
            done = set()

            blobranks = {}
            for i,(b,r) in enumerate(zip(srcblobs, rank)):
                if not b in blobranks:
                    blobranks[b] = []
                blobranks[b].append(r)

            for t in todo:
                bl = blobs[iy[t], ix[t]]
                if bl == 0:
                    # ??
                    done.add(t)
                    continue
                if rankmap[t] == min(blobranks[bl]):
                    #print('Source', t, 'has rank', rank[t], 'vs blob ranks', blobranks[bl])
                    segmap[blobs == bl] = t
                    #print('Source', t, 'is isolated at S/N', thresh)
                    done.add(t)
            todo.difference_update(done)

        self.segmap = segmap

        if self.plots:
            import pylab as plt
            plt.clf()
            dimshow(segmap)
            ax = plt.axis()
            from legacypipe.detection import plot_boundary_map
            plot_boundary_map(segmap >= 0)
            plt.plot(ix, iy, 'r.')
            plt.axis(ax)
            plt.title('Segmentation map')
            self.ps.savefig()

            plt.clf()
            dimshow(self.rgb)
            ax = plt.axis()
            for i in range(len(self.srcs)):
                plot_boundary_map(segmap == i)
            plt.plot(ix, iy, 'r.')
            plt.axis(ax)
            plt.title('Segments')
            self.ps.savefig()


    def run_model_selection(self, cat, Ibright, B, iterative_detection=True):
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

        models = SourceModels()
        # Remember original tim images
        models.save_images(self.tims)

        # Create initial models for each tim x each source
        models.create(self.tims, cat, subtract=True)

        N = len(cat)
        B.dchisq = np.zeros((N, 5), np.float32)
        B.all_models    = np.array([{} for i in range(N)])
        B.all_model_ivs = np.array([{} for i in range(N)])
        B.all_model_cpu = np.array([{} for i in range(N)])
        B.all_model_hit_limit = np.array([{} for i in range(N)])
        B.all_model_opt_steps = np.array([{} for i in range(N)])

        # Model selection for sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
            src = cat[srci]
            debug('Model selection for source %i of %i in blob %s; sourcei %i' %
                  (numi+1, len(Ibright), self.name, srci))
            cpu0 = time.process_time()

            if getattr(src, 'freezeparams', False):
                info('Frozen source', src, '-- keeping as-is!')
                B.sources[srci] = src
                continue
            
            # Add this source's initial model back in.
            models.add(srci, self.tims)

            if self.plots_single:
                import pylab as plt
                plt.figure(2)
                coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                           fill_holes=False)
                rgb = get_rgb(coimgs,self.bands)
                plt.imsave('blob-%s-%s-bdata.png' % (self.name, srci), rgb,
                           origin='lower')
                plt.figure(1)

            # Model selection for this source.
            keepsrc = self.model_selection_one_source(src, srci, models, B)
            B.sources[srci] = keepsrc
            cat[srci] = keepsrc

            models.update_and_subtract(srci, keepsrc, self.tims)

            if self.plots_single:
                plt.figure(2)
                tr = self.tractor(self.tims, cat)
                coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs,self.bands), ticks=False)
                plt.savefig('blob-%s-%i-sub.png' % (self.name, srci))
                plt.figure(1)

            cpu1 = time.process_time()
            B.cpu_source[srci] += (cpu1 - cpu0)

        # At this point, we have subtracted our best model fits for each source
        # to be kept; the tims contain residual images.

        if iterative_detection:

            if self.plots and False:
                # One plot per tim is a little much, even for me...
                import pylab as plt
                for tim in self.tims:
                    plt.clf()
                    plt.suptitle('Iterative detection: %s' % tim.name)
                    plt.subplot(2,2,1)
                    plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
                               vmin=-5.*tim.sig1, vmax=10.*tim.sig1)
                    plt.title('image')
                    plt.subplot(2,2,2)
                    plt.imshow(tim.getImage(), interpolation='nearest', origin='lower')
                    plt.title('image')
                    plt.colorbar()
                    plt.subplot(2,2,3)
                    plt.imshow(tim.getInvError(), interpolation='nearest', origin='lower')
                    plt.title('inverr')
                    plt.colorbar()
                    plt.subplot(2,2,4)
                    plt.imshow(tim.getImage() * (tim.getInvError() > 0), interpolation='nearest', origin='lower')
                    plt.title('image*(inverr>0)')
                    plt.colorbar()
                    self.ps.savefig()

            Bnew = self.iterative_detection(B, models)
            if Bnew is not None:
                from astrometry.util.fits import merge_tables
                # B.sources is a list of objects... merge() with
                # fillzero doesn't handle them well.
                srcs = B.sources
                newsrcs = Bnew.sources
                B.delete_column('sources')
                Bnew.delete_column('sources')
                # also scalars don't work well
                iblob = B.iblob
                B.delete_column('iblob')
                B = merge_tables([B, Bnew], columns='fillzero')
                B.sources = srcs + newsrcs
                B.iblob = iblob

        models.restore_images(self.tims)
        del models
        return B

    def iterative_detection(self, Bold, models):
        # Compute per-band detection maps
        from legacypipe.detection import sed_matched_filters, detection_maps, run_sed_matched_filters
        from astrometry.util.multiproc import multiproc

        if self.plots:
            coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                       fill_holes=False)
            import pylab as plt
            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.title('Iterative detection: residuals')
            self.ps.savefig()

        mp = multiproc()
        detmaps,detivs,satmaps = detection_maps(
            self.tims, self.blobwcs, self.bands, mp)

        # from runbrick.py
        from scipy.ndimage.morphology import binary_dilation
        satmaps = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]

        # Also compute detection maps on the (first-round) model images!
        # save tim.images (= residuals at this point)
        realimages = [tim.getImage() for tim in self.tims]
        for tim,mods in zip(self.tims, models.models):
            modimg = np.zeros_like(tim.getImage())
            for mod in mods:
                if mod is None:
                    continue
                mod.addTo(modimg)
            tim.data = modimg
        if self.plots:
            coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                       fill_holes=False)
            import pylab as plt
            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.title('Iterative detection: first-round models')
            self.ps.savefig()

        mod_detmaps,mod_detivs,_ = detection_maps(
            self.tims, self.blobwcs, self.bands, mp)
        # revert
        for tim,img in zip(self.tims, realimages):
            tim.data = img

        if self.plots:
            import pylab as plt
            plt.clf()
            dimshow(get_rgb(detmaps,self.bands), ticks=False)
            plt.title('Iterative detection: detection maps')
            self.ps.savefig()
            plt.clf()
            dimshow(get_rgb(mod_detmaps,self.bands), ticks=False)
            plt.title('Iterative detection: model detection maps')
            self.ps.savefig()

        # if self.plots:
        #     import pylab as plt
        #     plt.clf()
        #     for det,div,b in zip(detmaps, detivs, self.bands):
        #         plt.hist((det * np.sqrt(div)).ravel(), range=(-5,10),
        #                  bins=50, histtype='step', color=dict(z='m').get(b, b))
        #     plt.title('Detection pixel S/N')
        #     self.ps.savefig()

        detlogger = logging.getLogger('legacypipe.detection')
        detloglvl = detlogger.getEffectiveLevel()
        detlogger.setLevel(detloglvl + 10)

        SEDs = sed_matched_filters(self.bands)

        # Avoid re-detecting sources at positions close to initial
        # source positions (including ones that will get cut!)
        avoid_x = Bold.init_x
        avoid_y = Bold.init_y
        avoid_r = np.zeros(len(avoid_x), np.float32) + 2.
        nsigma = 6.

        Tnew,newcat,_ = run_sed_matched_filters(
            SEDs, self.bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r),
            self.blobwcs, nsigma=nsigma, saturated_pix=satmaps, veto_map=None,
            plots=False, ps=None, mp=mp)
            #plots=self.plots, ps=self.ps, mp=mp)

        detlogger.setLevel(detloglvl)

        if Tnew is None:
            debug('No iterative sources detected!')
            return None

        debug('Found', len(Tnew), 'new sources')
        Tnew.cut(self.refmap[Tnew.iby, Tnew.ibx] == 0)
        debug('Cut to', len(Tnew), 'on refmap')
        if len(Tnew) == 0:
            return None

        detsns = np.dstack([m*np.sqrt(iv) for m,iv in zip(detmaps, detivs)])
        modsns = np.dstack([m*np.sqrt(iv) for m,iv in zip(mod_detmaps, mod_detivs)])

        det_max = np.max(detsns[Tnew.iby, Tnew.ibx, :], axis=1)
        mod_max = np.max(modsns[Tnew.iby, Tnew.ibx, :], axis=1)
        det_sum = np.sum(detsns[Tnew.iby, Tnew.ibx, :], axis=1)
        mod_sum = np.sum(modsns[Tnew.iby, Tnew.ibx, :], axis=1)
        del detsns, modsns

        if self.plots:
            coimgs,cons = quick_coadds(self.tims, self.bands, self.blobwcs,
                                       fill_holes=False)
            import pylab as plt
            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            ax = plt.axis()
            crossa = dict(ms=10, mew=1.5)
            rr = np.array([s.getPosition().ra for s in Bold.sources
                           if s is not None])
            dd = np.array([s.getPosition().dec for s in Bold.sources
                           if s is not None])
            _,xx,yy = self.blobwcs.radec2pixelxy(rr, dd)
            plt.plot(xx-1, yy-1, 'r+', label='Old', **crossa)
            plt.plot(Tnew.ibx, Tnew.iby, '+', color=(0,1,0), label='New',
                     **crossa)
            plt.axis(ax)
            plt.legend()
            plt.title('Iterative detections')
            self.ps.savefig()

            plt.clf()
            plt.loglog(mod_max, det_max, 'k.')
            ax = plt.axis()
            plt.plot([1e-3, 1e6], [1e-3, 1e6], 'b--', lw=3, alpha=0.3)
            plt.axis(ax)
            plt.xlabel('Model detection S/N: max')
            plt.ylabel('Iterative detection S/N: max')
            self.ps.savefig()

            plt.clf()
            plt.loglog(mod_sum, det_sum, 'k.')
            ax = plt.axis()
            plt.plot([1e-3, 1e6], [1e-3, 1e6], 'b--', lw=3, alpha=0.3)
            plt.axis(ax)
            plt.xlabel('Model detection S/N: sum')
            plt.ylabel('Iterative detection S/N: sum')
            self.ps.savefig()

            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            ax = plt.axis()
            crossa = dict(ms=10, mew=1.5)
            plt.plot(xx-1, yy-1, 'r+', label='Old', **crossa)
            plt.plot(Tnew.ibx, Tnew.iby, '+', color=(0,1,0), label='New',
                     **crossa)
            for x,y,r1,r2 in zip(Tnew.ibx, Tnew.iby, det_max/np.maximum(mod_max, 1.), det_sum/np.maximum(mod_sum, len(self.bands))):
                plt.text(x, y, '%.1f, %.1f' % (r1,r2),
                         color='k', fontsize=10,
                         bbox=dict(facecolor='w', alpha=0.5))
            plt.axis(ax)
            plt.legend()
            plt.title('Iterative detections')
            self.ps.savefig()

        B = 0.2
        Tnew.cut(det_max > B * np.maximum(mod_max, 1.))
        debug('Cut to', len(Tnew), 'iterative sources compared to model detection map')
        if len(Tnew) == 0:
            return None

        info('Measuring', len(Tnew), 'iterative sources')

        from tractor import NanoMaggies, RaDecPos
        newsrcs = [PointSource(RaDecPos(t.ra, t.dec),
                               NanoMaggies(**dict([(b,1) for b in self.bands])))
                               for t in Tnew]
        # Save
        oldsrcs = self.srcs
        self.srcs = newsrcs
        
        Bnew = fits_table()
        Bnew.sources = newsrcs
        Bnew.Isrcs = np.array([-1]*len(Bnew))
        Bnew.cpu_source = np.zeros(len(Bnew), np.float32)
        Bnew.blob_symm_nimages = np.zeros(len(Bnew), np.int16)
        Bnew.blob_symm_npix    = np.zeros(len(Bnew), np.int32)
        Bnew.blob_symm_width   = np.zeros(len(Bnew), np.int16)
        Bnew.blob_symm_height  = np.zeros(len(Bnew), np.int16)
        Bnew.hit_limit = np.zeros(len(Bnew), bool)
        Bnew.brightblob = self.refmap[Tnew.iby, Tnew.ibx].astype(np.int16)
        # Be quieter during iterative detection!
        bloblogger = logging.getLogger('legacypipe.oneblob')
        loglvl = bloblogger.getEffectiveLevel()
        bloblogger.setLevel(loglvl + 10)

        # Run the whole oneblob pipeline on the iterative sources!
        Bnew = self.run(Bnew, iterative_detection=False, compute_metrics=False)

        bloblogger.setLevel(loglvl)

        # revert
        self.srcs = oldsrcs

        if len(Bnew) == 0:
            return None

        return Bnew

    def model_selection_one_source(self, src, srci, models, B):

        if self.bigblob:
            mods = [mod[srci] for mod in models.models]
            srctims,modelMasks = _get_subimages(self.tims, mods, src)

            # Create a little local WCS subregion for this source, by
            # resampling non-zero inverrs from the srctims into blobwcs
            insrc = np.zeros((self.blobh,self.blobw), bool)
            for tim in srctims:
                try:
                    Yo,Xo,Yi,Xi,nil = resample_with_wcs(
                        self.blobwcs, tim.subwcs, intType=np.int16)
                except:
                    continue
                insrc[Yo,Xo] |= (tim.inverr[Yi,Xi] > 0)

            if np.sum(insrc) == 0:
                # No source pixels touching blob... this can
                # happen when a source scatters outside the blob
                # in the fitting stage.  Drop the source here.
                return None

            yin = np.max(insrc, axis=1)
            xin = np.max(insrc, axis=0)
            yl,yh = np.flatnonzero(yin)[np.array([0,-1])]
            xl,xh = np.flatnonzero(xin)[np.array([0,-1])]
            del insrc

            srcwcs = self.blobwcs.get_subimage(xl, yl, 1+xh-xl, 1+yh-yl)
            srcwcs_x0y0 = (xl, yl)
            # A mask for which pixels in the 'srcwcs' square are occupied.
            srcblobmask = self.blobmask[yl:yh+1, xl:xh+1]
        else:
            modelMasks = models.model_masks(srci, src)
            srctims = self.tims
            srcwcs = self.blobwcs
            srcwcs_x0y0 = (0, 0)
            srcblobmask = self.blobmask

        if self.plots_per_source:
            # This is a handy blob-coordinates plot of the data
            # going into the fit.
            import pylab as plt
            plt.clf()
            nil,nil,coimgs,nil = quick_coadds(srctims, self.bands,self.blobwcs,
                                              fill_holes=False, get_cow=True)
            dimshow(get_rgb(coimgs, self.bands))
            ax = plt.axis()
            pos = src.getPosition()
            ok,x,y = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
            ix,iy = int(np.round(x-1)), int(np.round(y-1))
            plt.plot(x-1, y-1, 'r+')
            plt.axis(ax)
            plt.title('Model selection: data')
            self.ps.savefig()

        # Mask out other sources while fitting this one, by
        # finding symmetrized blobs of significant pixels
        mask_others = True
        if mask_others:
            from legacypipe.detection import detection_maps
            from astrometry.util.multiproc import multiproc
            from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
            from scipy.ndimage.measurements import label
            # Compute per-band detection maps
            mp = multiproc()
            detmaps,detivs,_ = detection_maps(
                srctims, srcwcs, self.bands, mp)
            # Compute the symmetric area that fits in this 'tim'
            pos = src.getPosition()
            ok,xx,yy = srcwcs.radec2pixelxy(pos.ra, pos.dec)
            bh,bw = srcblobmask.shape
            ix = int(np.clip(np.round(xx-1), 0, bw-1))
            iy = int(np.clip(np.round(yy-1), 0, bh-1))
            flipw = min(ix, bw-1-ix)
            fliph = min(iy, bh-1-iy)
            flipblobs = np.zeros(srcblobmask.shape, bool)
            # The slice where we can perform symmetrization
            slc = (slice(iy-fliph, iy+fliph+1),
                   slice(ix-flipw, ix+flipw+1))
            # Go through the per-band detection maps, marking significant pixels
            for i,(detmap,detiv) in enumerate(zip(detmaps,detivs)):
                sn = detmap * np.sqrt(detiv)
                flipsn = np.zeros_like(sn)
                # Symmetrize
                flipsn[slc] = np.minimum(sn[slc],
                                         np.flipud(np.fliplr(sn[slc])))
                # just OR the detection maps per-band...
                flipblobs |= (flipsn > 5.)

            flipblobs = binary_fill_holes(flipblobs)
            blobs,_ = label(flipblobs)
            goodblob = blobs[iy,ix]

            if self.plots_per_source and True:
                # This plot is about the symmetric-blob definitions
                # when fitting sources.
                import pylab as plt
                from legacypipe.detection import plot_boundary_map

                plt.clf()
                for i,(band,detmap,detiv) in enumerate(zip(self.bands, detmaps, detivs)):
                    if i >= 4:
                        break
                    detsn = detmap * np.sqrt(detiv)
                    plt.subplot(2,2, i+1)
                    mx = detsn.max()
                    dimshow(detsn, vmin=-2, vmax=max(8, mx))
                    ax = plt.axis()
                    plot_boundary_map(detsn >= 5.)
                    plt.plot(ix, iy, 'rx')
                    plt.plot([ix-flipw, ix-flipw, ix+flipw, ix+flipw, ix-flipw],
                             [iy-fliph, iy+fliph, iy+fliph, iy-fliph, iy-fliph], 'r-')
                    plt.axis(ax)
                    plt.title('det S/N: ' + band)
                plt.subplot(2,2,4)
                dimshow(flipblobs, vmin=0, vmax=1)
                plt.colorbar()
                ax = plt.axis()
                plot_boundary_map(blobs == goodblob)
                if binary_fill_holes(flipblobs)[iy,ix]:
                    fb = (blobs == goodblob)
                    di = binary_dilation(fb, iterations=4)
                    if np.any(di):
                        plot_boundary_map(di, rgb=(255,0,0))
                plt.plot(ix, iy, 'rx')
                plt.plot([ix-flipw, ix-flipw, ix+flipw, ix+flipw, ix-flipw],
                         [iy-fliph, iy+fliph, iy+fliph, iy-fliph, iy-fliph], 'r-')
                plt.axis(ax)
                plt.title('good blob')
                self.ps.savefig()

                plt.clf()
                plt.subplot(2,2,1)
                dimshow(blobs)
                plt.colorbar()
                plt.title('blob map; goodblob=%i' % goodblob)
                plt.subplot(2,2,2)
                dimshow(flipblobs, vmin=0, vmax=1)
                plt.colorbar()
                plt.title('symmetric blob mask: 1 = good; red=symm')
                ax = plt.axis()
                plt.plot(ix, iy, 'rx')
                plt.plot([ix-flipw-0.5, ix-flipw-0.5, ix+flipw+0.5, ix+flipw+0.5, ix-flipw-0.5],
                         [iy-fliph-0.5, iy+fliph+0.5, iy+fliph+0.5, iy-fliph-0.5, iy-fliph-0.5], 'r-')
                plt.axis(ax)

                plt.subplot(2,2,3)
                dh,dw = flipblobs.shape
                sx0,sy0 = srcwcs_x0y0
                dimshow(self.segmap[sy0:sy0+dh, sx0:sx0+dw])
                plt.title('Segmentation map')

                plt.subplot(2,2,4)
                dilated = binary_dilation(flipblobs, iterations=4)
                s = self.segmap[iy + sy0, ix + sx0]
                if s != -1:
                    dilated *= (self.segmap[sy0:sy0+dh, sx0:sx0+dw] == s)
                dimshow(dilated)
                if s != -1:
                    plt.title('Dilated goodblob * Segmentation map')
                else:
                    plt.title('Dilated goodblob (no Segmentation map)')

                self.ps.savefig()

            # If there is no longer a source detected at the original source
            # position, we want to drop this source.  However, saturation can
            # cause there to be no detection S/N because of masking, so do
            # a hole-fill before checking.
            if not flipblobs[iy,ix]:
                # The hole-fill can still fail (eg, in small test images) if
                # the bleed trail splits the blob into two pieces.
                # Skip this test for reference sources.
                if is_reference_source(src):
                    debug('Reference source center is outside symmetric blob; keeping')
                else:
                    debug('Source center is not in the symmetric blob mask; skipping')
                    return None

            if goodblob != 0:
                flipblobs = (blobs == goodblob)

            dilated = binary_dilation(flipblobs, iterations=4)
            if not np.any(dilated):
                debug('No pixels in dilated symmetric mask')
                return None

            dh,dw = flipblobs.shape
            sx0,sy0 = srcwcs_x0y0
            s = self.segmap[iy + sy0, ix + sx0]
            if s != -1:
                dilated *= (self.segmap[sy0:sy0+dh, sx0:sx0+dw] == s)

            if not np.any(dilated):
                debug('No pixels in segmented dilated symmetric mask')
                return None

            yin = np.max(dilated, axis=1)
            xin = np.max(dilated, axis=0)
            yl,yh = np.flatnonzero(yin)[np.array([0,-1])]
            xl,xh = np.flatnonzero(xin)[np.array([0,-1])]
            (oldx0,oldy0) = srcwcs_x0y0
            srcwcs = srcwcs.get_subimage(xl, yl, 1+xh-xl, 1+yh-yl)
            srcwcs_x0y0 = (oldx0 + xl, oldy0 + yl)
            srcblobmask = srcblobmask[yl:yh+1, xl:xh+1]
            dilated = dilated[yl:yh+1, xl:xh+1]
            flipblobs = flipblobs[yl:yh+1, xl:xh+1]

            saved_srctim_ies = []
            keep_srctims = []
            mm = []
            totalpix = 0
            for tim in srctims:
                # Zero out inverse-errors for all pixels outside
                # 'dilated'.
                try:
                    Yo,Xo,Yi,Xi,nil = resample_with_wcs(
                        tim.subwcs, srcwcs, intType=np.int16)
                except OverlapError:
                    continue
                ie = tim.getInvError()
                newie = np.zeros_like(ie)

                good, = np.nonzero(dilated[Yi,Xi] * (ie[Yo,Xo] > 0))
                if len(good) == 0:
                    debug('Tim has inverr all == 0')
                    continue
                yy = Yo[good]
                xx = Xo[good]
                newie[yy,xx] = ie[yy,xx]
                xl,xh = xx.min(), xx.max()
                yl,yh = yy.min(), yy.max()
                totalpix += len(xx)

                d = { src: ModelMask(xl, yl, 1+xh-xl, 1+yh-yl) }
                mm.append(d)

                saved_srctim_ies.append(ie)
                tim.inverr = newie
                keep_srctims.append(tim)

            srctims = keep_srctims
            modelMasks = mm

            B.blob_symm_nimages[srci] = len(srctims)
            B.blob_symm_npix[srci] = totalpix
            sh,sw = srcwcs.shape
            B.blob_symm_width [srci] = sw
            B.blob_symm_height[srci] = sh

            # if self.plots_per_source:
            #     from legacypipe.detection import plot_boundary_map
            #     plt.clf()
            #     dimshow(get_rgb(coimgs, self.bands))
            #     ax = plt.axis()
            #     plt.plot(x-1, y-1, 'r+')
            #     plt.axis(ax)
            #     sx0,sy0 = srcwcs_x0y0
            #     sh,sw = srcwcs.shape
            #     ext = [sx0, sx0+sw, sy0, sy0+sh]
            #     plot_boundary_map(flipblobs, rgb=(255,255,255), extent=ext)
            #     plot_boundary_map(dilated, rgb=(0,255,0), extent=ext)
            #     plt.title('symmetrized blobs')
            #     self.ps.savefig()
            #     nil,nil,coimgs,nil = quick_coadds(
            #         srctims, self.bands, self.blobwcs,
            #         fill_holes=False, get_cow=True)
            #     dimshow(get_rgb(coimgs, self.bands))
            #     ax = plt.axis()
            #     plt.plot(x-1, y-1, 'r+')
            #     plt.axis(ax)
            #     plt.title('Symmetric-blob masked')
            #     self.ps.savefig()
            #     plt.clf()
            #     for tim in srctims:
            #         ie = tim.getInvError()
            #         sigmas = (tim.getImage() * ie)[ie > 0]
            #         plt.hist(sigmas, range=(-5,5), bins=21, histtype='step')
            #         plt.axvline(np.mean(sigmas), alpha=0.5)
            #     plt.axvline(0., color='k', lw=3, alpha=0.5)
            #     plt.xlabel('Image pixels (sigma)')
            #     plt.title('Symmetrized pixel values')
            #     self.ps.savefig()
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
            coimgs,_ = quick_coadds(srctims, self.bands, srcwcs, fill_holes=False)
            dimshow(get_rgb(coimgs, self.bands))
            plt.title('Model selection: stage1 data (srcwcs)')
            self.ps.savefig()

        srctractor = self.tractor(srctims, [src])
        srctractor.setModelMasks(modelMasks)
        srccat = srctractor.getCatalog()

        ok,ix,iy = srcwcs.radec2pixelxy(src.getPosition().ra,
                                        src.getPosition().dec)
        ix = int(ix-1)
        iy = int(iy-1)
        # Start in blob
        sh,sw = srcwcs.shape
        if ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
            debug('Source is starting outside blob -- skipping.')
            if mask_others:
                for ie,tim in zip(saved_srctim_ies, srctims):
                    tim.inverr = ie
            return None

        from tractor import Galaxy
        is_galaxy = isinstance(src, Galaxy)
        x0,y0 = srcwcs_x0y0

        # Fitting behaviors based on geometric masks.
        force_pointsource_mask = (IN_BLOB['BRIGHT'] | IN_BLOB['CLUSTER'])
        if self.large_galaxies_force_pointsource:
            force_pointsource_mask |= IN_BLOB['GALAXY']
        force_pointsource = ((self.refmap[y0+iy,x0+ix] & force_pointsource_mask) > 0)

        fit_background_mask = IN_BLOB['MEDIUM']
        ### HACK
        if self.large_galaxies_force_pointsource:
            fit_background_mask |= IN_BLOB['GALAXY']
        fit_background = ((self.refmap[y0+iy,x0+ix] & fit_background_mask) > 0)

        if is_galaxy:
            fit_background = False

            # LSLGA galaxy: set the maximum allowed r_e.
            known_galaxy_logrmax = 0.
            if isinstance(src, (DevGalaxy,ExpGalaxy)):
                print('Known galaxy.  Initial shape:', src.shape)
                # MAGIC 2. = factor by which r_e is allowed to grow for an LSLGA galaxy.
                known_galaxy_logrmax = np.log(src.shape.re * 2.)
            else:
                print('WARNING: unknown galaxy type:', src)

        debug('Source at blob coordinates', x0+ix, y0+iy, '- forcing pointsource?', force_pointsource, ', is large galaxy?', is_galaxy, ', fitting sky background:', fit_background)

        if fit_background:
            for tim in srctims:
                tim.freezeAllBut('sky')
            srctractor.thawParam('images')
            skyparams = srctractor.images.getParams()

        enable_galaxy_cache()

        # Compute the log-likehood without a source here.
        srccat[0] = None

        if fit_background:
            srctractor.optimize_loop(**self.optargs)

        if self.plots_per_source:
            model_mod_rgb = {}
            model_resid_rgb = {}
            # the "none" model
            modimgs = list(srctractor.getModelImages())
            co,nil = quick_coadds(srctims, self.bands, srcwcs, images=modimgs)
            rgb = get_rgb(co, self.bands)
            model_mod_rgb['none'] = rgb
            res = [(tim.getImage() - mod) for tim,mod in zip(srctims, modimgs)]
            co,nil = quick_coadds(srctims, self.bands, srcwcs, images=res)
            rgb = get_rgb(co, self.bands)
            model_resid_rgb['none'] = rgb

        chisqs_none = _per_band_chisqs(srctractor, self.bands)

        nparams = dict(psf=2, rex=3, exp=5, dev=5, ser=6)
        # This is our "upgrade" threshold: how much better a galaxy
        # fit has to be versus psf
        galaxy_margin = 3.**2 + (nparams['exp'] - nparams['psf'])

        # *chisqs* is actually chi-squared improvement vs no source;
        # larger is a better fit.
        chisqs = dict(none=0)

        oldmodel, psf, rex, dev, exp = _initialize_models(src)
        ser = None

        trymodels = [('psf', psf)]

        if oldmodel == 'psf':
            if getattr(src, 'forced_point_source', False):
                # This is set in the GaiaSource contructor from
                # gaia.pointsource
                debug('Gaia source is forced to be a point source -- not trying other models')
            elif force_pointsource:
                # Geometric mask
                debug('Not computing galaxy models due to objects in blob')
            else:
                trymodels.append(('rex', rex))
                # Try galaxy models if rex > psf, or if bright.
                # The 'gals' model is just a marker
                trymodels.append(('gals', None))
        else:
            # If the source was initialized as a galaxy, try all models
            trymodels.extend([('rex', rex), ('dev', dev), ('exp', exp),
                              ('ser', None)])

        cputimes = {}
        for name,newsrc in trymodels:
            cpum0 = time.process_time()

            if name == 'gals':
                # If 'rex' was better than 'psf', or the source is
                # bright, try the galaxy models.
                chi_rex = chisqs.get('rex', 0)
                chi_psf = chisqs.get('psf', 0)
                margin = 1. # 1 parameter
                if chi_rex > (chi_psf+margin) or max(chi_psf, chi_rex) > 400:
                    trymodels.extend([
                        ('dev', dev), ('exp', exp), ('ser', None)])
                continue

            if name == 'ser' and newsrc is None:
                # Start at the better of exp or dev.
                smod = _select_model(chisqs, nparams, galaxy_margin)
                if smod not in ['dev', 'exp']:
                    continue
                if smod == 'dev':
                    newsrc = ser = SersicGalaxy(
                        dev.getPosition().copy(), dev.getBrightness().copy(),
                        dev.getShape().copy(), LegacySersicIndex(4.))
                elif smod == 'exp':
                    newsrc = ser = SersicGalaxy(
                        exp.getPosition().copy(), exp.getBrightness().copy(),
                        exp.getShape().copy(), LegacySersicIndex(1.))
                #print('Initialized SER model:', newsrc)

            srccat[0] = newsrc

            # Set maximum galaxy model sizes
            if is_galaxy:
                # This is a known large galaxy -- set max size based on initial size.
                logrmax = known_galaxy_logrmax
                if name in ('rex', 'exp', 'dev', 'ser'):
                    newsrc.shape.setMaxLogRadius(logrmax)
            else:
                # FIXME -- could use different fractions for deV vs exp (or comp)
                fblob = 0.8
                sh,sw = srcwcs.shape
                logrmax = np.log(fblob * max(sh, sw) * self.pixscale)
                if name in ['rex', 'exp', 'dev', 'ser']:
                    if logrmax < newsrc.shape.getMaxLogRadius():
                        newsrc.shape.setMaxLogRadius(logrmax)

            # Use the same modelMask shapes as the original source ('src').
            # Need to create newsrc->mask mappings though:
            mm = remap_modelmask(modelMasks, src, newsrc)
            srctractor.setModelMasks(mm)
            enable_galaxy_cache()

            if fit_background:
                # Reset sky params
                srctractor.images.setParams(skyparams)
                srctractor.thawParam('images')

            # First-round optimization (during model selection)
            R = srctractor.optimize_loop(**self.optargs)
            #print('Fit result:', newsrc)
            #print('Steps:', R['steps'])
            hit_limit = R.get('hit_limit', False)
            opt_steps = R.get('steps', -1)
            if hit_limit:
                if name in ['rex', 'exp', 'dev', 'ser']:
                    debug('Hit limit: r %.2f vs %.2f' %
                          (newsrc.shape.re, np.exp(logrmax)))
            ok,ix,iy = srcwcs.radec2pixelxy(newsrc.getPosition().ra,
                                            newsrc.getPosition().dec)
            ix = int(ix-1)
            iy = int(iy-1)
            sh,sw = srcblobmask.shape
            if ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
                # Exited blob!
                debug('Source exited sub-blob!')
                if mask_others:
                    for ie,tim in zip(saved_srctim_ies, srctims):
                        tim.inverr = ie
                continue

            disable_galaxy_cache()

            if self.plots_per_source:
                # save RGB images for the model
                modimgs = list(srctractor.getModelImages())
                co,nil = quick_coadds(srctims, self.bands, srcwcs, images=modimgs)
                rgb = get_rgb(co, self.bands)
                model_mod_rgb[name] = rgb
                res = [(tim.getImage() - mod) for tim,mod in zip(srctims, modimgs)]
                co,nil = quick_coadds(srctims, self.bands, srcwcs, images=res)
                rgb = get_rgb(co, self.bands)
                model_resid_rgb[name] = rgb

            # Compute inverse-variances for each source.
            # Convert to "vanilla" ellipse parameterization
            # (but save old shapes first)
            # we do this (rather than making a copy) because we want to
            # use the same modelMask maps.
            if isinstance(newsrc, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
                oldshape = newsrc.shape

            if fit_background:
                # We have to freeze the sky here before computing
                # uncertainties
                srctractor.freezeParam('images')

            nsrcparams = newsrc.numberOfParams()
            _convert_ellipses(newsrc)
            assert(newsrc.numberOfParams() == nsrcparams)

            # Compute a very approximate "fracin" metric (fraction of flux in masked model image
            # versus total flux of model), to avoid wild extrapolation when nearly unconstrained.
            fracin = dict([(b, []) for b in self.bands])
            fluxes = dict([(b, newsrc.getBrightness().getFlux(b)) for b in self.bands])
            for tim,mod in zip(srctims, srctractor.getModelImages()):
                f = (mod * (tim.getInvError() > 0)).sum() / fluxes[tim.band]
                #print('Model image for', tim.name, ': has fracin = %.3g' % f)
                fracin[tim.band].append(f)
            for band in self.bands:
                if len(fracin[band]) == 0:
                    continue
                f = np.mean(fracin[band])
                if f < 1e-6:
                    #print('Source', newsrc, ': setting flux in band', band,
                    #      'to zero based on fracin = %.3g' % f)
                    newsrc.getBrightness().setFlux(band, 0.)

            # Compute inverse-variances
            # This uses the second-round modelMasks.
            allderivs = srctractor.getDerivs()
            ivars = _compute_invvars(allderivs)
            assert(len(ivars) == nsrcparams)

            # If any fluxes have zero invvar, zero out the flux.
            params = newsrc.getParams()
            reset = False
            for i,(pname,p,iv) in enumerate(zip(newsrc.getParamNames(), newsrc.getParams(), ivars)):
                #print('  ', pname, '=', p, 'iv', iv, 'sigma', 1./np.sqrt(iv))
                if iv == 0:
                    #print('Resetting', pname, '=', 0)
                    params[i] = 0.
                    reset = True
            if reset:
                newsrc.setParams(params)
                allderivs = srctractor.getDerivs()
                ivars = _compute_invvars(allderivs)
                assert(len(ivars) == nsrcparams)

            B.all_model_ivs[srci][name] = np.array(ivars).astype(np.float32)
            B.all_models[srci][name] = newsrc.copy()
            assert(B.all_models[srci][name].numberOfParams() == nsrcparams)

            # Now revert the ellipses!
            if isinstance(newsrc, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
                newsrc.shape = oldshape

            # Use the original 'srctractor' here so that the different
            # models are evaluated on the same pixels.
            ch = _per_band_chisqs(srctractor, self.bands)
            chisqs[name] = _chisq_improvement(newsrc, ch, chisqs_none)
            cpum1 = time.process_time()
            B.all_model_cpu[srci][name] = cpum1 - cpum0
            cputimes[name] = cpum1 - cpum0
            B.all_model_hit_limit[srci][name] = hit_limit
            B.all_model_opt_steps[srci][name] = opt_steps

        if mask_others:
            for tim,ie in zip(srctims, saved_srctim_ies):
                # revert tim to original (unmasked-by-others)
                tim.inverr = ie

        # After model selection, revert the sky
        # (srctims=tims when not bigblob)
        if fit_background:
            srctractor.images.setParams(skyparams)

        # Actually select which model to keep.  This "modnames"
        # array determines the order of the elements in the DCHISQ
        # column of the catalog.
        modnames = ['psf', 'rex', 'dev', 'exp', 'ser']
        keepmod = _select_model(chisqs, nparams, galaxy_margin)

        if keepmod is None and getattr(src, 'reference_star', False):
            # Definitely keep ref stars (Gaia & Tycho)
            print('Forcing keeping reference source:', psf)
            keepmod = 'psf'

        keepsrc = {'none':None, 'psf':psf, 'rex':rex,
                   'dev':dev, 'exp':exp, 'ser':ser}[keepmod]
        bestchi = chisqs.get(keepmod, 0.)
        B.dchisq[srci, :] = np.array([chisqs.get(k,0) for k in modnames])
        #print('Keeping model', keepmod, '(chisqs: ', chisqs, ')')

        if keepsrc is not None and bestchi == 0.:
            # Weird edge case, or where some best-fit fluxes go
            # negative. eg
            # https://github.com/legacysurvey/legacypipe/issues/174
            debug('Best dchisq is 0 -- dropping source')
            keepsrc = None

        B.hit_limit[srci] = B.all_model_hit_limit[srci].get(keepmod, False)

        # This is the model-selection plot
        if self.plots_per_source:
            import pylab as plt
            plt.clf()
            rows,cols = 3, 6
            modnames = ['none', 'psf', 'rex', 'dev', 'exp', 'ser']
            # Top-left: image
            plt.subplot(rows, cols, 1)
            coimgs, cons = quick_coadds(srctims, self.bands, srcwcs)
            rgb = get_rgb(coimgs, self.bands)
            dimshow(rgb, ticks=False)
            # next over: rgb with same stretch as models
            plt.subplot(rows, cols, 2)
            rgb = get_rgb(coimgs, self.bands)
            dimshow(rgb, ticks=False)
            for imod,modname in enumerate(modnames):
                if modname != 'none' and not modname in chisqs:
                    continue
                axes = []
                # Second row: models
                plt.subplot(rows, cols, 1+imod+1*cols)
                rgb = model_mod_rgb[modname]
                dimshow(rgb, ticks=False)
                axes.append(plt.gca())
                plt.title(modname)
                # Third row: residuals (not chis)
                plt.subplot(rows, cols, 1+imod+2*cols)
                rgb = model_resid_rgb[modname]
                dimshow(rgb, ticks=False)
                axes.append(plt.gca())
                plt.title('chisq %.0f' % chisqs[modname], fontsize=8)
                # Highlight the model to be kept
                if modname == keepmod:
                    for ax in axes:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(2)
            plt.suptitle('Blob %s, src %i (psf: %s, fitbg: %s): keep %s\n%s\nwas: %s' %
                         (self.name, srci, force_pointsource, fit_background,
                          keepmod, str(keepsrc), str(src)), fontsize=10)
            self.ps.savefig()

        return keepsrc

    def _optimize_individual_sources(self, tr, cat, Ibright, cputime):
        # Single source (though this is coded to handle multiple sources)
        # Fit sources one at a time, but don't subtract other models
        cat.freezeAllParams()

        models = SourceModels()
        models.create(self.tims, cat)
        enable_galaxy_cache()

        for i in Ibright:
            cpu0 = time.process_time()
            cat.freezeAllBut(i)
            src = cat[i]
            if getattr(src, 'freezeparams', False):
                debug('Frozen source', src, '-- keeping as-is!')
                continue
            modelMasks = models.model_masks(0, cat[i])
            tr.setModelMasks(modelMasks)
            tr.optimize_loop(**self.optargs)
            cpu1 = time.process_time()
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
            cpu0 = time.process_time()
            src = cat[srci]
            if getattr(src, 'freezeparams', False):
                debug('Frozen source', src, '-- keeping as-is!')
                continue
            debug('Fitting source', srci, '(%i of %i in blob %s)' %
                  (numi+1, len(Ibright), self.name), ':', src)
            # Add this source's initial model back in.
            models.add(srci, self.tims)

            if self.bigblob:
                # Create super-local sub-sub-tims around this source
                # Make the subimages the same size as the modelMasks.
                mods = [mod[srci] for mod in models.models]
                srctims,modelMasks = _get_subimages(self.tims, mods, src)
                # We plots only the first & last three sources
                if self.plots_per_source and (numi < 3 or numi >= len(Ibright)-3):
                    import pylab as plt
                    plt.clf()
                    # Recompute coadds because of the subtract-all-and-readd shuffle
                    coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                                 fill_holes=False)
                    rgb = get_rgb(coimgs, self.bands)
                    dimshow(rgb)
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
            srctractor.setModelMasks(modelMasks)

            # First-round optimization
            #print('First-round initial log-prob:', srctractor.getLogProb())
            srctractor.optimize_loop(**self.optargs)
            #print('First-round final log-prob:', srctractor.getLogProb())

            # Re-remove the final fit model for this source
            models.update_and_subtract(srci, src, self.tims)

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            #print('Fitting source took', Time()-tsrc)
            #print(src)
            cpu1 = time.process_time()
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
            coimgs,_ = quick_coadds(tims, self.bands, wcs, images=mods,
                                    fill_holes=False)
            dimshow(get_rgb(coimgs,self.bands, **rgbkwargs_resid))
            return

        mods = None
        if model is not None:
            mods = list(model.getModelImages())
        coimgs,_ = quick_coadds(tims, self.bands, wcs, images=mods,
                                fill_holes=False)
        dimshow(get_rgb(coimgs,self.bands))

    def _initial_plots(self):
        import pylab as plt
        debug('Plotting blob image for blob', self.name)
        coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
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
        # ref sources
        for x,y,src in zip(x0,y0,self.srcs):
            if is_reference_source(src):
                plt.plot(x-1, y-1, 'o', mec='g', mfc='none')
        plt.axis(ax)
        plt.title('initial sources')
        self.ps.savefig()

    def create_tims(self, timargs):
        # In order to make multiprocessing easier, the one_blob method
        # is passed all the ingredients to make local tractor Images
        # rather than the Images themselves.  Here we build the
        # 'tims'.
        tims = []
        for (img, inverr, twcs, wcsobj, pcal, sky, subpsf, name,
             sx0, sx1, sy0, sy1,
             band, sig1, modelMinval, imobj) in timargs:
            # Mask out inverr for pixels that are not within the blob.
            try:
                Yo,Xo,Yi,Xi,_ = resample_with_wcs(wcsobj, self.blobwcs,
                                                  intType=np.int16)
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
            h,w = img.shape
            if h < 400 and w < 400:
                subpsf = subpsf.constantPsfAt(w/2., h/2.)

            tim = Image(data=img, inverr=inverr, wcs=twcs,
                        psf=subpsf, photocal=pcal, sky=sky, name=name)
            tim.band = band
            tim.sig1 = sig1
            tim.modelMinval = modelMinval
            tim.subwcs = wcsobj
            tim.meta = imobj
            tim.psf_sigma = imobj.fwhm / 2.35
            tim.dq = None
            tims.append(tim)
        return tims

def _convert_ellipses(src):
    if isinstance(src, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
        src.shape = src.shape.toEllipseE()
        if isinstance(src, RexGalaxy):
            src.shape.freezeParams('e1', 'e2')

def _compute_invvars(allderivs):
    ivs = []
    for derivs in allderivs:
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

def _argsort_by_brightness(cat, bands, ref_first=False):
    fluxes = []
    for src in cat:
        # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
        br = src.getBrightness()
        flux = sum([br.getFlux(band) for band in bands])
        if ref_first and is_reference_source(src):
            # Put the reference sources at the front of the list!
            flux += 1e6
        fluxes.append(flux)
    Ibright = np.argsort(-np.array(fluxes))
    return Ibright

def is_reference_source(src):
    return getattr(src, 'is_reference_source', False)

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

def _initialize_models(src):
    from legacypipe.survey import LogRadius
    if isinstance(src, PointSource):
        psf = src.copy()
        rex = RexGalaxy(src.getPosition(), src.getBrightness(),
                        LogRadius(-1.)).copy()
        # logr, ee1, ee2
        shape = LegacyEllipseWithPriors(-1., 0., 0.)
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        oldmodel = 'psf'
    elif isinstance(src, DevGalaxy):
        rex = RexGalaxy(src.getPosition(), src.getBrightness(),
                        LogRadius(np.log(src.getShape().re))).copy()
        dev = src.copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        oldmodel = 'dev'
    elif isinstance(src, ExpGalaxy):
        psf = PointSource(src.getPosition(), src.getBrightness()).copy()
        rex = RexGalaxy(src.getPosition(), src.getBrightness(),
                        LogRadius(np.log(src.getShape().re))).copy()
        dev = DevGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        exp = src.copy()
        oldmodel = 'exp'
    return oldmodel, psf, rex, dev, exp

def _get_subimages(tims, mods, src):
    subtims = []
    modelMasks = []
    #print('Big blob: trimming:')
    for tim,mod in zip(tims, mods):
        if mod is None:
            continue
        mh,mw = mod.shape
        if mh == 0 or mw == 0:
            continue
        # for modelMasks
        d = { src: ModelMask(0, 0, mw, mh) }
        modelMasks.append(d)

        x0,y0 = mod.x0 , mod.y0
        x1,y1 = x0 + mw, y0 + mh

        subtim = _get_subtim(tim, x0, x1, y0, y1)

        if subtim.shape != (mh,mw):
            print('Subtim was not the shape expected:', subtim.shape,
                  'image shape', tim.getImage().shape, 'slice y', y0,y1,
                  'x', x0,x1, 'mod shape', mh,mw)

        subtims.append(subtim)
    return subtims, modelMasks

def _get_subtim(tim, x0, x1, y0, y1):
    slc = slice(y0,y1), slice(x0, x1)
    subimg = tim.getImage()[slc]
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
    subtim.fulltim = tim
    subtim.meta = tim.meta
    subtim.psf_sigma = tim.psf_sigma
    if tim.dq is not None:
        subtim.dq = tim.dq[slc]
    else:
        subtim.dq = None
    return subtim


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

    def create(self, tims, srcs, subtract=False, modelmasks=None):
        '''
        Note that this modifies the *tims* if subtract=True.
        '''
        self.models = []
        for itim,tim in enumerate(tims):
            mods = []
            sh = tim.shape
            ie = tim.getInvError()
            for src in srcs:

                mm = None
                if modelmasks is not None:
                    mm = modelmasks[itim].get(src, None)

                mod = src.getModelPatch(tim, modelMask=mm)
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

    def update_and_subtract(self, i, src, tims, tim_ies=None, ps=None):
        for itim,(tim,mods) in enumerate(zip(tims, self.models)):
            if src is None:
                mods[i] = None
                continue

            if tim is None:
                continue
            mod = src.getModelPatch(tim)
            mods[i] = mod
            if mod is None:
                continue

            if tim_ies is not None:
                # Apply an extra mask (ie, the mask_others segmentation mask)
                ie = tim_ies[itim]
                if ie is None:
                    continue
                inslice, outslice = mod.getSlices(tim.shape)
                p = mod.patch[inslice]
                img = tim.getImage()
                img[outslice] -= p * (ie[outslice]>0)
            else:
                mod.addTo(tim.getImage(), scale=-1)
            
            # if mod.patch.max() > 1e6:
            #     if ps is not None:
            #         z = np.zeros_like(tim.getImage())
            #         import pylab as plt
            #         plt.clf()
            #         plt.suptitle('tim: %s' % tim.name)
            #         plt.subplot(2,2,1)
            #         plt.imshow(mod.patch, interpolation='nearest', origin='lower')
            #         plt.colorbar()
            #         plt.title('mod')
            #         plt.subplot(2,2,2)
            #         plt.imshow(tim.getImage(), interpolation='nearest', origin='lower')
            #         plt.colorbar()
            #         plt.title('tim (before)')
            #         mod.addTo(z, scale=1)
            #         plt.subplot(2,2,3)
            #         plt.imshow(z, interpolation='nearest', origin='lower')
            #         plt.colorbar()
            #         plt.title('mod')
            #         img = tim.getImage().copy()
            #         mod.addTo(img, scale=-1)
            #         plt.subplot(2,2,4)
            #         plt.imshow(img, interpolation='nearest', origin='lower')
            #         plt.colorbar()
            #         plt.title('tim-mod')
            #         ps.savefig()

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

def _select_model(chisqs, nparams, galaxy_margin):
    '''
    Returns keepmod (string), the name of the preferred model.
    '''
    keepmod = 'none'
    #print('_select_model: chisqs', chisqs)

    # This is our "detection threshold": 5-sigma in
    # *parameter-penalized* units; ie, ~5.2-sigma for point sources
    cut = 5.**2
    # Take the best of all models computed
    diff = max([chisqs[name] - nparams[name] for name in chisqs.keys()
                if name != 'none'] + [-1])

    if diff < cut:
        # Drop this source
        return keepmod

    # Now choose between point source and REX
    if 'psf' in chisqs and not 'rex' in chisqs:
        # bright stars / reference stars: we don't test the simple model.
        return 'psf'

    #print('PSF', chisqs.get('psf',0)-nparams['psf'], 'vs REX', chisqs.get('rex',0)-nparams['rex'])

    # Is PSF good enough to keep?
    if 'psf' in chisqs and (chisqs['psf']-nparams['psf'] >= cut):
        keepmod = 'psf'

    # Now choose between point source and REX
    if 'psf' in chisqs and (
            chisqs['psf']-nparams['psf'] >= chisqs.get('rex',0)-nparams['rex']):
        #print('Keeping PSF')
        keepmod = 'psf'
    elif 'rex' in chisqs and (
            chisqs['rex']-nparams['rex'] > chisqs.get('psf',0)-nparams['psf']):
        #print('REX is better fit than PSF.')
        oldkeepmod = keepmod
        keepmod = 'rex'
        # For REX, we also demand a fractionally better fit
        dchisq_psf = chisqs.get('psf',0)
        dchisq_rex = chisqs.get('rex',0)
        if dchisq_psf > 0 and (dchisq_rex - dchisq_psf) < (0.01 * dchisq_psf):
            #print('REX is not a fractionally better fit, keeping', oldkeepmod)
            keepmod = oldkeepmod

    if not ('exp' in chisqs or 'dev' in chisqs):
        #print('No EXP or DEV; keeping', keepmod)
        return keepmod

    # This is our "upgrade" threshold: how much better a galaxy
    # fit has to be versus psf
    cut = galaxy_margin

    # This is the "fractional" upgrade threshold for psf/rex to dev/exp:
    # 1% of psf vs nothing
    fcut = 0.01 * chisqs.get('psf', 0.)
    cut = max(cut, fcut)

    expdiff = chisqs.get('exp', 0) - chisqs[keepmod]
    devdiff = chisqs.get('dev', 0) - chisqs[keepmod]

    #print('EXP vs', keepmod, ':', expdiff)
    #print('DEV vs', keepmod, ':', devdiff)

    if not (expdiff > cut or devdiff > cut):
        #print('Keeping', keepmod)
        return keepmod

    if expdiff > devdiff:
        #print('Upgrading to EXP: diff', expdiff)
        keepmod = 'exp'
    else:
        #print('Upgrading to DEV: diff', expdiff)
        keepmod = 'dev'

    # Consider Sersic models
    if 'ser' not in chisqs:
        return keepmod
    serdiff = chisqs['ser'] - chisqs[keepmod]

    sermargin = 25.

    if serdiff < sermargin:
        return keepmod
    keepmod = 'ser'
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
    for img in tractor.images:
        chi = tractor.getChiImage(img=img)
        chisqs[img.band] = chisqs[img.band] + (chi ** 2).sum()
    return chisqs

def get_inblob_map(blobwcs, refs):
    bh,bw = blobwcs.shape
    bh = int(bh)
    bw = int(bw)
    blobmap = np.zeros((bh,bw), np.uint8)
    # circular/elliptical regions:
    for col,bit,ellipse in [('isbright', 'BRIGHT', False),
                            ('ismedium', 'MEDIUM', False),
                            ('iscluster', 'CLUSTER', False),
                            ('islargegalaxy', 'GALAXY', True),]:
        isit = refs.get(col)
        if not np.any(isit):
            debug('None marked', col)
            continue
        I, = np.nonzero(isit)
        debug(len(I), 'with', col, 'set')
        if len(I) == 0:
            continue

        thisrefs = refs[I]
        ok,xx,yy = blobwcs.radec2pixelxy(thisrefs.ra, thisrefs.dec)
        for x,y,ref in zip(xx,yy,thisrefs):
            # Cut to L1 rectangle
            xlo = int(np.clip(np.floor(x-1 - ref.radius_pix), 0, bw))
            xhi = int(np.clip(np.ceil (x   + ref.radius_pix), 0, bw))
            ylo = int(np.clip(np.floor(y-1 - ref.radius_pix), 0, bh))
            yhi = int(np.clip(np.ceil (y   + ref.radius_pix), 0, bh))
            #print('x range', xlo,xhi, 'y range', ylo,yhi)
            if xlo == xhi or ylo == yhi:
                continue

            bitval = np.uint8(IN_BLOB[bit])
            if not ellipse:
                rr = ((np.arange(ylo,yhi)[:,np.newaxis] - (y-1))**2 +
                      (np.arange(xlo,xhi)[np.newaxis,:] - (x-1))**2)
                masked = (rr <= ref.radius_pix**2)
            else:
                # *should* have ba and pa if we got here...
                xgrid,ygrid = np.meshgrid(np.arange(xlo,xhi), np.arange(ylo,yhi))
                dx = xgrid - (x-1)
                dy = ygrid - (y-1)
                debug('Galaxy: PA', ref.pa, 'BA', ref.ba, 'Radius', ref.radius, 'pix', ref.radius_pix)
                if not np.isfinite(ref.pa):
                    ref.pa = 0.
                v1x = -np.sin(np.deg2rad(ref.pa))
                v1y =  np.cos(np.deg2rad(ref.pa))
                v2x =  v1y
                v2y = -v1x
                dot1 = dx * v1x + dy * v1y
                dot2 = dx * v2x + dy * v2y
                r1 = ref.radius_pix
                r2 = ref.radius_pix * ref.ba
                masked = (dot1**2 / r1**2 + dot2**2 / r2**2 < 1.)

            blobmap[ylo:yhi, xlo:xhi] |= (bitval * masked)
    return blobmap
