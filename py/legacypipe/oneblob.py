import numpy as np
import time

from astrometry.util.ttime import Time
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow

from tractor import Tractor, PointSource, Image, Catalog, Patch, Galaxy
from tractor.galaxy import (DevGalaxy, ExpGalaxy,
                            disable_galaxy_cache, enable_galaxy_cache)
from tractor.patch import ModelMask
from tractor.sersic import SersicGalaxy

from legacypipe.survey import (RexGalaxy,
                               LegacyEllipseWithPriors, LegacySersicIndex, get_rgb)
from legacypipe.bits import IN_BLOB
from legacypipe.coadds import quick_coadds
from legacypipe.runbrick_plots import _plot_mods
from legacypipe.utils import get_cpu_arch

from legacypipe.utils import run_ps

rgbkwargs_resid = dict(resids=True)

import logging
logger = logging.getLogger('legacypipe.oneblob')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)
def is_debug():
    return logger.isEnabledFor(logging.DEBUG)

# Determines the order of elements in the DCHISQ array.
MODEL_NAMES = ['psf', 'rex', 'dev', 'exp', 'ser']

def one_blob(X):
    '''
    Fits sources contained within a "blob" of pixels.
    '''
    if X is None:
        return None
    (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
     srcs, bands, plots, ps, reoptimize, iterative, use_ceres, refmap,
     large_galaxies_force_pointsource, less_masking, frozen_galaxies) = X

    debug('Fitting blob %s: blobid %i, nsources %i, size %i x %i, %i images, %i frozen galaxies' %
          (nblob, iblob, len(Isrcs), blobw, blobh, len(timargs), len(frozen_galaxies)))

    if len(timargs) == 0:
        return None
    if len(Isrcs) == 0:
        return None

    assert(blobmask.shape == (blobh,blobw))
    assert(refmap.shape == (blobh,blobw))

    for g in frozen_galaxies:
        debug('Frozen galaxy:', g)

    LegacySersicIndex.stepsize = 0.001

    if plots:
        import pylab as plt
        plt.figure(2, figsize=(3,3))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.figure(1)

    t0 = time.process_time()
    # A local WCS for this blob
    blobwcs = brickwcs.get_subimage(bx0, by0, blobw, blobh)

    ob = OneBlob(nblob, blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, use_ceres, refmap,
                 large_galaxies_force_pointsource,
                 less_masking, frozen_galaxies)
    B = ob.init_table(Isrcs)
    B = ob.run(B, reoptimize=reoptimize, iterative_detection=iterative)
    ob.finalize_table(B, bx0, by0)

    t1 = time.process_time()
    B.cpu_blob = np.empty(len(B), np.float32)
    B.cpu_blob[:] = t1 - t0
    B.iblob = iblob
    return B

class OneBlob(object):
    def __init__(self, name, blobwcs, blobmask, timargs, srcs, bands,
                 plots, ps, use_ceres, refmap,
                 large_galaxies_force_pointsource,
                 less_masking, frozen_galaxies):
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
        self.less_masking = less_masking
        self.tims = create_tims(self.blobwcs, self.blobmask, timargs)
        self.total_pix = sum([np.sum(t.getInvError() > 0) for t in self.tims])
        self.plots2 = False
        alphas = [0.1, 0.3, 1.0]
        self.optargs = dict(priors=True, shared_params=False, alphas=alphas,
                            print_progress=True)
        self.blobh,self.blobw = blobmask.shape
        self.trargs = dict()
        self.frozen_galaxy_mods = []

        if len(frozen_galaxies):
            debug('Subtracting frozen galaxy models...')
            tr = Tractor(self.tims, Catalog(*frozen_galaxies))
            mm = []
            for tim in self.tims:
                mh,mw = tim.shape
                mm.append(dict([(g, ModelMask(0, 0, mw, mh)) for g in frozen_galaxies]))
            tr.setModelMasks(mm)
            if self.plots:
                mods = []
            for tim in self.tims:
                try:
                    mod = tr.getModelImage(tim)
                except:
                    print('Exception getting frozen-galaxies model.')
                    print('galaxies:', frozen_galaxies)
                    print('tim:', tim)
                    import traceback
                    traceback.print_exc()
                    continue
                self.frozen_galaxy_mods.append(mod)
                tim.data -= mod
                if self.plots:
                    mods.append(mod)
            if self.plots:
                import pylab as plt
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                        fill_holes=False)
                plt.clf()
                dimshow(get_rgb(coimgs, self.bands))
                plt.title('Subtracted frozen galaxies')
                self.ps.savefig()
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                        fill_holes=False)
                plt.clf()
                dimshow(get_rgb(coimgs, self.bands))
                plt.title('After subtracting frozen galaxies')
                self.ps.savefig()

        # if use_ceres:
        #     from tractor.ceres_optimizer import CeresOptimizer
        #     ceres_optimizer = CeresOptimizer()
        #     self.optargs.update(scale_columns=False,
        #                         scaled=False,
        #                         dynamic_scale=False)
        #     self.trargs.update(optimizer=ceres_optimizer)
        # else:
        #     self.optargs.update(dchisq = 0.1)

        from tractor.dense_optimizer import ConstrainedDenseOptimizer
        self.trargs.update(optimizer=ConstrainedDenseOptimizer())
        self.optargs.update(dchisq = 0.1)

    def init_table(self, Isrcs):
        # Per-source measurements for this blob
        B = fits_table()
        B.sources = self.srcs
        B.Isrcs = Isrcs
        # Did sources start within the blob?
        _,x0,y0 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in self.srcs]),
            np.array([src.getPosition().dec for src in self.srcs]))
        # blob-relative initial positions (zero-indexed)
        B.x0 = (x0 - 1.).astype(np.float32)
        B.y0 = (y0 - 1.).astype(np.float32)
        B.safe_x0 = np.clip(np.round(x0-1).astype(int), 0, self.blobw-1)
        B.safe_y0 = np.clip(np.round(y0-1).astype(int), 0, self.blobh-1)
        B.started_in_blob = self.blobmask[B.safe_y0, B.safe_x0]
        # This uses 'initial' pixel positions, because that's what determines
        # the fitting behaviors.
        return B

    def finalize_table(self, B, bx0, by0):
        _,x1,y1 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in B.sources]),
            np.array([src.getPosition().dec for src in B.sources]))
        B.finished_in_blob = self.blobmask[
            np.clip(np.round(y1-1).astype(int), 0, self.blobh-1),
            np.clip(np.round(x1-1).astype(int), 0, self.blobw-1)]
        assert(len(B.finished_in_blob) == len(B))
        assert(len(B.finished_in_blob) == len(B.started_in_blob))

        # Setting values here (after .run() has completed) means that iterative sources
        # (which get merged with the original table B) get values also.
        B.blob_x0     = np.zeros(len(B), np.int16) + bx0
        B.blob_y0     = np.zeros(len(B), np.int16) + by0
        B.blob_width  = np.zeros(len(B), np.int16) + self.blobw
        B.blob_height = np.zeros(len(B), np.int16) + self.blobh
        B.blob_npix   = np.zeros(len(B), np.int32) + np.sum(self.blobmask)
        B.blob_nimages= np.zeros(len(B), np.int16) + len(self.tims)
        B.blob_totalpix = np.zeros(len(B), np.int32) + self.total_pix
        B.cpu_arch = np.zeros(len(B), dtype='U3')
        B.cpu_arch[:] = get_cpu_arch()
        # Convert to whole-brick (zero-indexed) pixel positions.
        # (do this here rather than above to ease handling iterative detections)
        B.x0 += bx0
        B.y0 += by0
        # these are now in brick coords... rename for consistency in runbrick.py
        B.rename('x0', 'bx0')
        B.rename('y0', 'by0')

    def run(self, B, reoptimize=False, iterative_detection=True,
            compute_metrics=True):
        # The overall steps here are:
        # - fit initial fluxes for small number of sources that may need it
        # - optimize individual sources
        # - compute segmentation map
        # - model selection (including iterative detection)
        # - metrics

        #print('OneBlob run starting: srcs', self.srcs)
        #for src in self.srcs:
        #    print('OneBlob  ', src.getParams())

        trun = tlast = Time()
        # Not quite so many plots...
        self.plots1 = self.plots
        cat = Catalog(*self.srcs)

        N = len(B)
        B.cpu_source         = np.zeros(N, np.float32)
        B.force_keep_source  = np.zeros(N, bool)
        B.fit_background     = np.zeros(N, bool)
        B.forced_pointsource = np.zeros(N, bool)
        B.hit_limit          = np.zeros(N, bool)
        B.hit_ser_limit      = np.zeros(N, bool)
        B.hit_r_limit        = np.zeros(N, bool)
        B.blob_symm_width    = np.zeros(N, np.int16)
        B.blob_symm_height   = np.zeros(N, np.int16)
        B.blob_symm_npix     = np.zeros(N, np.int32)
        B.blob_symm_nimages  = np.zeros(N, np.int16)

        # Save initial fluxes for all sources (used if we force
        # keeping a reference star)
        for src in self.srcs:
            src.initial_brightness = src.brightness.copy()

        # Set the freezeparams field for each source.  (This is set for
        # large galaxies with the 'freeze' column set.)
        for src in self.srcs:
            src.freezeparams = getattr(src, 'freezeparams', False)

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

        tr = self.tractor(self.tims, cat)

        # Fit any sources marked with 'needs_initial_flux' -- saturated, and SGA
        fitflux = [src for src in cat if getattr(src, 'needs_initial_flux', False)]
        if len(fitflux):
            self._fit_fluxes(cat, self.tims, self.bands, fitcat=fitflux)
            if self.plots:
                self._plots(tr, 'Fitting initial fluxes')
        del fitflux

        if self.plots:
            self._plots(tr, 'Initial models')
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('Initial models')
            self.ps.savefig()

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

        if self.plots:
            self._plots(tr, 'After source fitting')
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('After source fitting')
            self.ps.savefig()
            # Plot source locations
            ax = plt.axis()
            _,xf,yf = self.blobwcs.radec2pixelxy(
                np.array([src.getPosition().ra  for src in self.srcs]),
                np.array([src.getPosition().dec for src in self.srcs]))
            plt.plot(xf-1, yf-1, 'r.', label='Sources')
            Ir = np.flatnonzero([is_reference_source(src) for src in self.srcs])
            if len(Ir):
                plt.plot(xf[Ir]-1, yf[Ir]-1, 'o', mec='g', mfc='none', ms=8, mew=2,
                         label='Ref source')
            plt.legend()
            plt.axis(ax)
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

        # Set any fitting behaviors based on geometric masks.

        # Fitting behaviors: force point-source
        force_pointsource_mask = (IN_BLOB['BRIGHT'] | IN_BLOB['CLUSTER'])
        # large_galaxies_force_pointsource is True by default.
        if self.large_galaxies_force_pointsource:
            force_pointsource_mask |= IN_BLOB['GALAXY']
        # Fit background?
        fit_background_mask = IN_BLOB['BRIGHT']
        if not self.less_masking:
            fit_background_mask |= IN_BLOB['MEDIUM']
        ### this variable *also* forces fitting the background.
        if self.large_galaxies_force_pointsource:
            fit_background_mask |= IN_BLOB['GALAXY']
        for srci,src in enumerate(cat):
            _,ix,iy = self.blobwcs.radec2pixelxy(src.getPosition().ra,
                                                 src.getPosition().dec)
            ix = int(np.clip(ix-1, 0, self.blobw-1))
            iy = int(np.clip(iy-1, 0, self.blobh-1))
            bits = self.refmap[iy, ix]
            force_pointsource = ((bits & force_pointsource_mask) > 0)
            fit_background = ((bits & fit_background_mask) > 0)
            is_galaxy = isinstance(src, Galaxy)
            if is_galaxy:
                fit_background = False
                force_pointsource = False
            B.forced_pointsource[srci] = force_pointsource
            B.fit_background[srci] = fit_background
            # Also set a parameter on 'src' for use in compute_segmentation_map()
            src.maskbits_forced_point_source = force_pointsource

        self.compute_segmentation_map()

        # Next, model selections: point source vs rex vs dev/exp vs ser.
        B = self.run_model_selection(cat, Ibright, B,
                                     iterative_detection=iterative_detection)

        debug('Blob', self.name, 'finished model selection:', Time()-tlast)
        tlast = Time()

        # Cut down to just the kept sources
        cat = B.sources
        I = np.array([i for i,s in enumerate(cat) if s is not None])
        B.cut(I)
        del I
        cat = Catalog(*B.sources)
        tr.catalog = cat

        if self.plots:
            self._plots(tr, 'After model selection')
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('After model selection')
            self.ps.savefig()
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr, addnoise=True)
            plt.title('After model selection (+noise)')
            self.ps.savefig()

        if self.plots_single:
            plt.figure(2)
            mods = list(tr.getModelImages())
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                    fill_holes=False)
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.savefig('blob-%s-model.png' % (self.name))
            res = [(tim.getImage() - mod) for tim,mod in zip(self.tims, mods)]
            coresids,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=res)
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
                if B.force_keep_source[isub]:
                    B.srcinvvars[isub] = np.zeros(nsrcparams, np.float32)
                    cat.freezeParam(isub)
                    continue
                _convert_ellipses(src)
                assert(src.numberOfParams() == nsrcparams)
                # Compute inverse-variances
                allderivs = tr.getDerivs()
                ivars = _compute_invvars(allderivs)
                del allderivs
                assert(len(ivars) == nsrcparams)
                B.srcinvvars[isub] = ivars
                assert(len(B.srcinvvars[isub]) == cat[isub].numberOfParams())
                cat.freezeParam(isub)
                del ivars

            # Check for sources with zero inverse-variance -- I think these
            # can be generated during the "Simultaneous re-opt" stage above --
            # sources can get scattered outside the blob.
            I, = np.nonzero([np.sum(iv) > 0 or force
                             for iv,force in zip(B.srcinvvars, B.force_keep_source)])
            if len(I) < len(B):
                debug('Keeping', len(I), 'of', len(B),'sources with non-zero ivar')
                B.cut(I)
                cat = Catalog(*B.sources)
                tr.catalog = cat
            del I

            M = _compute_source_metrics(B.sources, self.tims, self.bands, tr)
            for k,v in M.items():
                B.set(k, v)
            del M

        info('Blob', self.name, 'finished, total:', Time()-trun)
        return B

    def compute_segmentation_map(self):
        from functools import reduce
        from legacypipe.detection import detection_maps
        from astrometry.util.multiproc import multiproc
        from scipy.ndimage import binary_dilation

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

            if self.plots and False:
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

        if self.plots:
            import pylab as plt
            plt.clf()
            plt.imshow(saturated_pix, interpolation='nearest', origin='lower',
                       vmin=0, vmax=1, cmap='gray')
            plt.title('saturated pix')
            self.ps.savefig()

            plt.clf()
            plt.imshow(maxsn, interpolation='nearest', origin='lower')
            plt.title('max s/n for segmentation')
            self.ps.savefig()

        ok,ix,iy = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in self.srcs]),
            np.array([src.getPosition().dec for src in self.srcs]))
        ix = np.clip(np.round(ix)-1, 0, self.blobw-1).astype(np.int32)
        iy = np.clip(np.round(iy)-1, 0, self.blobh-1).astype(np.int32)

        # Do not compute segmentation map for sources in the CLUSTER mask
        # (or with very bad coords)
        Iseg, = np.nonzero(ok * ((self.refmap[iy, ix] & IN_BLOB['CLUSTER']) == 0))
        del ok
        # Zero out the S/N in CLUSTER mask
        maxsn[(self.refmap & IN_BLOB['CLUSTER']) > 0] = 0.
        # (also zero out the satmap in the CLUSTER mask)
        saturated_pix[(self.refmap & IN_BLOB['CLUSTER']) > 0] = False

        import heapq
        H,W = self.blobh, self.blobw
        segmap = np.empty((H,W), np.int32)
        segmap[:,:] = -1
        # Iseg are the indices in self.srcs of sources to segment
        sy = iy[Iseg]
        sx = ix[Iseg]
        segmap[sy, sx] = Iseg
        maxr2 = np.zeros(len(Iseg), np.int32)
        # Reference sources forced to be point sources get a max radius:
        ref_radius = 25
        for j,i in enumerate(Iseg):
            if getattr(self.srcs[i], 'forced_point_source', False):
                maxr2[j] = ref_radius**2
        # Sources inside maskbits masks that are forced to be point sources
        # also get a max radius.
        for j,i in enumerate(Iseg):
            if getattr(self.srcs[i], 'maskbits_forced_point_source', False):
                maxr2[j] = ref_radius**2

        mask = self.blobmask
        # Watershed by priority-fill.
        # values are (-sn, key, x, y, center_x, center_y, maxr2)
        q = [(-maxsn[y,x], segmap[y,x],x,y,x,y,r2)
             for x,y,r2 in zip(sx,sy,maxr2)]
        heapq.heapify(q)
        while len(q):
            _,key,x,y,cx,cy,r2 = heapq.heappop(q)
            segmap[y,x] = key
            # 4-connected neighbours
            for x,y in [(x, y-1), (x, y+1), (x-1, y), (x+1, y),]:
                # out of bounds?
                if x<0 or y<0 or x==W or y==H:
                    continue
                # not in blobmask?
                if not mask[y,x]:
                    continue
                # already queued or segmented?
                if segmap[y,x] != -1:
                    continue
                # outside the ref source radius?
                if r2 > 0 and (x-cx)**2 + (y-cy)**2 > r2:
                    continue
                # mark as queued
                segmap[y,x] = -2
                # enqueue!
                heapq.heappush(q, (-maxsn[y,x], key, x, y, cx, cy, r2))
        del q, maxr2
        del maxsn, saturated_pix

        # ensure that each source owns a tiny radius around its center
        # in the segmentation map.  If there is more than one source
        # in that radius, each pixel gets assigned to its nearest
        # source.
        radius = 5
        Ibright = _argsort_by_brightness([self.srcs[i] for i in Iseg], self.bands)
        _set_kingdoms(segmap, radius, Iseg[Ibright], ix, iy)

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
        # (model sizes are determined at this point)
        models.create(self.tims, cat, subtract=True)

        N = len(cat)
        B.dchisq = np.zeros((N, 5), np.float32)
        B.all_models    = np.array([{} for i in range(N)])
        B.all_model_ivs = np.array([{} for i in range(N)])
        B.all_model_cpu = np.array([{} for i in range(N)])
        B.all_model_hit_limit     = np.array([{} for i in range(N)])
        B.all_model_hit_r_limit   = np.array([{} for i in range(N)])
        B.all_model_opt_steps     = np.array([{} for i in range(N)])

        # Model selection for sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
            src = cat[srci]
            debug('Model selection for source %i of %i in blob %s; sourcei %i' %
                  (numi+1, len(Ibright), self.name, srci))
            cpu0 = time.process_time()

            if src.freezeparams:
                info('Frozen source', src, '-- keeping as-is!')
                B.sources[srci] = src
                continue

            # Add this source's initial model back in.
            models.add(srci, self.tims)

            if self.plots_single:
                import pylab as plt
                plt.figure(2)
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                        fill_holes=False)
                rgb = get_rgb(coimgs,self.bands)
                plt.imsave('blob-%s-%s-bdata.png' % (self.name, srci), rgb,
                           origin='lower')
                plt.figure(1)

            # Model selection for this source.
            keepsrc = self.model_selection_one_source(src, srci, models, B)

            # Definitely keep ref stars (Gaia & Tycho)
            if keepsrc is None and getattr(src, 'reference_star', False):
                debug('Dropped reference star:', src)
                src.brightness = src.initial_brightness
                debug('  Reset brightness to', src.brightness)
                src.force_keep_source = True
                keepsrc = src

            B.sources[srci] = keepsrc
            B.force_keep_source[srci] = getattr(keepsrc, 'force_keep_source', False)
            cat[srci] = keepsrc

            models.update_and_subtract(srci, keepsrc, self.tims)

            if self.plots_single:
                plt.figure(2)
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
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
                B = merge_tables([B, Bnew], columns='fillzero')
                # columns not in Bnew:
                # {'safe_x0', 'safe_y0', 'started_in_blob'}
                B.sources = srcs + newsrcs

        models.restore_images(self.tims)
        del models
        return B

    def iterative_detection(self, Bold, models):
        # Compute per-band detection maps
        from scipy.ndimage import binary_dilation
        from legacypipe.detection import sed_matched_filters, detection_maps, run_sed_matched_filters
        from astrometry.util.multiproc import multiproc

        if self.plots:
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
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
        satmaps = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]

        # Also compute detection maps on the (first-round) model images!
        # save tim.images (= residuals at this point)
        realimages = [tim.getImage() for tim in self.tims]
        for itim,(tim,mods) in enumerate(zip(self.tims, models.models)):
            modimg = np.zeros_like(tim.getImage())
            for mod in mods:
                if mod is None:
                    continue
                mod.addTo(modimg)
            if len(self.frozen_galaxy_mods):
                modimg += self.frozen_galaxy_mods[itim]
            tim.data = modimg
        if self.plots:
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
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
        avoid_x = Bold.safe_x0
        avoid_y = Bold.safe_y0
        avoid_r = np.zeros(len(avoid_x), np.float32) + 2.
        nsigma = 6.
        avoid_map = (self.refmap != 0)

        Tnew,_,_ = run_sed_matched_filters(
            SEDs, self.bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r),
            self.blobwcs, nsigma=nsigma, saturated_pix=satmaps, veto_map=avoid_map,
            plots=False, ps=None, mp=mp)

        detlogger.setLevel(detloglvl)

        if Tnew is None:
            debug('No iterative sources detected!')
            return None

        debug('Found', len(Tnew), 'new sources')
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
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
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

            plt.plot(Bold.safe_x0, Bold.safe_y0, 'o', ms=5, mec='r',
                     mfc='none', label='Avoid (r=2)')
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

        info('Blob %s:'%self.name, 'Measuring', len(Tnew), 'iterative sources')

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
        Bnew.x0 = Tnew.ibx.astype(np.float32)
        Bnew.y0 = Tnew.iby.astype(np.float32)
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

        # FIXME -- don't need these aliased variable names any more
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
            _,_,coimgs,_ = quick_coadds(srctims, self.bands,self.blobwcs,
                                        fill_holes=False, get_cow=True)
            dimshow(get_rgb(coimgs, self.bands))
            ax = plt.axis()
            pos = src.getPosition()
            _,x,y = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
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
            from scipy.ndimage import binary_dilation, binary_fill_holes
            from scipy.ndimage.measurements import label
            # Compute per-band detection maps
            mp = multiproc()
            detmaps,detivs,_ = detection_maps(
                srctims, srcwcs, self.bands, mp)
            # Compute the symmetric area that fits in this 'srcblobmask' region
            pos = src.getPosition()
            _,xx,yy = srcwcs.radec2pixelxy(pos.ra, pos.dec)
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
                # flipsn = np.zeros_like(sn)
                # # Symmetrize
                # flipsn[slc] = np.minimum(sn[slc],
                #                          np.flipud(np.fliplr(sn[slc])))
                # # just OR the detection maps per-band...
                # flipblobs |= (flipsn > 5.)

                # Symmetrize
                sn[slc] = np.minimum(sn[slc],
                                     np.flipud(np.fliplr(sn[slc])))
                # just OR the detection maps per-band...
                flipblobs |= (sn > 5.)

            flipblobs = binary_fill_holes(flipblobs)
            blobs,_ = label(flipblobs)
            goodblob = blobs[iy,ix]

            if self.plots_per_source and True:
                # This plot is about the symmetric-blob definitions
                # when fitting sources.
                import pylab as plt
                #from legacypipe.detection import plot_boundary_map
                # plt.clf()
                # for i,(band,detmap,detiv) in enumerate(zip(self.bands, detmaps, detivs)):
                #     if i >= 4:
                #         break
                #     detsn = detmap * np.sqrt(detiv)
                #     plt.subplot(2,2, i+1)
                #     mx = detsn.max()
                #     dimshow(detsn, vmin=-2, vmax=max(8, mx))
                #     ax = plt.axis()
                #     plot_boundary_map(detsn >= 5.)
                #     plt.plot(ix, iy, 'rx')
                #     plt.plot([ix-flipw, ix-flipw, ix+flipw, ix+flipw, ix-flipw],
                #              [iy-fliph, iy+fliph, iy+fliph, iy-fliph, iy-fliph], 'r-')
                #     plt.axis(ax)
                #     plt.title('det S/N: ' + band)
                # plt.subplot(2,2,4)
                # dimshow(flipblobs, vmin=0, vmax=1)
                # plt.colorbar()
                # ax = plt.axis()
                # plot_boundary_map(blobs == goodblob)
                # if binary_fill_holes(flipblobs)[iy,ix]:
                #     fb = (blobs == goodblob)
                #     di = binary_dilation(fb, iterations=4)
                #     if np.any(di):
                #         plot_boundary_map(di, rgb=(255,0,0))
                # plt.plot(ix, iy, 'rx')
                # plt.plot([ix-flipw, ix-flipw, ix+flipw, ix+flipw, ix-flipw],
                #          [iy-fliph, iy+fliph, iy+fliph, iy-fliph, iy-fliph], 'r-')
                # plt.axis(ax)
                # plt.title('good blob')
                # self.ps.savefig()

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
                mysegmap = self.segmap[sy0:sy0+dh, sx0:sx0+dw]
                # renumber for plotting
                _,S = np.unique(mysegmap, return_inverse=True)
                dimshow(S.reshape(mysegmap.shape), cmap='tab20',
                        interpolation='nearest', origin='lower')
                ax = plt.axis()
                plt.plot(ix, iy, 'kx', ms=15, mew=3)
                plt.axis(ax)
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
                    Yo,Xo,Yi,Xi,_ = resample_with_wcs(
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

        srctractor = self.tractor(srctims, [src])
        srctractor.setModelMasks(modelMasks)
        srccat = srctractor.getCatalog()

        is_galaxy = isinstance(src, Galaxy)
        force_pointsource = B.forced_pointsource[srci]
        fit_background = B.fit_background[srci]

        _,ix,iy = srcwcs.radec2pixelxy(src.getPosition().ra,
                                       src.getPosition().dec)
        ix = int(ix-1)
        iy = int(iy-1)
        # Start in blob
        sh,sw = srcwcs.shape
        if is_galaxy:
            # allow SGA galaxy sources to start outside the blob
            pass
        elif ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
            debug('Source is starting outside blob -- skipping.')
            if mask_others:
                for ie,tim in zip(saved_srctim_ies, srctims):
                    tim.inverr = ie
            return None

        if is_galaxy:
            # SGA galaxy: set the maximum allowed r_e.
            known_galaxy_logrmax = 0.
            if isinstance(src, (DevGalaxy,ExpGalaxy, SersicGalaxy)):
                print('Known galaxy.  Initial shape:', src.shape)
                # MAGIC 2. = factor by which r_e is allowed to grow for an SGA galaxy.
                known_galaxy_logrmax = np.log(src.shape.re * 2.)
            else:
                print('WARNING: unknown galaxy type:', src)

        x0,y0 = srcwcs_x0y0
        debug('Source at blob coordinates', x0+ix, y0+iy, '- forcing pointsource?', force_pointsource, ', is large galaxy?', is_galaxy, ', fitting sky background:', fit_background)


        if fit_background:
            for tim in srctims:
                tim.freezeAllBut('sky')
            srctractor.thawParam('images')
            # When we're fitting the background, using the sparse optimizer is critical
            # when we have a lot of images: we're adding Nimages extra parameters, touching
            # every pixel; you don't want Nimages x Npixels dense matrices.
            from tractor.lsqr_optimizer import LsqrOptimizer
            srctractor.optimizer = LsqrOptimizer()
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
            co,_ = quick_coadds(srctims, self.bands, srcwcs, images=modimgs)
            rgb = get_rgb(co, self.bands)
            model_mod_rgb['none'] = rgb
            res = [(tim.getImage() - mod) for tim,mod in zip(srctims, modimgs)]
            co,_ = quick_coadds(srctims, self.bands, srcwcs, images=res)
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
                debug('Not computing galaxy models due to being in a mask')
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
            print('OneBlob before model selection:', newsrc)
            try:
                R = srctractor.optimize_loop(**self.optargs)
            except Exception as e:
                print('Exception fitting source in model selection.  src:', newsrc)
                import traceback
                traceback.print_exc()
                raise(e)
                continue
            print('OneBlob after model selection:', newsrc)
            #print('Fit result:', newsrc)
            #print('Steps:', R['steps'])
            hit_limit = R.get('hit_limit', False)
            opt_steps = R.get('steps', -1)
            hit_ser_limit = False
            hit_r_limit = False
            #print('OneBlob steps:', opt_steps)
            #print('OneBlob hit limit:', hit_limit)
            if hit_limit:
                debug('Source', newsrc, 'hit limit:')
                if is_debug():
                    for nm,p,low,upp in zip(newsrc.getParamNames(), newsrc.getParams(),
                                            newsrc.getLowerBounds(), newsrc.getUpperBounds()):
                        debug('  ', nm, '=', p, 'bounds', low, upp)

                if name == 'ser':
                    si = newsrc.sersicindex
                    sival = si.getValue()
                    # Can end up close, but not exactly at a limit...
                    if min(sival - si.lower, si.upper - sival) < 1e-3:
                        hit_ser_limit = True
                        debug('Hit sersic limit')
                if name in ['rex', 'exp', 'dev', 'ser']:
                    shape = newsrc.shape
                    logr = shape.logre
                    if min(logr - shape.getLowerBounds()[0],
                           shape.getUpperBounds()[0] - logr) < 0.01:
                        hit_r_limit = True
                        debug('Hit radius limit')

            _,ix,iy = srcwcs.radec2pixelxy(newsrc.getPosition().ra,
                                           newsrc.getPosition().dec)
            ix = int(ix-1)
            iy = int(iy-1)
            sh,sw = srcblobmask.shape
            if is_galaxy:
                # Allow (SGA) galaxies to exit the blob
                pass
            elif ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
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
                co,_ = quick_coadds(srctims, self.bands, srcwcs, images=modimgs)
                rgb = get_rgb(co, self.bands)
                model_mod_rgb[name] = rgb
                res = [(tim.getImage() - mod) for tim,mod in zip(srctims, modimgs)]
                co,_ = quick_coadds(srctims, self.bands, srcwcs, images=res)
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

            # Compute a very approximate "fracin" metric (fraction of
            # flux in masked model image versus total flux of model),
            # to avoid wild extrapolation when nearly unconstrained.
            fracin = dict([(b, []) for b in self.bands])
            fluxes = dict([(b, newsrc.getBrightness().getFlux(b))
                           for b in self.bands])
            for tim,mod in zip(srctims, srctractor.getModelImages(sky=False)):
                f = (mod * (tim.getInvError() > 0)).sum() / fluxes[tim.band]
                fracin[tim.band].append(f)
            for band in self.bands:
                if len(fracin[band]) == 0:
                    continue
                f = np.mean(fracin[band])
                if f < 1e-6:
                    debug('Source', newsrc, ': setting flux in band', band,
                          'to zero based on fracin = %.3g' % f)
                    newsrc.getBrightness().setFlux(band, 0.)

            # Compute inverse-variances
            # This uses the second-round modelMasks.
            allderivs = srctractor.getDerivs()
            ivars = _compute_invvars(allderivs)
            assert(len(ivars) == nsrcparams)

            # If any fluxes have zero invvar, zero out the flux.
            params = newsrc.getParams()
            reset = False
            for i,(pname,iv) in enumerate(zip(newsrc.getParamNames(), ivars)):
                if iv == 0:
                    debug('Zeroing out flux', pname, 'based on iv==0')
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
            print('Chisq for', name, '=', chisqs[name])
            cpum1 = time.process_time()
            B.all_model_cpu[srci][name] = cpum1 - cpum0
            cputimes[name] = cpum1 - cpum0
            B.all_model_hit_limit  [srci][name] = hit_limit
            B.all_model_hit_r_limit[srci][name] = hit_r_limit
            B.all_model_opt_steps  [srci][name] = opt_steps
            if name == 'ser':
                B.hit_ser_limit[srci] = hit_ser_limit

        if mask_others:
            for tim,ie in zip(srctims, saved_srctim_ies):
                # revert tim to original (unmasked-by-others)
                tim.inverr = ie

        # After model selection, revert the sky
        if fit_background:
            srctractor.images.setParams(skyparams)

        # Actually select which model to keep.  The MODEL_NAMES
        # array determines the order of the elements in the DCHISQ
        # column of the catalog.
        keepmod = _select_model(chisqs, nparams, galaxy_margin)
        keepsrc = {'none':None, 'psf':psf, 'rex':rex,
                   'dev':dev, 'exp':exp, 'ser':ser}[keepmod]
        bestchi = chisqs.get(keepmod, 0.)
        B.dchisq[srci, :] = np.array([chisqs.get(k,0) for k in MODEL_NAMES])
        #print('Keeping model', keepmod, '(chisqs: ', chisqs, ')')

        if keepsrc is not None and bestchi == 0.:
            # Weird edge case, or where some best-fit fluxes go
            # negative. eg
            # https://github.com/legacysurvey/legacypipe/issues/174
            debug('Best dchisq is 0 -- dropping source')
            keepsrc = None

        B.hit_limit    [srci] = B.all_model_hit_limit    [srci].get(keepmod, False)
        B.hit_r_limit  [srci] = B.all_model_hit_r_limit  [srci].get(keepmod, False)
        if keepmod != 'ser':
            B.hit_ser_limit[srci] = False

        # This is the model-selection plot
        if self.plots_per_source:
            import pylab as plt
            plt.clf()
            rows,cols = 3, 6
            modnames = ['none', 'psf', 'rex', 'dev', 'exp', 'ser']
            # Top-left: image
            plt.subplot(rows, cols, 1)
            coimgs,_ = quick_coadds(srctims, self.bands, srcwcs)
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
            if src.freezeparams:
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
        # (modelMasks sizes are determined at this point)
        models.create(self.tims, cat, subtract=True)

        # For sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
            cpu0 = time.process_time()
            src = cat[srci]
            if src.freezeparams:
                debug('Frozen source', src, '-- keeping as-is!')
                continue
            debug('Fitting source', srci, '(%i of %i in blob %s)' %
                  (numi+1, len(Ibright), self.name), ':', src)
            # Add this source's initial model back in.
            models.add(srci, self.tims)

            is_galaxy = isinstance(src, Galaxy)
            if is_galaxy:
                # During SGA pre-burns, limit initial positions (fit
                # other parameters), to avoid problems like NGC0943,
                # where one galaxy in a pair moves a large distance to
                # fit the overall light profile.
                ra,dec = src.pos.getParams()
                cosdec = np.cos(np.deg2rad(dec))
                # max allowed motion in deg
                maxmove = 5. / 3600.
                src.pos.lowers = [ra - maxmove/cosdec, dec - maxmove]
                src.pos.uppers = [ra + maxmove/cosdec, dec + maxmove]

            # FIXME -- do we need to create this local 'srctrcator' any more?
            srctims = self.tims
            modelMasks = models.model_masks(srci, src)
            srctractor = self.tractor(srctims, [src])
            srctractor.setModelMasks(modelMasks)

            # First-round optimization
            #print('First-round initial log-prob:', srctractor.getLogProb())
            srctractor.optimize_loop(**self.optargs)
            #print('First-round final log-prob:', srctractor.getLogProb())

            if is_galaxy:
                # Drop limits on SGA positions
                src.pos.lowers = [None, None]
                src.pos.uppers = [None, None]

            # Re-remove the final fit model for this source
            models.update_and_subtract(srci, src, self.tims)

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            debug('Finished fitting:', src)
            cpu1 = time.process_time()
            cputime[srci] += (cpu1 - cpu0)

        models.restore_images(self.tims)
        del models

    def _fit_fluxes(self, cat, tims, bands, fitcat=None):
        if fitcat is None:
            fitcat = [src for src in cat if not src.freezeparams]
        if len(fitcat) == 0:
            return
        for src in fitcat:
            src.freezeAllBut('brightness')
        debug('Fitting fluxes for %i of %i sources' % (len(fitcat), len(cat)))
        for b in bands:
            for src in fitcat:
                src.getBrightness().freezeAllBut(b)
            # Images for this band
            btims = [tim for tim in tims if tim.band == b]
            btr = self.tractor(btims, fitcat)
            try:
                from tractor.ceres_optimizer import CeresOptimizer
                ceres_block = 8
                btr.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
            except ImportError:
                from tractor.lsqr_optimizer import LsqrOptimizer
                btr.optimizer = LsqrOptimizer()
            btr.optimize_forced_photometry(shared_params=False, wantims=False)
        for src in fitcat:
            src.thawAllParams()

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

    def _plot_coadd(self, tims, wcs, model=None, resid=None, addnoise=False):
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
                                fill_holes=False, addnoise=addnoise)
        dimshow(get_rgb(coimgs,self.bands))

    def _initial_plots(self):
        import pylab as plt
        debug('Plotting blob image for blob', self.name)
        coimgs,_,sat = quick_coadds(self.tims, self.bands, self.blobwcs,
                                    fill_holes=False, get_saturated=True)
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

        _,x0,y0 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in self.srcs]),
            np.array([src.getPosition().dec for src in self.srcs]))

        h,w = sat.shape
        ix = np.clip(np.round(x0)-1, 0, w-1).astype(int)
        iy = np.clip(np.round(y0)-1, 0, h-1).astype(int)
        srcsat = sat[iy,ix]

        ax = plt.axis()
        plt.plot(x0-1, y0-1, 'r.', label='Sources')
        if len(srcsat):
            plt.plot(x0[srcsat]-1, y0[srcsat]-1, 'o', mec='orange', mfc='none', ms=5, mew=2,
                     label='SATUR at center')
        # ref sources
        Ir = np.flatnonzero([is_reference_source(src) for src in self.srcs])
        if len(Ir):
            plt.plot(x0[Ir]-1, y0[Ir]-1, 'o', mec='g', mfc='none', ms=8, mew=2,
                         label='Ref source')
        plt.axis(ax)
        plt.title('initial sources')
        plt.legend()
        self.ps.savefig()

def create_tims(blobwcs, blobmask, timargs):
    from legacypipe.bits import DQ_BITS
    # In order to make multiprocessing easier, the one_blob method
    # is passed all the ingredients to make local tractor Images
    # rather than the Images themselves.  Here we build the
    # 'tims'.
    tims = []
    for (img, inverr, dq, twcs, wcsobj, pcal, sky, subpsf, name,
         band, sig1, imobj) in timargs:
        # Mask out inverr for pixels that are not within the blob.
        try:
            Yo,Xo,Yi,Xi,_ = resample_with_wcs(wcsobj, blobwcs,
                                              intType=np.int16)
        except OverlapError:
            continue
        if len(Yo) == 0:
            continue
        inverr2 = np.zeros_like(inverr)
        I = np.flatnonzero(blobmask[Yi,Xi])
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
        tim.subwcs = wcsobj
        tim.meta = imobj
        tim.psf_sigma = imobj.fwhm / 2.35
        tim.dq = dq
        tim.dq_saturation_bits = DQ_BITS['satur']
        tims.append(tim)
    return tims

def _set_kingdoms(segmap, radius, I, ix, iy):
    '''
    radius: int
    ix,iy: int arrays
    I: indices into ix,iy that will be placed into 'segmap'
    '''
    # ensure that each source owns a tiny radius around its center
    # in the segmentation map.  If there is more than one source
    # in that radius, each pixel gets assigned to its nearest
    # source.
    # 'kingdom' records the current distance to nearest source
    assert(radius < 255)
    kingdom = np.empty(segmap.shape, np.uint8)
    kingdom[:,:,] = 255
    H,W = segmap.shape
    xcoords = np.arange(W)
    ycoords = np.arange(H)
    for i in I:
        x,y = ix[i], iy[i]
        yslc = slice(max(0, y-radius), min(H, y+radius+1))
        xslc = slice(max(0, x-radius), min(W, x+radius+1))
        slc = (yslc, xslc)
        # Radius to nearest earlier source
        oldr = kingdom[slc]
        # Radius to new source
        newr = np.hypot(xcoords[np.newaxis, xslc] - x, ycoords[yslc, np.newaxis] - y)
        assert(newr.shape == oldr.shape)
        newr = (newr + 0.5).astype(np.uint8)
        # Pixels that are within range and closer to this source than any other.
        owned = (newr <= radius) * (newr < oldr)
        segmap[slc][owned] = i
        kingdom[slc][owned] = newr[owned]

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
                patch = tr.getModelPatch(tim, src)
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

    assert(np.all(np.isfinite(fracflux_den)))
    assert(np.all(np.isfinite(rchi2_den)))
    assert(np.all(np.isfinite(fracmasked_den)))

    fracflux   = np.zeros_like(fracflux_num)
    rchi2      = np.zeros_like(rchi2_num)
    fracmasked = np.zeros_like(fracmasked_num)
    # Avoid divide-by-zeros (these happen when, eg, we have no coverage in one band but
    # sources detected in another band, hence denominator is zero)
    I = np.flatnonzero(fracflux_den != 0)
    fracflux.flat[I] = fracflux_num.flat[I] / fracflux_den.flat[I]
    I = np.flatnonzero(rchi2_den != 0)
    rchi2.flat[I] = rchi2_num.flat[I] / rchi2_den.flat[I]
    I = np.flatnonzero(fracmasked_den != 0)
    fracmasked.flat[I] = fracmasked_num.flat[I] / fracmasked_den.flat[I]

    # fracin_{num,den} are in flux * nimages units
    tinyflux = 1e-9
    fracin     = fracin_num     / np.maximum(tinyflux, fracin_den)

    return dict(fracin=fracin, fracflux=fracflux, rchisq=rchi2,
                fracmasked=fracmasked)

def _initialize_models(src):
    from legacypipe.survey import LogRadius
    psf = None
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
    print('_select_model: chisqs', chisqs)

    # This is our "detection threshold": 5-sigma in
    # *parameter-penalized* units; ie, ~5.2-sigma for point sources
    cut = 5.**2
    # Take the best of all models computed
    diff = max([chisqs[name] - nparams[name] for name in chisqs.keys()
                if name != 'none'] + [-1])

    print('best fit source chisq: %.3f, vs threshold %.3f' % (diff, cut))
    if diff < cut:
        # Drop this source
        return keepmod

    # Now choose between point source and REX
    if 'psf' in chisqs and (not 'rex' in chisqs) and (not 'dev' in chisqs) and (not 'exp' in chisqs) and (not 'ser' in chisqs):
        # bright stars / reference stars: we don't compute the REX or any other models.
        # We also need to check existence of the *other* models because sometimes REX can fail
        # in ways where we don't even compute a chisq (eg, source leaves blob)
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
