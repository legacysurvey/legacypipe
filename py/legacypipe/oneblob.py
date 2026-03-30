import os
import time
import signal

from collections import namedtuple

import numpy as np

from astrometry.util.ttime import Time
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow

from tractor import Tractor, PointSource, Image, Catalog, Patch, Galaxy
from tractor.galaxy import DevGalaxy, ExpGalaxy
from tractor.patch import ModelMask
from tractor.sersic import SersicGalaxy

from legacypipe.survey import (LegacyEllipseWithPriors, LegacySersicIndex,
                               RexGalaxy, get_rgb)
from legacypipe.bits import REF_MAP_BITS
from legacypipe.coadds import quick_coadds
from legacypipe.runbrick_plots import _plot_mods
from legacypipe.utils import get_cpu_arch

import logging
logger = logging.getLogger('legacypipe.oneblob')
def error(*args):
    from legacypipe.utils import log_error
    log_error(logger, args)
    import traceback
    traceback.print_exc()
def warning(*args):
    from legacypipe.utils import log_warning
    log_warning(logger, args)
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

quit_now = False

def sigusr1(sig, stackframe):
    info('SIGUSR1 was received in worker PID %i' % os.getpid())
    global quit_now
    # only raise the exception once
    if not quit_now:
        quit_now = True
        raise QuitNowException()

# similar to KeyboardInterrupt, inherit from BaseException so that
# "try:except Exception" does not catch this.
class QuitNowException(BaseException):
    pass

_last_status_update = None
def status_update(s, force=False):
    global _last_status_update
    tnow = time.time()
    if force or (_last_status_update is None) or (tnow - _last_status_update > 10):
        from legacypipe.trackingpool import update_process_status
        update_process_status(dict(type='progress', message=s))
        _last_status_update = tnow

OneBlobArgs = namedtuple('OneBlobArgs', [
    'blobname', 'nblobs', 'iblob', 'Isrcs', 'brickwcs', 'bx0', 'by0', 'blobw', 'blobh', 'blobmask',
    'timargs',
    'srcs', 'bands', 'plots', 'ps', 'reoptimize', 'iterative', 'iterative_nsigma', 'use_ceres',
    'refmap', 'large_galaxies_force_pointsource', 'less_masking', 'frozen_galaxies',
    'halfdone', 'do_segmentation', 'bright_masking', 'galaxy_masking'])

def one_blob(args):
    '''
    Fits sources contained within a "blob" of pixels.
    '''
    if args is None:
        return None

    if quit_now:
        # This happens when the pool sends us one last task after the signal has been sent.
        info('Quit_now is set; not processing blob %s' % args.blobname)
        # don't return None -- this triggers different behavior
        raise QuitNowException()

    from legacypipe.runbrick import is_gpu_worker
    is_gpu = is_gpu_worker()

    pid = os.getpid()
    info('Fitting blob %s of %i: blobid %i, nsources %i, size %i x %i, %i images, %i frozen galaxies; pid %i; is GPU? %s' %
         (args.blobname, args.nblobs, args.iblob, len(args.Isrcs), args.blobw, args.blobh, len(args.timargs),
          len(args.frozen_galaxies), pid, is_gpu))

    if len(args.timargs) == 0:
        return None
    if len(args.Isrcs) == 0:
        return None

    assert(args.blobmask.shape == (args.blobh,args.blobw))
    assert(args.refmap.shape == (args.blobh,args.blobw))

    for g in args.frozen_galaxies:
        debug('Frozen galaxy:', g)

    LegacySersicIndex.stepsize = 0.001

    if args.plots:
        import pylab as plt
        plt.figure(2, figsize=(3,3))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.figure(1)

    t0 = time.process_time()
    # A local WCS for this blob
    blobwcs = args.brickwcs.get_subimage(args.bx0, args.by0, args.blobw, args.blobh)

    oldsigusr1 = signal.SIG_DFL
    B = None
    try:
        ob = None
        oldsigusr1 = signal.signal(signal.SIGUSR1, sigusr1)

        if args.halfdone is not None:
            # We're resuming from a blob-checkpoint
            ob = args.halfdone
            B = ob.B
            del ob.B
            N = len(B.sources)
            info('Blob %s: resuming from checkpoint: done %i/%i fitting, %i/%i model sel' %
                 (args.blobname, np.sum(B.done_fitting), N, np.sum(B.done_model_selection), N))
            ob.tims = create_tims(blobwcs, args.blobmask, args.timargs)
            ob.plots = args.plots
            ob.ps = args.ps
            # FIXME -- update more parameters??
            ob.large_galaxies_force_pointsource = args.large_galaxies_force_pointsource
            # (temporary) - set defaults from older checkpoints
            if not hasattr(ob, 'segmap'):
                ob.segmap = None
            if not hasattr(ob, 'saved_segmap'):
                ob.saved_segmap = None

        else:
            # we should just make OneBlob's constructor take a OneBlobArgs object!
            ob = OneBlob(args.blobname, args.nblobs, blobwcs, args.blobmask, args.timargs, args.bands,
                         args.plots, args.ps, args.use_ceres, args.refmap,
                         args.large_galaxies_force_pointsource,
                         args.less_masking, args.frozen_galaxies,
                         args.iterative_nsigma,
                         do_segmentation=args.do_segmentation,
                         iblob=args.iblob,
                         )
            B = ob.init_table(args.srcs, args.Isrcs)

        if is_gpu:
            from tractor.gpu_optimizer import GpuOptimizer
            opt = GpuOptimizer('cupy')
        else:
            #from tractor.smarter_dense_optimizer import SmarterDenseOptimizer
            #opt = SmarterDenseOptimizer()
            # Or: use the GPU code, but with Numpy instead of Cupy.
            from tractor.gpu_optimizer import GpuOptimizer
            opt = GpuOptimizer('numpy')

        # if use_ceres:
        #     from tractor.ceres_optimizer import CeresOptimizer
        #     ceres_optimizer = CeresOptimizer()
        #     self.optargs.update(scale_columns=False,
        #                         scaled=False,
        #                         dynamic_scale=False)
        #     self.trargs.update(optimizer=ceres_optimizer)

        ob.trargs.update(optimizer=opt)

        B = ob.run(B, reoptimize=args.reoptimize, iterative_detection=args.iterative,
                   galaxy_masking=args.galaxy_masking,
                   bright_masking=args.bright_masking,
                   )
        ob.finalize_table(B, args.bx0, args.by0)

        B.iblob = args.iblob
        B.ran_on_gpu[:] = is_gpu

    except QuitNowException:
        if ob is not None:
            info('Caught QuitNowException; returning checkpoint state for blob %s' % args.blobname)
            ob.B = B
        else:
            info('Caught QuitNowException; ob None for blob %s' % args.blobname)
        return ob
    finally:
        if B is not None:
            # increment timer
            if 'cpu_blob' in B.get_columns():
                t1 = time.process_time()
                B.cpu_blob[:] += (t1 - t0)
        # revert signals
        signal.signal(signal.SIGUSR1, oldsigusr1)

    return B

class CheckStep(object):
    def __init__(self, blobwcs, blobmask):
        self.blobwcs = blobwcs
        self.blobh, self.blobw = blobwcs.shape
        self.blobmask = blobmask

    def __call__(self, tractor=None, **kwargs):
        # Returns True if the step should be accepted.
        if tractor.isParamFrozen('catalog'):
            return True
        for src in tractor.catalog:
            if src is None:
                continue
            # ConstantSurfaceBrightness
            if not hasattr(src, 'pos'):
                continue
            pos = src.pos
            ok,ix,iy = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
            if not ok:
                info('Optimizer stepped so far that WCS failed!  Source: %s' % str(src))
                return False
            ix = int(ix - 1)
            iy = int(iy - 1)
            #debug('CheckStep: coords (%i,%i) in blob sized %ix%i: %s' %
            #     (ix, iy, self.blobw, self.blobh, str(src)))
            if ix < 0 or iy < 0 or ix >= self.blobw or iy >= self.blobh:
                # stepped outside the blob rectangle!
                info('Optimizer stepped to blob coord (%i, %i) - blob size %i x %i - reject!  Source: %s'
                     % (ix, iy, self.blobw, self.blobh, str(src)))
                return False
            if not self.blobmask[iy, ix]:
                # stepped outside the blob mask!
                info('Optimizer stepped to a pixel outside the blob mask - reject!  Source: %s' %
                     (str(src)))
                return False
        return True

class OneBlob(object):
    def __init__(self, name, nblobs, blobwcs, blobmask, timargs, bands,
                 plots, ps, use_ceres, refmap,
                 large_galaxies_force_pointsource,
                 less_masking, frozen_galaxies,
                 iterative_nsigma,
                 do_segmentation=True,
                 bright_masking=False,
                 iblob=None):
        self.name = name
        self.nblobs = nblobs
        self.prefix = 'Blob %s of %s:' % (name, nblobs)
        self.iblob = iblob #blob id
        self.blobwcs = blobwcs
        self.pixscale = self.blobwcs.pixel_scale()
        self.blobmask = blobmask
        self.blobh,self.blobw = blobmask.shape
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
        self.iterative_nsigma = iterative_nsigma
        self.tims = create_tims(self.blobwcs, self.blobmask, timargs)
        self.total_pix = sum([np.sum(t.getInvError() > 0) for t in self.tims])
        self.plots2 = False
        alphas = [0.1, 0.3, 1.0]
        self.do_segmentation = do_segmentation
        self.bright_masking = bright_masking

        self.segmap = None
        # When we do iterative detection, the original segmentation map gets saved here.
        self.saved_segmap = None

        # callback function for tractor.optimize_loop: bail out if the
        # optimizer moves a source center outside the blob
        check_step = CheckStep(self.blobwcs, self.blobmask)

        self.optargs = dict(priors=True, shared_params=False, alphas=alphas,
                            print_progress=True, check_step=check_step,
                            dchisq = 0.1)
        self.trargs = dict()
        self.frozen_galaxy_mods = []

        if len(frozen_galaxies):
            self.status('Subtracting %i frozen galaxy models...' % len(frozen_galaxies))
            tr = self.tractor(self.tims, Catalog(*frozen_galaxies))
            # FIXME -- can we set *max* sizes instead?
            # mm = []
            # for tim in self.tims:
            #     mh,mw = tim.shape
            #     mm.append(dict([(g, ModelMask(0, 0, mw, mh)) for g in frozen_galaxies]))
            # tr.setModelMasks(mm)
            if self.plots:
                mods = []
            for tim in self.tims:
                try:
                    mod = tr.getModelImage(tim)
                except Exception:
                    error('Exception getting frozen-galaxies model.  Galaxies:', frozen_galaxies,
                          'Tim:', tim)
                    continue
                self.frozen_galaxy_mods.append(mod)
                tim.setImage(tim.data - mod)
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

    def __getstate__(self):
        # Remove "tims" from the pickled object
        tims = self.tims
        self.tims = None
        R = super().__getstate__()
        self.tims = tims
        return R

    def info(self, *args):
        info(self.prefix, *args)
    def debug(self, *args):
        debug(self.prefix, *args)
    def status(self, *args, force=False):
        self.debug(*args)
        status_update(self.prefix + (' ' if len(self.prefix) else '') +
                      ' '.join(str(s) for s in args), force=force)

    def init_table(self, srcs, Isrcs):
        # Per-source measurements for this blob
        B = fits_table()
        B.sources = srcs
        B.Isrcs = Isrcs
        # Did sources start within the blob?
        _,x0,y0 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in srcs]),
            np.array([src.getPosition().dec for src in srcs]))
        # blob-relative initial positions (zero-indexed)
        B.x0 = (x0 - 1.).astype(np.float32)
        B.y0 = (y0 - 1.).astype(np.float32)
        B.safe_x0 = np.clip(np.round(x0-1).astype(int), 0, self.blobw-1)
        B.safe_y0 = np.clip(np.round(y0-1).astype(int), 0, self.blobh-1)
        # This uses 'initial' pixel positions, because that's what determines
        # the fitting behaviors.
        B.started_in_blob = self.blobmask[B.safe_y0, B.safe_x0]

        N = len(srcs)
        B.done_fitting = np.zeros(N, bool)
        B.done_model_selection = np.zeros(N, bool)
        B.cpu_source    = np.zeros(N, np.float32)
        B.hit_limit     = np.zeros(N, bool)
        B.hit_ser_limit = np.zeros(N, bool)
        B.hit_r_limit   = np.zeros(N, bool)
        B.dchisq = np.zeros((N, 5), np.float32)
        B.all_models    = np.array([{} for i in range(N)])
        B.all_model_ivs = np.array([{} for i in range(N)])
        B.all_model_cpu = np.array([{} for i in range(N)])
        B.all_model_hit_limit   = np.array([{} for i in range(N)])
        B.all_model_hit_r_limit = np.array([{} for i in range(N)])
        B.all_model_opt_steps   = np.array([{} for i in range(N)])
        B.selected_model_name = np.zeros(N, 'U4')
        B.force_keep_source  = np.zeros(N, bool)
        B.fit_background     = np.zeros(N, bool)
        B.forced_pointsource = np.zeros(N, bool)
        B.blob_symm_width    = np.zeros(N, np.int16)
        B.blob_symm_height   = np.zeros(N, np.int16)
        B.blob_symm_npix     = np.zeros(N, np.int32)
        B.blob_symm_nimages  = np.zeros(N, np.int16)
        B.cpu_blob = np.zeros(N, np.float32)
        B.ran_on_gpu = np.zeros(N, bool)
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
        B.blob_x0       = np.zeros(len(B), np.int16) + bx0
        B.blob_y0       = np.zeros(len(B), np.int16) + by0
        B.blob_width    = np.zeros(len(B), np.int16) + self.blobw
        B.blob_height   = np.zeros(len(B), np.int16) + self.blobh
        B.blob_npix     = np.zeros(len(B), np.int32) + np.sum(self.blobmask)
        B.blob_nimages  = np.zeros(len(B), np.int16) + len(self.tims)
        B.blob_totalpix = np.zeros(len(B), np.int32) + self.total_pix
        B.cpu_arch = np.zeros(len(B), dtype='U3')
        B.cpu_arch[:] = get_cpu_arch()
        # Convert to whole-brick (zero-indexed) pixel positions.
        # (do this here rather than above to ease handling iterative detections)
        B.bx0 = B.x0 + bx0
        B.by0 = B.y0 + by0
        B.delete_column('x0')
        B.delete_column('y0')

    def run(self, B, reoptimize=False, iterative_detection=True,
            compute_metrics=True, mask_others=False, is_iterative=False,
            galaxy_masking=False,
            bright_masking=False,
            ):
        # The overall steps here are:
        # - fit initial fluxes for small number of sources that may need it
        # - optimize individual sources
        # - compute segmentation map
        # - model selection (including iterative detection)
        # - metrics
        self.prefix = 'Blob %s%s of %s:' % (self.name, '-iter' if is_iterative else '', self.nblobs)

        self.status('Starting: %i sources' % (len(B.sources)))
        trun = tlast = Time()
        # Not quite so many plots...
        self.plots1 = self.plots

        for src in B.sources:
            # when resuming: src can be None
            if src is None:
                continue
            # when resuming, this will already have been set, don't re-set it.
            if not hasattr(src, 'initial_brightness'):
                # Save initial fluxes for all sources (used if we force
                # keeping a reference star)
                src.initial_brightness = src.brightness.copy()

            # Set the freezeparams field for each source.  (This is set for
            # large galaxies with the 'freeze' column set.)
            src.freezeparams = getattr(src, 'freezeparams', False)

        if self.plots:
            import pylab as plt
            self._initial_plots(B.sources)
            from legacypipe.detection import plot_boundary_map
            plt.clf()
            dimshow(self.rgb)
            ax = plt.axis()
            bitset = ((self.refmap & REF_MAP_BITS['MEDIUM']) != 0)
            plot_boundary_map(bitset, rgb=(255,0,0), iterations=2)
            bitset = ((self.refmap & REF_MAP_BITS['BRIGHT']) != 0)
            plot_boundary_map(bitset, rgb=(200,200,0), iterations=2)
            bitset = ((self.refmap & REF_MAP_BITS['GALAXY']) != 0)
            plot_boundary_map(bitset, rgb=(0,255,0), iterations=2)
            plt.axis(ax)
            plt.title('Reference-source Masks')
            self.ps.savefig()

        cat = Catalog(*B.sources)
        tr = self.tractor(self.tims, cat)

        if np.any(B.done_fitting):
            self.info('Skipping fitting initial fluxes (already finished that)')
        else:
            # Fit any sources marked with 'needs_initial_flux' -- saturated, and SGA
            fitflux = [src for src in cat
                       if (src is not None and getattr(src, 'needs_initial_flux', False))]
            if len(fitflux):
                self.status('Fitting initial fluxes for %i sources' % (len(fitflux)))
                self._fit_fluxes(cat, self.tims, self.bands, fitcat=fitflux)
                if self.plots:
                    self._plots(tr, 'After fitting initial fluxes')
                for src in fitflux:
                    src.needs_initial_flux = False
            del fitflux

        if self.plots:
            self._plots(tr, 'Initial models')
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('Initial models')
            self.ps.savefig()
            # Save the initial source locations for later plotting
            _,xfit0,yfit0 = self.blobwcs.radec2pixelxy(
                np.array([src.getPosition().ra  for src in cat if src is not None]),
                np.array([src.getPosition().dec for src in cat if src is not None]))

        # Optimize individual sources

        # The sizes of the model patches fit here are determined by the
        # sources themselves, ie by the size of the mod patch returned by
        #  src.getModelPatch(tim)
        if np.all(B.done_fitting):
            self.info('Skipping fitting individual sources (already finished that)')
        else:
            self.debug('Initial fitting')
            Ibright = _argsort_by_brightness(cat, self.bands, ref_first=True)
            if len(cat) > 1:
                self._optimize_individual_sources_subtract(
                    cat, Ibright, B.cpu_source, B.done_fitting)
            else:
                self.status('Fitting source')
                self._optimize_individual_sources(tr, cat, Ibright, B.cpu_source,
                                                  B.done_fitting)

        tr.setModelMasks(None)

        if self.plots:
            self._plots(tr, 'After source fitting')
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr)
            plt.title('After source fitting')
            self.ps.savefig()
            plt.clf()
            self._plot_coadd(self.tims, self.blobwcs, model=tr, inverr_mask=False)
            plt.title('After source fitting (not masking inverr)')
            self.ps.savefig()
            # Plot source locations
            ax = plt.axis()
            goodcat = [src for src in cat if src is not None]
            _,xf,yf = self.blobwcs.radec2pixelxy(
                np.array([src.getPosition().ra  for src in goodcat]),
                np.array([src.getPosition().dec for src in goodcat]))
            plt.plot(xf-1, yf-1, 'r.', label='Sources')
            plt.plot([xfit0-1, xf-1], [yfit0-1, yf-1], 'r-')
            plt.plot(xfit0-1, yfit0-1, 'o', mec='r', mfc='none')
            Ir = np.flatnonzero([is_reference_source(src) for src in goodcat])
            if len(Ir):
                plt.plot(xf[Ir]-1, yf[Ir]-1, 'o', mec='g', mfc='none', ms=8, mew=2,
                         label='Ref source')
            plt.legend()
            plt.axis(ax)
            plt.title('After source fitting: models')
            self.ps.savefig()
            if self.plots_single:
                plt.figure(2)
                mods = list(tr.getModelImages())
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=mods,
                                        fill_holes=False)
                dimshow(get_rgb(coimgs, self.bands), ticks=False)
                plt.savefig('blob-%s-initmodel.png' % (self.name))
                res = [(tim.getImage() - mod) for tim,mod in zip(self.tims, mods)]
                coresids,_ = quick_coadds(self.tims, self.bands, self.blobwcs, images=res)
                dimshow(get_rgb(coresids, self.bands, resids=True), ticks=False)
                plt.savefig('blob-%s-initresid.png' % (self.name))
                dimshow(get_rgb(coresids, self.bands), ticks=False)
                plt.savefig('blob-%s-initsub.png' % (self.name))
                plt.figure(1)

        self.debug('Finished initial fitting: %s' % (Time()-tlast))
        tlast = Time()

        # Set any fitting behaviors based on geometric masks.

        # Fitting behaviors: force point-source
        force_pointsource_mask = (REF_MAP_BITS['BRIGHT'] |
                                  REF_MAP_BITS['CLUSTER'] |
                                  REF_MAP_BITS['MCLOUDS'])
        # large_galaxies_force_pointsource is True by default.
        if self.large_galaxies_force_pointsource:
            force_pointsource_mask |= REF_MAP_BITS['GALAXY']
        # Fit background?
        fit_background_mask = REF_MAP_BITS['BRIGHT']
        if not self.less_masking:
            fit_background_mask |= REF_MAP_BITS['MEDIUM']
        # (we used to only turn this on if large_galaxies_force_pointsource)
        fit_background_mask |= REF_MAP_BITS['GALAXY']

        for srci,src in enumerate(cat):
            if src is None:
                continue
            pos = src.getPosition()
            ok,ix,iy = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
            if not ok:
                info('Source', src, 'has RA,Dec', pos, 'so weird that WCS fails!  Ignoring.')
                cat[srci] = None
                continue
            ix = int(np.clip(ix-1, 0, self.blobw-1))
            iy = int(np.clip(iy-1, 0, self.blobh-1))
            bits = self.refmap[iy, ix]
            force_pointsource = ((bits & force_pointsource_mask) > 0)
            fit_background = ((bits & fit_background_mask) > 0)
            is_galaxy = isinstance(src, Galaxy)

            # SGA sources that come in should never get forced to point-source via masks.
            if is_galaxy:
                fit_background = False
                force_pointsource = False
            B.forced_pointsource[srci] = force_pointsource
            B.fit_background[srci] = fit_background
            # Also set a parameter on 'src' for use in compute_segmentation_map()
            src.maskbits_forced_point_source = force_pointsource

        if not np.all(B.done_fitting):
            if self.do_segmentation:
                self.status('Computing segmentation map')
                self.segmap = self.compute_segmentation_map(cat)
                mask_others = False #self.do_segmentation
            # Next, model selections: point source vs rex vs dev/exp vs ser.
            self.debug('Starting model selection')
            Ibright = _argsort_by_brightness(cat, self.bands, ref_first=True)
            B = self.run_model_selection(cat, Ibright, B, segmap,
                                         iterative_detection=iterative_detection,
                                         galaxy_masking=galaxy_masking,
                                         bright_masking=bright_masking,
                                         mask_others=mask_others)
            self.debug('Finished model selection: %s' % (Time()-tlast))
        tlast = Time()
        self.segmap = None

        # Cut down to just the kept sources
        # note that "B" is returned by run_model_selection -- may be a new table with larger
        # size thanks to iterative detection.
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
            fakedone = np.zeros(len(B), bool)
            if len(cat) > 1:
                self._optimize_individual_sources_subtract(
                    cat, Ibright, B.cpu_source, fakedone)
            else:
                self._optimize_individual_sources(tr, cat, Ibright, B.cpu_source, fakedone)

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
            self.status('Computing metrics')
            # Compute variances on all parameters for the kept model
            B.srcinvvars = [None for i in range(len(B))]
            cat.thawAllRecursive()
            cat.freezeAllParams()
            for isub in range(len(B.sources)):
                if (isub+1)%1000 == 0:
                    self.status('Computing metrics: source %i of %i' % (isub+1, len(B.sources)))
                cat.thawParam(isub)
                src = cat[isub]
                if src is None:
                    cat.freezeParam(isub)
                    continue
                if getattr(src, 'freezeparams', False):
                    # SGA frozen-sources.  Do these get uncertainties measured?
                    _convert_ellipses(src)
                nsrcparams = src.numberOfParams()
                if B.force_keep_source[isub]:
                    B.srcinvvars[isub] = np.zeros(nsrcparams, np.float32)
                    cat.freezeParam(isub)
                    continue
                # Compute inverse-variances.  (We compute
                # inverse-variances during model selection, but that
                # is on a subset of pixels (segmentation, model masks,
                # etc).
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

            M = _compute_source_metrics(B.sources, self.tims, self.bands, tr, self.status)
            for k,v in M.items():
                B.set(k, v)
            del M

        self.info('Finished, total: %s' % (Time()-trun))
        self.status('Finished', force=True)
        return B

    def compute_segmentation_map(self, cat):
        from functools import reduce
        from legacypipe.detection import detection_maps
        from astrometry.util.multiproc import multiproc
        from scipy.ndimage import binary_dilation

        # Compute per-band detection maps
        mp = multiproc()
        detmaps,detivs,satmaps = detection_maps(self.tims, self.blobwcs, self.bands, mp)
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

        Iseg = np.flatnonzero([src is not None for src in cat])
        ok,ix,iy = self.blobwcs.radec2pixelxy(
            np.array([cat[i].getPosition().ra  for i in Iseg]),
            np.array([cat[i].getPosition().dec for i in Iseg]))
        Iseg = Iseg[ok]
        ix = ix[ok]
        iy = iy[ok]
        del ok
        assert(len(Iseg) == len(ix))
        ix = np.clip(np.round(ix)-1, 0, self.blobw-1).astype(np.int32)
        iy = np.clip(np.round(iy)-1, 0, self.blobh-1).astype(np.int32)

        # Do not compute segmentation map for sources in the CLUSTER mask
        I = ((self.refmap[iy, ix] & REF_MAP_BITS['CLUSTER']) == 0)
        Iseg = Iseg[I]
        ix = ix[I]
        iy = iy[I]
        del I

        # Zero out the S/N in CLUSTER mask
        maxsn[(self.refmap & REF_MAP_BITS['CLUSTER']) > 0] = 0.
        # (also zero out the satmap in the CLUSTER mask)
        saturated_pix[(self.refmap & REF_MAP_BITS['CLUSTER']) > 0] = False

        import heapq
        H,W = self.blobh, self.blobw
        segmap = np.empty((H,W), np.int32)
        segmap[:,:] = -1
        # Iseg are the indices in cat of sources to segment, sx,sy their coordinates
        sy = iy
        sx = ix
        segmap[sy, sx] = Iseg
        maxr2 = np.zeros(len(Iseg), np.int32)
        # Reference sources forced to be point sources get a max radius:
        ref_radius = 25
        for j,i in enumerate(Iseg):
            if getattr(cat[i], 'forced_point_source', False):
                maxr2[j] = ref_radius**2
            # Sources inside maskbits masks that are forced to be point sources
            # also get a max radius.
            if getattr(cat[i], 'maskbits_forced_point_source', False):
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
        radius = 10
        Ibright = _argsort_by_brightness([cat[i] for i in Iseg], self.bands)
        _set_kingdoms(segmap, radius, Iseg[Ibright], ix[Ibright], iy[Ibright])

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
            for i in range(len(cat)):
                plot_boundary_map(segmap == i)
            plt.plot(ix, iy, 'r.')
            plt.axis(ax)
            plt.title('Segments')
            self.ps.savefig()

        return segmap

    def run_model_selection(self, cat, Ibright, B, segmap,
                            iterative_detection=True,
                            galaxy_masking=False,
                            bright_masking=False,
                            mask_others=False):
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

        brightmap = None
        # Mask other blobs of bright pixels while fitting a source in a bright blob.
        if bright_masking and len(cat) > 1:
            from legacypipe.coadds import make_coadds
            from scipy.ndimage import label, binary_dilation, binary_fill_holes
            brightmap = np.zeros(self.blobwcs.shape, bool)
            co = make_coadds(self.tims, self.bands, self.blobwcs,
                             allmasks=False, mjdminmax=False)
            for im,iv in zip(co.coimgs, co.cowimgs):
                sn = im * np.sqrt(iv)
                brightmap |= (sn > 10.)
            brightmap = binary_dilation(brightmap, iterations=2)
            # fill holes for, eg, bright stars with saturated cores.
            # Should we explicitly fill SATUR?
            brightmap = binary_fill_holes(brightmap)
            brightmap,_ = label(brightmap)

            if self.plots:
                import pylab as plt
                plt.clf()
                plt.imshow(brightmap > 0, interpolation='nearest', origin='lower', vmin=0, vmax=1,
                           cmap='gray')
                plt.title('Brightmap')
                self.ps.savefig()

        # SGA/large-galaxy segmentation
        gal_segmap = None
        if galaxy_masking:
            # Find galaxies
            gal_inds = []
            gal_ix,gal_iy = [],[]
            for srci,src in enumerate(cat):
                if src is None:
                    continue
                is_galaxy = isinstance(src, Galaxy)
                if not is_galaxy:
                    continue
                pos = src.getPosition()
                _,ix,iy = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
                ix = int(np.clip(ix-1, 0, self.blobw-1))
                iy = int(np.clip(iy-1, 0, self.blobh-1))
                gal_inds.append(srci)
                gal_ix.append(ix)
                gal_iy.append(iy)
            # Only create galaxy segmentation map if >1 galaxies
            if len(gal_inds) > 1:
                gal_segmap = np.empty((self.blobh,self.blobw), np.int32)
                gal_segmap[:,:] = -1
                _set_kingdoms(gal_segmap, 65535, gal_inds, gal_ix, gal_iy)
            del gal_inds, gal_ix, gal_iy

            if self.plots and (gal_segmap is not None):
                import pylab as plt
                plt.clf()
                dimshow(gal_segmap)
                ax = plt.axis()
                from legacypipe.detection import plot_boundary_map
                plot_boundary_map(gal_segmap >= 0)
                plt.plot(ix, iy, 'r.')
                plt.axis(ax)
                plt.title('Galaxy segmentation map')
                self.ps.savefig()

        models = SourceModels()
        # Remember original tim images
        models.save_images(self.tims)

        # Create initial models for each tim x each source
        # (model sizes are determined at this point)
        self.debug('Creating (& subtracting) initial models for model selection...')
        models.create(self.tims, cat, subtract=True)

        # Model selection for sources, in decreasing order of brightness
        for numi,srci in enumerate(Ibright):
            if B.done_model_selection[srci]:
                self.debug('Already done model selection for srci %i' % srci)
                continue

            src = cat[srci]
            self.status('Model selection for source %i of %i' % (numi+1, len(Ibright)))
            self.debug('Initial model:', str(src))

            cpu0 = time.process_time()

            if src.freezeparams:
                self.debug('Frozen source, keeping as-is: %s' % (str(src)))
                B.sources[srci] = src
                B.done_model_selection[srci] = True
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
            pre = self.prefix
            self.prefix = '%s source %i of %i model sel' % (pre, numi+1, len(Ibright))
            plots = self.plots_per_source * (numi < 50)
            keepsrc = self.model_selection_one_source(src, srci, models, B, segmap,
                                                      mask_others=mask_others,
                                                      brightmap=brightmap,
                                                      gal_segmap=gal_segmap,
                                                      plots=plots)
            self.prefix = pre

            # Definitely keep ref stars (Gaia & Tycho)
            if keepsrc is None and getattr(src, 'reference_star', False):
                self.debug('Reference star would have been dropped: resetting brightness: %s' %
                           (str(src)))
                ## FIXME?  This initial_brightness is *before* even our initial round of fitting.
                ## FIXME?  Do we really want to subtract the *initial* brightness in update_and_subtract?
                src.brightness = src.initial_brightness
                src.force_keep_source = True
                keepsrc = src

            B.sources[srci] = keepsrc
            cat[srci] = keepsrc
            B.force_keep_source[srci] = getattr(keepsrc, 'force_keep_source', False)

            models.update_and_subtract(srci, keepsrc, self.tims)

            if self.plots_single:
                plt.figure(2)
                coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                           fill_holes=False)
                dimshow(get_rgb(coimgs,self.bands), ticks=False)
                plt.savefig('blob-%s-%i-sub.png' % (self.name, srci))
                plt.figure(1)
            if self.plots and (numi % 100 == 99):
                import pylab as plt
                plt.clf()
                tr = self.tractor(self.tims, cat)
                self._plot_coadd(self.tims, self.blobwcs, model=tr)
                del tr
                plt.title('Model selection after %i sources' % (numi+1))
                self.ps.savefig()

            cpu1 = time.process_time()
            B.cpu_source[srci] += (cpu1 - cpu0)
            B.done_model_selection[srci] = True

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

            oldprefix = self.prefix
            Bnew = self.iterative_detection(B, models)
            self.prefix = oldprefix

            if Bnew is not None:
                # B.sources is a list of objects... merge() with
                # fillzero doesn't handle them well.
                srcs = B.sources
                newsrcs = Bnew.sources
                B.delete_column('sources')
                Bnew.delete_column('sources')
                B = merge_tables([B, Bnew], columns='fillzero')
                B.sources = srcs + newsrcs
                del srcs, newsrcs
            del Bnew
        models.restore_images(self.tims)
        del models
        return B

    def iterative_detection(self, Bold, models):
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
        detmaps,detivs,satmaps = detection_maps(self.tims, self.blobwcs, self.bands, mp)

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
            tim.setImage(modimg)
        if self.plots:
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs,
                                    fill_holes=False)
            import pylab as plt
            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.title('Iterative detection: first-round models')
            self.ps.savefig()

        mod_detmaps,mod_detivs,_ = detection_maps(self.tims, self.blobwcs, self.bands, mp)
        # revert the tim image data
        for tim,img in zip(self.tims, realimages):
            tim.setImage(img)

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
        # magic number 4: matching r_excl in runbrick.py
        avoid_r = np.zeros(len(avoid_x), np.float32) + 4.
        avoid_map = (self.refmap != 0)

        Tnew, newsrcs, _ = run_sed_matched_filters(
            SEDs, self.bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r),
            self.blobwcs, nsigma=self.iterative_nsigma, saturated_pix=satmaps,
            veto_map=avoid_map, plots=False, ps=None, mp=mp)

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
                     mfc='none', label='Avoid (r=4)')
            plt.plot(xx-1, yy-1, 'r+', label='Old', **crossa)
            plt.axis(ax)
            plt.legend()
            plt.title('Iterative detections (avoid)')
            self.ps.savefig()

            plt.clf()
            plt.imshow(avoid_map, origin='lower', interpolation='nearest', vmin=0, vmax=1)
            plt.title('Iterative detection: avoid map')
            self.ps.savefig()

            plt.clf()
            dimshow(get_rgb(coimgs,self.bands), ticks=False)
            plt.plot(Bold.safe_x0, Bold.safe_y0, 'o', ms=5, mec='r',
                     mfc='none', label='Avoid (r=4)')
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
        keep = (det_max > B * np.maximum(mod_max, 1.))
        Tnew.cut(keep)
        debug('Cut to', len(Tnew), 'iterative sources compared to model detection map')
        if len(Tnew) == 0:
            return None
        newsrcs = [newsrcs[i] for i,k in enumerate(keep) if k]
        assert(len(Tnew) == len(newsrcs))

        self.info('Measuring %i iterative sources' % (len(Tnew)))

        isrcs = np.empty(len(newsrcs), np.int32)
        isrcs[:] = -1
        Bnew = self.init_table(newsrcs, isrcs)
        Bnew.x0 = Tnew.ibx.astype(np.float32)
        Bnew.y0 = Tnew.iby.astype(np.float32)
        # Be quieter during iterative detection!
        bloblogger = logging.getLogger('legacypipe.oneblob')
        loglvl = bloblogger.getEffectiveLevel()
        bloblogger.setLevel(loglvl + 10)

        # Run the whole oneblob pipeline on the iterative sources!
        self.saved_segmap = self.segmap
        self.segmap = None
        Bnew = self.run(Bnew, iterative_detection=False, compute_metrics=False,
                        mask_others=False, is_iterative=True)
        self.segmap = self.saved_segmap
        self.saved_segmap = None

        # revert
        bloblogger.setLevel(loglvl)

        if len(Bnew) == 0:
            return None
        return Bnew

    def model_selection_one_source(self, src, srci, models, B, segmap,
                                   mask_others=False,
                                   brightmap=None,
                                   gal_segmap=None,
                                   plots=False):
        blob_modelMasks = models.model_masks(srci, src)

        # Create tiny local tims corresponding to the modelMasks.
        srctims = []
        srcmm = []
        totalpix = 0
        for tim_mm, tim in zip(blob_modelMasks, self.tims):
            if not src in tim_mm:
                continue
            mm = tim_mm[src]
            x0,x1,y0,y1 = mm.extent
            slc = slice(y0, y1), slice(x0, x1)
            ie = tim.getInvError()[slc]
            if np.all(ie == 0):
                continue
            totalpix += np.sum(ie > 0)
            subtim = Image(data=tim.getImage()[slc],
                           inverr=ie,
                           wcs=tim.getWcs().shifted(x0, y0),
                           sky=tim.getSky().shifted(x0, y0),
                           psf=tim.getPsf().constantPsfAt((x0+x1-1)/2, (y0+y1-1)/2),
                           photocal=tim.getPhotoCal(),
                           name=tim.name,
                           )
            subtim.subwcs = tim.subwcs.get_subimage(x0, y0, x1-x0, y1-y0)
            subtim.sig1 = tim.sig1
            subtim.band = tim.band
            subtim.meta = tim.meta
            subtim.psf_sigma = tim.psf_sigma
            if tim.dq is not None:
                subtim.dq = tim.dq[slc]
            subtim.dq_saturation_bits = tim.dq_saturation_bits
            srctims.append(subtim)
            srcmm.append(dict({src: ModelMask(0, 0, x1-x0, y1-y0)}))
        modelMasks = srcmm
        del srcmm
        if len(srctims) == 0:
            debug('No images overlap source:', src)
            return None

        # Coordinates within the blob of the source mask
        sx0,sx1,sy0,sy1 = model_masks_to_blob_extent(srctims, modelMasks, src, self.blobwcs, to_int=True)
        srcwcs = self.blobwcs.get_subimage(sx0, sy0, sx1-sx0, sy1-sy0)
        srcblobmask = self.blobmask[sy0:sy1, sx0:sx1]
        sh,sw = srcblobmask.shape

        pos = src.getPosition()
        _,xx,yy = srcwcs.radec2pixelxy(pos.ra, pos.dec)
        ix = int(np.clip(np.round(xx-1), 0, sw-1))
        iy = int(np.clip(np.round(yy-1), 0, sh-1))
        in_bounds = (xx > -0.5) and (yy > -0.5) and (xx < sw-0.5) and (yy < sh-0.5)

        # an extra mask to apply, in srcblobwcs shape
        source_mask = None

        if mask_others:
            from legacypipe.detection import detection_maps
            from astrometry.util.multiproc import multiproc
            from scipy.ndimage import binary_dilation, binary_fill_holes
            from scipy.ndimage.measurements import label
            # Compute per-band detection maps
            mp = multiproc()
            detmaps,detivs,_ = detection_maps(srctims, srcwcs, self.bands, mp)
            # Compute the symmetric area that fits in this 'srcblobmask' region
            flipw = min(ix, sw-1-ix)
            fliph = min(iy, sh-1-iy)
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

            if plots:
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
                if segmap is not None:
                    dh,dw = flipblobs.shape
                    mysegmap = segmap[sy0:sy0+dh, sx0:sx0+dw]
                    # renumber for plotting
                    _,S = np.unique(mysegmap, return_inverse=True)
                    dimshow(S.reshape(mysegmap.shape), cmap='tab20',
                            interpolation='nearest', origin='lower')
                    ax = plt.axis()
                    plt.plot(ix, iy, 'kx', ms=15, mew=3)
                    plt.axis(ax)
                    plt.title('Segmentation map')
                else:
                    plt.title('(No segmentation map)')

                plt.subplot(2,2,4)
                dilated = binary_dilation(flipblobs, iterations=4)
                if segmap is None:
                    s = -1
                else:
                    s = segmap[iy + sy0, ix + sx0]
                if s != -1:
                    dilated *= (segmap[sy0:sy0+dh, sx0:sx0+dw] == s)
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
                if has_fixed_position(src):
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

            source_mask = dilated

        if segmap is not None:
            # Segmap is the size of the full blob, so apply a pixel offset
            s = segmap[iy + sy0, ix + sx0]
            if s != -1:
                if source_mask is None:
                    source_mask = srcblobmask & (segmap[sy0:sy0+sh, sx0:sx0+sw] == s)
                else:
                    source_mask &= (segmap[sy0:sy0+sh, sx0:sx0+sw] == s)

        is_galaxy = isinstance(src, Galaxy)
        if is_galaxy and (gal_segmap is not None):
            s = gal_segmap[iy + sy0, ix + sx0]
            if s != -1:
                if source_mask is None:
                    source_mask = srcblobmask & (gal_segmap[sy0:sy0+sh, sx0:sx0+sw] == s)
                else:
                    source_mask &= (gal_segmap[sy0:sy0+sh, sx0:sx0+sw] == s)

        if (brightmap is not None) and in_bounds:
            s = brightmap[iy + sy0, ix + sx0]
            if s == 0:
                # The current source is not in a bright blob.
                # Mask out all pixels in bright blobs
                brmask = (brightmap[sy0:sy0+sh, sx0:sx0+sw] == 0)
            else:
                # The current source is in a bright blob.
                # Mask out all pixels in *other* bright blobs
                brmask = ((brightmap[sy0:sy0+sh, sx0:sx0+sw] == 0) |
                          (brightmap[sy0:sy0+sh, sx0:sx0+sw] == s))
            if source_mask is None:
                source_mask = srcblobmask & brmask
            else:
                source_mask &= brmask

        if source_mask is not None:
            if not np.any(source_mask):
                debug('No pixels in single-source mask')
                return None

            keep_srctims = []
            keep_mm = []
            totalpix = 0
            for tim,tim_mm in zip(srctims, modelMasks):
                # Zero out inverse-errors for all pixels in the modelmask but
                # with source_mask=0.
                mm = tim_mm.get(src)
                if mm is None:
                    continue
                x0,x1,y0,y1 = mm.extent
                mmwcs = tim.subwcs.get_subimage(x0, y0, x1-x0, y1-y0)
                try:
                    Yo,Xo,Yi,Xi,_ = resample_with_wcs(mmwcs, srcwcs)
                except OverlapError:
                    continue
                ie = tim.getInvError()
                newie = np.zeros_like(ie)
                good, = np.nonzero(source_mask[Yi,Xi] * (ie[y0+Yo,x0+Xo] > 0))
                if len(good) == 0:
                    #debug('Tim has inverr all == 0')
                    continue
                yy = Yo[good]
                xx = Xo[good]
                newie[y0+yy,x0+xx] = ie[y0+yy,x0+xx]
                totalpix += len(xx)
                tim.setInvError(newie)
                keep_srctims.append(tim)
                keep_mm.append(tim_mm)
            srctims = keep_srctims
            modelMasks = keep_mm
            del keep_srctims, keep_mm

        B.blob_symm_nimages[srci] = len(srctims)
        B.blob_symm_npix   [srci] = totalpix
        B.blob_symm_width  [srci] = sw
        B.blob_symm_height [srci] = sh

        if plots:
            # This is a handy blob-coordinates plot of the data
            # going into the fit.
            import pylab as plt
            plt.clf()
            _,_,coimgs,_ = quick_coadds(srctims, self.bands, self.blobwcs,
                                        fill_holes=False, get_cow=True)
            dimshow(get_rgb(coimgs, self.bands))
            ax = plt.axis()
            pos = src.getPosition()
            _,x,y = self.blobwcs.radec2pixelxy(pos.ra, pos.dec)
            ix,iy = int(np.round(x-1)), int(np.round(y-1))
            plt.plot(x-1, y-1, 'r+', ms=10)
            ex0,ex1,ey0,ey1 = model_masks_to_blob_extent(srctims, modelMasks, src, self.blobwcs)
            plt.plot([ex0,ex0,ex1,ex1,ex0], [ey0,ey1,ey1,ey0,ey0], 'r-')
            plt.axis(ax)
            plt.title('Model selection: data')
            self.ps.savefig()

        is_galaxy = isinstance(src, Galaxy)
        force_pointsource = B.forced_pointsource[srci]
        fit_background = B.fit_background[srci]

        # Use per-image sky background level
        fit_sb = False
        fit_sky = fit_background

        #fit_sb = fit_background
        #fit_sky = False

        if fit_sb:
            # Fit the source + a constant surface brightness with the same bands as the source.
            from tractor.basics import ConstantSurfaceBrightness
            br = src.getBrightness().copy()
            br.setParams(np.zeros(br.numberOfParams()))
            sb = ConstantSurfaceBrightness(br)
            srctractor = self.tractor(srctims, [src, sb])
        else:
            srctractor = self.tractor(srctims, [src])
        srctractor.setModelMasks(modelMasks)
        srccat = srctractor.getCatalog()

        _,ix,iy = srcwcs.radec2pixelxy(src.getPosition().ra,
                                       src.getPosition().dec)
        ix = int(ix-1)
        iy = int(iy-1)
        sh,sw = srcwcs.shape
        optargs = self.optargs.copy()
        if is_galaxy:
            # allow SGA galaxy sources to start outside the blob
            optargs.update(check_step=None)
        elif has_fixed_position(src):
            # eg, Gaia sources - positions fixed, so no need to check
            optargs.update(check_step=None)
        elif ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
            debug('Source is starting outside blob -- skipping.')
            return None

        if is_galaxy:
            # SGA galaxy: set the maximum allowed r_e.
            known_galaxy_logrmax = 0.
            if isinstance(src, (DevGalaxy,ExpGalaxy, SersicGalaxy)):
                debug('Known galaxy.  Initial shape:', src.shape)
                # MAGIC 2. = factor by which r_e is allowed to grow for an SGA galaxy.
                known_galaxy_logrmax = np.log(src.shape.re * 2.)
            else:
                warning('WARNING: unknown galaxy type:', src)

        debug(('Source at blob coordinates %i,%i, local source coords %i,%i of %ix%i; ' +
               'forcing pointsource? %s, is large galaxy? %s, fitting sky background: %s') %
              (sx0+ix, sy0+iy, ix, iy, sw, sh, force_pointsource, is_galaxy, fit_background))

        opt = srctractor.optimizer
        opt.cache_image_params(srctractor)

        # Compute the log-likehood without a source here.
        srccat[0] = None

        if fit_sb:
            mm = remap_modelmask(modelMasks, src, srccat[1])
            srctractor.setModelMasks(mm)
            srctractor.optimize_loop(**optargs)
            # Save the const sb levels fit with no source?  or just zero?
            skyparams = srccat[1].getParams()
            debug('Fit background with no source: sb', srccat[1])
        if fit_sky:
            for tim in srctims:
                tim.freezeAllBut('sky')
            srctractor.thawParam('images')
            initial_skyparams = srctractor.images.getParams()
            # When we're fitting the background, using the sparse optimizer is critical
            # when we have a lot of images: we're adding Nimages extra parameters, touching
            # every pixel; you don't want Nimages x Npixels dense matrices.
            from tractor.lsqr_optimizer import LsqrOptimizer
            srctractor.optimizer = LsqrOptimizer()
            srctractor.optimize_loop(**optargs)
            skyparams = srctractor.images.getParams()
            debug('Fitting skies with no source:', skyparams)

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
                # This is set in the GaiaSource contructor from gaia.pointsource
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
                        dev.soft_shape.copy(), LegacySersicIndex(4.))
                elif smod == 'exp':
                    newsrc = ser = SersicGalaxy(
                        exp.getPosition().copy(), exp.getBrightness().copy(),
                        exp.soft_shape.copy(), LegacySersicIndex(1.))

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
                logrmax = np.log(fblob * max(sh, sw) * self.pixscale)
                if name in ['rex', 'exp', 'dev', 'ser']:
                    if logrmax < newsrc.shape.getMaxLogRadius():
                        newsrc.shape.setMaxLogRadius(logrmax)

            # Use the same modelMask shapes as the original source ('src').
            # Need to create newsrc->mask mappings though:
            if fit_sb:
                mm = remap_modelmasks(modelMasks, src, [newsrc, srccat[1]])
            else:
                mm = remap_modelmask(modelMasks, src, newsrc)
            srctractor.setModelMasks(mm)

            if fit_sb:
                # # Reset sky params
                srccat[1].setParams(skyparams)
            if fit_sky:
                srctractor.images.setParams(skyparams)

            # First-round optimization (during model selection)
            self.debug('Before model selection: %s' % (str(newsrc)))

            # Fit just the fluxes first...
            newsrc.freezeAllBut('brightness')
            srctractor.optimize_loop(**optargs)
            self.debug('After model selection (just fluxes): %s' % (str(newsrc)))
            newsrc.thawAllParams()

            try:
                R = srctractor.optimize_loop(**optargs)
            except Exception as e:
                error('Exception fitting source in model selection.  src:', newsrc)
                raise(e)
            self.debug('After  model selection: %s' % (str(newsrc)))
            hit_limit = R.get('hit_limit', False)
            opt_steps = R.get('steps', -1)
            hit_ser_limit = False
            hit_r_limit = False
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
            if is_galaxy:
                # Allow (SGA) galaxies to exit the blob
                pass
            elif is_reference_source(src):
                # and Gaia stars
                pass
            elif ix < 0 or iy < 0 or ix >= sw or iy >= sh or not srcblobmask[iy,ix]:
                # Exited blob!
                debug('Source exited sub-blob!')
                continue

            if plots:
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

            if fit_sb:
                # We have to freeze the sky here before computing
                # uncertainties
                srccat.freezeParam(1)
            if fit_sky:
                srctractor.freezeParam('images')

            # Convert to "vanilla" ellipse parameterization
            # (but save old shapes first)
            if isinstance(newsrc, (DevGalaxy, ExpGalaxy, SersicGalaxy)):
                newsrc.soft_shape = newsrc.shape
            nsrcparams = newsrc.numberOfParams()
            _convert_ellipses(newsrc)
            assert(newsrc.numberOfParams() == nsrcparams)

            # Compute a very approximate "fracin" metric (fraction of
            # flux in masked model image versus total flux of model),
            # to avoid wild extrapolation when nearly unconstrained.
            fracin = dict([(b, []) for b in self.bands])
            fluxes = dict([(b, newsrc.getBrightness().getFlux(b))
                           for b in self.bands])

            if fit_sb:
                # set the SB to zero before computing model metrics
                sb_fit = srccat[1].getParams()
                srccat[1].setParams(np.zeros(len(sb_fit)))

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
            allderivs = srctractor.getDerivs()
            ivars = _compute_invvars(allderivs)
            assert(len(ivars) == nsrcparams)

            # If any fluxes have zero invvar, zero out the flux.
            params = newsrc.getParams()
            reset = False
            for i,(pname,iv) in enumerate(zip(newsrc.getParamNames(), ivars)):
                # ugly...
                if iv == 0 and 'brightness' in pname:
                    debug('Zeroing out flux', pname, 'based on iv==0')
                    params[i] = 0.
                    reset = True
            if reset:
                newsrc.setParams(params)
                debug('Recomputing ivars with source parameters', newsrc)
                allderivs = srctractor.getDerivs()
                ivars = _compute_invvars(allderivs)
                assert(len(ivars) == nsrcparams)

            B.all_model_ivs[srci][name] = np.array(ivars).astype(np.float32)
            B.all_models[srci][name] = newsrc.copy()
            assert(B.all_models[srci][name].numberOfParams() == nsrcparams)

            if fit_sb:
                # Turn the background back on before measuring chi-sq
                srccat[1].setParams(sb_fit)

            ch = _per_band_chisqs(srctractor, self.bands)
            chisqs[name] = _chisq_improvement(newsrc, ch, chisqs_none)
            cpum1 = time.process_time()
            B.all_model_cpu        [srci][name] = cpum1 - cpum0
            B.all_model_hit_limit  [srci][name] = hit_limit
            B.all_model_hit_r_limit[srci][name] = hit_r_limit
            B.all_model_opt_steps  [srci][name] = opt_steps
            if name == 'ser':
                B.hit_ser_limit[srci] = hit_ser_limit

            # thaw the sky models for fitting the next model
            if fit_sb:
                srccat.thawParam(1)
            if fit_sky:
                srctractor.thawParam('images')

        opt.clear_cached_image_params()

        # Actually select which model to keep.  The MODEL_NAMES
        # array determines the order of the elements in the DCHISQ
        # column of the catalog.
        keepmod = _select_model(chisqs, nparams, galaxy_margin)
        keepsrc = {'none':None, 'psf':psf, 'rex':rex,
                   'dev':dev, 'exp':exp, 'ser':ser}[keepmod]
        bestchi = chisqs.get(keepmod, 0.)
        B.dchisq[srci, :] = np.array([chisqs.get(k,0) for k in MODEL_NAMES])
        B.selected_model_name[srci] = keepmod

        if keepsrc is not None and bestchi == 0.:
            # Weird edge case, or where some best-fit fluxes go
            # negative. eg
            # https://github.com/legacysurvey/legacypipe/issues/174
            debug('Best dchisq is 0 -- dropping source')
            keepsrc = None

        B.hit_limit  [srci] = B.all_model_hit_limit  [srci].get(keepmod, False)
        B.hit_r_limit[srci] = B.all_model_hit_r_limit[srci].get(keepmod, False)
        if keepmod != 'ser':
            B.hit_ser_limit[srci] = False

        if fit_sky:
            # Revert sky params back the way we found them.
            srctractor.images.setParams(initial_skyparams)

        # This is the model-selection plot
        if plots:
            import pylab as plt
            plt.clf()
            rows,cols = 3, 6
            modnames = ['none', 'psf', 'rex', 'dev', 'exp', 'ser']
            # Top-left: image in blob coords
            plt.subplot(rows, cols, 1)
            #coimgs,_ = quick_coadds(srctims, self.bands, self.blobwcs)
            coimgs,_ = quick_coadds(self.tims, self.bands, self.blobwcs)
            rgb = get_rgb(coimgs, self.bands)
            dimshow(rgb, ticks=False)
            if ey0 is not None:
                ax = plt.axis()
                plt.plot(np.array([ex0,ex0,ex1,ex1,ex0]), np.array([ey0,ey1,ey1,ey0,ey0]), 'r-')
                plt.axis(ax)
            # next: src coords
            plt.subplot(rows, cols, 2)

            coimgs,_ = quick_coadds(srctims, self.bands, srcwcs)
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

    def _optimize_individual_sources(self, tr, cat, Ibright, cputime, done_fitting):
        # Single source (though this is coded to handle multiple sources)
        # Fit sources one at a time, but don't subtract other models
        cat.freezeAllParams()

        models = SourceModels()
        models.create(self.tims, cat)

        for i in Ibright:
            if done_fitting[i]:
                debug('Already done fitting sourcei', i)
                continue
            cpu0 = time.process_time()
            cat.freezeAllBut(i)
            src = cat[i]
            if src.freezeparams:
                debug('Frozen source', src, '-- keeping as-is!')
                done_fitting[i] = True
                continue
            modelMasks = models.model_masks(i, src)
            tr.setModelMasks(modelMasks)

            opt = tr.optimizer
            opt.cache_image_params(tr)

            tr.optimize_loop(**self.optargs)

            opt.clear_cached_image_params()

            cpu1 = time.process_time()
            cputime[i] += (cpu1 - cpu0)
            done_fitting[i] = True

        tr.setModelMasks(None)

    def tractor(self, tims, cat):
        tr = Tractor(tims, cat, **self.trargs)
        tr.freezeParams('images')
        return tr

    def _optimize_individual_sources_subtract(self, cat, Ibright,
                                              cputime, done_fitting):
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
            if done_fitting[srci]:
                self.debug('Already done fitting sourcei %i' % (srci))
                continue

            cpu0 = time.process_time()
            src = cat[srci]
            if src.freezeparams:
                debug('Frozen source', src, '-- keeping as-is!')
                done_fitting[srci] = True
                continue
            self.status('Fitting source %i of %i' % (numi+1, len(Ibright)))

            modelMasks = models.model_masks(srci, src)
            # sub-select the images (and corresponding modelmasks) that actually overlap this source
            srctims = []
            srcmm = []
            for mm, tim in zip(modelMasks, self.tims):
                if src in mm:
                    srctims.append(tim)
                    srcmm.append(mm)
            srctractor = self.tractor(srctims, [src])
            srctractor.setModelMasks(srcmm)

            if len(srctims) == 0:
                # eg, bright Gaia stars from off the brick can end up here??
                debug('No images overlap this source; skipping')
                continue

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

            #debug('%i images overlap this source' % len(srctims))
            optargs = self.optargs.copy()
            if has_fixed_position(src):
                optargs.update(check_step=None)

            opt = srctractor.optimizer
            opt.cache_image_params(srctractor)

            if self.plots:
                mods = list(srctractor.getModelImages())
                mod0,_ = quick_coadds(srctims, self.bands, self.blobwcs, images=mods,
                                      fill_holes=False)
                rgb,_ = quick_coadds(srctims, self.bands, self.blobwcs,
                                      fill_holes=False)

            # First-round optimization
            srctractor.optimize_loop(**optargs)

            opt.clear_cached_image_params()

            if self.plots and (numi<20 or numi%20 == 0):
                mods = list(srctractor.getModelImages())
                mod1,_ = quick_coadds(srctims, self.bands, self.blobwcs, images=mods,
                                      fill_holes=False)
                import pylab as plt
                plt.clf()
                plt.subplot(1,3,1)
                dimshow(get_rgb(rgb, self.bands))
                plt.title('Data')
                ax = plt.axis()
                ex0,ex1,ey0,ey1 = model_masks_to_blob_extent(srctims, srcmm, src, self.blobwcs,
                                                             to_int=True)
                plt.plot([ex0,ex0,ex1,ex1,ex0], [ey0,ey1,ey1,ey0,ey0], 'r-')
                plt.axis(ax)
                plt.subplot(1,3,2)
                dimshow(get_rgb(mod0, self.bands))
                plt.title('Initial model')
                plt.subplot(1,3,3)
                dimshow(get_rgb(mod1, self.bands))
                plt.title('Fit model')
                plt.suptitle('Source fitting: %i of %i' % (numi+1, len(Ibright)))
                self.ps.savefig()

            if is_galaxy:
                # Drop limits on SGA positions
                src.pos.lowers = [None, None]
                src.pos.uppers = [None, None]

            # Re-remove the final fit model for this source
            models.update_and_subtract(srci, src, self.tims)

            srctractor.setModelMasks(None)

            self.debug('Finished fitting source %i of %i (source id %i): %s' %
                       (numi+1, len(Ibright), srci, str(src)))
            cpu1 = time.process_time()
            cputime[srci] += (cpu1 - cpu0)
            done_fitting[srci] = True

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

        if self.plots:
            bmods = []

        for b in bands:
            for src in fitcat:
                src.getBrightness().freezeAllBut(b)
            # Images for this band
            btims = [tim for tim in tims if tim.band == b]
            btr = self.tractor(btims, fitcat)

            # SmarterDenseOptimizer currently only handles _single sources_ so
            # cannot be used here! (and using a dense matrix isn't great for
            # forced photometry)
            got_ceres = False
            if self.use_ceres:
                try:
                    from tractor.ceres_optimizer import CeresOptimizer
                    ceres_block = 8
                    btr.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
                    got_ceres = True
                except ImportError:
                    pass
            if not got_ceres:
                from tractor.lsqr_optimizer import LsqrOptimizer
                btr.optimizer = LsqrOptimizer()

            debug('Fitting %i sources in %i image for band %s.  Optimizer: %s' %
                 (len(fitcat), len(btims), b, str(type(btr.optimizer))))

            if self.plots:
                mods = list(btr.getModelImages())
                mod0,_ = quick_coadds(btims, [b], self.blobwcs, images=mods,
                                        fill_holes=False)

            btr.optimize_forced_photometry(shared_params=False, wantims=False)

            if self.plots:
                mods = list(btr.getModelImages())
                mod1,_ = quick_coadds(btims, [b], self.blobwcs, images=mods,
                                        fill_holes=False)
                bmods.append((mod0, mod1))

            for src in fitcat:
                src.getBrightness().thawAllParams()
        for src in fitcat:
            src.thawAllParams()

        if self.plots:
            import pylab as plt
            plt.clf()
            R,C = 2, len(bands)
            for i,(b,(mod0,mod1)) in enumerate(zip(bands, bmods)):
                plt.subplot(R, C, i+1)
                dimshow(get_rgb(mod0, [b]))
                plt.title('Before (%s)' % b)
                plt.subplot(R, C, i+C+1)
                dimshow(get_rgb(mod1, [b]))
                plt.title('After (%s)' % b)
            plt.suptitle('Flux fitting')
            self.ps.savefig()

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

    def _plot_coadd(self, tims, wcs, model=None, resid=None, addnoise=False,
                    inverr_mask=True):
        if resid is not None:
            mods = list(resid.getChiImages())
            coimgs,_ = quick_coadds(tims, self.bands, wcs, images=mods,
                                    fill_holes=False)
            rgbkwargs_resid = dict(resids=True)
            dimshow(get_rgb(coimgs,self.bands, **rgbkwargs_resid))
            return

        mods = None
        if model is not None:
            mods = list(model.getModelImages())
        if not inverr_mask:
            _,_,coimgs = quick_coadds(tims, self.bands, wcs, images=mods,
                                      fill_holes=False, addnoise=addnoise,
                                      get_co2=True)
        else:
            coimgs,_ = quick_coadds(tims, self.bands, wcs, images=mods,
                                    fill_holes=False, addnoise=addnoise)
        dimshow(get_rgb(coimgs,self.bands))

    def _initial_plots(self, cat):
        import pylab as plt
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

        goodcat = [src for src in cat if src is not None]
        _,x0,y0 = self.blobwcs.radec2pixelxy(
            np.array([src.getPosition().ra  for src in goodcat]),
            np.array([src.getPosition().dec for src in goodcat]))

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
        Ir = np.flatnonzero([is_reference_source(src) for src in goodcat])
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
    I: indices into the original sources array that will be placed into 'segmap'
    '''
    # ensure that each source owns a tiny radius around its center
    # in the segmentation map.  If there is more than one source
    # in that radius, each pixel gets assigned to its nearest
    # source.
    # 'kingdom' records the current distance to nearest source
    if radius < 255:
        rmax = 255
        rtype = np.uint8
    else:
        rmax = 65535
        rtype = np.uint16
    assert(radius <= rmax)
    best_r = np.empty(segmap.shape, rtype)
    best_r[:,:,] = rmax
    H,W = segmap.shape
    xcoords = np.arange(W)
    ycoords = np.arange(H)
    for i,x,y in zip(I, ix, iy):
        yslc = slice(max(0, y-radius), min(H, y+radius+1))
        xslc = slice(max(0, x-radius), min(W, x+radius+1))
        slc = (yslc, xslc)
        # Radius to nearest earlier source
        oldr = best_r[slc]
        # Radius to new source
        newr = np.hypot(xcoords[np.newaxis, xslc] - x, ycoords[yslc, np.newaxis] - y)
        assert(newr.shape == oldr.shape)
        newr = np.minimum(newr + 0.5, rmax).astype(rtype)
        # Pixels that are within range and closer to this source than any other.
        owned = (newr <= radius) * (newr < oldr)
        segmap[slc][owned] = i
        best_r[slc][owned] = newr[owned]

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
        if src is None:
            fluxes.append(-1000.)
            continue
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

def has_fixed_position(src):
    if src is None:
        return False
    return (len(src.pos.getParams()) == 0)

def _compute_source_metrics(srcs, tims, bands, tr, status):
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
                if (isrc+1)%1000 == 0:
                    status('Computing frac metrics: band %s (%i of %i), source %i of %i, patches' %
                           (band, iband+1, len(bands), isrc+1, len(srcs)))
                patch = tr.getModelPatch(tim, src)
                if patch is None or patch.patch is None:
                    continue
                counts[isrc] = np.sum([np.abs(pcal.brightnessToCounts(b))
                                              for b in src.getBrightnesses()])
                if counts[isrc] == 0:
                    continue
                H,W = mod.shape
                ok = patch.clipTo(W,H)
                if (not ok) or np.all(patch.patch == 0):
                    continue
                srcmods[isrc] = patch
                patch.addTo(mod)

            # Now compute metrics for each source
            for isrc,patch in enumerate(srcmods):
                status('Computing frac metrics: band %s (%i of %i), source %i of %i, metrics' %
                       (band, iband+1, len(bands), isrc+1, len(srcs)))
                if patch is None:
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
                # Everything is made a bit complicated by the fact that model patches
                # can go negative (because pixelized PSF models are noisy).
                # Usually PSF models get normalized, so sum(model)/counts = 1.
                # counts (ie, source's total flux) can also go negative!
                fin = np.clip(np.sum(patch) / counts[isrc], 0., 1.)

                fracflux_num[isrc,iband] += (fin *
                    np.sum((mod[slc] - patch) * np.abs(patch)) /
                    np.sum(patch**2))
                fracflux_den[isrc,iband] += fin

                fracmasked_num[isrc,iband] += np.clip(
                    np.sum((tim.getInvError()[slc] == 0) * patch) /
                    counts[isrc], 0., 1.)
                fracmasked_den[isrc,iband] += fin

                fracin_num[isrc,iband] += np.abs(np.sum(patch))
                fracin_den[isrc,iband] += np.abs(counts[isrc])

            # Compute rchisq in a separate loop because we want the
            # sky model added in here, but not for the other metrics!
            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2

            for isrc,patch in enumerate(srcmods):
                status('Computing frac metrics: band %s (%i of %i), source %i of %i, rchi2' %
                       (band, iband+1, len(bands), isrc+1, len(srcs)))
                if patch is None:
                    continue
                slc = patch.getSlice(mod)
                # We compute numerator and denom separately to handle
                # edge objects, where sum(patch.patch) < counts.
                # Also, to normalize by the number of images.  (Being
                # on the edge of an image is like being in half an
                # image.)
                rchi2_num[isrc,iband] += (np.abs(np.sum(chisq[slc] * patch.patch) /
                                                 np.abs(counts[isrc])))
                # If the source is not near an image edge,
                # sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.abs(np.sum(patch.patch)) / np.abs(counts[isrc])

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
    elif isinstance(src, RexGalaxy):
        psf = PointSource(src.getPosition(), src.getBrightness()).copy()
        rex = src.copy()
        shape = LegacyEllipseWithPriors(src.shape.re, 0., 0.)
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        oldmodel = 'rex'
    elif isinstance(src, DevGalaxy):
        psf = PointSource(src.getPosition(), src.getBrightness()).copy()
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
            tim.setImage(img.copy())

    def restore_images(self, tims):
        for tim,img in zip(tims, self.orig_images):
            tim.setImage(img)

    def create(self, tims, srcs, subtract=False, modelmasks=None):
        '''
        Note that this modifies the *tims* if subtract=True.
        '''
        self.models = []
        for itim,tim in enumerate(tims):
            mods = []
            sh = tim.shape
            ie = tim.getInvError()
            #for srci,src in enumerate(srcs):
            for src in srcs:
                if src is None:
                    continue
                mm = None
                if modelmasks is not None:
                    mm = modelmasks[itim].get(src, None)
                mod = src.getModelPatch(tim, modelMask=mm)
                if mod is not None and mod.patch is not None:
                    if not np.all(np.isfinite(mod.patch)):
                        warning('Non-finite mod patch.  Source:', src, 'tim:', tim,
                                'PSF:', tim.getPsf())
                    assert(np.all(np.isfinite(mod.patch)))

                    mod = _clip_model_to_blob(mod, sh, ie)
                    if subtract and mod is not None:
                        mod.addTo(tim.getImage(), scale=-1)
                        tim.setImage(tim.data)
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
                tim.setImage(tim.data)

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
                tim.setImage(tim.data)
            else:
                mod.addTo(tim.getImage(), scale=-1)
                tim.setImage(tim.data)

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

def remap_modelmasks(modelMasks, oldsrc, newsrcs):
    mm = []
    for mim in modelMasks:
        d = dict()
        mm.append(d)
        try:
            for newsrc in newsrcs:
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
    debug('_select_model: chisqs', chisqs)

    # This is our "detection threshold": 5-sigma in
    # *parameter-penalized* units; ie, ~5.2-sigma for point sources
    cut = 5.**2
    # Take the best of all models computed
    diff = max([chisqs[name] - nparams[name] for name in chisqs.keys()
                if name != 'none'] + [-1])

    #debug('best fit source chisq: %.3f, vs threshold %.3f' % (diff, cut))
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

def model_masks_to_blob_extent(tims, modelMasks, src, wcs, to_int=False):
    xlo = xhi = ylo = yhi = None
    for tim,mm in zip(tims, modelMasks):
        mask = mm.get(src, None)
        if mask is None:
            continue
        x0,x1,y0,y1 = mask.extent
        # FITS coordinates, inclusive
        x0 += 1
        y0 += 1
        r,d = tim.subwcs.pixelxy2radec([x0, x0, x1, x1], [y0, y1, y1, y0])
        ok,x,y = wcs.radec2pixelxy(r, d)
        assert(all(ok))
        x -= 1
        y -= 1
        if xlo is None or min(x) < xlo:
            xlo = min(x)
        if xhi is None or max(x) > xhi:
            xhi = max(x)
        if ylo is None or min(y) < ylo:
            ylo = min(y)
        if yhi is None or max(y) > yhi:
            yhi = max(y)
    if xlo is None:
        return None,None,None,None
    if to_int:
        h,w = wcs.shape
        ylo = int(np.clip(np.floor(ylo), 0, h))
        yhi = int(np.clip(np.ceil (yhi)+1, 0, h))
        xlo = int(np.clip(np.floor(xlo), 0, w))
        xhi = int(np.clip(np.ceil (xhi)+1, 0, w))

    return xlo,xhi,ylo,yhi
