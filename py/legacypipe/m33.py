import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import pylab as plt
from collections import Counter

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects

from astrometry.util.plotutils import dimshow

from tractor import Tractor
from tractor.psf import PixelizedPSF
from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
from tractor import Image
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor.lsqr_optimizer import LsqrOptimizer
from tractor.sersic import SersicGalaxy, SersicIndex
from tractor import Catalog

from legacypipe.reference import get_reference_map
from legacypipe.survey import LegacySurveyWcs
from legacypipe.survey import LegacySurveyWcs
from legacypipe.bits import DQ_BITS
from legacypipe.survey import get_rgb

def stage_m33_a(survey=None,
                refstars=None,
                T_dup=None,
                T_clusters=None,
                refcat=None,
               tycho_stars=True,
               gaia_stars=True,
               large_galaxies=True,
               star_clusters=True,
               brick=None,
               brickname=None,
               brickid=None,
               pixscale=None,
               targetwcs=None,
               bands=None,
               version_header=None,
               plots=False, ps=None,
               record_event=None,
               **kwargs):
    #from legacypipe.reference import get_reference_sources
    from legacypipe.runbrick import stage_refs

    # I previously ran stage_refs pointing to SGA rather than SGA-parent.  Fix that by
    # replacing the large-galaxy entries in 'refs'
    os.environ['LARGEGALAXIES_CAT'] = '/global/cfs/cdirs/cosmo/staging/largegalaxies/v3.0/SGA-parent-v3.0.kd.fits'
    R = stage_refs(survey=survey, brick=brick, brickname=brickname, brickid=brickid,
                   pixscale=pixscale, targetwcs=targetwcs, bands=bands,
                   version_header=version_header, tycho_stars=tycho_stars, gaia_stars=gaia_stars,
                   large_galaxies=large_galaxies, star_clusters=star_clusters,
                   plots=plots, ps=ps, record_event=record_event)
    return R



def stage_m33_b(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):
    # This is coming in after 'fit_on_coadds', so we have not done source detection
    # or blobs yet.

    # Fit out Gaia stars before proceeding to do something with M33 itself.
    from tractor import Tractor
    I = np.flatnonzero(np.logical_or(refstars.isgaia, refstars.istycho))
    print(len(I), 'stars')
    cat = [refcat[i] for i in I]
    print('Sources:')
    for src in cat:
        src.freezeAllBut('brightness')
        #print(' ', src.numberOfParams(), src)
    from tractor import ceres
    ceres_block = 8
    from tractor.ceres_optimizer import CeresOptimizer

    # we're running after fit_on_coadds, so assume one tim per band.
    assert(len(tims) == 3)
    for band,tim in zip(bands, tims):
        print('Band', band)
        for src in cat:
            src.brightness.freezeAllBut(band)
        tr = Tractor([tim], cat)
        tr.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
        tr.freezeParams('images')
        print('Forced phot...')
        tr.optimize_forced_photometry(shared_params=False, wantims=False)
        print('Finished forced phot')

    for src in cat:
        src.brightness.thawAllParams()
        src.thawAllParams()
    return dict(refcat=refcat)

def stage_m33_c(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):

    # Subtract already-fit Gaia models
    from tractor import Tractor
    I = np.flatnonzero(np.logical_or(refstars.isgaia, refstars.istycho))
    print(len(I), 'stars')
    cat = [refcat[i] for i in I]

    if plots:
        #from legacypipe.coadds import quick_coadds
        from legacypipe.survey import get_rgb
        from astrometry.util.plotutils import dimshow
        import pylab as plt
        plt.clf()
        dimshow(get_rgb([tim.data for tim in tims], bands))
        ps.savefig()

        mods = []
    
    # we're running after fit_on_coadds, so assume one tim per band.
    assert(len(tims) == 3)
    for band,tim in zip(bands, tims):
        print('Band', band)
        tr = Tractor([tim], cat)
        print('Getting model image...')
        mod = tr.getModelImage(0)
        if plots:
            mods.append(mod)
        tim.data -= mod

    if plots:
        plt.clf()
        dimshow(get_rgb(mods, bands))
        ps.savefig()
        plt.clf()
        dimshow(get_rgb([tim.data for tim in tims], bands))
        ps.savefig()
        
    return dict(tims=tims)


class Duck(object):
    pass

def stage_m33_d(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):
    # Downsample tims.
    D = 4

    subtims = []
    for band,tim in zip(bands, tims):
        print('Downsampling tim band', band)
        H,W = tim.shape
        sh,sw = H//D, W//D
        print('Full size', H,W, 'to', sh,sw)
        # blur image and invvar.
        simg = gaussian_filter(tim.data,   1.0 * D)[::D, ::D][:sh,:sw]
        siv  = gaussian_filter(tim.invvar, 1.0 * D)[::D, ::D][:sh,:sw]

        # pretend we did binning
        simg *= D*D
        siv  /= D*D

        # Zero out DQ bits where we have no coverage in the coadd!
        from functools import reduce
        allbits = reduce(np.bitwise_or, DQ_BITS.values())
        #print('allbits: 0x%x' % allbits)
        #print('Most common DQ values:', Counter(tim.dq.ravel()).most_common(10))
        tim.dq[tim.dq == allbits] = 0
        
        subdq = np.zeros((sh,sw), tim.dq.dtype)
        for i in range(D):
            for j in range(D):
                ### OR??
                subdq |= tim.dq[i::D, j::D][:sh,:sw]
        
        #simg = np.zeros((sh,sw), np.float32)
        #siv  = np.zeros((sh,sw), np.float32)
        # blur image first
        #blur = gaussian_filter(simg, 1.0 * D)
        # for i in range(D):
        #     for j in range(D):
        #         subiv = tim.inverr[i::D, j::D][:sh,:sw]**2
        #         siv += subiv
        #         simg += tim.data[i::D, j::D][:sh,:sw] # * subiv
        #simg /= siv
        swcs = tim.subwcs.get_subimage(0,0,sw*D,sh*D).scale(1./D)
        twcs = LegacySurveyWcs(swcs, tim.wcs.tai)
        psfimg = tim.psf.img
        spsf = gaussian_filter(psfimg, 1.0 * D)
        ph,pw = psfimg.shape
        cx,cy = pw//2, ph//2
        subpsf = spsf[cy%D::D, cx%D::D]
        subpsf /= subpsf.sum()
        sph,spw = subpsf.shape
        print('PSF:', ph,pw, '->', sph,spw)
        subpsf_sigma = np.hypot(tim.psf_sigma, 1.0 * D) / float(D)
        subpsf = PixelizedPSF(subpsf)
        subpsf = HybridPixelizedPSF(subpsf,
                                    gauss=NCircularGaussianPSF([subpsf_sigma], [1.]))

        subobj = Duck()
        subobj.fwhm = subpsf_sigma * 2.35
        subobj.pixscale = tim.imobj.pixscale * D
        
        subtim = Image(simg, invvar=siv, psf=subpsf, wcs=twcs, photocal=tim.photocal)
        subtim.subwcs = swcs
        subtim.psf_sigma = subpsf_sigma
        subtim.sig1 = 1./np.median(subtim.inverr)
        subtim.imobj = subobj
        subtim.time = tim.time
        subtim.dq = subdq
        subtim.dq_saturation_bits = tim.dq_saturation_bits
        subtim.band = tim.band
        print('sig1:', tim.sig1, '->', subtim.sig1)
        print('psf_sigma:', tim.psf_sigma, '->', subtim.psf_sigma)
    
        subtims.append(subtim)

    H,W = targetwcs.shape
    sh,sw = H//D, W//D
    subwcs = targetwcs.get_subimage(0,0,sw*D,sh*D).scale(1./D)
        
    return dict(tims=subtims, targetwcs=subwcs, H=H, W=W)


def stage_m33_e(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):

    from scipy.ndimage.measurements import label, find_objects
    from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
    from legacypipe.detection import detection_maps

    H,W = targetwcs.shape
    for tim in tims:
        print('wcs', tim.subwcs.shape, 'tim', tim.shape, 'img', tim.data.shape)
        assert(tim.subwcs.shape == tim.data.shape)

    # Run source detection just far enough to define the blobs.
    print('Detection maps...')
    detmaps, detivs, satmaps = detection_maps(tims, targetwcs, bands, mp,
                                              apodize=10)
    saturated_pix = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]
    SEDs = survey.sed_matched_filters(bands)

    hotmap = np.zeros((H,W), bool)
    
    hot = np.zeros((H,W), bool)
    for sedname,sed in SEDs:
        for iband in range(len(bands)):
            if sed[iband] == 0:
                continue
            if np.all(detivs[iband] == 0):
                continue
            allzero = False
            break
        if allzero:
            print('SED', sedname, 'has all zero weight')
            continue

        sedmap = np.zeros((H,W), np.float32)
        sediv  = np.zeros((H,W), np.float32)
        if saturated_pix is not None:
            satur  = np.zeros((H,W), bool)
        for iband in range(len(bands)):
            if sed[iband] == 0:
                continue
            # We convert the detmap to canonical band via
            #   detmap * w
            # And the corresponding change to sig1 is
            #   sig1 * w
            # So the invvar-weighted sum is
            #    (detmap * w) / (sig1**2 * w**2)
            #  = detmap / (sig1**2 * w)
            sedmap += detmaps[iband] * detivs[iband] / sed[iband]
            sediv  += detivs [iband] / sed[iband]**2
            if saturated_pix is not None:
                satur |= saturated_pix[iband]
        sedmap /= np.maximum(1e-16, sediv)
        sedsn   = sedmap * np.sqrt(sediv)
        del sedmap
        peaks = (sedsn > nsigma)
        # zero out the edges -- larger margin here?
        peaks[0 ,:] = 0
        peaks[:, 0] = 0
        peaks[-1,:] = 0
        peaks[:,-1] = 0
        # Label the N-sigma blobs at this point... we'll use this to build
        # "sedhot", which in turn is used to define the blobs that we will
        # optimize simultaneously.  This also determines which pixels go
        # into the fitting!
        dilate = 8
        sedhot = binary_fill_holes(binary_dilation(peaks, iterations=dilate))
        hotmap |= sedhot

    blobs,nblobs = label(hotmap)
    blobslices = find_objects(blobs)

    print('Found', nblobs, 'blobs')

    if plots:
        plt.clf()
        plt.imshow(blobs, interpolation='nearest', origin='lower')
        plt.title('Blobs')
        ps.savefig()
    
    return dict(H=H, W=W,
        blobmap=blobs, blobslices=blobslices)


def stage_m33_f(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
                gaia_stars=True,
                blobmap=None, blobslices=None,
               **kwargs):

    # Select the galaxy-masked area
    # (hack)
    galblob = blobmap[H//2, W//2]
    print('Large galaxy blob:', galblob)
    print('Number of pixels:', np.sum(blobmap == galblob))
    print('blobmap range:', blobmap.min(), blobmap.max())
    print('blobslices length:', len(blobslices))
    
    galslc = blobslices[galblob-1]
    print('Large galaxy slice:', galslc)

    Igals = np.flatnonzero(refstars.islargegalaxy)
    igal = np.argsort(-refstars.radius[Igals])[0]
    igal = Igals[igal]
    print('Largest galaxy: radius', refstars.radius[igal])
    # table of one element
    refgal = refstars[np.array([igal])]
    print('ref gal:', refgal)

    # make the mask at 2 x radius
    rgcopy = refgal.copy()
    rgcopy.radius *= 2.0
    galmap = (get_reference_map(targetwcs, rgcopy) != 0)

    refgalslc = find_objects(galmap.astype(int))[0]
    print('ref gal slice:', refgalslc)

    # find the other galaxy within the mask
    _,xx,yy = targetwcs.radec2pixelxy(refstars.ra, refstars.dec)
    H,W = targetwcs.shape
    xx = np.clip((xx - 1).astype(int), 0, W-1)
    yy = np.clip((yy - 1).astype(int), 0, H-1)
    Iothers = np.flatnonzero(galmap[yy[Igals], xx[Igals]] *
                             (refstars.ref_id[Igals] != refgal.ref_id[0]))
    Iothers = Igals[Iothers]
    print('Galaxies within M33 mask:', Iothers)
    #refothers = refstars[Iothers]
    #catothers = [refcat[i] for i in Iothers]
    
    if plots:
        plt.clf()
        plt.imshow(blobmap == galblob, interpolation='nearest', origin='lower')
        plt.title('Blob map for galaxy')
        ps.savefig()

        plt.clf()
        plt.imshow(galmap, interpolation='nearest', origin='lower')
        #ax = plt.axis()
        #plt.plot(xx, yy, 'ro')
        #plt.axis(ax)
        plt.title('Reference galaxy ellipse')
        ps.savefig()


    galsrc = refcat[igal]
    print('Large galaxy source:', galsrc)

    if False:
        sersrc = SersicGalaxy(galsrc.pos, galsrc.brightness, galsrc.shape, SersicIndex(1.0))
        refcat[igal] = sersrc
        galsrc = refcat[igal]
        print('Switched to:', galsrc)

    slc = refgalslc

    subtims = []
    for band,tim in zip(bands, tims):
        # Make a copy of the image because we're going to subtract out models below!
        simg = tim.data[slc].copy()
        sie = tim.inverr[slc].copy()

        # Zero out inverr outside the galaxy ellipse!!
        sie[np.logical_not(galmap[slc])] = 0.

        subdq = tim.dq[slc]
        sy,sx = slc
        y0,y1 = sy.start,sy.stop
        x0,x1 = sx.start,sx.stop
        subwcs = tim.subwcs.get_subimage(x0, y0, x1-x0, y1-y0)
        twcs = LegacySurveyWcs(subwcs, tim.wcs.tai)
        subtim = Image(simg, inverr=sie, psf=tim.psf, wcs=twcs, photocal=tim.photocal)
        subtim.subwcs = subwcs
        subtim.psf_sigma = tim.psf_sigma
        subtim.sig1 = tim.sig1
        subtim.imobj = tim.imobj
        subtim.time = tim.time
        subtim.dq = subdq
        subtim.dq_saturation_bits = tim.dq_saturation_bits
        subtim.band = tim.band
        subtims.append(subtim)

    tr = Tractor(subtims, [galsrc], optimizer=ConstrainedDenseOptimizer())
    tr.freezeParam('images')
    print('Optimizing!')
    tr.optimize_loop(dchisq=0.1, shared_params=False, priors=True, alphas=[0.1, 0.3, 1.0])

    # Fit other galaxy
    tr.optimizer = LsqrOptimizer()
    for i in Iothers:
        print('Fitting galaxy:', refcat[i])
        # subtract out previous fit galaxy
        mods = list(tr.getModelImages())
        for mod,subtim in zip(mods, subtims):
            subtim.data -= mod
        tr.catalog = Catalog(refcat[i])
        print('Optimizing!')
        tr.optimize_loop(dchisq=0.1, shared_params=False, priors=True, alphas=[0.1, 0.3, 1.0])
    
    return dict(refgalslc=refgalslc, galsrc=galsrc, refcat=refcat, galmap=galmap,
                Iothers=Iothers)
        
def stage_m33_g(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
                gaia_stars=True,
                blobmap=None, blobslices=None,

                galsrc=None,
                **kwargs):

    print('Fit M33:', galsrc)

    Igals = np.flatnonzero(refstars.islargegalaxy)
    igal = np.argsort(-refstars.radius[Igals])[0]
    igal = Igals[igal]
    print('Largest galaxy: radius', refstars.radius[igal])
    # table of one element
    refgal = refstars[np.array([igal])]
    galmap = (get_reference_map(targetwcs, refgal) != 0)
    # find the other galaxy within the mask
    _,xx,yy = targetwcs.radec2pixelxy(refstars.ra, refstars.dec)
    H,W = targetwcs.shape
    xx = np.clip((xx - 1).astype(int), 0, W-1)
    yy = np.clip((yy - 1).astype(int), 0, H-1)
    Iothers = np.flatnonzero(galmap[yy[Igals], xx[Igals]] *
                             (refstars.ref_id[Igals] != refgal.ref_id[0]))
    Iothers = Igals[Iothers]
    print('Galaxies within M33 mask:', Iothers)

    if plots:
        tr = Tractor(tims, [galsrc], optimizer=ConstrainedDenseOptimizer())
        tr.freezeParam('images')
        mods = list(tr.getModelImages())
        plt.clf()
        dimshow(get_rgb([tim.data for tim in tims], bands))
        ps.savefig()
        plt.clf()
        dimshow(get_rgb(mods, bands))
        ps.savefig()
        plt.clf()
        dimshow(get_rgb([tim.data - mod for tim,mod in zip(tims,mods)], bands, resids=True))
        ps.savefig()

    from legacypipe.catalog import prepare_fits_catalog
    from legacypipe.utils import copy_header_with_wcs
    from tractor import Catalog

    # Igals = np.flatnonzero(refstars.islargegalaxy)
    # igal = np.argsort(-refstars.radius[Igals])[0]
    # igal = Igals[igal]
    # print('galsrc:', galsrc)

    cat = Catalog(galsrc, *[refcat[i] for i in Iothers])
    T = refstars[np.append(np.array([igal]), Iothers)]

    for src in cat:
        src.shape = src.shape.toEllipseE()

    #invvars = np.zeros(cat.numberOfParams(), np.float32)
    invvars=None

    T = prepare_fits_catalog(cat, invvars, T, bands, save_invvars=False)

    # Set Sersic indices for all galaxy types.
    # sigh, bytes vs strings.  In py3, T.type (dtype '|S3') are bytes.
    T.sersic[np.array([t in ['DEV',b'DEV'] for t in T.type])] = 4.0
    T.sersic[np.array([t in ['EXP',b'EXP'] for t in T.type])] = 1.0
    T.sersic[np.array([t in ['REX',b'REX'] for t in T.type])] = 1.0

    T.writeto('cat-m33.fits')

    T.dchisq = np.zeros((len(T),5), np.float32)
    for c in ['rchisq', 'fracflux', 'fracmasked', 'fracin',
              'nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth',
              'fiberflux', 'fibertotflux',
              'flux_ivar']:
        T.set(c, np.zeros((len(T),3), np.float32))

    for c in ['mjd_min', 'mjd_max', 'nea_g', 'nea_r', 'nea_z',
              'blob_nea_g', 'blob_nea_r', 'blob_nea_z',
              'shape_r_ivar', 'shape_e1_ivar', 'shape_e2_ivar', 'sersic_ivar']:
        T.set(c, np.zeros(len(T), np.float32))
    # The "format_catalog" code expects all lower-case column names...
    for c in T.columns():
        if c != c.lower():
            T.rename(c, c.lower())
    from legacypipe.format_catalog import format_catalog
    outfn = 'tractor-m33-ser.fits'
    release = 9999
    format_catalog(T, None, None, bands, 'grz', outfn, release)

    # if plots:
    #     import pylab as plt
    #     for band,tim in zip(bands,tims):
    #         plt.clf()
    #         plt.hist((tim.data * tim.inverr).ravel(), range=(-5,5), bins=50)
    #         plt.xlabel('Pixel S/N')
    #         plt.title('Scaled image: %s band' % band)
    #         ps.savefig()
    #     for band,tim in zip(bands,tims):
    #         plt.clf()
    #         plt.imshow((tim.dq & DQ_BITS['satur']) > 0, cmap='gray', vmin=0, vmax=1, origin='lower')
    #         plt.colorbar()
    #         plt.title('SATUR pix: %s band' % band)
    #         ps.savefig()
    # 
    #     for band,tim in zip(bands,tims):
    #         plt.clf()
    #         plt.imshow(tim.inverr, cmap='gray', origin='lower')
    #         plt.colorbar()
    #         plt.title('Inverr: %s band' % band)
    #         ps.savefig()
    # # we blurred by a sigma=D gaussian,
    # psfnorm = 1./(2. * np.sqrt(np.pi) * 4)
    # for band,tim in zip(bands, tims):
    #     #tim.data *= 1./(psfnorm**2)
    #     #tim.inverr *= psfnorm
    #     tim.data *= (D*D)
    #     tim.inverr *= (1./D)

def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs

    parser = get_parser()
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    verbose = optdict.pop('verbose')
    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1,0]:
        return kwargs

    import logging
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    kwargs.update(
        prereqs_update={'m33_a': 'fit_on_coadds',
                        'm33_b': 'm33_a',
                        'm33_c': 'm33_b',
                        'm33_d': 'm33_c',
                        'm33_e': 'm33_d',
                        'm33_f': 'm33_e',
                        'm33_g': 'm33_f',
                        'srcs': 'm33_d',
        },
        stagefunc_vars={'stage_m33_a':stage_m33_a,
                        'stage_m33_b':stage_m33_b,
                        'stage_m33_c':stage_m33_c,
                        'stage_m33_d':stage_m33_d,
                        'stage_m33_e':stage_m33_e,
                        'stage_m33_f':stage_m33_f,
                        'stage_m33_g':stage_m33_g,
        },
    )
    
    run_brick(opt.brick, survey, **kwargs)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())


    # python -u legacypipe/m33.py --radec 23.461713020137164 30.662208518892047 --width 28501 --height 28501 --pixscale 0.262 --outdir $CSCRATCH/NGC0598_GROUP --survey-dir /global/cfs/cdirs/cosmo/work/legacysurvey/dr9m --run south --skip-calibs --checkpoint $CSCRATCH/NGC0598_GROUP/NGC0598_GROUP-largegalaxy-checkpoint.p --pickle "$CSCRATCH/NGC0598_GROUP/NGC0598_GROUP-largegalaxy-%%(stage)s.p" --fit-on-coadds --saddle-fraction 0.2 --saddle-min 4.0 --stage m33_a
