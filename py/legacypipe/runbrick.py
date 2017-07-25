'''
Main "pipeline" script for the Legacy Survey (DECaLS, MzLS)
data reductions.

For calling from other scripts, see:

- :py:func:`run_brick`

Or for much more fine-grained control, see the individual stages:

- :py:func:`stage_tims`
- :py:func:`stage_image_coadds`
- :py:func:`stage_srcs`
- :py:func:`stage_fitblobs`
- :py:func:`stage_coadds`
- :py:func:`stage_wise_forced`
- :py:func:`stage_writecat`

To see the code we run on each "blob" of pixels, see "oneblob.py".

- :py:func:`one_blob`

'''
from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import sys
import os

import pylab as plt
import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.ttime import Time
from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

from tractor import Tractor, PointSource, Image, NanoMaggies, Catalog, RaDecPos
from tractor.ellipses import EllipseE
from tractor.galaxy import (DevGalaxy, ExpGalaxy, FixedCompositeGalaxy,
                            FracDev, disable_galaxy_cache)

from legacypipe.survey import (
    get_rgb, imsave_jpeg, LegacySurveyData, on_bricks_dependencies)
from legacypipe.cpimage import CP_DQ_BITS
from legacypipe.utils import (
    RunbrickError, NothingToDoError, iterwrapper, find_unique_pixels)
from legacypipe.coadds import make_coadds, write_coadd_images, quick_coadds

# RGB image args used in the tile viewer:
rgbkwargs = dict(mnmx=(-1,100.), arcsinh=1.)
rgbkwargs_resid = dict(mnmx=(-5,5))

# Memory Limits
import resource
def set_ulimit(gigabytes):
    rsrc = resource.RLIMIT_AS #used if RLIMIT_VMEM not found
    #soft, hard = resource.getrlimit(rsrc)
    soft, hard = resource.getrlimit(rsrc)
    soft= gigabytes * 1024**3
    resource.setrlimit(rsrc, (soft,hard))

def get_ulimit():
    for name, desc in [
        ('RLIMIT_AS', 'VMEM'),
        ('RLIMIT_CORE', 'core file size'),
        ('RLIMIT_CPU',  'CPU time'),
        ('RLIMIT_FSIZE', 'file size'),
        ('RLIMIT_DATA', 'heap size'),
        ('RLIMIT_STACK', 'stack size'),
        ('RLIMIT_RSS', 'resident set size'),
        ('RLIMIT_NPROC', 'number of processes'),
        ('RLIMIT_NOFILE', 'number of open files'),
        ('RLIMIT_MEMLOCK', 'lockable memory address'),
        ]:
        limit_num = getattr(resource, name)
        soft, hard = resource.getrlimit(limit_num)
        print('Maximum %-25s (%-15s) : %20s %20s' % (desc, name, soft, hard))

def runbrick_global_init():
    print('Starting process', os.getpid(), Time()-Time())
    disable_galaxy_cache()

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               survey=None,
               ra=None, dec=None,
               plots=False, ps=None,
               target_extent=None, program_name='runbrick.py',
               bands=['g','r','z'],
               do_calibs=True,
               splinesky=True,
               gaussPsf=False, pixPsf=False, hybridPsf=False,
               constant_invvar=False,
               depth_cut = True,
               read_image_pixels = True,
               mp=None,
               record_event=None,
               **kwargs):
    '''
    This is the first stage in the pipeline.  It
    determines which CCD images overlap the brick or region of
    interest, runs calibrations for those images if necessary, and
    then reads the images, creating `tractor.Image` ("tractor image"
    or "tim") objects for them.

    PSF options:

    - *gaussPsf*: boolean.  Single-component circular Gaussian, with
      width set from the header FWHM value.  Useful for quick
      debugging.

    - *pixPsf*: boolean.  Pixelized PsfEx model.

    - *hybridPsf*: boolean.  Hybrid Pixelized PsfEx / Gaussian approx model.

    Sky:

    - *splinesky*: boolean.  Use SplineSky model, rather than ConstantSky?

    '''
    from legacypipe.survey import (
        get_git_version, get_version_header, wcs_for_brick, read_one_tim)
    t0 = tlast = Time()
    assert(survey is not None)

    record_event and record_event('stage_tims: starting')

    # Get brick object
    if ra is not None:
        from legacypipe.survey import BrickDuck
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck(ra, dec, brickname)
    else:
        brick = survey.get_brick_by_name(brickname)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)
    brickid = brick.brickid
    brickname = brick.brickname

    print('Got brick:', Time()-t0)

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    # custom brick -- set RA,Dec bounds
    if ra is not None:
        brick.ra1,nil  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,nil  = targetwcs.pixelxy2radec(1, H/2)
        nil, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        nil, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)

    print('Got brick wcs:', Time()-t0)

    # Create FITS header with version strings
    gitver = get_git_version()
    print('Got git version:', Time()-t0)

    version_header = get_version_header(program_name, survey.survey_dir,
                                        git_version=gitver)

    # Get DESICONDA version, and read file $DESICONDA/pkg_list.txt for
    # other package versions.
    default_ver = 'UNAVAILABLE'
    depnum = 0
    desiconda = os.environ.get('DESICONDA', default_ver)
    verstr = os.path.basename(desiconda)
    version_header.add_record(dict(name='DEPNAM%02i' % depnum, value='desiconda',
                                   comment='Name of dependency product'))
    version_header.add_record(dict(name='DEPVER%02i' % depnum, value=verstr,
                                   comment='Version of dependency product'))
    depnum += 1

    if desiconda != default_ver:
        fn = os.path.join(desiconda, 'pkg_list.txt')
        vers = {}
        if not os.path.exists(fn):
            print('Warning: expected file $DESICONDA/pkg_list.txt to exist but it does not!')
        else:
            # Read version numbers
            for line in open(fn):
                words = line.strip().split('=')
                if len(words) >= 2:
                    vers[words[0]] = words[1]

        for pkg in ['astropy', 'matplotlib', 'mkl', 'numpy', 'python', 'scipy']:
            verstr = vers.get(pkg, default_ver)
            version_header.add_record(dict(name='DEPNAM%02i' % depnum, value=pkg,
                                           comment='Name of dependency product'))
            version_header.add_record(dict(name='DEPVER%02i' % depnum, value=verstr,
                                           comment='Version of dependency product'))
            depnum += 1
            if verstr == default_ver:
                print('Warning: failed to get version string for "%s"' % pkg)
    # Get additional version strings from modules.
    for dep in ['unwise_coadds', 'unwise_coadds_timeresolved']:
        # Look in the OS environment variables for modules-style
        # $scipy_VERSION => 0.15.1_5a3d8dfa-7.1
        verstr = os.environ.get('%s_VERSION' % dep, default_ver)
        if verstr == default_ver:
            print('Warning: failed to get version string for "%s"' % dep)
        version_header.add_record(dict(name='DEPNAM%02i' % depnum, value=dep,
                                    comment='Name of dependency product'))
        version_header.add_record(dict(name='DEPVER%02i' % depnum, value=verstr,
                                    comment='Version of dependency product'))
        depnum += 1

    version_header.add_record(dict(name='BRICKNAM', value=brickname,
                                comment='LegacySurvey brick RRRr[pm]DDd'))
    version_header.add_record(dict(name='BRICKID' , value=brickid,
                                comment='LegacySurvey brick id'))
    version_header.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min'))
    version_header.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max'))
    version_header.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min'))
    version_header.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max'))
    version_header.add_record(dict(name='BRICKRA' , value=brick.ra,
                                comment='Brick center'))
    version_header.add_record(dict(name='BRICKDEC', value=brick.dec,
                                comment='Brick center'))

    # Add NOAO-requested headers
    version_header.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'),
        comment='[h] RA Brick center'))
    version_header.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'),
        comment='[deg] Dec Brick center'))
    version_header.add_record(dict(
        name='CENTRA', value=brick.ra, comment='[deg] Brick center RA'))
    version_header.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='[deg] Brick center Dec'))
    for i,(r,d) in enumerate(targetrd[:4]):
        version_header.add_record(dict(
            name='CORN%iRA' %(i+1), value=r, comment='[deg] Brick corner RA'))
        version_header.add_record(dict(
            name='CORN%iDEC'%(i+1), value=d, comment='[deg] Brick corner Dec'))

    print('Got FITS header:', Time()-t0)

    # Find CCDs
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    print(len(ccds), 'CCDs touching target WCS')

    print('Got CCDs:', Time()-t0)

    # Sort images by band -- this also eliminates images whose
    # *filter* string is not in *bands*.
    print('Unique filters:', np.unique(ccds.filter))
    ccds.cut(np.in1d(ccds.filter, bands))
    print('Cut on filter:', len(ccds), 'CCDs remain.')

    print('Cutting out non-photometric CCDs...')
    I = survey.photometric_ccds(ccds)
    if I is None:
        print('None cut')
    else:
        print(len(I), 'of', len(ccds), 'CCDs are photometric')
        ccds.cut(I)

    print('Applying CCD cuts...')
    ccds.ccd_cuts = survey.ccd_cuts(ccds)
    cutvals = ccds.ccd_cuts
    print('CCD cut bitmask values:', cutvals)
    ccds.cut(cutvals == 0)
    print(len(ccds), 'CCDs survive cuts')

    print('Cutting on CCDs to be used for fitting...')
    I = survey.ccds_for_fitting(brick, ccds)
    if I is not None:
        print('Cutting to', len(I), 'of', len(ccds), 'CCDs for fitting.')
        ccds.cut(I)

    if depth_cut:
        # If we have many images, greedily select images until we have
        # reached our target depth
        print('Cutting to CCDs required to hit our depth targets')
        keep_ccds = make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                                   plots, ps, splinesky, gaussPsf, pixPsf,
                                   do_calibs, gitver, targetwcs)
        ccds.cut(np.array(keep_ccds))
        print('Cut to', len(ccds), 'CCDs required to reach depth targets')

    # Create Image objects for each CCD
    ims = []
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        if survey.cache_dir is not None:
            im.check_for_cached_files(survey)
        ims.append(im)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
              'seeing %.2f' % (ccd.fwhm*im.pixscale),
              'object', getattr(ccd, 'object', None))

    print('Cut CCDs:', Time()-t0)

    tnow = Time()
    print('[serial tims] Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if do_calibs:
        from legacypipe.survey import run_calibs
        record_event and record_event('stage_tims: starting calibs')
        kwa = dict(git_version=gitver)
        if gaussPsf:
            kwa.update(psfex=False)
        if splinesky:
            kwa.update(splinesky=True)
        # Run calibrations
        args = [(im, kwa) for im in ims]
        mp.map(run_calibs, args)
        tnow = Time()
        print('[parallel tims] Calibrations:', tnow-tlast)
        tlast = tnow
        #record_event and record_event('stage_tims: done calibs')

    print('Calibs:', Time()-t0)

    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                                hybridPsf=hybridPsf,
                                splinesky=splinesky,
                                constant_invvar=constant_invvar,
                                pixels=read_image_pixels))
                                for im in ims]
    record_event and record_event('stage_tims: starting read_tims')
    tims = list(mp.map(read_one_tim, args))
    record_event and record_event('stage_tims: done read_tims')

    tnow = Time()
    print('[parallel tims] Read', len(ccds), 'images:', tnow-tlast)
    tlast = tnow

    # Cut the table of CCDs to match the 'tims' list
    I = np.array([i for i,tim in enumerate(tims) if tim is not None])
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))
    if len(tims) == 0:
        raise NothingToDoError('No photometric CCDs touching brick.')

    # Count pixels
    npix = 0
    for tim in tims:
        h,w = tim.shape
        npix += h*w
    print('Total of', npix, 'pixels read')

    # Check calibration product versions
    for tim in tims:
        for cal,ver in [('sky', tim.skyver), ('wcs', tim.wcsver),
                        ('psf', tim.psfver)]:
            if tim.plver.strip() != ver[1].strip():
                print(('Warning: image "%s" PLVER is "%s" but %s calib was run'
                      +' on PLVER "%s"') % (str(tim), tim.plver, cal, ver[1]))

    # Add additional columns to the CCDs table.
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                            for tim in tims]).astype(np.int16)
    ccds.ccd_y1 = np.array([tim.y0 + tim.shape[0]
                            for tim in tims]).astype(np.int16)
    rd = np.array([[tim.subwcs.pixelxy2radec(1, 1)[-2:],
                    tim.subwcs.pixelxy2radec(1, y1-y0)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, 1)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, y1-y0)[-2:]]
                    for tim,x0,y0,x1,y1 in
                    zip(tims, ccds.ccd_x0+1, ccds.ccd_y0+1,
                        ccds.ccd_x1, ccds.ccd_y1)])
    ok,x,y = targetwcs.radec2pixelxy(rd[:,:,0], rd[:,:,1])
    ccds.brick_x0 = np.floor(np.min(x, axis=1)).astype(np.int16)
    ccds.brick_x1 = np.ceil (np.max(x, axis=1)).astype(np.int16)
    ccds.brick_y0 = np.floor(np.min(y, axis=1)).astype(np.int16)
    ccds.brick_y1 = np.ceil (np.max(y, axis=1)).astype(np.int16)
    ccds.sig1 = np.array([tim.sig1 for tim in tims])
    ccds.psfnorm = np.array([tim.psfnorm for tim in tims])
    ccds.galnorm = np.array([tim.galnorm for tim in tims])
    ccds.propid = np.array([tim.propid for tim in tims])
    ccds.plver  = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.wcsver = np.array([tim.wcsver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.wcsplver = np.array([tim.wcsver[1] for tim in tims])
    ccds.psfplver = np.array([tim.psfver[1] for tim in tims])

    # Cut "bands" down to just the bands for which we have images.
    timbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in timbands]
    print('Cut bands to', bands)

    if plots:
        # Pixel histograms of subimages.
        for b in bands:
            sig1 = np.median([tim.sig1 for tim in tims if tim.band == b])
            plt.clf()
            for tim in tims:
                if tim.band != b:
                    continue
                # broaden range to encompass most pixels... only req'd
                # when sky is bad
                lo,hi = -5.*sig1, 5.*sig1
                pix = tim.getImage()[tim.getInvError() > 0]
                lo = min(lo, np.percentile(pix, 5))
                hi = max(hi, np.percentile(pix, 95))
                plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                         alpha=0.5, label=tim.name)
            plt.legend()
            plt.xlabel('Pixel values')
            plt.title('Pixel distributions: %s band' % b)
            ps.savefig()

            plt.clf()
            lo,hi = -5., 5.
            for tim in tims:
                if tim.band != b:
                    continue
                ie = tim.getInvError()
                pix = (tim.getImage() * ie)[ie > 0]
                plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                         alpha=0.5, label=tim.name)
            plt.legend()
            plt.xlabel('Pixel values (sigma)')
            plt.xlim(lo,hi)
            plt.title('Pixel distributions: %s band' % b)
            ps.savefig()

    if plots and False:
        # Plot image pixels, invvars, masks
        for tim in tims:
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(tim.getImage(), vmin=-3.*tim.sig1, vmax=10.*tim.sig1)
            plt.title('image')
            plt.subplot(2,2,2)
            dimshow(tim.getInvError(), vmin=0, vmax=1.1/tim.sig1)
            plt.title('inverr')
            if tim.dq is not None:
                plt.subplot(2,2,3)
                dimshow(tim.dq, vmin=0, vmax=tim.dq.max())
                plt.title('DQ')
                plt.subplot(2,2,3)
                #nil,udq = np.unique(tim.dq, return_inverse=True)
                dimshow(((tim.dq & tim.dq_saturation_bits) > 0),
                        vmin=0, vmax=1.5, cmap='hot')
                plt.title('SATUR')
            plt.suptitle(tim.name)
            ps.savefig()

            if tim.dq is not None:
                plt.clf()
                bitmap = dict([(v,k) for k,v in CP_DQ_BITS.items()])
                k = 1
                for i in range(12):
                    bitval = 1 << i
                    if not bitval in bitmap:
                        continue
                    plt.subplot(3,3,k)
                    k+=1
                    plt.imshow((tim.dq & bitval) > 0,
                               vmin=0, vmax=1.5, cmap='hot')
                    plt.title(bitmap[bitval])
                plt.suptitle('Mask planes: %s' % tim.name)
                ps.savefig()

    # Add header cards about which bands and cameras are involved.
    for band in 'grz':
        hasit = band in bands
        version_header.add_record(dict(
            name='BRICK_%s' % band.upper(), value=hasit,
            comment='Does band %s touch this brick?' % band))

        cams = np.unique([tim.imobj.camera for tim in tims
                          if tim.band == band])
        version_header.add_record(dict(
            name='CAMS_%s' % band.upper(), value=' '.join(cams),
            comment='Cameras contributing band %s' % band))
    version_header.add_record(dict(name='BRICKBND', value=''.join(bands),
                                   comment='Bands touching this brick'))
    version_header.add_record(dict(name='NBANDS', value=len(bands),
                                   comment='Number of bands in this catalog'))
    for i,band in enumerate(bands):
        version_header.add_record(dict(name='BAND%i' % i, value=band,
                                       comment='Band name in this catalog'))

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick',
            'target_extent', 'ccds', 'bands', 'survey']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                   plots, ps, splinesky, gaussPsf, pixPsf, do_calibs,
                   gitver, targetwcs, get_depth_maps=False):
    from legacypipe.survey import wcs_for_brick
    from collections import Counter

    # Add some margin to our DESI depth requirements
    margin = 0.5
    target_depth_map = dict(g=24.0 + margin, r=23.4 + margin, z=22.5 + margin)

    #target_percentiles = np.array([2, 5, 10])
    #target_ddepths = np.array([-0.6, -0.3, 0.])

    # List extra (redundant) target percentiles so that increasing the depth at
    # any of these percentiles causes the image to be kept.
    target_percentiles = np.array(list(range(2, 10)) +
                                  list(range(10, 30, 5)) +
                                  list(range(30, 101, 10)))
    target_ddepths = np.zeros(len(target_percentiles), np.float32)
    target_ddepths[target_percentiles < 10] = -0.3
    target_ddepths[target_percentiles <  5] = -0.6
    #print('Target percentiles:', target_percentiles)
    #print('Target ddepths:', target_ddepths)

    cH,cW = H//10, W//10
    coarsewcs = targetwcs.scale(0.1)
    coarsewcs.imagew = cW
    coarsewcs.imageh = cH

    # Unique pixels in this brick (U: cH x cW boolean)
    U = find_unique_pixels(coarsewcs, cW, cH, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    keep_ccds = []

    # Sort CCDs by a fast proxy for depth.
    pixscale = 3600. * np.sqrt(np.abs(ccds.cd1_1*ccds.cd2_2 - ccds.cd1_2*ccds.cd2_1))
    seeing = ccds.fwhm * pixscale

    depthmaps = []

    # Compute the rectangle in *coarsewcs* covered by each CCD
    slices = []
    for ccd in ccds:
        wcs = survey.get_approx_wcs(ccd)
        hh,ww = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,ww,ww,1], [1,1,hh,hh])
        ok,xx,yy = coarsewcs.radec2pixelxy(rr, dd)
        y0 = int(np.round(np.clip(yy.min(), 0, cH-1)))
        y1 = int(np.round(np.clip(yy.max(), 0, cH-1)))
        x0 = int(np.round(np.clip(xx.min(), 0, cW-1)))
        x1 = int(np.round(np.clip(xx.max(), 0, cW-1)))
        if y0 == y1 or x0 == x1:
            slices.append(None)
            continue
        # Check whether this CCD overlaps the unique area of this brick...
        if not np.any(U[y0:y1+1, x0:x1+1]):
            print('No overlap with unique area for CCD', ccd.expnum, ccd.ccdname)
            slices.append(None)
            continue
        slices.append((slice(y0, y1+1), slice(x0, x1+1)))

    for band in bands:
        # scalar
        target_depth = target_depth_map[band]
        # vector
        target_depths = target_depth + target_ddepths

        depthiv = np.zeros((cH,cW), np.float32)
        depthmap = np.zeros_like(depthiv)
        depthvalue = np.zeros_like(depthiv)
        last_pcts = np.zeros_like(target_depths)
        # indices of CCDs we still want to look at in the current band
        b_inds = np.where(ccds.filter == band)[0]
        print(len(b_inds), 'CCDs in', band, 'band')
        b_inds = np.array([i for i in b_inds if slices[i] is not None])
        print(len(b_inds), 'CCDs in', band, 'band overlap target')
        # CCDs that we will try before searching for good ones -- CCDs
        # from the same exposure number as CCDs we have chosen to
        # take.
        try_ccds = set()

        plot_vals = []

        if plots:
            plt.clf()
            for i in b_inds:
                sy,sx = slices[i]
                x0,x1 = sx.start, sx.stop
                y0,y1 = sy.start, sy.stop
                plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-', alpha=0.5)
            plt.title('CCDs overlapping brick: %i in %s band' % (len(b_inds), band))
            ps.savefig()

        while len(b_inds):
            if len(try_ccds) == 0:
                # Choose the next CCD to look at in this band.
    
                # A rough point-source depth proxy would be:
                # metric = np.sqrt(ccds.extime[b_inds]) / seeing[b_inds]
                # If we want to put more weight on choosing good-seeing images, we could do:
                metric = np.sqrt(ccds.exptime[b_inds]) / seeing[b_inds]**2
                # This metric is *BIG* for *GOOD* ccds!
    
                # Here, we try explicitly to include CCDs that cover
                # pixels that are still shallow by the largest amount
                # for the largest number of percentiles of interest;
                # note that pixels with no coverage get depth 0, so
                # score high in this metric.
                #
                # The value is the depth still required to hit the
                # target, summed over percentiles of interest
                # (for pixels unique to this brick)
                depthvalue[:,:] = 0.
                active = (last_pcts < target_depths)
                for d,pct in zip(target_depths[active], last_pcts[active]):
                    #print('target percentile depth', d, 'has depth', pct)
                    depthvalue += U * np.maximum(0, d - depthmap)
                ccdvalue = np.zeros(len(b_inds), np.float32)
                for j,i in enumerate(b_inds):
                    #ccdvalue[j] = np.sum(depthvalue[slices[i]])
                    # mean -- we want the most bang for the buck per pixel?
                    ccdvalue[j] = np.mean(depthvalue[slices[i]])
                metric *= ccdvalue

                # *ibest* is an index into b_inds
                ibest = np.argmax(metric)
                # *iccd* is an index into ccds.
                iccd = b_inds[ibest]
                ccd = ccds[iccd]
                print('Chose best CCD: seeing', seeing[iccd], 'exptime', ccds.exptime[iccd], 'with value', ccdvalue[ibest])

            else:
                iccd = try_ccds.pop()
                ccd = ccds[iccd]
                print('Popping CCD from use_ccds list')

            # remove *iccd* from b_inds
            b_inds = b_inds[b_inds != iccd]

            im = survey.get_image_object(ccd)
            print('Band', band, 'expnum', im.expnum, 'exptime', im.exptime, 'seeing', im.fwhm*im.pixscale, 'arcsec')

            # HACK HACK HACK -- these are missing 2017-07-12
            if im.expnum in [401266, 400580, 222612]:
                print('HACK -- skipping expnum', im.expnum)
                continue

            if do_calibs:
                kwa = dict(git_version=gitver)
                if gaussPsf:
                    kwa.update(psfex=False)
                if splinesky:
                    kwa.update(splinesky=True)
                im.run_calibs(**kwa)

            im.check_for_cached_files(survey)
            print(im)
            print('Reading WCS from', im.imgfn, 'HDU', im.hdu)
            wcs = im.get_wcs()
            x0,x1,y0,y1,slc = im.get_image_extent(wcs=wcs, radecpoly=targetrd)
            if x0==x1 or y0==y1:
                print('No actual overlap')
                continue
            wcs = wcs.get_subimage(int(x0), int(y0), int(x1-x0), int(y1-y0))

            skysig1 = im.get_sky_sig1(splinesky=splinesky)
            if skysig1 is None:
                iv = im.read_invvar(slc=slc)
                dq = im.read_dq(slc=slc)
                skysig1 = 1./np.sqrt(np.median(iv[dq == 0]))
            # skysig1 is in image counts; scale to nanomaggies
            zpscale = NanoMaggies.zeropointToScale(im.ccdzpt)
            skysig1 /= zpscale

            psf = im.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf)
            psf = psf.constantPsfAt((x1-x0)//2, (y1-y0)//2)

            # create a fake tim to compute galnorm
            from tractor import (PixPos, Flux, ModelMask, LinearPhotoCal, Image,
                                 NullWCS)
            from legacypipe.survey import SimpleGalaxy

            h,w = 50,50
            gal = SimpleGalaxy(PixPos(w//2,h//2), Flux(1.))
            tim = Image(data=np.zeros((h,w), np.float32),
                        psf=psf, wcs=NullWCS(pixscale=im.pixscale))
            mm = ModelMask(0, 0, w, h)
            galmod = gal.getModelPatch(tim, modelMask=mm).patch
            galmod = np.maximum(0, galmod)
            galmod /= galmod.sum()
            galnorm = np.sqrt(np.sum(galmod**2))
            detiv = 1. / (skysig1 / galnorm)**2
            galdepth = -2.5 * (np.log10(5. * skysig1 / galnorm) - 9.)
            print('Galdepth for this CCD:', galdepth)

            # Add this image the the depth map...
            from astrometry.util.resample import resample_with_wcs, OverlapError
            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(coarsewcs, wcs)
                print(len(Yo), 'of', (cW*cH), 'pixels covered by this image')
            except OverlapError:
                print('No overlap')
                continue
            depthiv[Yo,Xo] += detiv

            # compute the new depth map & percentiles (including the proposed new CCD)
            depthmap[:,:] = 0.
            depthmap[depthiv > 0] = 22.5 - 2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
            depthpcts = np.percentile(depthmap[U], target_percentiles)

            for i,(p,d,t) in enumerate(zip(target_percentiles, depthpcts, target_depths)):
                print('  pct % 3i, prev %5.2f -> %5.2f vs target %5.2f %s' % (p, last_pcts[i], d, t, ('ok' if d >= t else '')))

            keep = False
            # Did we increase the depth of any target percentile that did not already exceed its target depth?
            if np.any((depthpcts > last_pcts) * (last_pcts < target_depths)):
                keep = True

            # Add any other CCDs from this same expnum to the try_ccds list.
            # (before making the plot)
            I = np.where(ccd.expnum == ccds.expnum[b_inds])[0]
            try_ccds.update(b_inds[I])
            print('Adding', len(I), 'CCDs with the same expnum to try_ccds list')

            if plots:
                cc = '1' if keep else '0'
                xx = [Xo.min(), Xo.min(), Xo.max(), Xo.max(), Xo.min()]
                yy = [Yo.min(), Yo.max(), Yo.max(), Yo.min(), Yo.min()]
                plot_vals.append(((xx,yy,cc),(last_pcts,depthpcts,keep),im.ccdname))

            if plots and (
                (len(try_ccds) == 0) or np.all(depthpcts >= target_depths)):
                plt.clf()

                plt.subplot2grid((2,2),(0,0))
                plt.imshow(depthvalue, interpolation='nearest', origin='lower',
                           vmin=0)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('heuristic value')

                plt.subplot2grid((2,2),(0,1))
                plt.imshow(depthmap, interpolation='nearest', origin='lower',
                           vmin=target_depth - 2, vmax=target_depth + 0.5)
                ax = plt.axis()
                for (xx,yy,cc) in [p[0] for p in plot_vals]:
                    plt.plot(xx,yy, '-', color=cc, lw=3)
                plt.axis(ax)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('depth map')

                plt.subplot2grid((2,2),(1,0), colspan=2)
                ax = plt.gca()
                plt.plot(target_percentiles, target_depths, 'ro', label='Target')
                plt.plot(target_percentiles, target_depths, 'r-')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    plt.plot(target_percentiles, lp, 'k-',
                             label='Previous percentiles')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    cc = 'b' if k else 'r'
                    plt.plot(target_percentiles, dp, '-', color=cc,
                             label='Depth percentiles')
                ccdnames = ','.join([p[2] for p in plot_vals])
                plot_vals = []

                plt.ylim(target_depth - 2, target_depth + 0.5)
                plt.xscale('log')
                plt.xlabel('Percentile')
                plt.ylabel('Depth')
                plt.title('depth percentiles')
                plt.suptitle('%s %i-%s, exptime %.0f, seeing %.2f, band %s' %
                             (im.camera, im.expnum, ccdnames, im.exptime,
                              im.pixscale * im.fwhm, band))
                ps.savefig()

            if not keep:
                print('Not keeping this exposure')
                depthiv[Yo,Xo] -= detiv
                continue

            keep_ccds.append(iccd)
            last_pcts = depthpcts

            if np.all(depthpcts >= target_depths):
                print('Reached all target depth percentiles for band', band)
                break

        if get_depth_maps:
            if np.any(depthiv > 0):
                depthmap[:,:] = 0.
                depthmap[depthiv > 0] = 22.5 -2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
                depthmap[np.logical_not(U)] = np.nan
                depthmaps.append((band, depthmap.copy()))

        if plots:
            I = np.where(ccds.filter == band)[0]
            plt.clf()
            plt.plot(seeing[I], ccds.exptime[I], 'k.')
            # which CCDs from this band are we keeping?
            kept = np.array(keep_ccds)
            if len(kept):
                kept = kept[ccds.filter[kept] == band]
                plt.plot(seeing[kept], ccds.exptime[kept], 'ro')
            plt.xlabel('Seeing (arcsec)')
            plt.ylabel('Exptime (sec)')
            plt.title('CCDs kept for band %s' % band)
            yl,yh = plt.ylim()
            plt.ylim(0, np.max(ccds.exptime[I]) * 1.1)
            ps.savefig()

    if get_depth_maps:
        return (keep_ccds, depthmaps)
    return keep_ccds

def stage_mask_junk(tims=None, targetwcs=None, W=None, H=None, bands=None,
                    mp=None, nsigma=None, plots=None, ps=None, record_event=None,
                    **kwargs):
    '''
    This pipeline stage tries to detect artifacts in the individual
    exposures, by running a detection step and removing blobs with
    large axis ratio (long, thin objects, often satellite trails).
    '''
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import label, find_objects, center_of_mass
    from scipy.linalg import svd

    record_event and record_event('stage_mask_junk: starting')

    if plots:
        coimgs,cons = quick_coadds(tims, bands, targetwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Before mask_junk')
        ps.savefig()

    allss = []
    for tim in tims:
        # Create a detection map for this image and detect blobs
        det = tim.data * (tim.inverr > 0)
        det = gaussian_filter(det, tim.psf_sigma) / tim.psfnorm**2
        detsig1 = tim.sig1 / tim.psfnorm
        det = (det > (nsigma * detsig1))
        det = binary_fill_holes(det)
        timblobs,timnblobs = label(det)
        timslices = find_objects(timblobs)

        if plots:
            zeroed = np.zeros(tim.shape, bool)

        for i,slc in enumerate(timslices):
            # Compute moments for each blob
            inblob = timblobs[slc]
            inblob = (inblob == (i+1))
            cy,cx = center_of_mass(inblob)
            bh,bw = inblob.shape
            xx = np.arange(bw)
            yy = np.arange(bh)
            ninblob = float(np.sum(inblob))
            cxx = np.sum(((xx - cx)**2)[np.newaxis,:] * inblob) / ninblob
            cyy = np.sum(((yy - cy)**2)[:,np.newaxis] * inblob) / ninblob
            cxy = np.sum((yy - cy)[:,np.newaxis] *
                         (xx - cx)[np.newaxis,:] * inblob) / ninblob
            C = np.array([[cxx, cxy],[cxy, cyy]])
            u,s,v = svd(C)
            allss.append(np.sqrt(np.abs(s)))

            # For an ellipse, this gives the major and minor axes
            # (ie, end to end, not semi-major/minor)
            ss = np.sqrt(s)
            major = 4. * ss[0]
            minor = 4. * ss[1]
            # Remove only long, thin objects.
            if not (major > 200 and minor/major < 0.1):
                continue
            # Zero it out!
            tim.inverr[slc] *= np.logical_not(inblob)
            if tim.dq is not None:
                # Add to dq mask bits
                tim.dq[slc] |= CP_DQ_BITS['longthin']

            rd = tim.wcs.pixelToPosition(cx, cy)
            ra,dec = rd.ra, rd.dec
            ok,bx,by = targetwcs.radec2pixelxy(ra, dec)
            print('Zeroing out a source with major/minor axis', major, '/',
                  minor, 'at centroid RA,Dec=(%.4f,%.4f), brick coords %i,%i'
                  % (ra, dec, bx, by))

            if plots:
                zeroed[slc] = np.logical_not(inblob)

        if plots:
            if np.sum(zeroed) > 0:
                plt.clf()
                from scipy.ndimage.morphology import binary_dilation
                zout = (binary_dilation(zeroed, structure=np.ones((3,3)))
                        - zeroed)
                dimshow(tim.getImage(), vmin=-3.*tim.sig1, vmax=10.*tim.sig1)
                outline = np.zeros(zout.shape+(4,), np.uint8)
                outline[:,:,1] = zout*255
                outline[:,:,3] = zout*255
                dimshow(outline)
                plt.title('Masked: ' + tim.name)
                ps.savefig()

    if plots:
        coimgs,cons = quick_coadds(tims, bands, targetwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('After mask_junk')
        ps.savefig()

        allss = np.array(allss)
        mx = max(allss.max(), 100) * 1.1
        plt.clf()
        plt.plot([0, mx], [0, mx], 'k-', alpha=0.2)
        plt.plot(allss[:,1], allss[:,0], 'b.')
        plt.ylabel('Major axis size (pixels)')
        plt.xlabel('Minor axis size (pixels)')
        plt.axis('scaled')
        plt.axis([0, mx, 0, mx])
        ps.savefig()

        plt.clf()
        plt.plot(allss[:,0], (allss[:,1]+1.) / (allss[:,0]+1.), 'b.')
        plt.xlabel('Major axis size (pixels)')
        plt.ylabel('Axis ratio')
        plt.axis([0, mx, 0, 1])
        plt.axhline(0.1, color='r', alpha=0.2)
        plt.axvline(200, color='r', alpha=0.2)
        ps.savefig()

    return dict(tims=tims)

def stage_image_coadds(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobs=None, lanczos=True, ccds=None,
                       mp=None, record_event=None,
                       **kwargs):
    record_event and record_event('stage_image_coadds: starting')
    '''
    Immediately after reading the images, we can create coadds of just
    the image products.  Later, full coadds including the models will
    be created (in `stage_coadds`).  But it's handy to have the coadds
    early on, to diagnose problems or just to look at the data.
    '''
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=version_header)

    record_event and record_event('stage_mask_junk: starting')
            
    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs),
                    mp=mp)

    # Sims: coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_coadd, nil = quick_coadds(
            tims, bands, targetwcs, images=[tim.sims_image for tim in tims])
        image_coadd,nil = quick_coadds(
            tims, bands, targetwcs, images=[tim.data - tim.sims_image
                                            for tim in tims])

    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    coadd_list= [('image',C.coimgs,rgbkwargs)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, rgbkwargs))

    for name,ims,rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            print('Wrote', out.fn)

        # Blob-outlined version
        if blobs is not None:
            from scipy.ndimage.morphology import binary_dilation
            outline = (binary_dilation(blobs >= 0, structure=np.ones((3,3)))
                       - (blobs >= 0))
            # coadd_bw
            if len(rgb.shape) == 2:
                rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
            # Outline in green
            rgb[:,:,0][outline] = 0
            rgb[:,:,1][outline] = 1
            rgb[:,:,2][outline] = 0

            with survey.write_output(name+'blob-jpeg', brick=brickname) as out:
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                print('Wrote', out.fn)
        del rgb
    return None

def _median_smooth_detmap(X):
    from scipy.ndimage.filters import median_filter
    from legacypipe.survey import bin_image
    (detmap, detiv, binning) = X
    # Bin down before median-filtering, for speed.
    binned,nil = bin_image(detmap, detiv, binning)
    smoo = median_filter(binned, (50,50))
    return smoo

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               mp=None, nsigma=None,
               on_bricks=False,
               allow_missing_brickq=-1,
               survey=None, brick=None,
               record_event=None,
               **kwargs):
    '''
    In this stage we run SED-matched detection to find objects in the
    images.  For each object detected, a `tractor` source object is
    created: a `tractor.PointSource`, `tractor.ExpGalaxy`,
    `tractor.DevGalaxy`, or `tractor.FixedCompositeGalaxy` object.  In
    this stage, the sources are also split into "blobs" of overlapping
    pixels.  Each of these blobs will be processed independently.
    '''
    from legacypipe.detection import (detection_maps, sed_matched_filters,
                        run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, find_objects, center_of_mass

    record_event and record_event('stage_srcs: starting')

    tlast = Time()

    # existing sources that should be avoided when detecting new
    # sources.
    avoid_x, avoid_y = [],[]

    if on_bricks:
        avoid = _subtract_onbricks_sources(
            survey, brick, allow_missing_brickq, targetwcs, tims, bands,
            plots, ps, mp)
        if avoid is not None:
            avoid_x,avoid_y = avoid

    record_event and record_event('stage_srcs: detection maps')
    print('Rendering detection maps...')
    detmaps, detivs, satmap = detection_maps(tims, targetwcs, bands, mp)
    tnow = Time()
    print('[parallel srcs] Detmaps:', tnow-tlast)
    tlast = tnow
    record_event and record_event('stage_srcs: sources')

    # Median-smooth detection maps
    binning = 4
    smoos = mp.map(_median_smooth_detmap,
                   [(m,iv,binning) for m,iv in zip(detmaps, detivs)])
    tnow = Time()
    print('[parallel srcs] Median-filter detmaps:', tnow-tlast)
    tlast = tnow

    for i,(detmap,detiv,smoo) in enumerate(zip(detmaps, detivs, smoos)):
        # Subtract binned median image.
        S = binning
        for ii in range(S):
            for jj in range(S):
                sh,sw = detmap[ii::S, jj::S].shape
                detmap[ii::S, jj::S] -= smoo[:sh,:sw]

    # Handle the margin of interpolated (masked) pixels around
    # saturated pixels
    saturated_pix = binary_dilation(satmap > 0, iterations=10)
    
    # Read Tycho-2 stars and use as saturated sources.
    tycho = fits_table(survey.find_file('tycho2'))
    print('Read', len(tycho), 'Tycho-2 stars')
    ok,xx,yy = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    ## ibx = integer brick x
    tycho.ibx = np.round(xx-1.).astype(int)
    tycho.iby = np.round(yy-1.).astype(int)
    margin = 10
    tycho.cut(ok * (tycho.ibx > -margin) * (tycho.ibx < W+margin) *
              (tycho.iby > -margin) * (tycho.iby < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    tycho.ibx = np.clip(tycho.ibx, 0, W-1)
    tycho.iby = np.clip(tycho.iby, 0, H-1)
    del ok,xx,yy

    # Saturated blobs -- create a source for each, except for those
    # that already have a Tycho-2 star
    satblobs,nsat = label(satmap > 0)
    if len(tycho):
        # Build a map from old "satblobs" to new; identity to start
        remap = np.arange(nsat+1)
        # Drop blobs that contain a Tycho-2 star
        zeroout = satblobs[tycho.iby, tycho.ibx]
        remap[zeroout] = 0
        # Renumber them to be contiguous
        I = np.flatnonzero(remap)
        nsat = len(I)
        remap[I] = 1 + np.arange(nsat)
        satblobs = remap[satblobs]
        del remap, zeroout, I

    # Add sources for any remaining saturated blobs
    satyx = center_of_mass(satmap, labels=satblobs, index=np.arange(nsat)+1)
    if len(satyx) == 0:
        Tsat = tycho
    else:
        Tsat = fits_table()
        # NOTE, satyx is in y,x order (center_of_mass)
        Tsat.ibx = np.array([x for y,x in satyx]).astype(int)
        Tsat.iby = np.array([y for y,x in satyx]).astype(int)
        Tsat.ra,Tsat.dec = targetwcs.pixelxy2radec(Tsat.ibx+1, Tsat.iby+1)
        print('Adding', len(Tsat), 'additional saturated stars')
        # MAGIC mag for a saturated star
        Tsat.mag = np.zeros(len(Tsat), np.float32) + 15.
        Tsat = merge_tables([tycho, Tsat], columns='fillzero')
    del satyx
        
    satcat = []
    if len(Tsat):
        avoid_x.extend(Tsat.ibx)
        avoid_y.extend(Tsat.iby)
        # Create catalog entries...
        for r,d,m in zip(Tsat.ra, Tsat.dec, Tsat.mag):
            fluxes = dict([(band, NanoMaggies.magToNanomaggies(m))
                           for band in bands])
            assert(np.all(np.isfinite(list(fluxes.values()))))
            satcat.append(PointSource(RaDecPos(r, d),
                                      NanoMaggies(order=bands, **fluxes)))

    if plots:
        plt.clf()
        dimshow(satmap)
        plt.title('satmap')
        ps.savefig()

        rgb = get_rgb(detmaps, bands)
        plt.clf()
        dimshow(rgb)
        plt.title('detmaps')
        ps.savefig()

        rgb[:,:,0][saturated_pix] = 0
        rgb[:,:,1][saturated_pix] = 1
        rgb[:,:,2][saturated_pix] = 0
        plt.clf()
        dimshow(rgb)
        ax = plt.axis()
        if len(Tsat):
            plt.plot(Tsat.ibx, Tsat.iby, 'ro')
        plt.axis(ax)
        plt.title('detmaps & saturated')
        ps.savefig()

    # SED-matched detections
    record_event and record_event('stage_srcs: SED-matched')
    print('Running source detection at', nsigma, 'sigma')
    SEDs = survey.sed_matched_filters(bands)
    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, (avoid_x,avoid_y), targetwcs,
        nsigma=nsigma, saturated_pix=saturated_pix, plots=plots, ps=ps, mp=mp)
    if Tnew is None:
        raise NothingToDoError('No sources detected.')
    Tnew.delete_column('peaksn')
    Tnew.delete_column('apsn')
    del detmaps
    del detivs

    # Merge newly detected sources with existing saturated source list
    T = merge_tables([Tsat, Tnew], columns='fillzero')
    cat = Catalog(*(satcat + newcat))
    cat.freezeAllParams()

    assert(len(T) > 0)
    assert(len(cat) == len(T))
    
    tnow = Time()
    print('[serial srcs] Peaks:', tnow-tlast)
    tlast = tnow

    if plots:
        if coimgs is None:
            coimgs,cons = quick_coadds(tims, bands, targetwcs)
        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Catalog + SED-matched detections')
        ps.savefig()
        ax = plt.axis()
        p1 = plt.plot(T.ibx, T.iby, 'r+', **crossa)
        p2 = plt.plot(Tnew.ibx, Tnew.iby, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('Catalog + SED-matched detections')
        plt.figlegend((p1[0], p2[0]), ('SDSS', 'New'), 'upper left')
        ps.savefig()
        # for x,y in zip(peakx,peaky):
        #     plt.text(x+5, y, '%.1f' % (hot[np.clip(int(y),0,H-1),
        #                                  np.clip(int(x),0,W-1)]), color='r',
        #              ha='left', va='center')
        # ps.savefig()

    # Segment, and record which sources fall into each blob
    blobs,blobsrcs,blobslices = segment_and_group_sources(
        hot, T, name=brickname, ps=ps, plots=plots)
    del hot

    tnow = Time()
    print('[serial srcs] Blobs:', tnow-tlast)
    tlast = tnow

    keys = ['T', 'tims', 'blobsrcs', 'blobslices', 'blobs', 'cat',
            'ps', 'tycho']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def _subtract_onbricks_sources(survey, brick, allow_missing_brickq,
                               targetwcs, tims, bands, plots, ps, mp):
    # Check tractor-on-bricks dependencies
    bricks = on_bricks_dependencies(brick, survey)
    if len(bricks) == 0:
        return None
    from legacypipe.catalog import read_fits_catalog
    B = []
    for b in bricks:
        fn = survey.find_file('tractor', brick=b.brickname)
        print('Looking for', fn)
        if not os.path.exists(fn):
            print('File does not exist:', fn)
            if b.brickq <= allow_missing_brickq:
                print(('  (allowing this missing brick (brickq = %i) ' +
                       'because of --allow-missing-brickq %i)') % 
                       (b.brickq, allow_missing_brickq))
                continue
        B.append(fits_table(fn))
    del bricks
    B = [b for b in B if b is not None]
    if len(B) == 0:
        return None

    try:
        B = merge_tables(B)
    except:
        print('Error merging brick tables:')
        import traceback
        traceback.print_exc()
        print('Retrying with fillzero...')
        B = merge_tables(B, columns='fillzero')
    print('Total of', len(B), 'sources from neighbouring bricks')
    # Keep only sources that are primary in their own brick
    B.cut(B.brick_primary)
    print(len(B), 'are BRICK_PRIMARY')
    # HACK -- Keep only sources within a margin of this brick
    ok,B.xx,B.yy = targetwcs.radec2pixelxy(B.ra, B.dec)
    margin = 20
    B.cut((B.xx >= 1-margin) * (B.xx < W+margin) *
          (B.yy >= 1-margin) * (B.yy < H+margin))
    print(len(B), 'are within this image + margin')
    B.cut((B.out_of_bounds == False) * (B.left_blob == False))
    print(len(B), 'do not have out_of_bounds or left_blob set')
    if len(B) == 0:
        return None

    # Note that we shouldn't need to drop sources that are within this
    # current brick's unique area, because we cut to sources that are
    # BRICK_PRIMARY within their own brick.

    # Create sources for these catalog entries
    ### see forced-photom-decam.py for some additional patchups?

    bcat = read_fits_catalog(B)
    print('Created', len(bcat), 'tractor catalog objects')

    # Add the new sources to the 'avoid_[xy]' lists, which are
    # existing sources that should be avoided when detecting new
    # faint sources.
    avoid_x = np.round(B.xx - 1).astype(int)
    avoid_y = np.round(B.yy - 1).astype(int)

    print('Subtracting tractor-on-bricks sources belonging to other bricks')
    ## HACK -- note that this is going to screw up fracflux and
    ## other metrics for sources in this brick that overlap
    ## subtracted sources.
    if plots:
        mods = []
        # Before...
        coimgs,cons = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Before subtracting tractor-on-bricks marginal sources')
        ps.savefig()

    for i,src in enumerate(bcat):
        print(' ', i, src)

    tlast = Time()
    mods = mp.map(_get_mod, [(tim, bcat) for tim in tims])
    tnow = Time()
    print('[parallel srcs] Getting tractor-on-bricks model images:',tnow-tlast)
    tlast = tnow
    for tim,mod in zip(tims, mods):
        tim.data -= mod
    if not plots:
        del mods

    if plots:
        coimgs,cons = quick_coadds(tims, bands, targetwcs, images=mods)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Marginal sources subtracted off')
        ps.savefig()
        coimgs,cons = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('After subtracting off marginal sources')
        ps.savefig()

    return avoid_x,avoid_y

def stage_fitblobs(T=None,
                   brickname=None,
                   brickid=None,
                   brick=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None, blobradec=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   tycho=None,
                   rex=False,
                   record_event=None,
                   **kwargs):
    '''
    This is where the actual source fitting happens.
    The `one_blob` function is called for each "blob" of pixels with
    the sources contained within that blob.
    '''
    tlast = Time()
    for tim in tims:
        assert(np.all(np.isfinite(tim.getInvError())))

    record_event and record_event('stage_fitblobs: starting')

    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    if plots:
        coimgs,cons = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0],'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        for i,Isrcs in enumerate(blobsrcs):
            for isrc in Isrcs:
                src = cat[isrc]
                ra,dec = src.getPosition().ra, src.getPosition().dec
                ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                plt.text(x, y, 'b%i/s%i' % (i,isrc),
                         ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs + Sources')
        ps.savefig()

        plt.clf()
        dimshow(blobs)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0,bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        plt.clf()
        dimshow(blobs != -1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0,by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()


    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()

    tnow = Time()
    print('[serial fitblobs]:', tnow-tlast)
    tlast = tnow

    # Were we asked to only run a subset of blobs?
    keepblobs = None
    if blobradec is not None:
        # blobradec is a list like [(ra0,dec0), ...]
        rd = np.array(blobradec)
        ok,x,y = targetwcs.radec2pixelxy(rd[:,0], rd[:,1])
        x = (x - 1).astype(int)
        y = (y - 1).astype(int)
        blobxy = list(zip(x, y))
        print('Blobradec -> blobxy:', len(blobxy), 'points')

    if blobxy is not None:
        # blobxy is a list like [(x0,y0), (x1,y1), ...]
        keepblobs = []
        for x,y in blobxy:
            x,y = int(x), int(y)
            if x < 0 or x >= W or y < 0 or y >= H:
                print('Warning: clipping blob x,y to brick bounds', x,y)
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)
            blob = blobs[y,x]
            if blob >= 0:
                keepblobs.append(blob)
            else:
                print('WARNING: blobxy', x,y, 'is not in a blob!')
        keepblobs = np.unique(keepblobs)

    if blob0 is not None or (nblobs is not None and nblobs < len(blobslices)):
        if blob0 is None:
            blob0 = 0
        if nblobs is None:
            nblobs = len(blobslices) - blob0
        keepblobs = np.arange(blob0, blob0+nblobs)

    # keepblobs can be None or empty list
    if keepblobs is not None and len(keepblobs):
        # 'blobs' is an image with values -1 for no blob, or the index of the
        # blob.  Create a map from old 'blobs+1' to new 'blobs+1'.  +1  so that
        # -1 is a valid index.
        NB = len(blobslices)
        blobmap = np.empty(NB+1, int)
        blobmap[:] = -1
        blobmap[keepblobs + 1] = np.arange(len(keepblobs))
        # apply the map!
        blobs = blobmap[blobs + 1]

        # 'blobslices' and 'blobsrcs' are lists
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]

        # one more place where blob numbers are recorded...
        T.blob = blobs[T.iby, T.ibx]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if plots:
        plt.clf()
        dimshow(blobs>=0, vmin=0, vmax=1)
        ax = plt.axis()
        plt.plot(tycho.ibx, tycho.iby, 'ro')
        for x,y,mag in zip(tycho.ibx,tycho.iby,tycho.mag):
            plt.text(x, y, '%.1f' % (mag),
                     color='r', fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.5))
        plt.axis(ax)
        plt.title('Tycho-2 stars')
        ps.savefig()

    if checkpoint_filename is None:
        # Run one_blob on each blob!
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho, brick, rex)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))
        R = mp.map(_bounce_one_blob, blobiter)
    else:
        from astrometry.util.ttime import CpuMeas
        from astrometry.util.file import pickle_to_file, trymakedirs

        # Check for existing checkpoint file.
        R = []
        if os.path.exists(checkpoint_filename):
            from astrometry.util.file import unpickle_from_file
            print('Reading', checkpoint_filename)
            try:
                R = unpickle_from_file(checkpoint_filename)
                print('Read', len(R), 'results from checkpoint file',
                      checkpoint_filename)
            except:
                import traceback
                print('Failed to read checkpoint file ' + checkpoint_filename)
                traceback.print_exc()

        def _write_checkpoint(R, checkpoint_filename):
            fn = checkpoint_filename + '.tmp'
            print('Writing checkpoint', fn)
            pickle_to_file(R, fn)
            print('Wrote checkpoint to', fn)
            os.rename(fn, checkpoint_filename)
            print('Renamed temp checkpoint', fn, 'to', checkpoint_filename)

        # Create the iterator over blobs to process
        skipblobs = [B.iblob for B in R if B is not None]
        R = [r for r in R if r is not None]
        print('Skipping', len(skipblobs), 'blobs from checkpoint file')
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho, brick, rex, skipblobs=skipblobs)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))

        d = os.path.dirname(checkpoint_filename)
        if len(d) and not os.path.exists(d):
            trymakedirs(d)

        # Begin running one_blob on each blob...
        Riter = mp.imap_unordered(_bounce_one_blob, blobiter)
        # measure wall time and write out checkpoint file periodically.
        last_checkpoint = CpuMeas()
        while True:
            import multiprocessing
            # Time to write a checkpoint file?
            tnow = CpuMeas()
            dt = tnow.wall_seconds_since(last_checkpoint)
            if dt >= checkpoint_period:
                # Write checkpoint!
                try:
                    _write_checkpoint(R, checkpoint_filename)
                    last_checkpoint = tnow
                    dt = 0.
                except:
                    print('Failed to rename checkpoint file',
                          checkpoint_filename)
                    import traceback
                    traceback.print_exc()
            # Wait for results (with timeout)
            try:
                if mp.is_multiproc():
                    timeout = max(1, checkpoint_period - dt)
                    r = Riter.next(timeout)
                else:
                    r = Riter.next()
                R.append(r)
            except StopIteration:
                print('Done')
                break
            except multiprocessing.TimeoutError:
                # print('Timed out waiting for result')
                continue

        # Write checkpoint when done!
        _write_checkpoint(R, checkpoint_filename)
            
    print('[parallel fitblobs] Fitting sources took:', Time()-tlast)

    # Repackage the results from one_blob...
    
    # one_blob can reduce the number and change the types of sources.
    # Reorder the sources:
    assert(len(R) == len(blobsrcs))
    # Drop now-empty blobs.
    R = [r for r in R if r is not None and len(r)]
    if len(R) == 0:
        raise NothingToDoError('No sources passed significance tests.')
    # Sort results R by 'iblob'
    J = np.argsort([B.iblob for B in R])
    R = [R[j] for j in J]
    # Merge results R into one big table
    BB = merge_tables(R)
    del R
    # Pull out the source indices...
    II = BB.Isrcs
    newcat = BB.sources
    # ... and make the table T parallel with BB.
    T.cut(II)

    assert(len(T) == len(newcat))
    print('Old catalog:', len(cat))
    print('New catalog:', len(newcat))
    assert(len(newcat) > 0)
    cat = Catalog(*newcat)
    ns,nb = BB.fracflux.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.fracmasked.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.fracin.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.rchi2.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.dchisqs.shape
    assert(ns == len(cat))
    assert(nb == 5) # ptsrc, simple, dev, exp, comp

    # Renumber blobs to make them contiguous.
    oldblob = T.blob
    ublob,iblob = np.unique(T.blob, return_inverse=True)
    del ublob
    assert(len(iblob) == len(T))
    T.blob = iblob.astype(np.int32)

    # write out blob map
    if write_metrics:
        # Build map from (old+1) to new blob numbers, for the blob image.
        blobmap = np.empty(blobs.max()+2, int)
        # make sure that dropped blobs -> -1
        blobmap[:] = -1
        # in particular,
        blobmap[0] = -1
        blobmap[oldblob + 1] = iblob
        blobs = blobmap[blobs+1]
        del blobmap

        # copy version_header before modifying it.
        hdr = fitsio.FITSHDR()
        for r in version_header.records():
            hdr.add_record(r)
        # Plug the WCS header cards into these images
        targetwcs.add_to_header(hdr)
        hdr.delete('IMAGEW')
        hdr.delete('IMAGEH')
        hdr.add_record(dict(name='IMTYPE', value='blobmap',
                            comment='LegacySurvey image type'))
        hdr.add_record(dict(name='EQUINOX', value=2000.))

        with survey.write_output('blobmap', brick=brickname) as out:
            out.fits.write(blobs, header=hdr)
    del iblob, oldblob
    blobs = None

    T.brickid   = np.zeros(len(T), np.int32) + brickid
    T.brickname = np.array([brickname] * len(T))
    if len(T.brickname) == 0:
        T.brickname = T.brickname.astype('S8')
    T.objid     = np.arange(len(T)).astype(np.int32)

    # How many sources in each blob?
    from collections import Counter
    ninblob = Counter(T.blob)
    T.ninblob = np.array([ninblob[b] for b in T.blob]).astype(np.int16)
    del ninblob

    # Copy blob results to table T
    T.tycho2inblob = BB.hastycho
    T.dchisq       = BB.dchisqs.astype(np.float32)
    T.left_blob    = np.logical_and(BB.started_in_blob,
                                    np.logical_not(BB.finished_in_blob))
    for k in ['fracflux', 'fracin', 'fracmasked', 'rchi2', 'cpu_source',
              'cpu_blob', 'blob_width', 'blob_height', 'blob_npix',
              'blob_nimages', 'blob_totalpix']:
        T.set(k, BB.get(k))

    # Compute MJD_MIN, MJD_MAX
    T.mjd_min = np.empty(len(T), np.float64)
    T.mjd_min[:] = np.inf
    T.mjd_max = np.empty(len(T), np.float64)
    T.mjd_max[:] = -np.inf
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    for tim in tims:
        ok,x,y = tim.subwcs.radec2pixelxy(ra, dec)
        x -= 1
        y -= 1
        I = np.flatnonzero(ok * (x >= 0.5) * (x <= W-0.5) *
                           (y >= 0.5) * (y <= H-0.5))
        T.mjd_min[I] = np.minimum(T.mjd_min[I], tim.time.toMjd())
        T.mjd_max[I] = np.maximum(T.mjd_max[I], tim.time.toMjd())

    invvars = np.hstack(BB.srcinvvars)
    assert(cat.numberOfParams() == len(invvars))

    if write_metrics or get_all_models:
        TT,hdr = _format_all_models(T, newcat, BB, bands, rex)
        if get_all_models:
            all_models = TT
        if write_metrics:
            primhdr = fitsio.FITSHDR()
            for r in version_header.records():
                primhdr.add_record(r)
                primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                                        comment='NOAO data product type'))

            with survey.write_output('all-models', brick=brickname) as out:
                TT.writeto(None, fits_object=out.fits, header=hdr,
                           primheader=primhdr)

    keys = ['cat', 'invvars', 'T', 'blobs']
    if get_all_models:
        keys.append('all_models')
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def _format_all_models(T, newcat, BB, bands, rex):
    from legacypipe.catalog import prepare_fits_catalog, fits_typemap
    from astrometry.util.file import pickle_to_file

    TT = fits_table()
    # Copy only desired columns...
    for k in ['blob', 'brickid', 'brickname', 'dchisq', 'objid',
              'cpu_source', 'cpu_blob', 'ninblob',
              'blob_width', 'blob_height', 'blob_npix', 'blob_nimages',
              'blob_totalpix']:
        TT.set(k, T.get(k))
    TT.type = np.array([fits_typemap[type(src)] for src in newcat])

    hdr = fitsio.FITSHDR()

    if rex:
        simpname = 'rex'
    else:
        simpname = 'simple'
    srctypes = ['ptsrc', simpname, 'dev','exp','comp']

    for srctype in srctypes:
        # Create catalog with the fit results for each source type
        xcat = Catalog(*[m.get(srctype,None) for m in BB.all_models])
        # NOTE that for Rex, the shapes have been converted to EllipseE
        # and the e1,e2 params are frozen.

        namemap = dict(ptsrc='psf', simple='simp')
        prefix = namemap.get(srctype,srctype)

        allivs = np.hstack([m.get(srctype,[]) for m in BB.all_model_ivs])
        assert(len(allivs) == xcat.numberOfParams())
        
        TT,hdr = prepare_fits_catalog(xcat, allivs, TT, hdr, bands, None,
                                      prefix=prefix+'_')
        TT.set('%s_cpu' % prefix,
               np.array([m.get(srctype,0) 
                         for m in BB.all_model_cpu]).astype(np.float32))

    # remove silly columns
    for col in TT.columns():
        # all types
        if '_type' in col:
            TT.delete_column(col)
            continue
        # shapes for shapeless types
        if (('psf_' in col or 'simp_' in col) and
            ('shape' in col or 'fracDev' in col)):
            TT.delete_column(col)
            continue
        # shapeDev for exp sources, vice versa
        if (('exp_' in col and 'Dev' in col) or
            ('dev_' in col and 'Exp' in col) or
            ('rex_' in col and 'Dev' in col)):
            TT.delete_column(col)
            continue
    TT.delete_column('dev_fracDev')
    TT.delete_column('dev_fracDev_ivar')
    if rex:
        TT.delete_column('rex_shapeExp_e1')
        TT.delete_column('rex_shapeExp_e2')
        TT.delete_column('rex_shapeExp_e1_ivar')
        TT.delete_column('rex_shapeExp_e2_ivar')
    return TT,hdr

def _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims, cat, bands,
               plots, ps, simul_opt, use_ceres, tycho, brick, rex,
               skipblobs=[]):
    from collections import Counter
    H,W = targetwcs.shape
    tychoblobs = set(blobs[tycho.iby, tycho.ibx])
    # Remove -1 = no blob from set; not strictly necessary, just cosmetic
    try:
        tychoblobs.remove(-1)
    except:
        pass
    print('Blobs containing Tycho-2 stars:', tychoblobs)

    # sort blobs by size so that larger ones start running first
    blobvals = Counter(blobs[blobs>=0])
    blob_order = np.array([i for i,npix in blobvals.most_common()])

    U = find_unique_pixels(targetwcs, W, H, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)

    for nblob,iblob in enumerate(blob_order):
        if iblob in skipblobs:
            print('Skipping blob', iblob)
            continue

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        assert(len(Isrcs) > 0)

        tblob = Time()
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        # Here we assume the "blobs" array has been remapped so that
        # -1 means "no blob", while 0 and up label the blobs, thus
        # iblob equals the value in the "blobs" map.
        blobmask = (blobs[bslc] == iblob)

        # If the blob is solely outside the unique region of this brick,
        # skip it!
        if np.all(U[bslc][blobmask] == False):
            print('This blob is completely outside the unique region of this brick -- skipping')
            yield None
            continue

        # find one pixel within the blob, for debugging purposes
        onex = oney = None
        for y in range(by0, by1):
            ii = np.flatnonzero(blobmask[y-by0,:])
            if len(ii) == 0:
                continue
            onex = bx0 + ii[0]
            oney = y
            break

        hastycho = iblob in tychoblobs

        print('Blob', nblob+1, 'of', len(blobslices), ': blob id:', iblob,
              'sources:', len(Isrcs), 'size:', blobw, 'x', blobh,
              #'center', (bx0+bx1)/2, (by0+by1)/2,
              'brick X: %i,%i, Y: %i,%i' % (bx0,bx1,by0,by1),
              'npix:', np.sum(blobmask),
              'one pixel:', onex,oney, 'has Tycho-2 star:', hastycho)

        # Here we cut out subimages for the blob...
        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])
        subtimargs = []
        for itim,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            #print('blob extent in pixel space of', tim.name, ': x',
            # (sx0,sx1), 'y', (sy0,sy1), 'tim shape', (h,w))
            if sx1 < 0 or sy1 < 0 or sx0 > w or sy0 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subie  = tim.getInvError()[subslc]
            subwcs = tim.getWcs().shifted(sx0, sy0)
            # Note that we *don't* shift the PSF here -- we do that
            # in the one_blob code.
            subsky = tim.getSky().shifted(sx0, sy0)
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            if hasattr(tim.psf, 'clear_cache'):
                tim.psf.clear_cache()
            subtimargs.append((subimg, subie, subwcs, tim.subwcs,
                               tim.getPhotoCal(),
                               subsky, tim.psf, tim.name, sx0, sx1, sy0, sy1,
                               tim.band, tim.sig1, tim.modelMinval,
                               tim.imobj))

        yield (nblob, iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh,
               blobmask, subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
               simul_opt, use_ceres, hastycho, rex)

def _bounce_one_blob(X):
    ''' This just wraps the one_blob function, for debugging &
    multiprocessing purposes.
    '''
    from legacypipe.oneblob import one_blob
    try:
        return one_blob(X)
    except:
        import traceback
        print('Exception in one_blob:')
        if X is not None:
            print('(iblob = %i)' % (X[0]))
        traceback.print_exc()
        raise

def _get_mod(X):
    (tim, srcs) = X
    t0 = Time()
    tractor = Tractor([tim], srcs)

    if hasattr(tim, 'modelMinval'):
        print('tim modelMinval', tim.modelMinval)
        minval = tim.modelMinval
    else:
        # this doesn't really help when using pixelized PSFs / FFTs
        tim.modelMinval = minval = tim.sig * 0.1

    for src in srcs:
        from tractor.galaxy import ProfileGalaxy
        if not isinstance(src, ProfileGalaxy):
            continue
        px,py = tim.wcs.positionToPixel(src.getPosition())
        h = src._getUnitFluxPatchSize(tim, px, py, minval)
        if h > 512:
            print('halfsize', h, 'for', src)
            src.halfsize = 512

    mod = tractor.getModelImage(0)
    print('Getting model for', tim, ':', Time()-t0)
    return mod

def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 T=None, cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                 mp=None, on_bricks=None,
                 record_event=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage, we have all the source
    model fits, and we can create coadds of the images, model, and
    residuals.  We also perform aperture photometry in this stage.
    '''
    from legacypipe.survey import apertures_arcsec
    from functools import reduce
    tlast = Time()

    record_event and record_event('stage_coadds: starting')

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    tnow = Time()
    print('[serial coadds]:', tnow-tlast)
    tlast = tnow
    # Render model images...
    record_event and record_event('stage_coadds: model images')
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])

    tnow = Time()
    print('[parallel coadds] Getting model images:', tnow-tlast)
    tlast = tnow

    # Compute source pixel positions
    assert(len(T) == len(cat))
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
    
    # Get integer brick pixel coords for each source, for referencing maps
    T.out_of_bounds = reduce(np.logical_or, [xx < 0.5, yy < 0.5,
                                             xx > W+0.5, yy > H+0.5])
    ixy = (np.clip(np.round(xx - 1), 0, W-1).astype(int),
           np.clip(np.round(yy - 1), 0, H-1).astype(int))
    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale
    # Aperture photometry locations
    apxy = np.vstack((xx - 1., yy - 1.)).T
    del xx,yy,ok,ra,dec

    record_event and record_event('stage_coadds: coadds')
    C = make_coadds(tims, bands, targetwcs, mods=mods, xy=ixy,
                    ngood=True, detmaps=True, psfsize=True, lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs),
                    plots=False, ps=ps, mp=mp)
    record_event and record_event('stage_coadds: extras')
    
    # Coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_mods = [tim.sims_image for tim in tims]
        T_sims_coadds = make_coadds(tims, bands, targetwcs, mods=sims_mods,
                                    lanczos=lanczos, mp=mp)
        sims_coadd = T_sims_coadds.comods
        del T_sims_coadds
        image_only_mods= [tim.data-tim.sims_image for tim in tims]
        T_image_coadds = make_coadds(tims, bands, targetwcs,
                                     mods=image_only_mods,
                                     lanczos=lanczos, mp=mp)
        image_coadd= T_image_coadds.comods
        del T_image_coadds
    ###

    for c in ['nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth']:
        T.set(c, C.T.get(c))
    # store galaxy sim bounding box in Tractor cat
    if 'sims_xy' in C.T.get_columns():
        T.set('sims_xy', C.T.get('sims_xy'))

    # Compute depth histogram
    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    coadd_list= [('image', C.coimgs,   rgbkwargs),
                 ('model', C.comods,   rgbkwargs),
                 ('resid', C.coresids, rgbkwargs_resid)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, rgbkwargs))

    for name,ims,rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        if on_bricks and name == 'image':
            # Do not overwrite the image.jpg file if it exists (eg,
            # written during image_coadds stage), because in the later
            # stage_srcs, we subtract the overlapping sources from
            # other bricks, modifying the images.
            fn = survey.find_file(name + '-jpeg', brick=brickname, output=True)
            if os.path.exists(fn):
                print('Not overwriting existing image %s: on_bricks' % fn)
                continue

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            print('Wrote', out.fn)
        del rgb

    if plots:
        plt.clf()
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        ok,x0,y0 = targetwcs.radec2pixelxy(T.orig_ra, T.orig_dec)
        ok,x1,y1 = targetwcs.radec2pixelxy(ra, dec)
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        #plt.plot(np.vstack((x0,x1))-1, np.vstack((y0,y1))-1, 'r-')
        for xx0,yy0,xx1,yy1 in zip(x0,y0,x1,y1):
            plt.plot([xx0-1,xx1-1], [yy0-1,yy1-1], 'r-')
        plt.plot(x1-1, y1-1, 'r.')
        plt.axis(ax)
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        ps.savefig()

        for i,(src,x,y,rr,dd) in enumerate(zip(cat, x1, y1, ra, dec)):
            ee = []
            ec = []
            cc = None
            green = (0.2,1,0.2)
            if isinstance(src, PointSource):
                plt.plot(x, y, 'o', mfc=green, mec='k', alpha=0.6)
            elif isinstance(src, ExpGalaxy):
                ee = [src.shape]
                cc = '0.8'
                ec = [cc]
            elif isinstance(src, DevGalaxy):
                ee = [src.shape]
                cc = green
                ec = [cc]
            elif isinstance(src, FixedCompositeGalaxy):
                ee = [src.shapeExp, src.shapeDev]
                cc = 'm'
                ec = ['m', 'c']
            else:
                print('Unknown type:', src)
                continue

            for e,c in zip(ee, ec):
                G = e.getRaDecBasis()
                angle = np.linspace(0, 2.*np.pi, 60)
                xy = np.vstack((np.append([0,0,1], np.sin(angle)),
                                np.append([0,1,0], np.cos(angle)))).T
                rd = np.dot(G, xy.T).T
                r = rr + rd[:,0] * np.cos(np.deg2rad(dd))
                d = dd + rd[:,1]
                ok,xx,yy = targetwcs.radec2pixelxy(r, d)
                x1,x2,x3 = xx[:3]
                y1,y2,y3 = yy[:3]
                plt.plot([x3, x1, x2], [y3, y1, y2], '-', color=c)
                plt.plot(x1, y1, '.', color=cc, ms=3, alpha=0.6)
                xx = xx[3:]
                yy = yy[3:]
                plt.plot(xx, yy, '-', color=c)
        plt.axis(ax)
        ps.savefig()

    tnow = Time()
    print('[serial coadds] Aperture photometry, wrap-up', tnow-tlast)

    return dict(T=T, AP=C.AP, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec)

def _depth_histogram(brick, targetwcs, bands, detivs, galdetivs):
    # Compute the brick's unique pixels.
    U = None
    if hasattr(brick, 'ra1'):
        print('Computing unique brick pixels...')
        H,W = targetwcs.shape
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        U = np.flatnonzero(U)
        print(len(U), 'of', W*H, 'pixels are unique to this brick')

    # depth histogram bins
    depthbins = np.arange(20, 25.001, 0.1)
    depthbins[0] = 0.
    depthbins[-1] = 100.
    D = fits_table()
    D.depthlo = depthbins[:-1].astype(np.float32)
    D.depthhi = depthbins[1: ].astype(np.float32)

    for band,detiv,galdetiv in zip(bands,detivs,galdetivs):
        for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
            # compute stats for 5-sigma detection
            with np.errstate(divide='ignore'):
                depth = 5. / np.sqrt(det)
            # that's flux in nanomaggies -- convert to mag
            depth = -2.5 * (np.log10(depth) - 9)
            # no coverage -> very bright detection limit
            depth[np.logical_not(np.isfinite(depth))] = 0.
            if U is not None:
                depth = depth.flat[U]
            if len(depth):
                print(band, name, 'band depth map: percentiles',
                      np.percentile(depth, np.arange(0,101, 10)))
            # histogram
            D.set('counts_%s_%s' % (name, band),
                  np.histogram(depth, bins=depthbins)[0].astype(np.int32))
    return D

def stage_wise_forced(
    cat=None,
    T=None,
    targetwcs=None,
    W=None, H=None,
    brickname=None,
    unwise_dir=None,
    unwise_tr_dir=None,
    brick=None,
    wise_ceres=True,
    mp=None,
    record_event=None,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    from wise.forcedphot import unwise_tiles_touching_wcs

    record_event and record_event('stage_wise_forced: starting')

    # Here we assume the targetwcs is axis-aligned and that the
    # edge midpoints yield the RA,Dec limits (true for TAN).
    r,d = targetwcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                                  np.array([H/2, H/2, 1,   H  ]))
    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [r[0], r[1], d[2], d[3]]

    tiles = unwise_tiles_touching_wcs(targetwcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    wcat = []
    for src in cat:
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)

    # PSF broadening in post-reactivation data, by band.
    # From Aaron's email to decam-chatter, 2016-12-12.
    broadening = { 1: 1.0294, 2: 1.0253, 3: None, 4: None }

    # Create list of groups-of-tiles to photometer
    args = []
    # Skip if $UNWISE_COADDS_DIR or --unwise-dir not set.
    if unwise_dir is not None:
        for band in [1,2,3,4]:
            args.append((wcat, tiles, band, roiradec, unwise_dir, wise_ceres,
                         broadening[band]))

    # Add time-resolved WISE coadds
    # Skip if $UNWISE_COADDS_TIMERESOLVED_DIR or --unwise-tr-dir not set.
    eargs = []
    if unwise_tr_dir is not None:
        tdir = unwise_tr_dir
        W = fits_table(os.path.join(tdir, 'time_resolved_neo1-atlas.fits'))
        print('Read', len(W), 'time-resolved WISE coadd tiles')
        W.cut(np.array([t in tiles.coadd_id for t in W.coadd_id]))
        print('Cut to', len(W), 'time-resolved vs', len(tiles), 'full-depth')
        assert(len(W) == len(tiles))
        # this ought to be enough for anyone =)
        Nepochs = 7
        # Add time-resolved coadds
        for band in [1,2]:
            # W1 is bit 0 (value 0x1), W2 is bit 1 (value 0x2)
            bitmask = (1 << (band-1))
            #ntiles,nepochs = W.epoch_bitmask.shape
            for e in range(Nepochs):
                # Which tiles have images for this epoch?
                I = np.flatnonzero(W.epoch_bitmask[:,e] & bitmask)
                if len(I) == 0:
                    continue
                print('Epoch %i: %i tiles:' % (e, len(I)), W.coadd_id[I])
                edir = os.path.join(tdir, 'e%03i' % e)
                eargs.append((e,(wcat, tiles[I], band, roiradec, edir,
                                 wise_ceres, broadening[band])))

    # Run the forced photometry!
    record_event and record_event('stage_wise_forced: photometry')
    phots = mp.map(_unwise_phot, args + [a for e,a in eargs])
    record_event and record_event('stage_wise_forced: results')

    # Unpack results...
    WISE = None
    if len(phots):
        WISE = phots[0]
    if WISE is not None:
        for i,p in enumerate(phots[1:len(args)]):
            if p is None:
                (wcat,tiles,band) = args[i+1][:3]
                print('"None" result from WISE forced phot:', tiles, band)
                continue
            WISE.add_columns_from(p)
        WISE.rename('tile', 'unwise_tile')

        # Look up mask bits
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        WISE.wise_mask = np.zeros((len(T), 4), np.uint8)
        for tile in tiles.coadd_id:
            from astrometry.util.util import Tan
            # unwise_dir can be a colon-separated list of paths
            found = False
            for d in unwise_dir.split(':'):
                fn = os.path.join(d, tile[:3], tile,
                                  'unwise-%s-msk.fits.gz' % tile)
                print('Looking for unWISE mask file', fn)
                if os.path.exists(fn):
                    found = True
                    break
            if not found:
                print('unWISE mask file for tile', tile, 'does not exist')
                continue
            # read header to pull out WCS (because it's compressed)
            M,hdr = fitsio.read(fn, header=True)
            wcs = Tan(*[float(hdr[k]) for k in
                        ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                         'CD1_1', 'CD1_2', 'CD2_1','CD2_2','NAXIS2','NAXIS1']])
            #print('Read WCS header', wcs)
            ok,xx,yy = wcs.radec2pixelxy(ra, dec)
            hh,ww = wcs.get_height(), wcs.get_width()
            #print('unWISE image size', hh,ww)
            xx = np.round(xx - 1).astype(int)
            yy = np.round(yy - 1).astype(int)
            I = np.flatnonzero(ok * (xx >= 0)*(xx < ww) * (yy >= 0)*(yy < hh))
            print(len(I), 'sources are within tile', tile)
            if len(I) == 0:
                continue
            # Reference the mask image M at yy,xx indices
            Mi = M[yy[I], xx[I]]
            # unpack mask bits
            for band in [1,2,3,4]:
                sd1 = (np.bitwise_and(Mi, 2**(2*(band-1)  )) != 0).astype(int)
                sd2 = (np.bitwise_and(Mi, 2**(2*(band-1)+1)) != 0).astype(int)
                WISE.wise_mask[I, band-1] = sd1 + 2*sd2

    # Unpack time-resolved results...
    WISE_T = None
    if len(phots) > len(args):
        WISE_T = phots[len(args)]
    if WISE_T is not None:
        WISE_T = fits_table()
        phots = phots[len(args):]
        for (e,a),phot in zip(eargs, phots):
            print('Epoch', e, 'photometry:')
            if phot is None:
                print('Failed.')
                continue
            phot.about()
            phot.delete_column('tile')
            for c in phot.columns():
                if not c in WISE_T.columns():
                    x = phot.get(c)
                    WISE_T.set(c, np.zeros((len(x), Nepochs), x.dtype))
                X = WISE_T.get(c)
                X[:,e] = phot.get(c)

    return dict(WISE=WISE, WISE_T=WISE_T)

def _unwise_phot(X):
    from wise.forcedphot import unwise_forcedphot
    (wcat, tiles, band, roiradec, unwise_dir, wise_ceres, broadening) = X
    try:
        W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec, bands=[band],
            unwise_dir=unwise_dir, use_ceres=wise_ceres,
            psf_broadening=broadening)
    except:
        import traceback
        print('unwise_forcedphot failed:')
        traceback.print_exc()

        if wise_ceres:
            print('Trying without Ceres...')
            W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                                  bands=[band], unwise_dir=unwise_dir,
                                  use_ceres=False, psf_broadening=broadening)
        W = None
    return W

def stage_writecat(
    survey=None,
    version_header=None,
    T=None,
    WISE=None,
    WISE_T=None,
    AP=None,
    apertures_arcsec=None,
    cat=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False,
    brickname=None,
    brickid=None,
    brick=None,
    invvars=None,
    allbands='ugrizY',
    record_event=None,
    **kwargs):
    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''
    from legacypipe.catalog import prepare_fits_catalog

    record_event and record_event('stage_writecat: starting')

    TT = T.copy()
    for k in ['ibx','iby']:
        TT.delete_column(k)

    assert(AP is not None)
    # How many apertures?
    ap = AP.get('apflux_img_%s' % bands[0])
    n,A = ap.shape
    TT.apflux       = np.zeros((len(TT), len(bands), A), np.float32)
    TT.apflux_ivar  = np.zeros((len(TT), len(bands), A), np.float32)
    TT.apflux_resid = np.zeros((len(TT), len(bands), A), np.float32)
    for iband,band in enumerate(bands):
        TT.apflux      [:,iband,:] = AP.get('apflux_img_%s'      % band)
        TT.apflux_ivar [:,iband,:] = AP.get('apflux_img_ivar_%s' % band)
        TT.apflux_resid[:,iband,:] = AP.get('apflux_resid_%s'    % band)

    hdr = fs = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs)

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

    for i,ap in enumerate(apertures_arcsec):
        primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                comment='Aperture radius, in arcsec'))

    # Record the meaning of mask bits
    bits = list(CP_DQ_BITS.values())
    bits.sort()
    bitmap = dict((v,k) for k,v in CP_DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if bit in bitmap:
            primhdr.add_record(dict(name='MASKB%i' % i, value=bitmap[bit],
                                    comment='Mask bit 2**%i=%i meaning' %
                                    (i, bit)))

    # Brick pixel positions
    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
    T2.bx0 = (bx - 1.).astype(np.float32)
    T2.by0 = (by - 1.).astype(np.float32)
    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

    T2.delete_column('orig_ra')
    T2.delete_column('orig_dec')
    
    T2.brick_primary = ((T2.ra  >= brick.ra1 ) * (T2.ra  < brick.ra2) *
                        (T2.dec >= brick.dec1) * (T2.dec < brick.dec2))

    if WISE is not None:
        # Convert WISE fluxes from Vega to AB.
        # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
        vega_to_ab = dict(w1=2.699,
                          w2=3.339,
                          w3=5.174,
                          w4=6.620)

        for band in [1,2,3,4]:
            primhdr.add_record(dict(
                name='WISEAB%i' % band, value=vega_to_ab['w%i' % band],
                comment='WISE Vega to AB conv for band %i' % band))

        T2.wise_coadd_id = WISE.unwise_tile
        for band in [1,2,3,4]:
            dm = vega_to_ab['w%i' % band]
            fluxfactor = 10.** (dm / -2.5)
            c = 'w%i_nanomaggies' % band
            WISE.set(c, WISE.get(c) * fluxfactor)
            if WISE_T is not None and band <= 2:
                WISE_T.set(c, WISE_T.get(c) * fluxfactor)
            c = 'w%i_nanomaggies_ivar' % band
            WISE.set(c, WISE.get(c) / fluxfactor**2)
            if WISE_T is not None and band <= 2:
                WISE_T.set(c, WISE_T.get(c) / fluxfactor**2)

        T2.wise_flux = np.vstack([
            WISE.w1_nanomaggies, WISE.w2_nanomaggies,
            WISE.w3_nanomaggies, WISE.w4_nanomaggies]).T
        T2.wise_flux_ivar = np.vstack([
            WISE.w1_nanomaggies_ivar, WISE.w2_nanomaggies_ivar,
            WISE.w3_nanomaggies_ivar, WISE.w4_nanomaggies_ivar]).T
        T2.wise_nobs = np.vstack([
            WISE.w1_nexp, WISE.w2_nexp, WISE.w3_nexp, WISE.w4_nexp]).T
        T2.wise_fracflux = np.vstack([
            WISE.w1_profracflux,WISE.w2_profracflux,
            WISE.w3_profracflux,WISE.w4_profracflux]).T
        T2.wise_rchi2 = np.vstack([
            WISE.w1_prochi2, WISE.w2_prochi2,
            WISE.w3_prochi2, WISE.w4_prochi2]).T

        T2.wise_mask = WISE.wise_mask
        
        if WISE_T is not None:
            T2.wise_lc_flux = np.hstack(
                (WISE_T.w1_nanomaggies[:,np.newaxis,:],
                 WISE_T.w2_nanomaggies[:,np.newaxis,:]))
            T2.wise_lc_flux_ivar = np.hstack(
                (WISE_T.w1_nanomaggies_ivar[:,np.newaxis,:],
                 WISE_T.w2_nanomaggies_ivar[:,np.newaxis,:]))
            T2.wise_lc_nobs = np.hstack(
                (WISE_T.w1_nexp[:,np.newaxis,:],
                 WISE_T.w2_nexp[:,np.newaxis,:]))
            T2.wise_lc_fracflux = np.hstack(
                (WISE_T.w1_profracflux[:,np.newaxis,:],
                 WISE_T.w2_profracflux[:,np.newaxis,:]))
            T2.wise_lc_rchi2 = np.hstack(
                (WISE_T.w1_prochi2[:,np.newaxis,:],
                 WISE_T.w2_prochi2[:,np.newaxis,:]))
            T2.wise_lc_mjd = np.hstack(
                (WISE_T.w1_mjd[:,np.newaxis,:],
                 WISE_T.w2_mjd[:,np.newaxis,:]))
            print('WISE light-curve shapes:', WISE_T.w1_nanomaggies.shape)

    with survey.write_output('tractor-intermediate', brick=brickname) as out:
        T2.writeto(None, fits_object=out.fits, primheader=primhdr, header=hdr)

    ### FIXME -- convert intermediate tractor catalog to final, for now...
    ### FIXME -- note that this is now the only place where 'allbands' is used.

    # The "format_catalog" code expects all lower-case column names...
    for c in T2.columns():
        if c != c.lower():
            T2.rename(c, c.lower())
    from legacypipe.format_catalog import format_catalog
    with survey.write_output('tractor', brick=brickname) as out:
        format_catalog(T2, hdr, primhdr, allbands, None,
                       write_kwargs=dict(fits_object=out.fits))

    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in survey.output_file_hashes.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

    # write fits file with galaxy-sim stuff (xy bounds of each sim)
    if 'sims_xy' in T.get_columns(): 
        sims_data = fits_table()
        sims_data.sims_xy = T.sims_xy
        with survey.write_output('galaxy-sims', brick=brickname) as out:
            sims_data.writeto(None, fits_object=out.fits)

    record_event and record_event('stage_writecat: done')

    return dict(T2=T2)

def run_brick(brick, survey, radec=None, pixscale=0.262,
              width=3600, height=3600,
              zoom=None,
              bands=None,
              allbands=None,
              depth_cut=True,
              nblobs=None, blob=None, blobxy=None, blobradec=None,
              nsigma=6,
              simul_opt=False,
              wise=True,
              lanczos=True,
              early_coadds=False,
              blob_image=False,
              do_calibs=True,
              write_metrics=True,
              on_bricks=False,
              allow_missing_brickq=-1,
              gaussPsf=False,
              pixPsf=False,
              hybridPsf=False,
              rex=False,
              splinesky=False,
              constant_invvar=False,
              ceres=True,
              wise_ceres=True,
              unwise_dir=None,
              unwise_tr_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plot_base=None, plot_number=0,
              record_event=None,
    # These are for the 'stages' infrastructure
              pickle_pat='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=[], forceall=False, write_pickles=True,
              checkpoint_filename=None,
              checkpoint_period=None,
              prereqs_update=None,
              stagefunc = None,
              ):
    '''
    Run the full Legacy Survey data reduction pipeline.

    The pipeline is built out of "stages" that run in sequence.  By
    default, this function will cache the result of each stage in a
    (large) pickle file.  If you re-run, it will read from the
    prerequisite pickle file rather than re-running the prerequisite
    stage.  This can yield faster debugging times, but you almost
    certainly want to turn it off (with `writePickles=False,
    forceall=True`) in production.

    Parameters
    ----------
    brick : string
        Brick name such as '2090m065'.  Can be None if *radec* is given.
    survey : a "LegacySurveyData" object (see common.LegacySurveyData), which is in
        charge of the list of bricks and CCDs to be handled, and where output files
        should be written.
    radec : tuple of floats (ra,dec)
        RA,Dec center of the custom region to run.
    pixscale : float
        Brick pixel scale, in arcsec/pixel.  Default = 0.262
    width, height : integers
        Brick size in pixels.  Default of 3600 pixels (with the default pixel
        scale of 0.262) leads to a slight overlap between bricks.
    zoom : list of four integers
        Pixel coordinates [xlo,xhi, ylo,yhi] of the brick subimage to run.
    bands : string
        Filter (band) names to include; default is "grz".

    Notes
    -----
    You must specify the region of sky to work on, via one of:

    - *brick*: string, brick name such as '2090m065'
    - *radec*: tuple of floats; RA,Dec center of the custom region to run

    If *radec* is given, *brick* should be *None*.  If *brick* is given,
    that brick`s RA,Dec center will be looked up in the
    survey-bricks.fits file.

    You can also change the size of the region to reduce:

    - *pixscale*: float, brick pixel scale, in arcsec/pixel.
    - *width* and *height*: integers; brick size in pixels.  3600 pixels
      (with the default pixel scale of 0.262) leads to a slight overlap
      between bricks.
    - *zoom*: list of four integers, [xlo,xhi, ylo,yhi] of the brick
      subimage to run.

    If you want to measure only a subset of the astronomical objects,
    you can use:

    - *nblobs*: None or int; for debugging purposes, only fit the
       first N blobs.
    - *blob*: int; for debugging purposes, start with this blob index.
    - *blobxy*: list of (x,y) integer tuples; only run the blobs
      containing these pixels.
    - *blobradec*: list of (RA,Dec) tuples; only run the blobs
      containing these coordinates.

    Other options:

    - *nsigma*: float; detection threshold in sigmas.

    - *simul_opt*: boolean; during fitting, if a blob contains multiple
      sources, run a step of fitting the sources simultaneously?

    - *wise*: boolean; run WISE forced photometry?

    - *early_coadds*: boolean; generate the early coadds?

    - *do_calibs*: boolean; run the calibration preprocessing steps?

    - *write_metrics*: boolean; write out a variety of useful metrics

    - *on_bricks*: boolean; tractor-on-bricks?

    - *gaussPsf*: boolean; use a simpler single-component Gaussian PSF model?

    - *pixPsf*: boolean; use the pixelized PsfEx PSF model and FFT convolution?

    - *hybridPsf*: boolean; use combo pixelized PsfEx + Gaussian approx model
    
    - *splinesky*: boolean; use the splined sky model (default is constant)?

    - *ceres*: boolean; use Ceres Solver when possible?

    - *wise_ceres*: boolean; use Ceres Solver for unWISE forced photometry?

    - *unwise_dir*: string; where to look for unWISE coadd files.
      This may be a colon-separated list of directories to search in
      order.

    - *unwise_tr_dir*: string; where to look for time-resolved
      unWISE coadd files.  This may be a colon-separated list of
      directories to search in order.

    - *threads*: integer; how many CPU cores to use

    Plotting options:

    - *coadd_bw*: boolean: if only one band is available, make B&W coadds?
    - *plots*: boolean; make a bunch of plots?
    - *plots2*: boolean; make a bunch more plots?
    - *plot_base*: string, default brick-BRICK, the plot filename prefix.
    - *plot_number*: integer, default 0, starting number for plot filenames.

    Options regarding the "stages":

    - *pickle_pat*: string; filename for 'pickle' files
    - *stages*: list of strings; stages (functions stage_*) to run.

    - *force*: list of strings; prerequisite stages that will be run
      even if pickle files exist.
    - *forceall*: boolean; run all stages, ignoring all pickle files.
    - *write_pickles*: boolean; write pickle files after each stage?

    Raises
    ------
    RunbrickError
        If an invalid brick name is given.
    NothingToDoError
        If no CCDs, or no photometric CCDs, overlap the given brick or region.

    '''
    from astrometry.util.stages import CallGlobalTime, runstage
    from astrometry.util.multiproc import multiproc
    from legacypipe.utils import MyMultiproc

    print('Total Memory Available to Job:')
    get_ulimit()

    # *initargs* are passed to the first stage (stage_tims)
    # so should be quantities that shouldn't get updated from their pickled
    # values.
    initargs = {}
    # *kwargs* update the pickled values from previous stages
    kwargs = {}

    forceStages = [s for s in stages]
    forceStages.extend(force)
    if forceall:
        kwargs.update(forceall=True)

    if allbands is not None:
        kwargs.update(allbands=allbands)

    if radec is not None:
        print('RA,Dec:', radec)
        assert(len(radec) == 2)
        ra,dec = radec
        try:
            ra = float(ra)
        except:
            from astrometry.util.starutil_numpy import hmsstring2ra
            ra = hmsstring2ra(ra)
        try:
            dec = float(dec)
        except:
            from astrometry.util.starutil_numpy import dmsstring2dec
            dec = dmsstring2dec(dec)
        print('Parsed RA,Dec', ra,dec)
        initargs.update(ra=ra, dec=dec)
        if brick is None:
            brick = ('custom-%06i%s%05i' %
                         (int(1000*ra), 'm' if dec < 0 else 'p',
                          int(1000*np.abs(dec))))
    initargs.update(brickname=brick, survey=survey)

    if stagefunc is None:
        stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plot_base is None:
        plot_base = plot_base_default
    ps = PlotSequence(plot_base % dict(brick=brick))
    initargs.update(ps=ps)
    if plot_number:
        ps.skipto(plot_number)

    kwargs.update(ps=ps, nsigma=nsigma,
                  gaussPsf=gaussPsf, pixPsf=pixPsf, hybridPsf=hybridPsf,
                  rex=rex,
                  constant_invvar=constant_invvar,
                  depth_cut=depth_cut,
                  splinesky=splinesky,
                  simul_opt=simul_opt,
                  use_ceres=ceres,
                  wise_ceres=wise_ceres,
                  do_calibs=do_calibs,
                  write_metrics=write_metrics,
                  lanczos=lanczos,
                  on_bricks=on_bricks,
                  allow_missing_brickq=allow_missing_brickq,
                  unwise_dir=unwise_dir,
                  unwise_tr_dir=unwise_tr_dir,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  force=forceStages, write=write_pickles,
                  record_event=record_event)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)

    if threads and threads > 1:
        # py3: imported TimingPool
        if sys.version_info[0] >= 3:
            #from legacypipe.timingpool import TimingPool, TimingPoolMeas
            from multiprocessing.pool import Pool
            pool = Pool(processes=threads, initializer=runbrick_global_init, initargs=[])
        else:
            from astrometry.util.timingpool import TimingPool, TimingPoolMeas
            pool = TimingPool(threads, initializer=runbrick_global_init,
                              initargs=[])
            Time.add_measurement(TimingPoolMeas(pool, pickleTraffic=False))
        mp = MyMultiproc(None, pool=pool)
    else:
        mp = MyMultiproc(init=runbrick_global_init, initargs=[])
        pool = None
    kwargs.update(mp=mp)

    if nblobs is not None:
        kwargs.update(nblobs=nblobs)
    if blob is not None:
        kwargs.update(blob0=blob)
    if blobxy is not None:
        kwargs.update(blobxy=blobxy)
    if blobradec is not None:
        kwargs.update(blobradec=blobradec)

    pickle_pat = pickle_pat % dict(brick=brick)

    prereqs = {
        'tims':None,
        'mask_junk': 'tims',
        'srcs': 'mask_junk',

        # fitblobs: see below

        'coadds': 'fitblobs',

        # wise_forced: see below

        'fitplots': 'fitblobs',
        'psfplots': 'tims',
        'initplots': 'srcs',

        }

    if early_coadds:
        if blob_image:
            prereqs.update({
                'image_coadds':'srcs',
                'fitblobs':'image_coadds',
                })
        else:
            prereqs.update({
                'image_coadds':'mask_junk',
                'srcs':'image_coadds',
                'fitblobs':'srcs',
                })
    else:
        prereqs.update({
            'fitblobs':'srcs',
            })

    if wise:
        prereqs.update({
            'wise_forced': 'coadds',
            'writecat': 'wise_forced',
            })
    else:
        prereqs.update({
            'writecat': 'coadds',
            })

    if prereqs_update is not None:
        prereqs.update(prereqs_update)

    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)
    if bands is not None:
        initargs.update(bands=bands)

    t0 = Time()
    
    def mystagefunc(stage, **kwargs):
        # Update the (pickled) survey output directory, so that running
        # with an updated --output-dir overrides the pickle file.
        picsurvey = kwargs.get('survey',None)
        if picsurvey is not None:
            picsurvey.output_dir = survey.output_dir

        mp.start_subphase('stage ' + stage)
        R = stagefunc(stage, **kwargs)
        sys.stdout.flush()
        sys.stderr.flush()
        print('Resources for stage', stage, ':')
        mp.report(threads)
        mp.finish_subphase()
        return R
    
    for stage in stages:
        runstage(stage, pickle_pat, mystagefunc, prereqs=prereqs,
                 initial_args=initargs, **kwargs)

    print('All done:', Time()-t0)
    mp.report(threads)


def get_parser():
    import argparse
    de = ('Main "pipeline" script for the Legacy Survey ' +
          '(DECaLS, MzLS, Bok) data reductions.')

    ep = '''
e.g., to run a small field containing a cluster:

python -u legacypipe/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle

'''
    parser = argparse.ArgumentParser(description=de,epilog=ep)

    parser.add_argument('-r', '--run', default=None,
                        help='Set the run type to execute')

    parser.add_argument(
        '-f', '--force-stage', dest='force', action='append', default=[],
        help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_argument('-F', '--force-all', dest='forceall',
                        action='store_true', help='Force all stages to run')
    parser.add_argument('-s', '--stage', dest='stage', default=[],
                        action='append', help="Run up to the given stage(s)")
    parser.add_argument('-n', '--no-write', dest='write', default=True,
                        action='store_false')
    parser.add_argument('-w', '--write-stage', action='append', default=None,
                        help='Write a pickle for a given stage: eg "tims", "image_coadds", "srcs"')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument(
        '--checkpoint', dest='checkpoint_filename', default=None,
        help='Write to checkpoint file?')
    parser.add_argument(
        '--checkpoint-period', type=int, default=None,
        help='Period for writing checkpoint files, in seconds; default 600')

    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')

    parser.add_argument(
        '--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')

    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')
    parser.add_argument(
        '--survey-dir', type=str, default=None,
        help='Override the $LEGACY_SURVEY_DIR environment variable')

    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to search for cached files')

    parser.add_argument('--threads', type=int, help='Run multi-threaded')
    parser.add_argument('-p', '--plots', dest='plots', action='store_true',
                        help='Per-blob plots?')
    parser.add_argument('--plots2', action='store_true',
                        help='More plots?')

    parser.add_argument(
        '-P', '--pickle', dest='pickle_pat',
        help='Pickle filename pattern, default %(default)s',
        default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    parser.add_argument('--plot-base',
                        help='Base filename for plots, default brick-BRICK')
    parser.add_argument('--plot-number', type=int, default=0,
                        help='Set PlotSequence starting number')

    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')

    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')

    #parser.add_argument('--no-ceres', dest='ceres', default=True,
    #                    action='store_false', help='Do not use Ceres Solver')
    parser.add_argument('--ceres', default=False, action='store_true',
                        help='Use Ceres Solver for all optimization?')

    parser.add_argument('--no-wise-ceres', dest='wise_ceres', default=True,
                        action='store_false',
                        help='Do not use Ceres Solver for unWISE forced phot')
    
    parser.add_argument('--nblobs', type=int,help='Debugging: only fit N blobs')
    parser.add_argument('--blob', type=int, help='Debugging: start with blob #')
    parser.add_argument(
        '--blobxy', type=int, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing pixel <bx> <by>; '+
              'this option can be repeated to run multiple blobs.'))
    parser.add_argument(
        '--blobradec', type=float, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing RA,Dec <ra> <dec>; '+
              'this option can be repeated to run multiple blobs.'))

    parser.add_argument(
        '--check-done', default=False, action='store_true',
        help='Just check for existence of output files for this brick?')
    parser.add_argument('--skip', default=False, action='store_true',
                        help='Quit if the output catalog already exists.')
    parser.add_argument('--skip-coadd', default=False, action='store_true',
                        help='Quit if the output coadd jpeg already exists.')

    parser.add_argument(
        '--skip-calibs', dest='do_calibs', default=True, action='store_false',
        help='Do not run the calibration steps')

    parser.add_argument('--skip-metrics', dest='write_metrics', default=True,
                        action='store_false',
                        help='Do not generate the metrics directory and files')

    parser.add_argument('--nsigma', type=float, default=6.0,
                        help='Set N sigma source detection thresh')

    parser.add_argument(
        '--simul-opt', action='store_true', default=False,
        help='Do simultaneous optimization after model selection')

    parser.add_argument('--no-wise', dest='wise', default=True,
                        action='store_false',
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')
    parser.add_argument(
        '--unwise-tr-dir', default=None,
        help='Base directory for unWISE time-resolved coadds; may be a colon-separated list')

    parser.add_argument('--early-coadds', action='store_true', default=False,
                        help='Make early coadds?')
    parser.add_argument('--blob-image', action='store_true', default=False,
                        help='Create "imageblob" image?')

    parser.add_argument(
        '--no-lanczos', dest='lanczos', action='store_false', default=True,
        help='Do nearest-neighbour rather than Lanczos-3 coadds')
    
    parser.add_argument('--gpsf', action='store_true', default=False,
                        help='Use a fixed single-Gaussian PSF')

    # parser.add_argument('--hybrid-psf', dest='hybridPsf', action='store_true', default=False,
    #                     help='Use a hybrid pixelized/Gaussian PSF model')
    parser.add_argument('--no-hybrid-psf', dest='hybridPsf', default=True,
                        action='store_false',
                        help="Don't use a hybrid pixelized/Gaussian PSF model")

    #parser.add_argument('--rex', action='store_true', default=False,
    #                    help='Use REX as simple galaxy models, rather than SIMP')
    parser.add_argument('--simp', dest='rex', default=True,
                        action='store_false',
                        help='Use SIMP rather than REX')
    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--bands', default=None,
                        help='Set the list of bands (filters) that are included in processing: comma-separated list, default "g,r,z"')

    # parser.add_argument('--no-depth-cut', dest='depth_cut', default=True,
    #                     action='store_false', help='Do not cut to the set of CCDs required to reach our depth target')
    parser.add_argument('--depth-cut', default=False, action='store_true',
                        help='Cut to the set of CCDs required to reach our depth target')
    
    parser.add_argument(
        '--on-bricks', default=False, action='store_true',
        help='Enable Tractor-on-bricks edge handling?')

    parser.add_argument(
        '--allow-missing-brickq', type=int, choices=[0,1,2], default=-1,
        help='Do not fail if a prerequisite brick of given brickq is missing.')

    return parser

def get_runbrick_kwargs(brick=None,
                        radec=None,
                        run=None,
                        survey_dir=None,
                        output_dir=None,
                        cache_dir=None,
                        check_done=False,
                        skip=False,
                        skip_coadd=False,
                        stage=[],
                        unwise_dir=None,
                        unwise_tr_dir=None,
                        write_stage=None,
                        write=True,
                        gpsf=False,
                        bands=None,
                        **opt):
    if brick is not None and radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return None, -1
    opt.update(radec=radec)

    from legacypipe.runs import get_survey
    survey = get_survey(run,
                        survey_dir=survey_dir,
                        output_dir=output_dir,
                        cache_dir=cache_dir)
    print('Got survey:', survey)
    
    if check_done or skip or skip_coadd:
        if skip_coadd:
            fn = survey.find_file('image-jpeg', output=True, brick=brick)
        else:
            fn = survey.find_file('tractor', output=True, brick=brick)
        print('Checking for', fn)
        exists = os.path.exists(fn)
        if skip_coadd and exists:
            return survey,0
        if exists:
            try:
                T = fits_table(fn)
                print('Read', len(T), 'sources from', fn)
            except:
                print('Failed to read file', fn)
                import traceback
                traceback.print_exc()
                exists = False

        if skip:
            if exists:
                return survey,0
        elif check_done:
            if not exists:
                print('Does not exist:', fn)
                return survey,-1
            print('Found:', fn)
            return survey,0

    if len(stage) == 0:
        stage.append('writecat')

    opt.update(stages=stage)

    # Remove opt values that are None.
    toremove = [k for k,v in opt.items() if v is None]
    for k in toremove:
        del opt[k]

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)
    if unwise_tr_dir is None:
        unwise_tr_dir = os.environ.get('UNWISE_COADDS_TIMERESOLVED_DIR', None)
    opt.update(unwise_dir=unwise_dir, unwise_tr_dir=unwise_tr_dir)

    # list of strings if -w / --write-stage is given; False if
    # --no-write given; True by default.
    if write_stage is not None:
        write_pickles = write_stage
    else:
        write_pickles = write
    opt.update(write_pickles=write_pickles)

    opt.update(gaussPsf=gpsf,
               pixPsf=not gpsf)

    if bands is not None:
        bands = bands.split(',')
    opt.update(bands=bands)
    opt.update(splinesky=True)
    return survey, opt

def main(args=None):
    import logging
    from astrometry.util.ttime import MemMeas, CpuMeas
    import datetime

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
    if args is None:
        print('Command-line args:', sys.argv)
    else:
        print('Args:', args)
    print()
    print('Slurm cluster:', os.environ.get('SLURM_CLUSTER_NAME', 'none'))
    print('Job id:', os.environ.get('JOB_ID', 'none'))
    print('Array task id:', os.environ.get('ARRAY_TASK_ID', 'none'))
    print()

    parser = get_parser()

    parser.add_argument(
        '--ps', help='Run "ps" and write results to given filename?')
    parser.add_argument(
        '--ps-t0', type=int, default=0, help='Unix-time start for "--ps"')

    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    ps_file = optdict.pop('ps', None)
    ps_t0   = optdict.pop('ps_t0', 0)
    verbose = optdict.pop('verbose')

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.on_bricks:
        # Quickly check for existence of required neighboring catalogs
        # before starting.
        brick = survey.get_brick_by_name(opt.brick)
        bricks = on_bricks_dependencies(brick, survey)
        print('Checking for catalogs for bricks:',
              ', '.join([b.brickname for b in bricks]))
        allexist = True
        for b in bricks:
            fn = survey.find_file('tractor', brick=b.brickname)
            print('File', fn)
            if not os.path.exists(fn):
                print('File', fn, 'does not exist (required for --on-bricks)')
                if b.brickq <= opt.allow_missing_brickq:
                    print(('  (allowing this missing brick (brickq = %i) ' +
                           'because of --allow-missing-brickq %i)') % 
                           (b.brickq, opt.allow_missing_brickq))
                else:
                    allexist = False
                continue
            try:
                T = fits_table(fn)
            except:
                print('Failed to open file', fn, '(reqd for --on-bricks)')
                allexist = False

        if not allexist:
            return -1
                
    Time.add_measurement(MemMeas)
    if opt.plots:
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if ps_file is not None:
        import threading
        from collections import deque
        from legacypipe.utils import run_ps_thread
        ps_shutdown = threading.Event()
        ps_queue = deque()
        def record_event(msg):
            from time import time
            ps_queue.append((time(), msg))
        kwargs.update(record_event=record_event)
        if ps_t0 > 0:
            record_event('start')

        ps_thread = threading.Thread(
            target=run_ps_thread,
            args=(os.getpid(), os.getppid(), ps_file, ps_shutdown, ps_queue),
            name='run_ps')
        # ps_thread.daemon = True
        print('Starting thread to run "ps"')
        ps_thread.start()

    rtn = -1
    try:
        run_brick(opt.brick, survey, **kwargs)
        rtn = 0
    except NothingToDoError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = 0
    except RunbrickError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = -1

    if ps_file is not None:
        # Try to shut down ps thread gracefully
        ps_shutdown.set()
        print('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.isAlive():
            print('ps thread is still alive.')

    return rtn

if __name__ == '__main__':
    sys.exit(main())

# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800
# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270
