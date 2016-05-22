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

To see the code we run on each "blob" of pixels,

- :py:func:`one_blob`

'''
from __future__ import print_function

# python -u legacypipe/runbrick.py -b 2437p082 --zoom 2575 2675 400 500 -P "pickles/zoom2-%(brick)s-%%(stage)s.pickle" > log 2>&1 &

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import os

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.ttime import Time
from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

from tractor import Tractor, PointSource, Image, NanoMaggies, Catalog, RaDecPos
from tractor.ellipses import EllipseE
from tractor.galaxy import (DevGalaxy, ExpGalaxy, FixedCompositeGalaxy, SoftenedFracDev,
                            FracDev, disable_galaxy_cache)

from legacypipe.common import (
    tim_get_resamp, get_rgb, imsave_jpeg, LegacySurveyData,
    on_bricks_dependencies)
from legacypipe.cpimage import CP_DQ_BITS
from legacypipe.utils import RunbrickError, NothingToDoError, iterwrapper
from legacypipe.coadds import make_coadds, write_coadd_images, quick_coadds

## GLOBALS!  Oh my!
nocache = True

# RGB image args used in the tile viewer:
rgbkwargs = dict(mnmx=(-1,100.), arcsinh=1.)
rgbkwargs_resid = dict(mnmx=(-5,5))

def runbrick_global_init():
    t0 = Time()
    print('Starting process', os.getpid(), Time()-t0)
    if nocache:
        disable_galaxy_cache()

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               survey=None,
               ra=None, dec=None,
               plots=False, ps=None, survey_dir=None, outdir=None,
               target_extent=None, pipe=False, program_name='runbrick.py',
               bands='grz',
               do_calibs=True,
               splinesky=True,
               gaussPsf=False, pixPsf=False,
               use_blacklist = True,
               mp=None,
               rsync=False,
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

    - *pixPsf*: boolean.  Pixelized PsfEx model, evaluated at the
      image center.  Uses the FFT-based galaxy convolution code.

    Sky:

    - *splinesky*: boolean.  Use SplineSky model, rather than ConstantSky?

    '''
    from legacypipe.common import (get_git_version, get_version_header, wcs_for_brick,
                                   read_one_tim)
    t0 = tlast = Time()

    if survey is None:
        survey = LegacySurveyData(survey_dir=survey_dir, output_dir=outdir)

    if ra is not None:
        from legacypipe.common import BrickDuck
        
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck()
        brick.ra  = ra
        brick.dec = dec
        brickid = brick.brickid = -1
        brick.brickname = brickname
    else:
        brick = survey.get_brick_by_name(brickname)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)

        print('Chosen brick:')
        brick.about()
        brickid = brick.brickid
        brickname = brick.brickname

    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    if ra is not None:
        brick.ra1,nil  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,nil  = targetwcs.pixelxy2radec(1, H/2)
        nil, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        nil, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)
        #print('RA1,RA2', brick.ra1, brick.ra2)
        #print('Dec1,Dec2', brick.dec1, brick.dec2)

    gitver = get_git_version()

    version_hdr = get_version_header(program_name, survey.survey_dir,
                                     git_version=gitver)
    for i,dep in enumerate(['numpy', 'scipy', 'wcslib', 'astropy', 'photutils',
                            'ceres', 'sextractor', 'psfex', 'astrometry_net',
                            'tractor', 'fitsio', 'unwise_coadds']):
        # Look in the OS environment variables for modules-style
        # $scipy_VERSION => 0.15.1_5a3d8dfa-7.1
        default_ver = 'UNAVAILABLE'
        verstr = os.environ.get('%s_VERSION' % dep, default_ver)
        if verstr == default_ver:
            print('Warning: failed to get version string for "%s"' % dep)
        version_hdr.add_record(dict(name='DEPNAM%02i' % i, value=dep,
                                    comment='Name of dependency product'))
        version_hdr.add_record(dict(name='DEPVER%02i' % i, value=verstr,
                                    comment='Version of dependency product'))

    version_hdr.add_record(dict(name='BRICKNAM', value=brickname,
                                comment='LegacySurvey brick RRRr[pm]DDd'))
    version_hdr.add_record(dict(name='BRICKID' , value=brickid,
                                comment='LegacySurvey brick id'))
    version_hdr.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min'))
    version_hdr.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max'))
    version_hdr.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min'))
    version_hdr.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max'))
    version_hdr.add_record(dict(name='BRICKRA' , value=brick.ra,
                                comment='Brick center'))
    version_hdr.add_record(dict(name='BRICKDEC', value=brick.dec,
                                comment='Brick center'))

    # NOAO-requested headers
    version_hdr.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'),
        comment='[h] RA Brick center'))
    version_hdr.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'),
        comment='[deg] Dec Brick center'))

    version_hdr.add_record(dict(
        name='CENTRA', value=brick.ra, comment='[deg] Brick center RA'))
    version_hdr.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='[deg] Brick center Dec'))

    for i,(r,d) in enumerate(targetrd[:4]):
        version_hdr.add_record(dict(
            name='CORN%iRA' % (i+1), value=r, comment='[deg] Brick corner RA'))
        version_hdr.add_record(dict(
            name='CORN%iDEC' %(i+1), value=d, comment='[deg] Brick corner Dec'))

    # print('Version header:')
    # print(version_hdr)

    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    print(len(ccds), 'CCDs touching target WCS')

    if use_blacklist:
        I = survey.apply_blacklist(ccds)
        ccds.cut(I)
        print(len(ccds), 'CCDs not in blacklisted propids (too many exposures!)')

    # Sort images by band -- this also eliminates images whose
    # *image.filter* string is not in *bands*.
    print('Unique filters:', np.unique(ccds.filter))
    ccds.cut(np.hstack([np.flatnonzero(ccds.filter == band) for band in bands]))
    print('Cut on filter:', len(ccds), 'CCDs remain.')
    
    print('Cutting out non-photometric CCDs...')
    I = survey.photometric_ccds(ccds)
    print(len(I), 'of', len(ccds), 'CCDs are photometric')
    ccds.cut(I)

    ims = []
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        ims.append(im)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid)

    tnow = Time()
    print('[serial tims] Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if rsync:
        # Check for existence of calibration files & rsync missing ones.
        reqd = []
        for im in ims:
            if not gaussPsf:
                reqd.append(im.psffn)
            if splinesky:
                reqd.append(im.splineskyfn)
            else:
                reqd.append(im.skyfn)
            reqd.append(im.pvwcsfn)
        # Just request missing files?
        reqd = [fn for fn in reqd if not os.path.exists(fn)]
        print('Required calib files:', reqd)
        print('Calib dir:', survey.get_calib_dir())
        caldir = survey.get_calib_dir() + '/'

        if len(reqd):
            reqd = [fn.replace(caldir, '') for fn in reqd]
            cmd = 'rsync -LRrv edison:/scratch1/scratchdirs/desiproc/decals-dir/calib/./"{%s}" %s' % (','.join(reqd), caldir)
            print(cmd)
            os.system(cmd)

        # Also grab image files
        reqd = []
        for im in ims:
            reqd.extend([im.imgfn, im.dqfn, im.wtfn])
        reqd = [fn for fn in reqd if not os.path.exists(fn)]
        print('Required image files:', reqd)
        imgdir = survey.get_image_dir() + '/'

        if len(reqd):
            reqd = [fn.replace(imgdir, '') for fn in reqd]
            cmd = 'rsync -LRrv edison:/scratch1/scratchdirs/desiproc/images/./"{%s}" %s' % (','.join(reqd), imgdir)
            print(cmd)
            os.system(cmd)

    if do_calibs:
        from legacypipe.common import run_calibs

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

    if False and plots:
        # Here, we read full images (not just subimages) to plot their
        # pixel distributions
        sig1s = dict([(b,[]) for b in bands])
        allpix = []
        for im in ims:
            subtim = im.get_tractor_image(splinesky=True, gaussPsf=True, subsky=False,
                                          radecpoly=targetrd)
            if subtim is None:
                continue
            fulltim = im.get_tractor_image(splinesky=True, gaussPsf=True)
            sig1s[fulltim.band].append(fulltim.sig1)
            allpix.append((fulltim.band, fulltim.name, im.exptime,
                           fulltim.getImage()[fulltim.getInvError() > 0]))
        sig1s = dict([(b, np.median(sig1s[b])) for b in sig1s.keys()])
        for b in bands:
            plt.clf()
            lp,lt = [],[]
            s = sig1s[b]
            for band,name,exptime,pix in allpix:
                if band != b:
                    continue
                # broaden range to encompass most pixels... only req'd when sky is bad
                lo,hi = -5.*s, 5.*s
                lo = min(lo, np.percentile(pix, 5))
                hi = max(hi, np.percentile(pix, 95))
                n,bb,p = plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                                 alpha=0.5)
                lp.append(p[0])
                lt.append('%s: %.0f s' % (name, exptime))
            plt.legend(lp, lt)
            plt.xlabel('Pixel values')
            plt.title('Pixel distributions: %s band' % b)
            ps.savefig()


    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf,
                                pixPsf=pixPsf, splinesky=splinesky)) for im in ims]
    tims = mp.map(read_one_tim, args)

    tnow = Time()
    print('[parallel tims] Read', len(ccds), 'images:', tnow-tlast)
    tlast = tnow

    # Cut the table of CCDs to match the 'tims' list
    I = np.flatnonzero(np.array([tim is not None for tim in tims]))
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))

    if len(tims) == 0:
        raise NothingToDoError('No photometric CCDs touching brick.')

    npix = 0
    for tim in tims:
        h,w = tim.shape
        npix += h*w
    print('Total of', npix, 'pixels read')

    for tim in tims:
        for cal,ver in [('sky', tim.skyver), ('wcs', tim.wcsver), ('psf', tim.psfver)]:
            if tim.plver != ver[1]:
                print('Warning: image "%s" PLVER is "%s" but %s calib was run on PLVER "%s"' %
                      (str(tim), tim.plver, cal, ver[1]))

    # Add additional columns to the CCDs table.
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                            for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
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
    ccds.plver = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.wcsver = np.array([tim.wcsver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.wcsplver = np.array([tim.wcsver[1] for tim in tims])
    ccds.psfplver = np.array([tim.psfver[1] for tim in tims])

    if plots and False:
        # Pixel histograms of subimages.
        for b in bands:
            sig1 = np.median([tim.sig1 for tim in tims if tim.band == b])
            plt.clf()
            for tim in tims:
                if tim.band != b:
                    continue
                # broaden range to encompass most pixels... only req'd when sky is bad
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

    if plots:
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
                dimshow(((tim.dq & tim.dq_saturation_bits) > 0), vmin=0, vmax=1.5, cmap='hot')
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
                    plt.imshow((tim.dq & bitval) > 0, vmin=0, vmax=1.5, cmap='hot')
                    plt.title(bitmap[bitval])
                plt.suptitle('Mask planes: %s' % tim.name)
                ps.savefig()

    if not pipe:
        # save resampling params
        tims_compute_resamp(mp, tims, targetwcs)
        tnow = Time()
        print('Computing resampling:', tnow-tlast)
        tlast = tnow
        # Produce per-band coadds, for plots
        coimgs,cons = quick_coadds(tims, bands, targetwcs)
        tnow = Time()
        print('Coadds:', tnow-tlast)
        tlast = tnow

    # Cut "bands" down to just the bands for which we have images.
    timbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in timbands]
    print('Cut bands to', bands)

    for band in 'grz':
        hasit = band in bands
        version_hdr.add_record(dict(name='BRICK_%s' % band.upper(), value=hasit,
                                    comment='Does band %s touch this brick?' % band))

        cams = np.unique([tim.imobj.camera for tim in tims
                          if tim.band == band])
        version_hdr.add_record(dict(name='CAMS_%s' % band.upper(),
                                    value=' '.join(cams),
                                    comment='Cameras contributing band %s' % band))
        
    version_hdr.add_record(dict(name='BRICKBND', value=''.join(bands),
                                comment='Bands touching this brick'))

    version_header = version_hdr

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick',
            'target_extent', 'ccds', 'bands', 'survey']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict([(k,locals()[k]) for k in keys])
    return rtn

def stage_mask_junk(tims=None, targetwcs=None, W=None, H=None, bands=None,
                    mp=None, nsigma=None, plots=None, ps=None, **kwargs):
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import label, find_objects, center_of_mass
    from scipy.linalg import svd

    if plots:
        coimgs,cons = quick_coadds(tims, bands, targetwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Before')
        ps.savefig()

    allss = []
    for tim in tims:
        # detection map
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
            if not (major > 200 and minor/major < 0.1):
                continue
            # Zero it out!
            tim.inverr[slc] *= np.logical_not(inblob)
            if tim.dq is not None:
                # Add to dq mask bits
                tim.dq[slc] |= CP_DQ_BITS['longthin']

            ra,dec = tim.wcs.pixelToPosition(cx, cy)
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
        plt.title('After')
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
                       **kwargs):
    '''
    Immediately after reading the images, we
    create coadds of just the image products.  Later, full coadds
    including the models will be created (in `stage_coadds`).  But
    it's handy to have the coadds early on, to diagnose problems or
    just to look at the data.
    '''

    if plots and False:

        for band in bands:

            # Plot sky-subtracted traces (projections)
            plt.clf()

            rr = []
            lp,lt = [],[]

            ppp = []

            cc = ['b','r','g','m','k']
            ci = 0
            for j,tim in enumerate(tims):
                if tim.band != band:
                    continue
                R = tim_get_resamp(tim, targetwcs)
                if R is None:
                    continue
                Yo,Xo,Yi,Xi = R

                proj = np.zeros((H,W), np.float32)
                haveproj = np.zeros((H,W), bool)

                proj[Yo,Xo] = tim.data[Yi,Xi]
                haveproj[Yo,Xo] = (tim.inverr[Yi,Xi] > 0)

                xx,ylo,ymed,yhi = [],[],[],[]
                for i in range(W):
                    I = np.flatnonzero(haveproj[:,i])
                    if len(I) == 0:
                        continue
                    xx.append(i)
                    y = proj[I,i]
                    lo,m,hi = np.percentile(y, [16, 50, 84])
                    ylo.append(lo)
                    ymed.append(m)
                    yhi.append(hi)

                if len(xx):
                    color = cc[ci % len(cc)]
                    pargs = dict(color=color)
                    ci += 1
                    p = plt.plot(xx, ymed, '-', alpha=0.5, zorder=10, **pargs)
                    plt.plot([xx[0], xx[-1]], [ymed[0], ymed[-1]], 'o', zorder=20, **pargs)
                    plt.plot(xx, ylo, '-', alpha=0.25, zorder=5, **pargs)
                    plt.plot(xx, yhi, '-', alpha=0.25, zorder=5, **pargs)

                    lp.append(p[0])
                    lt.append(tim.name)

                    rr.extend([h-l for h,l in zip(yhi,ylo)])

                    ppp.append((xx, ymed, color, tim.name))

            yrange = np.median(rr)
            plt.ylim(-yrange, +yrange)
            plt.xlim(0, W)
            plt.title('Per-column medians for %s band' % band)
            plt.legend(lp, lt, loc='upper right')
            ps.savefig()

            plt.clf()
            nplots = len(ppp)
            for i,(xx, ymed, color, name) in enumerate(ppp):
                dy = 0.25 * (i - float(nplots)/2) * yrange/2.
                plt.plot(xx, ymed + dy, color=color)
                plt.axhline(dy, color='k', alpha=0.1)
                plt.text(np.median(xx), np.median(ymed)+dy, name, ha='center', va='bottom', color='k',
                         bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
            plt.ylim(-yrange, +yrange)
            plt.title('Per-column medians for %s band' % band)
            ps.savefig()

    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(out.fn, primheader=version_header)
        print('Wrote', out.fn)
            
    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, lanczos=lanczos,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims, targetwcs))

    # if plots:
    #     for k,v in CP_DQ_BITS.items():
    #         plt.clf()
    #         dimshow(ormask & v, vmin=0, vmax=v)
    #         plt.title('OR mask, %s band: %s' % (band, k))
    #         ps.savefig()
    #         plt.clf()
    #         dimshow(andmask & v, vmin=0, vmax=v)
    #         plt.title('AND mask, %s band: %s' % (band,k))
    #         ps.savefig()
    
    if True:
        # Compute the brick's unique pixels.
        U = None
        if hasattr(brick, 'ra1'):
            print('Computing unique brick pixels...')
            xx,yy = np.meshgrid(np.arange(W), np.arange(H))
            rr,dd = targetwcs.pixelxy2radec(xx+1, yy+1)
            U = np.flatnonzero((rr >= brick.ra1 ) * (rr < brick.ra2 ) *
                               (dd >= brick.dec1) * (dd < brick.dec2))
            print(len(U), 'of', W*H, 'pixels are unique to this brick')

        # depth histogram bins
        depthbins = np.arange(19.9, 25.101, 0.1)
        depthbins[0] = 0.
        depthbins[-1] = 100.
        D = fits_table()
        D.depthlo = depthbins[:-1].astype(np.float32)
        D.depthhi = depthbins[1: ].astype(np.float32)

        for band,detiv,galdetiv in zip(bands, C.detivs, C.galdetivs):
            for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
                # compute stats for 5-sigma detection
                depth = 5. / np.sqrt(np.maximum(det, 1e-20))
                # that's flux in nanomaggies -- convert to mag
                depth = -2.5 * (np.log10(depth) - 9)
                # no coverage -> very bright detection limit
                depth[det == 0] = 0.
                if U is not None:
                    depth = depth.flat[U]
                print(band, name, 'band depth map: deciles',
                      np.percentile(depth, np.arange(0, 101, 10)))
                # histogram
                D.set('counts_%s_%s' % (name, band),
                      np.histogram(depth, bins=depthbins)[0].astype(np.int32))

        del U
        del depth
        del det
        del detiv
        del galdetiv

        with survey.write_output('depth-table', brick=brickname) as out:
            D.writeto(out.fn)
            print('Wrote', out.fn)
        del D

    #rgbkwargs2 = dict(mnmx=(-3., 3.))
    #rgbkwargs2 = dict(mnmx=(-2., 10.))
    for name,ims,rgbkw in [('image',C.coimgs,rgbkwargs),
        #('image2',C.coimgs,rgbkwargs2),
                           ]:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            i = 'zrg'.index(bands[0])
            rgb = rgb[:,:,i]
            kwa = dict(cmap='gray')

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            print('Wrote', out.fn)

        # Blob-outlined version
        if blobs is not None:
            from scipy.ndimage.morphology import binary_dilation
            outline = (binary_dilation(blobs >= 0, structure=np.ones((3,3)))
                       - (blobs >= 0))
            # coadd_bw?
            if len(rgb.shape) == 2:
                rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
            # Outline in green
            rgb[:,:,0][outline] = 0
            rgb[:,:,1][outline] = 1
            rgb[:,:,2][outline] = 0

            with survey.write_output(name + 'blob-jpeg', brick=brickname) as out:
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                print('Wrote', out.fn)
        del rgb

    return None

def _median_smooth_detmap(X):
    from scipy.ndimage.filters import median_filter
    from legacypipe.common import bin_image
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
               pipe=False, brickname=None,
               mp=None, nsigma=5,
               on_bricks=False,
               survey=None, brick=None,
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

    tlast = Time()
    avoid_x, avoid_y = [],[]
    if on_bricks:
        bricks = on_bricks_dependencies(brick, survey)
        if len(bricks) == 0:
            on_bricks = False
    if on_bricks:
        from legacypipe.desi_common import read_fits_catalog
        B = []
        for b in bricks:
            fn = survey.find_file('tractor', brick=b.brickname)
            print('Looking for', fn)
            B.append(fits_table(fn))
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

        # Note that we shouldn't need to drop sources that are within this
        # current brick's unique area, because we cut to sources that are
        # BRICK_PRIMARY within their own brick.

        # Create sources for these catalog entries
        ### see forced-photom-decam.py for some additional patchups?

        B.shapeexp = np.vstack((B.shapeexp_r, B.shapeexp_e1, B.shapeexp_e2)).T
        B.shapedev = np.vstack((B.shapedev_r, B.shapedev_e1, B.shapedev_e2)).T
        bcat = read_fits_catalog(B, ellipseClass=EllipseE)
        print('Created', len(bcat), 'tractor catalog objects')

        # Add the new sources to the 'avoid_[xy]' lists, which are
        # existing sources that should be avoided when detecting new
        # faint sources.
        avoid_x.extend(np.round(B.xx - 1).astype(int))
        avoid_y.extend(np.round(B.yy - 1).astype(int))

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
        print('[parallel srcs] Getting tractor-on-bricks model images:', tnow-tlast)
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

    if plots and False:
        for tim in tims:
            ivx = tim.imobj.read_invvar(clip=False, slice=tim.slice)
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(tim.getInvvar(), vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            plt.title('Tim invvar')
            plt.subplot(2,2,2)
            dimshow(tim.getInvvar() == 0, vmin=0, vmax=1, ticks=False)
            plt.title('Tim invvar == 0')
            plt.subplot(2,2,3)
            dimshow(ivx * tim.zpscale**2, vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            plt.title('Orig invvar')
            plt.subplot(2,2,4)
            dimshow(np.logical_or(ivx == 0, tim.dq != 0),
                    vmin=0, vmax=1, ticks=False)
            plt.title('Orig invvar == 0 or DQ')
            #dimshow(tim.dq != 0, vmin=0, vmax=1, ticks=False)
            #plt.title('DQ')
            plt.suptitle('Tim ' + tim.name)
            ps.savefig()

    if plots:
        for tim in tims:
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(tim.getInvvar(), vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            h,w = tim.shape
            rgba = np.zeros((h,w,4), np.uint8)
            rgba[:,:,0][tim.inverr == 0] = 255
            rgba[:,:,3][tim.inverr == 0] = 255
            dimshow(rgba)
            plt.title('Tim invvar')
            plt.subplot(2,2,2)

            mx = tim.getImage().max()
            ima = dict(vmin=-2.*tim.sig1, vmax=mx,
                       ticks=False)
            dimshow(tim.getImage(), **ima)
            plt.title('Tim image')
            if tim.dq is not None:
                plt.subplot(2,2,3)
                dimshow((tim.dq & tim.dq_saturation_bits > 0), vmin=0, vmax=1,
                        ticks=False)
                plt.title('SATUR')
            plt.suptitle('Tim ' + tim.name)
            ps.savefig()

    print('Rendering detection maps...')
    detmaps, detivs, satmap = detection_maps(tims, targetwcs, bands, mp)
    tnow = Time()
    print('[parallel srcs] Detmaps:', tnow-tlast)

    tlast = tnow
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

    # Read Tycho-2 stars
    tycho = fits_table(os.path.join(survey.get_survey_dir(), 'tycho2.fits.gz'))
    print('Read', len(tycho), 'Tycho-2 stars')
    ok,tycho.tx,tycho.ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    margin = 100
    tycho.cut(ok * (tycho.tx > -margin) * (tycho.tx < W+margin) *
              (tycho.ty > -margin) * (tycho.ty < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    del ok

    Tsat = fits_table()
    # Add sources for Tycho-2 stars
    if len(tycho):
        Tsat.tx = tycho.tx
        Tsat.ty = tycho.ty
        Tsat.mag = tycho.mag
    
    # Saturated blobs -- create a source for each, except for those
    # that already have a Tycho-2 star
    satblobs,nsat = label(satmap > 0)
    print('Satblobs:', satblobs.shape, satblobs.dtype)
    print('nsat', nsat, 'max', satblobs.max(), 'vals', np.unique(satblobs))
    if len(Tsat):
        # Build a map from old "satblobs" to new; identity to start
        remap = np.arange(nsat+1)
        # Drop blobs that contain a Tycho-2 star
        itx = np.clip(np.round(Tsat.tx), 0, W-1).astype(int)
        ity = np.clip(np.round(Tsat.ty), 0, H-1).astype(int)
        zeroout = satblobs[ity, itx]
        remap[zeroout] = 0
        # Renumber them to be contiguous
        I = np.flatnonzero(remap)
        nsat = len(I)
        remap[I] = 1 + np.arange(nsat)
        satblobs = remap[satblobs]
        print('Remapped satblobs:', satblobs.shape, satblobs.dtype)
        print('nsat', nsat, 'max', satblobs.max(), 'vals', np.unique(satblobs))
        del remap, itx, ity, zeroout, I

    # Add sources for any remaining saturated blobs
    satyx = center_of_mass(satmap, labels=satblobs, index=np.arange(nsat)+1)
    # NOTE, satyx is in y,x order (center_of_mass)
    satx = np.array([x for y,x in satyx]).astype(int)
    saty = np.array([y for y,x in satyx]).astype(int)
    del satyx

    if len(satx):
        print('Adding', len(satx), 'additional saturated stars')
        Tsat2 = fits_table()
        Tsat2.tx = satx
        Tsat2.ty = saty
        # MAGIC mag for a saturated star
        Tsat2.mag = np.zeros(len(satx)) + 15.
        Tsat = merge_tables([Tsat, Tsat2], columns='fillzero')
        del Tsat2
    del satx,saty
        
    if len(Tsat):
        Tsat.ra,Tsat.dec = targetwcs.pixelxy2radec(Tsat.tx+1, Tsat.ty+1)
        Tsat.itx = np.clip(np.round(Tsat.tx), 0, W-1).astype(int)
        Tsat.ity = np.clip(np.round(Tsat.ty), 0, H-1).astype(int)
        avoid_x.extend(Tsat.itx)
        avoid_y.extend(Tsat.ity)

        satcat = []
        for r,d,m in zip(Tsat.ra, Tsat.dec, Tsat.mag):
            fluxes = dict([(band, NanoMaggies.magToNanomaggies(m))
                           for band in bands])
            assert(np.all(np.isfinite(fluxes.values())))
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

        print('rgb', rgb.dtype)
        rgb[:,:,0][saturated_pix] = 0
        rgb[:,:,1][saturated_pix] = 1
        rgb[:,:,2][saturated_pix] = 0
        plt.clf()
        dimshow(rgb)
        ax = plt.axis()
        if len(Tsat):
            plt.plot(Tsat.tx, Tsat.ty, 'ro')
        plt.axis(ax)
        plt.title('detmaps & saturated')
        ps.savefig()

    # SED-matched detections
    print('Running source detection at', nsigma, 'sigma')
    SEDs = sed_matched_filters(bands)
    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, (avoid_x,avoid_y), targetwcs,
        nsigma=nsigma, saturated_pix=saturated_pix, plots=plots, ps=ps, mp=mp)
    if Tnew is None:
        raise NothingToDoError('No sources detected.')
    Tnew.delete_column('peaksn')
    Tnew.delete_column('apsn')

    TT = []
    cats = []
    if len(Tsat):
        TT.append(Tsat)
        cats.extend(satcat)
    TT.append(Tnew)
    cats.extend(newcat)

    T = merge_tables(TT, columns='fillzero')
    cat = Catalog(*cats)
    del TT
    del cats

    if pipe:
        del detmaps
        del detivs

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
        p1 = plt.plot(T.tx, T.ty, 'r+', **crossa)
        p2 = plt.plot(Tnew.tx, Tnew.ty, '+', color=(0,1,0), **crossa)
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

    for i,Isrcs in enumerate(blobsrcs):
        if not (Isrcs.dtype in [int, np.int64]):
            print('Isrcs dtype', Isrcs.dtype)
            print('i:', i)
            print('Isrcs:', Isrcs)
            print('blobslice:', blobslices[i])

    cat.freezeAllParams()

    tnow = Time()
    print('[serial srcs] Blobs:', tnow-tlast)
    tlast = tnow

    keys = ['T', 'tims', 'blobsrcs', 'blobslices', 'blobs', 'cat',
            'ps', 'tycho']
    if not pipe:
        keys.extend(['detmaps', 'detivs'])
    rtn = dict([(k,locals()[k]) for k in keys])
    return rtn

def _write_fitblobs_pickle(fn, data):
    from astrometry.util.file import pickle_to_file
    tmpfn = fn + '.tmp'
    pickle_to_file(data, tmpfn)
    os.rename(tmpfn, fn)
    print('Wrote', fn)
        
def stage_fitblobs(T=None,
                   brickname=None,
                   brickid=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   allbands = 'ugrizY',
                   tycho=None,
                   **kwargs):
    '''
    This is where the actual source fitting happens.
    The `one_blob` function is called for each "blob" of pixels with
    the sources contained within that blob.
    '''
    tlast = Time()
    for tim in tims:
        assert(np.all(np.isfinite(tim.getInvError())))

    if write_pickle_filename is not None:
        #import multiprocessing
        #write_pool = multiprocessing.Pool(1)
        import threading
        keys = ['T', 'blobsrcs', 'blobslices', 'blobs', 'cat',
                'targetwcs', 'W', 'H', 'bands', 'tims', 'survey',
                'brickname', 'brickid', 'brick', 'version_header', 'ccds']
        L = locals()
        vals = {}
        for k in keys:
            if k in L:
                vals[k] = L[k]
            else:
                if k in kwargs:
                    vals[k] = kwargs[k]
                else:
                    print('Missing key:', k)
        write_thread = threading.Thread(
            target=_write_fitblobs_pickle,
            args=(write_pickle_filename, vals), name='write_pickle')
        print('Starting thread to write fitblobs pickle')
        write_thread.start()
        
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

    keepblobs = None

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

    if keepblobs is not None:
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
        T.blob = blobs[T.ity, T.itx]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if plots:
        ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
        plt.clf()
        dimshow(blobs>=0, vmin=0, vmax=1)
        ax = plt.axis()
        plt.plot(tx-1, ty-1, 'ro')
        for x,y,mag in zip(tx,ty,tycho.mag):
            plt.text(x, y, '%.1f' % (mag),
                     color='r', fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.5))
        plt.axis(ax)
        plt.title('Tycho-2 stars')
        ps.savefig()

    if checkpoint_filename is None:
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))
        R = mp.map(_bounce_one_blob, blobiter)
    else:
        from astrometry.util.ttime import CpuMeas

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
                R = []

        skipblobs = [B.iblob for B in R]
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho, skipblobs=skipblobs)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))

        Riter = mp.imap_unordered(_bounce_one_blob, blobiter)
        # we'll actually measure wall time -- CpuMeas is just mis-named
        last_checkpoint = CpuMeas()
        while True:
            import multiprocessing
            from astrometry.util.file import pickle_to_file, trymakedirs

            d = os.path.dirname(checkpoint_filename)
            if len(d) and not os.path.exists(d):
                trymakedirs(d)
            
            tnow = CpuMeas()
            dt = tnow.wall_seconds_since(last_checkpoint)
            if dt >= checkpoint_period:
                # Write checkpoint!
                fn = checkpoint_filename + '.tmp'
                # (this happens out here in the main process, while the worker
                # processes continue in the background.)
                print('Writing checkpoint', fn)
                pickle_to_file(R, fn)
                print('Wrote checkpoint to', fn)
                try:
                    os.rename(fn, checkpoint_filename)
                    print('Renamed temp checkpoint', fn, 'to', checkpoint_filename)
                    last_checkpoint = tnow
                    dt = 0.
                except:
                    print('Failed to rename checkpoint file', fn)
                    import traceback
                    traceback.print_exc()
            try:
                if mp.is_multiproc():
                    timeout = max(1, checkpoint_period - dt)
                    r = Riter.next(timeout)
                else:
                    r = Riter.next()
                R.append(r)
                print('Result r:', type(r))
            except StopIteration:
                print('Done')
                break
            except multiprocessing.TimeoutError:
                print('Timed out waiting for result')
                continue

    print('[parallel fitblobs] Fitting sources took:', Time()-tlast)

    ## This used to be in fitblobs_finish

    # one_blob can reduce the number and change the types of sources!
    # Reorder the sources here...
    assert(len(R) == len(blobsrcs))

    # Drop now-empty blobs.
    R = [r for r in R if r is not None and len(r)]
    if len(R) > 0:
        J = np.argsort([B.iblob for B in R])
        R = [R[j] for j in J]
        BB = merge_tables(R)
    else:
        BB = fits_table()
        BB.Isrcs = []
        BB.sources = []
        nb = len(bands)
        BB.fracflux   = np.zeros((0,nb))
        BB.fracmasked = np.zeros((0,nb))
        BB.fracin     = np.zeros((0,nb))
        BB.rchi2      = np.zeros((0,nb))
        BB.dchisqs    = np.zeros((0,5))
        BB.flags      = np.zeros((0,5))
        BB.all_models = np.array([])
        BB.all_model_flags   = np.array([])
        BB.all_model_fluxivs = np.array([])
        BB.all_model_cpu     = np.array([])
        BB.cpu_blob         = []
        BB.cpu_source       = []
        BB.blob_width   = []
        BB.blob_height  = []
        BB.blob_npix    = []
        BB.blob_nimages = []
        BB.blob_totalpix = []
        BB.started_in_blob  = []
        BB.finished_in_blob = []
        BB.hastycho         = []
        BB.srcinvvars = [np.zeros((0,))]
    del R
    II = BB.Isrcs
    newcat = BB.sources
    T.cut(II)

    assert(len(T) == len(newcat))
    print('Old catalog:', len(cat))
    print('New catalog:', len(newcat))
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
    assert(len(BB.flags) == len(cat))

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

        with survey.write_output('blobmap', brick=brickname) as out:
            fitsio.write(out.fn, blobs, header=version_header, clobber=True)
            print('Wrote', out.fn)
        del blobmap
    del iblob, oldblob
    blobs = None

    T.brickid   = np.zeros(len(T), np.int32) + brickid
    T.brickname = np.array([brickname] * len(T))
    T.objid     = np.arange(len(T)).astype(np.int32)

    # How many sources in each blob?
    from collections import Counter
    ninblob = Counter(T.blob)
    T.ninblob = np.array([ninblob[b] for b in T.blob]).astype(np.int16)
    T.tycho2inblob = BB.hastycho
    T.dchisq       = BB.dchisqs.astype(np.float32)
    T.decam_flags  = BB.flags
    T.left_blob    = np.logical_and(BB.started_in_blob,
                                    np.logical_not(BB.finished_in_blob))
    for k in ['fracflux', 'fracin', 'fracmasked', 'rchi2', 'cpu_source',
              'cpu_blob', 'blob_width', 'blob_height', 'blob_npix',
              'blob_nimages', 'blob_totalpix']:
        T.set(k, BB.get(k))

    invvars = np.hstack(BB.srcinvvars)
    assert(cat.numberOfParams() == len(invvars))

    if write_metrics or get_all_models:
        from desi_common import prepare_fits_catalog, fits_typemap
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
        for srctype in ['ptsrc', 'simple', 'dev','exp','comp']:
            xcat = Catalog(*[m.get(srctype,None) for m in BB.all_models])
            # Convert shapes to EllipseE types
            if srctype in ['dev','exp']:
                for src in xcat:
                    if src is None:
                        continue
                    src.shape = src.shape.toEllipseE()
            elif srctype == 'comp':
                for src in xcat:
                    if src is None:
                        continue
                    src.shapeDev = src.shapeDev.toEllipseE()
                    src.shapeExp = src.shapeExp.toEllipseE()
                    src.fracDev = FracDev(src.fracDev.clipped())

            xcat.thawAllRecursive()

            namemap = dict(ptsrc='psf', simple='simp')
            prefix = namemap.get(srctype,srctype)

            allivs = np.hstack([m.get(srctype,[]) for m in BB.all_model_ivs])
            assert(len(allivs) == xcat.numberOfParams())
            
            TT,hdr = prepare_fits_catalog(xcat, allivs, TT, hdr, bands, None,
                                          allbands=allbands, prefix=prefix+'_')
            TT.set('%s_flags' % prefix,
                   np.array([m.get(srctype,0)
                             for m in BB.all_model_flags]))
            TT.set('%s_cpu' % prefix,
                   np.array([m.get(srctype,0)
                             for m in BB.all_model_cpu]).astype(np.float32))

        TT.delete_column('psf_shapeExp')
        TT.delete_column('psf_shapeDev')
        TT.delete_column('psf_fracDev')
        TT.delete_column('psf_shapeExp_ivar')
        TT.delete_column('psf_shapeDev_ivar')
        TT.delete_column('psf_fracDev_ivar')
        TT.delete_column('psf_type')
        TT.delete_column('simp_shapeExp')
        TT.delete_column('simp_shapeDev')
        TT.delete_column('simp_fracDev')
        TT.delete_column('simp_shapeExp_ivar')
        TT.delete_column('simp_shapeDev_ivar')
        TT.delete_column('simp_fracDev_ivar')
        TT.delete_column('simp_type')
        TT.delete_column('dev_shapeExp')
        TT.delete_column('dev_shapeExp_ivar')
        TT.delete_column('dev_fracDev')
        TT.delete_column('dev_fracDev_ivar')
        TT.delete_column('dev_type')
        TT.delete_column('exp_shapeDev')
        TT.delete_column('exp_shapeDev_ivar')
        TT.delete_column('exp_fracDev')
        TT.delete_column('exp_fracDev_ivar')
        TT.delete_column('exp_type')
        TT.delete_column('comp_type')
        # Unpack ellipses
        TT.dev_shape_r  = TT.dev_shapeDev[:,0]
        TT.dev_shape_e1 = TT.dev_shapeDev[:,1]
        TT.dev_shape_e2 = TT.dev_shapeDev[:,2]
        TT.exp_shape_r  = TT.exp_shapeExp[:,0]
        TT.exp_shape_e1 = TT.exp_shapeExp[:,1]
        TT.exp_shape_e2 = TT.exp_shapeExp[:,2]
        TT.comp_shapeDev_r  = TT.comp_shapeDev[:,0]
        TT.comp_shapeDev_e1 = TT.comp_shapeDev[:,1]
        TT.comp_shapeDev_e2 = TT.comp_shapeDev[:,2]
        TT.comp_shapeExp_r  = TT.comp_shapeExp[:,0]
        TT.comp_shapeExp_e1 = TT.comp_shapeExp[:,1]
        TT.comp_shapeExp_e2 = TT.comp_shapeExp[:,2]
        TT.delete_column('dev_shapeDev')
        TT.delete_column('exp_shapeExp')
        TT.delete_column('comp_shapeDev')
        TT.delete_column('comp_shapeExp')
        TT.dev_shape_r_ivar  = TT.dev_shapeDev_ivar[:,0]
        TT.dev_shape_e1_ivar = TT.dev_shapeDev_ivar[:,1]
        TT.dev_shape_e2_ivar = TT.dev_shapeDev_ivar[:,2]
        TT.exp_shape_r_ivar  = TT.exp_shapeExp_ivar[:,0]
        TT.exp_shape_e1_ivar = TT.exp_shapeExp_ivar[:,1]
        TT.exp_shape_e2_ivar = TT.exp_shapeExp_ivar[:,2]
        TT.comp_shapeDev_r_ivar  = TT.comp_shapeDev_ivar[:,0]
        TT.comp_shapeDev_e1_ivar = TT.comp_shapeDev_ivar[:,1]
        TT.comp_shapeDev_e2_ivar = TT.comp_shapeDev_ivar[:,2]
        TT.comp_shapeExp_r_ivar  = TT.comp_shapeExp_ivar[:,0]
        TT.comp_shapeExp_e1_ivar = TT.comp_shapeExp_ivar[:,1]
        TT.comp_shapeExp_e2_ivar = TT.comp_shapeExp_ivar[:,2]
        TT.delete_column('dev_shapeDev_ivar')
        TT.delete_column('exp_shapeExp_ivar')
        TT.delete_column('comp_shapeDev_ivar')
        TT.delete_column('comp_shapeExp_ivar')
        
        if get_all_models:
            all_models = TT
        if write_metrics:
            primhdr = fitsio.FITSHDR()
            for r in version_header.records():
                primhdr.add_record(r)
                primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                                        comment='Band order in array values'))
                primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                                        comment='NOAO data product type'))

            with survey.write_output('all-models', brick=brickname) as out:
                TT.writeto(out.fn, header=hdr)
                print('Wrote', out.fn)

    keys = ['cat', 'invvars', 'T', 'allbands', 'blobs']
    if get_all_models:
        keys.append('all_models')
    rtn = dict([(k,locals()[k]) for k in keys])
    return rtn

def _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims, cat, bands,
               plots, ps, simul_opt, use_ceres, tycho, skipblobs=[]):

    ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    tx = np.round(tx-1).astype(int)
    ty = np.round(ty-1).astype(int)
    # FIXME -- Cut or clip to targetwcs region?
    H,W = targetwcs.shape
    tx = np.clip(tx, 0, W-1)
    ty = np.clip(ty, 0, H-1)
    tychoblobs = set(blobs[ty, tx])
    # Remove -1 = no blob from set; not strictly necessary, just cosmetic
    try:
        tychoblobs.remove(-1)
    except:
        pass
    print('Blobs containing Tycho-2 stars:', tychoblobs)

    # sort blobs by size so that larger ones start running first
    from collections import Counter
    blobvals = Counter(blobs[blobs>=0])
    blob_order = np.array([i for i,npix in blobvals.most_common()])

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

        print('Blob', nblob+1, 'of', len(blobslices), ': blob', iblob,
              len(Isrcs), 'sources, size', blobw, 'x', blobh,
              #'center', (bx0+bx1)/2, (by0+by1)/2,
              'brick X %i,%i, Y %i,%i' % (bx0,bx1,by0,by1),
              'npix', np.sum(blobmask),
              'one pixel:', onex,oney, 'has Tycho-2 star:', hastycho)

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
               simul_opt, use_ceres, hastycho)

def _bounce_one_blob(X):

    # iblob = X[0]
    # fn = 'blob-%i.pickle' % iblob
    # from astrometry.util.file import pickle_to_file
    # pickle_to_file(X, fn)
    # print('Wrote', fn)
    from oneblob import one_blob
    
    try:
        return one_blob(X)
    except:
        import traceback
        print('Exception in one_blob: (iblob = %i)' % (X[0]))
        traceback.print_exc()
        #print 'CARRYING ON...'
        #return None
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

    #######
    # from tractor.galaxy import fft_timing
    # from astrometry.util.file import pickle_to_file
    # for row in fft_timing:
    #     print(row)
    # pickle_to_file(fft_timing,
    #                ('fft-timing-%s.pickle' % 
    # str(tim).replace('Image ','').replace(' ','')))
    #######

    return mod


def stage_blobiter(T=None,
                   brickname=None,
                   brickid=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   allbands = 'ugrizY',
                   tycho=None,
                   **kwargs):
    '''
    Try pre-computing the arguments to the oneblob function; is it a bottleneck in
    many-threads?
    '''
    tlast = Time()
    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1
    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()

    keepblobs = None
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

    if keepblobs is not None:
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
        T.blob = blobs[T.ity, T.itx]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                          cat, bands, plots, ps, simul_opt, use_ceres,
                          tycho)

    blobargs = list(blobiter)
    print('Computing blob args took', Time()-tlast)
    return dict(blobargs=blobargs)
    

def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 T=None, cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                 mp=None, on_bricks=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage (and
    `stage_fitblobs_finish` reformatting stage), we have all the
    source model fits, and we can create coadds of the images, model,
    and residuals.  We also perform aperture photometry in this stage.
    '''
    from legacypipe.common import apertures_arcsec
    tlast = Time()

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(out.fn, primheader=primhdr)
        print('Wrote', out.fn)

    tnow = Time()
    print('[serial coadds]:', tnow-tlast)
    tlast = tnow
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])
    tnow = Time()
    print('[parallel coadds] Getting model images:', tnow-tlast)
    tlast = tnow

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    # Look up number of images overlapping each source's position.
    assert(len(T) == len(cat))
    rr = np.array([s.getPosition().ra  for s in cat])
    dd = np.array([s.getPosition().dec for s in cat])
    ok,ix,iy = targetwcs.radec2pixelxy(rr, dd)
    T.oob = reduce(np.logical_or, [ix < 0.5, iy < 0.5, ix > W+0.5, iy > H+0.5])
    ix = np.clip(np.round(ix - 1), 0, W-1).astype(int)
    iy = np.clip(np.round(iy - 1), 0, H-1).astype(int)

    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale

    # Aperture photometry locations
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
    apxy = np.vstack((xx - 1., yy - 1.)).T

    nap = len(apertures)
    if len(xx) == 0:
        apertures = None
        apxy = None
    del xx,yy,ok,ra,dec

    C = make_coadds(tims, bands, targetwcs, mods=mods, xy=(ix,iy),
                    ngood=True, detmaps=True, psfsize=True, lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims, targetwcs),
                    plots=False, ps=ps)

    for c in ['nobs', 'anymask', 'allmask', 'psfsize', 'depth', 'galdepth']:
        T.set(c, C.T.get(c))

    if apertures is None:
        # empty table when 0 sources.
        C.AP = fits_table()
        for band in bands:
            C.AP.set('apflux_img_%s' % band, np.zeros((0,nap)))
            C.AP.set('apflux_img_ivar_%s' % band, np.zeros((0,nap)))
            C.AP.set('apflux_resid_%s' % band, np.zeros((0,nap)))

    # Compute the brick's unique pixels.
    U = None
    if hasattr(brick, 'ra1'):
        print('Computing unique brick pixels...')
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        rr,dd = targetwcs.pixelxy2radec(xx+1, yy+1)
        U = np.flatnonzero((rr >= brick.ra1 ) * (rr < brick.ra2 ) *
                           (dd >= brick.dec1) * (dd < brick.dec2))
        print(len(U), 'of', W*H, 'pixels are unique to this brick')

    # depth histogram bins
    depthbins = np.arange(20, 25.001, 0.1)
    depthbins[0] = 0.
    depthbins[-1] = 100.
    D = fits_table()
    D.depthlo = depthbins[:-1].astype(np.float32)
    D.depthhi = depthbins[1: ].astype(np.float32)

    for band,detiv,galdetiv in zip(bands, C.detivs, C.galdetivs):
        for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
            # compute stats for 5-sigma detection
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

    del U
    del depth
    del det
    del detiv
    del galdetiv

    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(out.fn)
        print('Wrote', out.fn)
    del D

    for name,ims,rgbkw in [('image', C.coimgs,   rgbkwargs),
                           ('model', C.comods,   rgbkwargs),
                           ('resid', C.coresids, rgbkwargs_resid),
                           ]:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            i = 'zrg'.index(bands[0])
            rgb = rgb[:,:,i]
            kwa = dict(cmap='gray')

        if on_bricks and name == 'image':
            # Do not overwrite the image.jpg file if it exists (eg, written during
            # image_coadds stage), because in the later stage_srcs, we subtract the
            # overlapping sources from other bricks, modifying the images.
            fn = survey.find_file(name + '-jpeg', brick=brickname, output=True)
            if os.path.exists(fn):
                print('Not overwriting existing image %s because on_bricks is set' % fn)
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


def stage_wise_forced(
    cat=None,
    T=None,
    targetwcs=None,
    W=None, H=None,
    brickname=None,
    unwise_dir=None,
    unwise_w12_dir=None,
    unwise_w34_dir=None,
    brick=None,
    use_ceres=True,
    mp=None,
    rsync=False,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    from wise.forcedphot import unwise_tiles_touching_wcs

    if unwise_dir is None:
        unwise_dir = 'unwise-coadds'
    if unwise_w12_dir is None:
        unwise_w12_dir = unwise_dir
    if unwise_w34_dir is None:
        unwise_w34_dir = unwise_dir

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

    if rsync:
        reqd = []
        basedirs12 = unwise_w12_dir.split(':')
        basedirs34 = unwise_w34_dir.split(':')
        for tile in tiles.coadd_id:
            for (band,basedirs) in [(1,basedirs12), (2,basedirs12), (3,basedirs34),(4,basedirs34)]:
                for tag in ['img-m', 'invvar-m', 'n-m', 'n-u']:
                    fnpart = os.path.join(tile[:3], tile,
                                          'unwise-%s-w%i-%s.fits' % (tile, band, tag))
                    found = False
                    for basedir in basedirs:
                        fn = os.path.join(basedir, fnpart)
                        if os.path.exists(fn) or os.path.exists(fn + '.gz'):
                            found = True
                            break
                    if not found:
                        reqd.append(fnpart)
        # HACK ... hard-coded paths
        reqd12 = [fn for fn in reqd if 'w1' in fn or 'w2' in fn]
        if len(reqd12):
            cmd = ('rsync -LRrv edison:/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/./"{%s}*" %s'
                   % (','.join(reqd12), '/global/cscratch1/sd/desiproc/unwise-coadds-fulldepth-zp'))
            print(cmd)
            os.system(cmd)
            #
        reqd34 = [fn for fn in reqd if 'w3' in fn or 'w4' in fn]
        if len(reqd34):
            cmd = ('rsync -LRrv edison:/scratch1/scratchdirs/desiproc/unwise-coadds/./"{%s}*" %s'
                   % (','.join(reqd34), '/global/cscratch1/sd/desiproc/unwise-coadds'))
            print(cmd)
            os.system(cmd)

            
    args = []
    for band in [1,2]:
        args.append((wcat, tiles, band, roiradec, unwise_w12_dir, use_ceres))
    for band in [3,4]:
        args.append((wcat, tiles, band, roiradec, unwise_w34_dir, use_ceres))

    # Time-resolved WISE coadds too
    tdir = os.environ['UNWISE_COADDS_TIMERESOLVED_DIR']
    W = fits_table(os.path.join(tdir, 'time_resolved_neo1-atlas.fits'))
    print('Read', len(W), 'time-resolved WISE coadd tiles')
    W.cut(np.array([t in tiles.coadd_id for t in W.coadd_id]))
    print('Cut to', len(W), 'time-resolved vs', len(tiles), 'full-depth')
    assert(len(W) == len(tiles))

    # this ought to be enough for anyone =)
    Nepochs = 5

    eargs = []
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
            eargs.append((e, (wcat, tiles[I], band, roiradec, edir, use_ceres)))
        
    phots = mp.map(_unwise_phot, args + [a for e,a in eargs])
    WISE = phots[0]
    if WISE is not None:
        for p in phots[1:len(args)]:
            WISE.add_columns_from(p)
        WISE.rename('tile', 'unwise_tile')

    WISE_T = phots[len(args)]
    if WISE_T is not None:
        WT = fits_table()
        phots = phots[len(args):]
        for (e,a),phot in zip(eargs, phots):
            print('Epoch', e, 'photometry:')
            if phot is None:
                print('Failed.')
                continue
            phot.about()
            phot.delete_column('tile')
            for c in phot.columns():
                if not c in WT.columns():
                    x = phot.get(c)
                    WT.set(c, np.zeros((len(x), Nepochs), x.dtype))
                X = WT.get(c)
                X[:,e] = phot.get(c)
        WISE_T = WT

    return dict(WISE=WISE, WISE_T=WISE_T)

def _unwise_phot(X):
    from wise.forcedphot import unwise_forcedphot

    (wcat, tiles, band, roiradec, unwise_dir, use_ceres) = X

    try:
        W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec, bands=[band],
                              unwise_dir=unwise_dir, use_ceres=use_ceres)
    except:
        import traceback
        print('unwise_forcedphot failed:')
        traceback.print_exc()

        if use_ceres:
            print('Trying without Ceres...')
            W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                                  bands=[band], unwise_dir=unwise_dir,
                                  use_ceres=False)
        W = None
    return W


'''
Write catalog output
'''
def stage_writecat(
    survey=None,
    version_header=None,
    T=None,
    WISE=None,
    WISE_T=None,
    AP=None,
    apertures_arcsec=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False,
    brickname=None,
    brickid=None,
    brick=None,
    invvars=None,
    allbands=None,
    **kwargs):
    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''
    from desi_common import prepare_fits_catalog
    from tractor.sfd import SFDMap
    
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        if k in TT.get_columns():
            TT.delete_column(k)
    TT.tx = TT.tx.astype(np.float32)
    TT.ty = TT.ty.astype(np.float32)

    # Set fields such as TT.decam_rchi2, etc -- fields with 6-band values
    B = np.array([allbands.index(band) for band in bands])
    atypes = dict(nobs=np.uint8, anymask=TT.anymask.dtype,
                  allmask=TT.allmask.dtype)
    for k in ['rchi2', 'fracflux', 'fracmasked', 'fracin', 'nobs',
              'anymask', 'allmask', 'psfsize', 'depth', 'galdepth']:
        t = atypes.get(k, np.float32)
        A = np.zeros((len(TT), len(allbands)), t)
        A[:,B] = TT.get(k)
        TT.set('decam_'+k, A)
        TT.delete_column(k)

    TT.rename('oob', 'out_of_bounds')

    if AP is not None:
        # How many apertures?
        ap = AP.get('apflux_img_%s' % bands[0])
        n,A = ap.shape
        TT.decam_apflux = np.zeros((len(TT), len(allbands), A), np.float32)
        TT.decam_apflux_ivar  = np.zeros((len(TT), len(allbands), A),
                                         np.float32)
        TT.decam_apflux_resid = np.zeros((len(TT), len(allbands), A),
                                         np.float32)
        for iband,band in enumerate(bands):
            i = allbands.index(band)
            TT.decam_apflux      [:,i,:] = AP.get('apflux_img_%s'      % band)
            TT.decam_apflux_ivar [:,i,:] = AP.get('apflux_img_ivar_%s' % band)
            TT.decam_apflux_resid[:,i,:] = AP.get('apflux_resid_%s'    % band)

    cat.thawAllRecursive()
    hdr = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs,
                                  allbands=allbands)

    # mod
    T2.ra += (T2.ra <   0) * 360.
    T2.ra -= (T2.ra > 360) * 360.

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)

    primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                            comment='Band order in array values'))

    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

    if AP is not None:
        for i,ap in enumerate(apertures_arcsec):
            primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                    comment='Aperture radius, in arcsec'))

    bits = CP_DQ_BITS.values()
    bits.sort()
    bitmap = dict((v,k) for k,v in CP_DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if bit in bitmap:
            primhdr.add_record(dict(name='MASKB%i' % i, value=bitmap[bit],
                                    comment='Mask bit 2**%i=%i meaning' %
                                    (i, bit)))

    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
    T2.bx0 = (bx - 1.).astype(np.float32)
    T2.by0 = (by - 1.).astype(np.float32)
    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

    T2.brick_primary = ((T2.ra  >= brick.ra1 ) * (T2.ra  < brick.ra2) *
                        (T2.dec >= brick.dec1) * (T2.dec < brick.dec2))
    T2.ra_ivar  = T2.ra_ivar .astype(np.float32)
    T2.dec_ivar = T2.dec_ivar.astype(np.float32)

    # Unpack shape columns
    T2.shapeExp_r  = T2.shapeExp[:,0]
    T2.shapeExp_e1 = T2.shapeExp[:,1]
    T2.shapeExp_e2 = T2.shapeExp[:,2]
    T2.shapeDev_r  = T2.shapeDev[:,0]
    T2.shapeDev_e1 = T2.shapeDev[:,1]
    T2.shapeDev_e2 = T2.shapeDev[:,2]
    T2.shapeExp_r_ivar  = T2.shapeExp_ivar[:,0]
    T2.shapeExp_e1_ivar = T2.shapeExp_ivar[:,1]
    T2.shapeExp_e2_ivar = T2.shapeExp_ivar[:,2]
    T2.shapeDev_r_ivar  = T2.shapeDev_ivar[:,0]
    T2.shapeDev_e1_ivar = T2.shapeDev_ivar[:,1]
    T2.shapeDev_e2_ivar = T2.shapeDev_ivar[:,2]

    T2.decam_flux[T2.decam_flux_ivar == 0] = 0.

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
            
            
    print('Reading SFD maps...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, T2.ra, T2.dec, get_ebv=True)
    ext = ext.astype(np.float32)
    decam_extinction = ext[:,:len(allbands)]
    wise_extinction = ext[:,len(allbands):]
    T2.ebv = ebv.astype(np.float32)
    T2.decam_mw_transmission = 10.**(-decam_extinction / 2.5)
    if WISE is not None:
        T2.wise_mw_transmission  = 10.**(-wise_extinction  / 2.5)

    cols = [
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'ninblob',
        'tycho2inblob', 'type', 'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by', 'bx0', 'by0', 'left_blob', 'out_of_bounds',
        'dchisq', 'ebv', 
        'cpu_source', 'cpu_blob',
        'blob_width', 'blob_height', 'blob_npix', 'blob_nimages',
        'blob_totalpix',
        'decam_flux', 'decam_flux_ivar',
        ]

    if AP is not None:
        cols.extend(['decam_apflux', 'decam_apflux_resid','decam_apflux_ivar'])

    cols.extend(['decam_mw_transmission', 'decam_nobs',
        'decam_rchi2', 'decam_fracflux', 'decam_fracmasked', 'decam_fracin',
        'decam_anymask', 'decam_allmask', 'decam_psfsize' ])

    if WISE is not None:
        cols.extend([
            'wise_flux', 'wise_flux_ivar',
            'wise_mw_transmission', 'wise_nobs', 'wise_fracflux', 'wise_rchi2'])

    if WISE_T is not None:
        cols.extend([
            'wise_lc_flux', 'wise_lc_flux_ivar',
            'wise_lc_nobs', 'wise_lc_fracflux', 'wise_lc_rchi2', 'wise_lc_mjd'])
        
    cols.extend([
        'fracdev', 'fracDev_ivar', 'shapeexp_r', 'shapeexp_r_ivar',
        'shapeexp_e1', 'shapeexp_e1_ivar',
        'shapeexp_e2', 'shapeexp_e2_ivar',
        'shapedev_r',  'shapedev_r_ivar',
        'shapedev_e1', 'shapedev_e1_ivar',
        'shapedev_e2', 'shapedev_e2_ivar',])

    # TUNIT cards.
    deg='deg'
    degiv='1/deg^2'
    flux = 'nanomaggy'
    fluxiv = '1/nanomaggy^2'
    units = dict(
        ra=deg, dec=deg, ra_ivar=degiv, dec_ivar=degiv, ebv='mag',
        decam_flux=flux, decam_flux_ivar=fluxiv,
        decam_apflux=flux, decam_apflux_ivar=fluxiv, decam_apflux_resid=flux,
        wise_flux=flux, wise_flux_ivar=fluxiv,
        wise_lc_flux=flux, wise_lc_flux_ivar=fluxiv,
        shapeexp_r='arcsec', shapeexp_r_ivar='1/arcsec^2',
        shapedev_r='arcsec', shapedev_r_ivar='1/arcsec^2')

    for i,col in enumerate(cols):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    # match case to T2.
    cc = T2.get_columns()
    cclower = [c.lower() for c in cc]
    for i,c in enumerate(cols):
        if (not c in cc) and c in cclower:
            j = cclower.index(c)
            cols[i] = cc[j]

    with survey.write_output('tractor', brick=brickname) as out:
        T2.writeto(out.fn, primheader=primhdr, header=hdr, columns=cols)
        print('Wrote', out.fn)

    # produce per-brick sha1sums file
    hashfn = survey.find_file('sha1sum-brick', brick=brickname, output=True)
    cmd = 'sha1sum -b ' + ' '.join(survey.output_files) + ' > ' + hashfn
    print('Checksums:', cmd)
    os.system(cmd)

    return dict(T2=T2)

def _bounce_tim_get_resamp(X):
    (tim, targetwcs) = X
    return tim_get_resamp(tim, targetwcs)

def tims_compute_resamp(mp, tims, targetwcs, force=False):
    if mp is None:
        from utils import MyMultiproc
        mp = MyMultiproc()
    if force:
        for tim in tims:
            if hasattr(tim, 'resamp'):
                del tim.resamp
    R = mp.map(_bounce_tim_get_resamp, [(tim,targetwcs) for tim in tims])
    for tim,r in zip(tims, R):
        tim.resamp = r

def run_brick(brick, radec=None, pixscale=0.262,
              width=3600, height=3600,
              zoom=None,
              bands=None,
              blacklist=True,
              nblobs=None, blob=None, blobxy=None,
              pipe=True, nsigma=6,
              simulOpt=False,
              wise=True,
              lanczos=True,
              early_coadds=True,
              blob_image=False,
              do_calibs=True,
              write_metrics=True,
              on_bricks=False,
              gaussPsf=False,
              pixPsf=False,
              splinesky=False,
              ceres=True,
              outdir=None,
              survey=None, survey_dir=None,
              unwise_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plotbase=None, plotnumber=0,
              rsync=False,
              picklePattern='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=[], forceAll=False, writePickles=True,
              checkpoint_filename=None,
              checkpoint_period=None,
              fitblobs_prereq_filename=None):
    '''
    Run the full Legacy Survey data reduction pipeline.

    The pipeline is built out of "stages" that run in sequence.  By
    default, this function will cache the result of each stage in a
    (large) pickle file.  If you re-run, it will read from the
    prerequisite pickle file rather than re-running the prerequisite
    stage.  This can yield faster debugging times, but you almost
    certainly want to turn it off (with `writePickles=False,
    forceAll=True`) in production.

    Parameters
    ----------
    brick : string
        Brick name such as '2090m065'.  Can be None if *radec* is given.
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

    Other options:

    - *pipe*: boolean; "pipeline mode"; avoid computing non-essential
      things.

    - *nsigma*: float; detection threshold in sigmas.

    - *simulOpt*: boolean; during fitting, if a blob contains multiple
      sources, run a step of fitting the sources simultaneously?

    - *wise*: boolean; run WISE forced photometry?

    - *early_coadds*: boolean; generate the early coadds?

    - *do_calibs*: boolean; run the calibration preprocessing steps?

    - *write_metrics*: boolean; write out a variety of useful metrics

    - *on_bricks*: boolean; tractor-on-bricks?

    - *gaussPsf*: boolean; use a simpler single-component Gaussian PSF model?

    - *pixPsf*: boolean; use the pixelized PsfEx PSF model and FFT convolution?

    - *splinesky*: boolean; use the splined sky model (default is constant)?

    - *ceres*: boolean; use Ceres Solver when possible?

    - *outdir*: string; base directory for output files; default "."

    - *survey*: a "LegacySurveyData" object (see common.LegacySurveyData), which is in
      charge of the list of bricks and CCDs to be handled, and also
      creates DecamImage objects.

    - *survey_dir*: string; default $LEGACY_SURVEY_DIR environment variable;
      where to look for files including calibration files, tables of
      CCDs and bricks, image data, etc.

    - *unwise_dir*: string; default unwise-coadds; where to look for
      unWISE coadd files.  This may be a colon-separated list of
      directories to search in order.

    - *threads*: integer; how many CPU cores to use

    Plotting options:

    - *coadd_bw*: boolean: if only one band is available, make B&W coadds?
    - *plots*: boolean; make a bunch of plots?
    - *plots2*: boolean; make a bunch more plots?
    - *plotbase*: string, default brick-BRICK, the plot filename prefix.
    - *plotnumber*: integer, default 0, starting number for plot filenames.

    Options regarding the "stages":

    - *picklePattern*: string; filename for 'pickle' files
    - *stages*: list of strings; stages (functions stage_*) to run.

    - *force*: list of strings; prerequisite stages that will be run
      even if pickle files exist.
    - *forceAll*: boolean; run all stages, ignoring all pickle files.
    - *writePickles*: boolean; write pickle files after each stage?

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

    initargs = {}
    kwargs = {}

    forceStages = [s for s in stages]
    forceStages.extend(force)

    if forceAll:
        kwargs.update(forceall=True)

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
    initargs.update(brickname=brick,
                    survey=survey)

    stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plotbase is None:
        plotbase = plot_base_default
    ps = PlotSequence(plotbase % dict(brick=brick))
    initargs.update(ps=ps)

    if plotnumber:
        ps.skipto(plotnumber)

    kwargs.update(ps=ps, nsigma=nsigma, gaussPsf=gaussPsf, pixPsf=pixPsf,
                  use_blacklist=blacklist,
                  splinesky=splinesky,
                  simul_opt=simulOpt, pipe=pipe,
                  use_ceres=ceres,
                  do_calibs=do_calibs,
                  write_metrics=write_metrics,
                  lanczos=lanczos,
                  on_bricks=on_bricks,
                  outdir=outdir, survey_dir=survey_dir, unwise_dir=unwise_dir,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  rsync=rsync,
                  force=forceStages, write=writePickles)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)
    if fitblobs_prereq_filename is not None:
        kwargs.update(write_pickle_filename=fitblobs_prereq_filename)

    if threads and threads > 1:
        from astrometry.util.timingpool import TimingPool, TimingPoolMeas
        pool = TimingPool(threads, initializer=runbrick_global_init,
                          initargs=[], taskqueuesize=2*threads)
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

    picklePattern = picklePattern % dict(brick=brick)

    prereqs = {
        'tims':None,
        'mask_junk': 'tims',
        'srcs': 'mask_junk',

        # fitblobs: see below

        'coadds': 'fitblobs',

        'blobiter': 'srcs',
        
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

    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)
    if bands is not None:
        initargs.update(bands=bands)

    t0 = Time()

    def mystagefunc(stage, **kwargs):
        mp.start_subphase('stage ' + stage)
        if pool is not None:
            print('At start of stage', stage, ':')
            print(pool.get_pickle_traffic_string())
        R = stagefunc(stage, **kwargs)
        print('Resources for stage', stage, ':')
        mp.report(threads)
        if pool is not None:
            print('At end of stage', stage, ':')
            print(pool.get_pickle_traffic_string())
        mp.finish_subphase()
        return R
    
    for stage in stages:
        #runstage(stage, picklePattern, stagefunc, prereqs=prereqs,
        runstage(stage, picklePattern, mystagefunc, prereqs=prereqs,
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

    parser.add_argument('--checkpoint', default=None,
                        help='Write to checkpoint file?')
    parser.add_argument(
        '--checkpoint-period', type=int, default=None,
        help='Period for writing checkpoint files, in seconds; default 600')

    parser.add_argument('--fitblobs-prereq', default=None,
                        help='Write pickle file containing prereqs for the fitblobs stage')

    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')

    parser.add_argument(
        '--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')

    parser.add_argument('-d', '--outdir', help='Set output base directory', default='.')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')

    parser.add_argument('--threads', type=int, help='Run multi-threaded')
    parser.add_argument('-p', '--plots', dest='plots', action='store_true',
                        help='Per-blob plots?')
    parser.add_argument('--plots2', action='store_true',
                        help='More plots?')

    parser.add_argument(
        '-P', '--pickle', dest='picklepat',
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
                        help='Use Ceres Solver?')
    
    parser.add_argument('--nblobs', type=int,help='Debugging: only fit N blobs')
    parser.add_argument('--blob', type=int, help='Debugging: start with blob #')
    parser.add_argument(
        '--blobxy', type=int, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing pixel <bx> <by>; '+
              'this option can be repeated to run multiple blobs.'))

    parser.add_argument('--pipe', default=False, action='store_true',
                        help='"pipeline" mode')

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

    parser.add_argument('--nsigma', type=float,
                        help='Set N sigma source detection thresh')

    parser.add_argument(
        '--simul-opt', action='store_true', default=False,
        help='Do simultaneous optimization after model selection')

    parser.add_argument('--no-wise', action='store_true', default=False,
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')

    parser.add_argument('--no-early-coadds', action='store_true', default=False,
                        help='Skip making the early coadds')
    parser.add_argument('--blob-image', action='store_true', default=False,
                        help='Create "imageblob" image?')

    parser.add_argument(
        '--no-lanczos', dest='lanczos', action='store_false', default=True,
        help='Do nearest-neighbour rather than Lanczos-3 coadds')
    
    parser.add_argument('--gpsf', action='store_true', default=False,
                        help='Use a fixed single-Gaussian PSF')

    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--bands', default=None,
                        help='Limit the bands that are included; default "grz"')

    parser.add_argument(
        '--no-blacklist', dest='blacklist', default=True, action='store_false',
        help='Do not blacklist some proposals?')

    parser.add_argument(
        '--rsync', default=False, action='store_true',
        help=('Rather than running calibrations, rsync from NERSC.  Also '+
              'rsync missing image file inputs'))

    parser.add_argument(
        '--on-bricks', default=False, action='store_true',
        help='Enable Tractor-on-bricks edge handling?')
    
    return parser

def get_runbrick_kwargs(opt):
    if opt.brick is not None and opt.radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return -1

    if opt.check_done or opt.skip or opt.skip_coadd:
        outdir = opt.outdir
        if outdir is None:
            outdir = '.'
        brickname = opt.brick
        if opt.skip_coadd:
            fn = os.path.join(outdir, 'coadd', brickname[:3], brickname,
                              'legacysurvey-%s-image.jpg' % brickname)
        else:
            fn = os.path.join(outdir, 'tractor', brickname[:3],
                              'tractor-%s.fits' % brickname)
        print('Checking for', fn)
        exists = os.path.exists(fn)
        if opt.skip_coadd and exists:
            return 0
        if exists:
            try:
                T = fits_table(fn)
                print('Read', len(T), 'sources from', fn)
            except:
                print('Failed to read file', fn)
                exists = False

        if opt.skip:
            if exists:
                return 0
        elif opt.check_done:
            if not exists:
                print('Does not exist:', fn)
                return -1
            print('Found:', fn)
            return 0

    if len(opt.stage) == 0:
        opt.stage.append('writecat')

    kwa = {}
    if opt.nsigma:
        kwa.update(nsigma=opt.nsigma)
    if opt.no_wise:
        kwa.update(wise=False)
    if opt.no_early_coadds:
        kwa.update(early_coadds=False)
    if opt.blob_image:
        kwa.update(blob_image=True)

    if opt.unwise_dir is None:
        opt.unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)

    # list of strings if -w / --write-stage is given; False if
    # --no-write given; True by default.
    if opt.write_stage is not None:
        writeStages = opt.write_stage
    else:
        writeStages = opt.write

    opt.pixpsf = not opt.gpsf
        
    kwa.update(
        radec=opt.radec, pixscale=opt.pixscale,
        width=opt.width, height=opt.height, zoom=opt.zoom,
        blacklist=opt.blacklist,
        threads=opt.threads, ceres=opt.ceres,
        do_calibs=opt.do_calibs,
        write_metrics=opt.write_metrics,
        on_bricks=opt.on_bricks,
        gaussPsf=opt.gpsf, pixPsf=opt.pixpsf, splinesky=True,
        simulOpt=opt.simul_opt,
        nblobs=opt.nblobs, blob=opt.blob, blobxy=opt.blobxy,
        pipe=opt.pipe, outdir=opt.outdir, survey_dir=opt.survey_dir,
        unwise_dir=opt.unwise_dir,
        plots=opt.plots, plots2=opt.plots2,
        coadd_bw=opt.coadd_bw,
        bands=opt.bands,
        lanczos=opt.lanczos,
        plotbase=opt.plot_base, plotnumber=opt.plot_number,
        force=opt.force, forceAll=opt.forceall,
        stages=opt.stage, writePickles=writeStages,
        picklePattern=opt.picklepat,
        rsync=opt.rsync,
        checkpoint_filename=opt.checkpoint,
        checkpoint_period=opt.checkpoint_period,
        fitblobs_prereq_filename=opt.fitblobs_prereq,
        )
    return kwa
    
def main(args=None):
    import logging
    from astrometry.util.ttime import MemMeas, CpuMeas
    import datetime

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
    print('Command-line args:', sys.argv)
    print()

    parser = get_parser()
    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1, 0]:
        return kwargs

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.on_bricks:
        # Quickly check for existence of required neighboring catalogs
        # before starting.
        survey = LegacySurveyData(survey_dir=opt.survey_dir, output_dir=opt.outdir)
        brick = survey.get_brick_by_name(opt.brick)
        bricks = on_bricks_dependencies(brick, survey)
        print('Checking for catalogs for bricks:',
              ', '.join([b.brickname for b in bricks]))
        allexist = True
        for b in bricks:
            fn = survey.find_file('tractor', brick=b.brickname)
            print('File', fn)
            if not os.path.exists(fn):
                allexist = False
                print('File', fn, 'does not exist (required for --on-bricks)')
                continue
            try:
                T = fits_table(fn)
            except:
                print('Failed to open file', fn, '(reqd for --on-bricks)')
                allexist = False

        if not allexist:
            return -1
                
    Time.add_measurement(MemMeas)
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)

    try:
        run_brick(opt.brick, **kwargs)
    except NothingToDoError as e:
        print()
        print(e.message)
        print()
        return 0
    except RunbrickError as e:
        print()
        print(e.message)
        print()
        return -1
    return 0

def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

if __name__ == '__main__':
    #sys.settrace(trace)
    sys.exit(main())


# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800

# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270

# 80, 380, 2970, 3270

    # not (380, 480, 0, 100)
# decmin 0.0851497
# decmax 0.106983
# ramin 0.228344
# ramax 0.250178
