from __future__ import print_function
import numpy as np
import os

from astrometry.util.fits import fits_table
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.util import Tan
from astrometry.util.miscutils import polygon_area
import tractor.sfd

def annotate(ccds, survey, camera, mp=None, normalizePsf=True, carryOn=True):
    if mp is None:
        from astrometry.util.multiproc import multiproc
        mp = multiproc()

    # File from the "observing" svn repo:
    from pkg_resources import resource_filename

    if camera == 'mosaic':
        tilefile = resource_filename('legacyzpts', 'data/mosaic-tiles_obstatus.fits')
    elif camera == '90prime':
        tilefile = None
    elif camera == 'decam':
        tilefile = resource_filename('legacyzpts', 'data/decam-tiles_obstatus.fits')
    else:
        tilefile = None

    if tilefile is not None:
        if os.path.isfile(tilefile):
            print('Reading {}'.format(tilefile))
            tiles = fits_table(tilefile)
        else:
            print('Required tile file {} not found!'.format(tilefile))
            raise IOError
    else:
        tiles = None

    if tiles is not None:
        # Map tile IDs back to index in the obstatus file.
        tileid_to_index = np.empty(max(tiles.tileid)+1, int)
        tileid_to_index[:] = -1
        tileid_to_index[tiles.tileid] = np.arange(len(tiles))

    anns = mp.map(annotate_one_ccd, [
        (ccd, survey, normalizePsf, carryOn) for ccd in ccds])

    gaussgalnorm = np.zeros(len(ccds), np.float32)
    for iccd,ann in enumerate(anns):
        tileid = ann.pop('tileid', -1)
        if tileid > -1:
            index = tileid_to_index[tileid]
            if index == -1:
                print('Tile ID not found in obstatus file')
            else:
                tile = tiles[index]
                assert(tile.tileid == tileid)
                ccds.tileid  [iccd] = tile.tileid
                ccds.tilepass[iccd] = tile.get('pass')
                ccds.tileebv [iccd] = tile.ebv_med
        gaussgalnorm[iccd] = ann.pop('gaussgalnorm', 0)
        for k,v in ann.items():
            ccds.get(k)[iccd] = v

    sfd = tractor.sfd.SFDMap()
    allbands = 'ugrizY'
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, ccds.ra_center,
                             ccds.dec_center, get_ebv=True)

    ext[np.logical_not(ccds.annotated),:] = 0.
    ebv[np.logical_not(ccds.annotated)] = 0.

    ext = ext.astype(np.float32)
    ccds.ebv = ebv.astype(np.float32)
    ccds.decam_extinction = ext[:,:len(allbands)]
    ccds.wise_extinction = ext[:,len(allbands):]

    # Depth
    with np.errstate(invalid='ignore', divide='ignore'):
        detsig1 = ccds.sig1 / ccds.psfnorm_mean
        depth = 5. * detsig1
        # that's flux in nanomaggies -- convert to mag
        ccds.psfdepth = -2.5 * (np.log10(depth) - 9)

        detsig1 = ccds.sig1 / ccds.galnorm_mean
        depth = 5. * detsig1
        # that's flux in nanomaggies -- convert to mag
        ccds.galdepth = -2.5 * (np.log10(depth) - 9)

        # Depth using Gaussian FWHM.
        psf_sigma = ccds.fwhm / 2.35
        gnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        detsig1 = ccds.sig1 / gnorm
        depth = 5. * detsig1
        # that's flux in nanomaggies -- convert to mag
        ccds.gausspsfdepth = -2.5 * (np.log10(depth) - 9)

        # Gaussian galaxy depth
        detsig1 = ccds.sig1 / gaussgalnorm
        depth = 5. * detsig1
        # that's flux in nanomaggies -- convert to mag
        ccds.gaussgaldepth = -2.5 * (np.log10(depth) - 9)

    # NaN depths -> 0
    for X in [ccds.psfdepth, ccds.galdepth, ccds.gausspsfdepth, ccds.gaussgaldepth]:
        X[np.logical_not(np.isfinite(X))] = 0.

def annotate_one_ccd(X):
    ccd, survey, normalizePsf, carryOn = X
    print('Annotating CCD', ccd.image_filename.strip(), 'expnum', ccd.expnum,
          'CCD', ccd.ccdname)
    result = {}
    try:
        im = survey.get_image_object(ccd)
    except:
        print('Failed to get_image_object()')
        import traceback
        traceback.print_exc()
        if carryOn:
            return result
        else:
            raise

    X = im.get_good_image_subregion()
    reg = [-1,-1,-1,-1]
    for i,x in enumerate(X):
        if x is not None:
            reg[i] = x
    result.update(good_region=reg)

    kwargs = dict(pixPsf=True, subsky=False,
                  pixels=False, dq=False, invvar=False,
                  normalizePsf=normalizePsf,
                  no_remap_invvar=True,
                  trim_edges=False)
    psf = None
    wcs = None
    sky = None

    if ccd.ccdnastrom == 0: # something went terribly wrong
        print('ccdnastrom == 0; bailing on annotation')
        return result

    try:
        tim = im.get_tractor_image(**kwargs)
    except:
        print('Failed to get_tractor_image')
        import traceback
        traceback.print_exc()
        if carryOn:
            return result
        else:
            raise

    if tim is None:
        print('Failed to get_tractor_image; bailing on annotation')
        return result

    psf = tim.psf
    wcs = tim.wcs.wcs
    sky = tim.sky
    hdr = tim.primhdr

    result.update(humidity = hdr.get('HUMIDITY'),
                  outtemp = hdr.get('OUTTEMP'),
                  plver = tim.plver,
                  procdate = tim.procdate,
                  plprocid = tim.plprocid)

    # parse 'DECaLS_15150_r' to get tile number
    obj = ccd.object.strip()
    words = obj.split('_')
    if len(words) == 3 and words[0] in ('DECaLS', 'MzLS', 'MOSAIC'):
        try:
            tileid = int(words[1])
            result.update(tileid=tileid)
        except:
            import traceback
            traceback.print_exc()

    # Instantiate PSF on a grid
    S = 32
    W,H = ccd.width, ccd.height
    xx = np.linspace(1+S, W-S, 5)
    yy = np.linspace(1+S, H-S, 5)
    xx,yy = np.meshgrid(xx, yy)
    psfnorms = []
    galnorms = []
    for x,y in zip(xx.ravel(), yy.ravel()):
        tim.psf = psf.constantPsfAt(x, y)
        try:
            p = im.psf_norm(tim, x=x, y=y)
            g = im.galaxy_norm(tim, x=x, y=y)
            psfnorms.append(p)
            galnorms.append(g)
        except:
            pass

    tim.psf = psf
    result.update(psfnorm_mean = np.mean(psfnorms),
                  psfnorm_std  = np.std (psfnorms),
                  galnorm_mean = np.mean(galnorms),
                  galnorm_std  = np.std (galnorms))

    # PSF in center of field
    cx,cy = (W+1)/2., (H+1)/2.
    p = psf.getPointSourcePatch(cx, cy).patch
    ph,pw = p.shape
    px,py = np.meshgrid(np.arange(pw), np.arange(ph))
    psum = np.sum(p)
    p /= psum
    # centroids
    cenx = np.sum(p * px)
    ceny = np.sum(p * py)
    # second moments
    x2 = np.sum(p * (px - cenx)**2)
    y2 = np.sum(p * (py - ceny)**2)
    xy = np.sum(p * (px - cenx)*(py - ceny))
    # semi-major/minor axes and position angle
    theta = np.rad2deg(np.arctan2(2 * xy, x2 - y2) / 2.)
    theta = np.abs(theta) * np.sign(xy)
    s = np.sqrt(((x2 - y2)/2.)**2 + xy**2)
    a = np.sqrt((x2 + y2) / 2. + s)
    b = np.sqrt((x2 + y2) / 2. - s)
    ell = 1. - b/a
    result.update(psf_mx2 = x2,
                  psf_my2 = y2,
                  psf_mxy = xy,
                  psf_a   = a,
                  psf_b   = b,
                  psf_theta = theta,
                  psf_ell   = ell)
    # Galaxy norm using Gaussian approximation of PSF.
    realpsf = tim.psf

    tim.psf = im.read_psf_model(0, 0, gaussPsf=True,
                                psf_sigma=tim.psf_sigma)
    result.update(gaussgalnorm = im.galaxy_norm(tim, x=cx, y=cy))
    tim.psf = realpsf

    has_skygrid = hasattr(sky, 'evaluateGrid')

    # Sky -- evaluate on a grid (every ~10th pixel)
    if has_skygrid:
        skygrid = sky.evaluateGrid(np.linspace(0, ccd.width-1,  int(1+ccd.width/10)),
                                   np.linspace(0, ccd.height-1, int(1+ccd.height/10)))
        result.update(meansky = np.mean(skygrid),
                      stdsky  = np.std(skygrid),
                      maxsky  = skygrid.max(),
                      minsky  = skygrid.min())
    else:
        skyval = sky.getConstant()
        result.update(meansky = skyval,
                      stdsky  = 0.,
                      maxsky  = skyval,
                      minsky  = skyval)

    # WCS
    r0,d0 = wcs.pixelxy2radec(1, 1)
    r1,d1 = wcs.pixelxy2radec(1, H)
    r2,d2 = wcs.pixelxy2radec(W, H)
    r3,d3 = wcs.pixelxy2radec(W, 1)
    result.update(ra0=r0, dec0=d0, ra1=r1, dec1=d1,
                  ra2=r2, dec2=d2, ra3=r3, dec3=d3)

    midx, midy = (W+1)/2., (H+1)/2.
    rc,dc  = wcs.pixelxy2radec(midx, midy)
    ra,dec = wcs.pixelxy2radec([1,W,midx,midx], [midy,midy,1,H])

    result.update(dra  = max(degrees_between(ra, dc+np.zeros_like(ra) , rc, dc)),
                  ddec = max(degrees_between(rc+np.zeros_like(dec),dec, rc, dc)),
                  ra_center = rc,
                  dec_center = dc)

    # Compute scale change across the chip
    # how many pixels to step
    step = 10
    xx = np.linspace(1+step, W-step, 5)
    yy = np.linspace(1+step, H-step, 5)
    xx,yy = np.meshgrid(xx, yy)
    pixscale = []
    for x,y in zip(xx.ravel(), yy.ravel()):
        sx = [x-step, x-step, x+step, x+step, x-step]
        sy = [y-step, y+step, y+step, y-step, y-step]
        sr,sd = wcs.pixelxy2radec(sx, sy)
        rc,dc = wcs.pixelxy2radec(x, y)
        # project around a tiny little TAN WCS at (x,y), with 1" pixels
        locwcs = Tan(rc, dc, 0., 0., 1./3600, 0., 0., 1./3600, 1., 1.)
        _,lx,ly = locwcs.radec2pixelxy(sr, sd)
        A = polygon_area((lx, ly))
        pixscale.append(np.sqrt(A / (2*step)**2))
    result.update(pixscale_mean = np.mean(pixscale),
                  pixscale_std  = np.std(pixscale),
                  pixscale_min  = min(pixscale),
                  pixscale_max  = max(pixscale))
    result.update(annotated=True)
    print('Finished annotation')
    return result

def init_annotations(ccds):
    ccds.annotated = np.zeros(len(ccds), bool)

    ccds.good_region = np.empty((len(ccds), 4), np.int16)
    ccds.good_region[:,:] = -1

    ccds.ra0  = np.zeros(len(ccds), np.float64)
    ccds.dec0 = np.zeros(len(ccds), np.float64)
    ccds.ra1  = np.zeros(len(ccds), np.float64)
    ccds.dec1 = np.zeros(len(ccds), np.float64)
    ccds.ra2  = np.zeros(len(ccds), np.float64)
    ccds.dec2 = np.zeros(len(ccds), np.float64)
    ccds.ra3  = np.zeros(len(ccds), np.float64)
    ccds.dec3 = np.zeros(len(ccds), np.float64)

    ccds.dra  = np.zeros(len(ccds), np.float32)
    ccds.ddec = np.zeros(len(ccds), np.float32)
    ccds.ra_center  = np.zeros(len(ccds), np.float64)
    ccds.dec_center = np.zeros(len(ccds), np.float64)

    ccds.meansky = np.zeros(len(ccds), np.float32)
    ccds.stdsky  = np.zeros(len(ccds), np.float32)
    ccds.maxsky  = np.zeros(len(ccds), np.float32)
    ccds.minsky  = np.zeros(len(ccds), np.float32)

    ccds.pixscale_mean = np.zeros(len(ccds), np.float32)
    ccds.pixscale_std  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_max  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_min  = np.zeros(len(ccds), np.float32)

    ccds.psfnorm_mean = np.zeros(len(ccds), np.float32)
    ccds.psfnorm_std  = np.zeros(len(ccds), np.float32)
    ccds.galnorm_mean = np.zeros(len(ccds), np.float32)
    ccds.galnorm_std  = np.zeros(len(ccds), np.float32)

    # 2nd moments
    ccds.psf_mx2 = np.zeros(len(ccds), np.float32)
    ccds.psf_my2 = np.zeros(len(ccds), np.float32)
    ccds.psf_mxy = np.zeros(len(ccds), np.float32)
    #
    ccds.psf_a = np.zeros(len(ccds), np.float32)
    ccds.psf_b = np.zeros(len(ccds), np.float32)
    ccds.psf_theta = np.zeros(len(ccds), np.float32)
    ccds.psf_ell   = np.zeros(len(ccds), np.float32)

    ccds.humidity = np.zeros(len(ccds), np.float32)
    ccds.outtemp  = np.zeros(len(ccds), np.float32)

    ccds.tileid   = np.zeros(len(ccds), np.int32)
    ccds.tilepass = np.zeros(len(ccds), np.uint8)
    ccds.tileebv  = np.zeros(len(ccds), np.float32)
