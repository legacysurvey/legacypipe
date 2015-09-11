from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt

import os
import tempfile
import time

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.plotutils import dimshow
from astrometry.util.util import Tan, Sip, anwcs_t
from astrometry.util.ttime import CpuMeas, Time
from astrometry.util.starutil_numpy import degrees_between, hmsstring2ra, dmsstring2dec
from astrometry.util.miscutils import polygons_intersect, estimate_mode, clip_polygon, clip_wcs
from astrometry.util.resample import resample_with_wcs,OverlapError

from tractor.basics import ConstantSky, NanoMaggies, ConstantFitsWcs, LinearPhotoCal, PointSource, RaDecPos
from tractor.engine import Image, Catalog, Patch
from tractor.galaxy import enable_galaxy_cache, disable_galaxy_cache
from tractor.utils import get_class_from_name
from tractor.ellipses import EllipseESoft
from tractor.sfd import SFDMap

from .utils import EllipseWithPriors

# search order: $TMPDIR, $TEMP, $TMP, then /tmp, /var/tmp, /usr/tmp
tempdir = tempfile.gettempdir()

# From: http://www.noao.edu/noao/staff/fvaldes/CPDocPrelim/PL201_3.html
# 1   -- detector bad pixel           InstCal
# 1   -- detector bad pixel/no data   Resampled
# 1   -- No data                      Stacked
# 2   -- saturated                    InstCal/Resampled
# 4   -- interpolated                 InstCal/Resampled
# 16  -- single exposure cosmic ray   InstCal/Resampled
# 64  -- bleed trail                  InstCal/Resampled
# 128 -- multi-exposure transient     InstCal/Resampled 
CP_DQ_BITS = dict(badpix=1, satur=2, interp=4, cr=16, bleed=64,
                  trans=128,
                  edge = 256,
                  edge2 = 512) # in z-band images?

# Ugly hack: for sphinx documentation, the astrometry and tractor (and
# other) packages are replaced by mock objects.  But you can't
# subclass a mock object correctly, so we have to un-mock
# EllipseWithPriors here.
if 'Mock' in str(type(EllipseWithPriors)):
    class duck(object):
        pass
    EllipseWithPriors = duck

class LegacyEllipseWithPriors(EllipseWithPriors):
    # Prior on (softened) ellipticity: Gaussian with this standard deviation
    ellipticityStd = 0.25

from tractor.galaxy import ExpGalaxy
from tractor.ellipses import EllipseE
class DesiGalaxy(ExpGalaxy):
    shape = EllipseE(0.45, 0., 0.)
    
    def __init__(self, *args):
        super(DesiGalaxy, self).__init__(*args)
        self.shape = DesiGalaxy.shape

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'DesiGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(DesiGalaxy, self).isParamFrozen(pname)
    
class BrickDuck(object):
    pass

def get_git_version():
    from astrometry.util.run_command import run_command
    rtn,version,err = run_command('git describe')
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    return version

def get_version_header(program_name, decals_dir, git_version=None):
    import datetime

    if program_name is None:
        import sys
        program_name = sys.argv[0]

    if git_version is None:
        git_version = get_git_version()
        #print('Version:', git_version)

    hdr = fitsio.FITSHDR()

    for s in [
        'Data product of the DECam Legacy Survey (DECaLS)',
        'Full documentation at http://legacysurvey.org',
        ]:
        hdr.add_record(dict(name='COMMENT', value=s, comment=s))
    hdr.add_record(dict(name='LEGPIPEV', value=git_version,
                        comment='legacypipe git version'))
    hdr.add_record(dict(name='DECALSV', value=decals_dir,
                        comment='DECaLS version'))
    hdr.add_record(dict(name='DECALSDR', value='DR2',
                        comment='DECaLS release name'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS',
                        comment='DECam Legacy Survey'))

    # Requested by NOAO
    hdr.add_record(dict(name='SURVEYID', value='DECam Legacy Survey (DECaLS)',
                        comment='Survey name'))
    hdr.add_record(dict(name='DRVERSIO', value='DR2',
                        comment='Survey data release number'))
    hdr.add_record(dict(name='OBSTYPE', value='object',
                        comment='Observation type'))
    hdr.add_record(dict(name='PROCTYPE', value='tile',
                        comment='Processing type'))
    
    import socket
    hdr.add_record(dict(name='HOSTNAME', value=socket.gethostname(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='HOSTFQDN', value=socket.getfqdn(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='NERSC', value=os.environ.get('NERSC_HOST', 'none'),
                        comment='NERSC machine where runbrick.py was run'))
    return hdr

class MyFITSHDR(fitsio.FITSHDR):
    '''
    This is copied straight from fitsio, simply removing "BUNIT" from
    the list of headers to remove.  This is required to format the
    tractor catalogs the way we want them.
    '''
    def clean(self):
        """
        Remove reserved keywords from the header.
        
        These are keywords that the fits writer must write in order
        to maintain consistency between header and data.
        """

        rmnames = ['SIMPLE','EXTEND','XTENSION','BITPIX','PCOUNT','GCOUNT',
                   'THEAP',
                   'EXTNAME',
                   #'BUNIT',
                   'BSCALE','BZERO','BLANK',
                   'ZQUANTIZ','ZDITHER0','ZIMAGE','ZCMPTYPE',
                   'ZSIMPLE','ZTENSION','ZPCOUNT','ZGCOUNT',
                   'ZBITPIX','ZEXTEND',
                   #'FZTILELN','FZALGOR',
                   'CHECKSUM','DATASUM']
        self.delete(rmnames)

        r = self._record_map.get('NAXIS',None)
        if r is not None:
            naxis = int(r['value'])
            self.delete('NAXIS')

            rmnames = ['NAXIS%d' % i for i in xrange(1,naxis+1)]
            self.delete(rmnames)

        r = self._record_map.get('ZNAXIS',None)
        self.delete('ZNAXIS')
        if r is not None:

            znaxis = int(r['value'])

            rmnames = ['ZNAXIS%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZTILE%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZNAME%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZVAL%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)

        
        r = self._record_map.get('TFIELDS',None)
        if r is not None:
            tfields = int(r['value'])
            self.delete('TFIELDS')

            if tfields > 0:

                nbase = ['TFORM','TTYPE','TDIM','TUNIT','TSCAL','TZERO',
                         'TNULL','TDISP','TDMIN','TDMAX','TDESC','TROTA',
                         'TRPIX','TRVAL','TDELT','TCUNI',
                         #'FZALG'
                        ]
                for i in xrange(1,tfields+1):
                    names=['%s%d' % (n,i) for n in nbase]
                    self.delete(names)

def bin_image(data, invvar, S):
    # rebin image data
    H,W = data.shape
    sH,sW = (H+S-1)/S, (W+S-1)/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    newiv = np.zeros((sH,sW), dtype=invvar.dtype)
    for i in range(S):
        for j in range(S):
            iv = invvar[i::S, j::S]
            subh,subw = iv.shape
            newdata[:subh,:subw] += data[i::S, j::S] * iv
            newiv  [:subh,:subw] += iv
    newdata /= (newiv + (newiv == 0)*1.)
    newdata[newiv == 0] = 0.
    return newdata,newiv

def segment_and_group_sources(image, T, name=None, ps=None, plots=False):
    '''
    *image*: binary image that defines "blobs"
    *T*: source table; only ".itx" and ".ity" elements are used (x,y integer pix pos).  Note: ".blob" field is added.
    *name*: for debugging only

    Returns: (blobs, blobsrcs, blobslices)

    *blobs*: image, values -1 = no blob, integer blob indices
    *blobsrcs*: list of np arrays of integers, elements in T within each blob
    *blobslices*: list of slice objects for blob bounding-boxes.
    
    '''
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import label, find_objects

    emptyblob = 0

    image = binary_fill_holes(image)

    blobs,nblobs = label(image)
    print('N detected blobs:', nblobs)
    H,W = image.shape
    del image

    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]

    if plots:
        plt.clf()
        dimshow(blobs > 0, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1), ha='center', va='bottom', color='r')
        plt.plot(T.itx, T.ity, 'rx')
        for i,t in enumerate(T):
            plt.text(t.itx, t.ity, 'src %i' % i, color='red', ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    # Find sets of sources within blobs
    blobsrcs = []
    keepslices = []
    blobmap = {}
    dropslices = {}
    for blob in range(1, nblobs+1):
        Isrcs = np.flatnonzero(T.blob == blob)
        if len(Isrcs) == 0:
            #print('Blob', blob, 'has no sources')
            blobmap[blob] = -1
            dropslices[blob] = blobslices[blob-1]
            continue
        blobmap[blob] = len(blobsrcs)
        blobsrcs.append(Isrcs)
        bslc = blobslices[blob-1]
        keepslices.append(bslc)

    blobslices = keepslices

    # Find sources that do not belong to a blob and add them as
    # singleton "blobs"; otherwise they don't get optimized.
    # for sources outside the image bounds, what should we do?
    inblobs = np.zeros(len(T), bool)
    for Isrcs in blobsrcs:
        inblobs[Isrcs] = True
    noblobs = np.flatnonzero(np.logical_not(inblobs))
    del inblobs
    # Add new fake blobs!
    for ib,i in enumerate(noblobs):
        #S = 3
        S = 5
        bslc = (slice(np.clip(T.ity[i] - S, 0, H-1), np.clip(T.ity[i] + S+1, 0, H)),
                slice(np.clip(T.itx[i] - S, 0, W-1), np.clip(T.itx[i] + S+1, 0, W)))

        # Does this new blob overlap existing blob(s)?
        oblobs = np.unique(blobs[bslc])
        oblobs = oblobs[oblobs != emptyblob]

        #print('This blob overlaps existing blobs:', oblobs)
        if len(oblobs) > 1:
            print('WARNING: not merging overlapping blobs like maybe we should')
        if len(oblobs):
            blob = oblobs[0]
            #print('Adding source to existing blob', blob)
            blobs[bslc][blobs[bslc] == emptyblob] = blob
            blobindex = blobmap[blob]
            if blobindex == -1:
                # the overlapping blob was going to be dropped -- restore it.
                blobindex = len(blobsrcs)
                blobmap[blob] = blobindex
                blobslices.append(dropslices[blob])
                blobsrcs.append(np.array([], np.int64))
            # Expand the existing blob slice to encompass this new source
            oldslc = blobslices[blobindex]
            sy,sx = oldslc
            oy0,oy1, ox0,ox1 = sy.start,sy.stop, sx.start,sx.stop
            sy,sx = bslc
            ny0,ny1, nx0,nx1 = sy.start,sy.stop, sx.start,sx.stop
            newslc = slice(min(oy0,ny0), max(oy1,ny1)), slice(min(ox0,nx0), max(ox1,nx1))
            blobslices[blobindex] = newslc
            # Add this source to the list of source indices for the existing blob.
            blobsrcs[blobindex] = np.append(blobsrcs[blobindex], np.array([i]))

        else:
            # Set synthetic blob number
            blob = nblobs+1 + ib
            blobs[bslc][blobs[bslc] == emptyblob] = blob
            blobmap[blob] = len(blobsrcs)
            blobslices.append(bslc)
            blobsrcs.append(np.array([i]))
    #print('Added', len(noblobs), 'new fake singleton blobs')

    # Remap the "blobs" image so that empty regions are = -1 and the blob values
    # correspond to their indices in the "blobsrcs" list.
    if len(blobmap):
        maxblob = max(blobmap.keys())
    else:
        maxblob = 0
    maxblob = max(maxblob, blobs.max())
    bm = np.zeros(maxblob + 1, int)
    for k,v in blobmap.items():
        bm[k] = v
    bm[0] = -1

    # DEBUG
    if plots:
        fitsio.write('blobs-before-%s.fits' % name, blobs, clobber=True)

    # Remap blob numbers
    blobs = bm[blobs]

    if plots:
        fitsio.write('blobs-after-%s.fits' % name, blobs, clobber=True)

    if plots:
        plt.clf()
        dimshow(blobs > -1, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1), ha='center', va='bottom', color='r')
        plt.plot(T.itx, T.ity, 'rx')
        for i,t in enumerate(T):
            plt.text(t.itx, t.ity, 'src %i' % i, color='red', ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    for j,Isrcs in enumerate(blobsrcs):
        for i in Isrcs:
            #assert(blobs[T.ity[i], T.itx[i]] == j)
            if (blobs[T.ity[i], T.itx[i]] != j):
                print('---------------------------!!!--------------------------')
                print('Blob', j, 'sources', Isrcs)
                print('Source', i, 'coords x,y', T.itx[i], T.ity[i])
                print('Expected blob value', j, 'but got', blobs[T.ity[i], T.itx[i]])

    T.blob = blobs[T.ity, T.itx]
    assert(len(blobsrcs) == len(blobslices))

    return blobs, blobsrcs, blobslices

def tim_get_resamp(tim, targetwcs):
    if hasattr(tim, 'resamp'):
        return tim.resamp

    try:
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, tim.subwcs, [], 2)
    except OverlapError:
        print('No overlap')
        return None
    if len(Yo) == 0:
        return None
    resamp = [x.astype(np.int16) for x in (Yo,Xo,Yi,Xi)]
    return resamp

def get_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None):
    '''
    Given a list of images in the given bands, returns a scaled RGB
    image.

    *imgs*  a list of numpy arrays, all the same size, in nanomaggies
    *bands* a list of strings, eg, ['g','r','z']
    *mnmx*  = (min,max), values that will become black/white *after* scaling. Default is (-3,10)
    *arcsinh* use nonlinear scaling as in SDSS
    *scales*

    Returns a (H,W,3) numpy array with values between 0 and 1.
    '''
    bands = ''.join(bands)

    grzscales = dict(g = (2, 0.0066),
                      r = (1, 0.01),
                      z = (0, 0.025),
                      )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
        elif bands == 'urz':
            scales = dict(u = (2, 0.0066),
                          r = (1, 0.01),
                          z = (0, 0.025),
                          )
        elif bands == 'gri':
            # scales = dict(g = (2, 0.004),
            #               r = (1, 0.0066),
            #               i = (0, 0.01),
            #               )
            scales = dict(g = (2, 0.002),
                          r = (1, 0.004),
                          i = (0, 0.005),
                          )
        else:
            scales = grzscales
        
    h,w = imgs[0].shape
    rgb = np.zeros((h,w,3), np.float32)
    # Convert to ~ sigmas
    for im,band in zip(imgs, bands):
        plane,scale = scales[band]
        rgb[:,:,plane] = (im / scale).astype(np.float32)

    if mnmx is None:
        mn,mx = -3, 10
    else:
        mn,mx = mnmx

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        rgb = nlmap(rgb)
        mn = nlmap(mn)
        mx = nlmap(mx)

    rgb = (rgb - mn) / (mx - mn)
    return np.clip(rgb, 0., 1.)
    

def switch_to_soft_ellipses(cat):
    from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
    from tractor.ellipses import EllipseESoft
    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseESoft.fromEllipseE(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeDev = EllipseESoft.fromEllipseE(src.shapeDev)
            src.shapeExp = EllipseESoft.fromEllipseE(src.shapeExp)

def brick_catalog_for_radec_box(ralo, rahi, declo, dechi,
                                decals, catpattern, bricks=None):
    '''
    Merges multiple Tractor brick catalogs to cover an RA,Dec
    bounding-box.

    No cleverness with RA wrap-around; assumes ralo < rahi.

    decals: Decals object
    
    bricks: table of bricks, eg from Decals.get_bricks()

    catpattern: filename pattern of catalog files to read,
        eg "pipebrick-cats/tractor-phot-%06i.its"
    
    '''
    assert(ralo < rahi)
    assert(declo < dechi)

    if bricks is None:
        bricks = decals.get_bricks_readonly()
    I = decals.bricks_touching_radec_box(bricks, ralo, rahi, declo, dechi)
    print(len(I), 'bricks touch RA,Dec box')
    TT = []
    hdr = None
    for i in I:
        brick = bricks[i]
        fn = catpattern % brick.brickid
        print('Catalog', fn)
        if not os.path.exists(fn):
            print('Warning: catalog does not exist:', fn)
            continue
        T = fits_table(fn, header=True)
        if T is None or len(T) == 0:
            print('Warning: empty catalog', fn)
            continue
        T.cut((T.ra  >= ralo ) * (T.ra  <= rahi) *
              (T.dec >= declo) * (T.dec <= dechi))
        TT.append(T)
    if len(TT) == 0:
        return None
    T = merge_tables(TT)
    # arbitrarily keep the first header
    T._header = TT[0]._header
    return T
    
def ccd_map_image(valmap, empty=0.):
    '''
    valmap: { 'N7' : 1., 'N8' : 17.8 }

    Returns: a numpy image (shape (12,14)) with values mapped to their CCD locations.
    '''
    img = np.empty((12,14))
    img[:,:] = empty
    for k,v in valmap.items():
        x0,x1,y0,y1 = ccd_map_extent(k)
        #img[y0+6:y1+6, x0+7:x1+7] = v
        img[y0:y1, x0:x1] = v
    return img

def ccd_map_center(ccdname):
    x0,x1,y0,y1 = ccd_map_extent(ccdname)
    return (x0+x1)/2., (y0+y1)/2.

def ccd_map_extent(ccdname, inset=0.):
    assert(ccdname.startswith('N') or ccdname.startswith('S'))
    num = int(ccdname[1:])
    assert(num >= 1 and num <= 31)
    if num <= 7:
        x0 = 7 - 2*num
        y0 = 0
    elif num <= 13:
        x0 = 6 - (num - 7)*2
        y0 = 1
    elif num <= 19:
        x0 = 6 - (num - 13)*2
        y0 = 2
    elif num <= 24:
        x0 = 5 - (num - 19)*2
        y0 = 3
    elif num <= 28:
        x0 = 4 - (num - 24)*2
        y0 = 4
    else:
        x0 = 3 - (num - 28)*2
        y0 = 5
    if ccdname.startswith('N'):
        (x0,x1,y0,y1) = (x0, x0+2, -y0-1, -y0)
    else:
        (x0,x1,y0,y1) = (x0, x0+2, y0, y0+1)

    # Shift from being (0,0)-centered to being aligned with the ccd_map_image() image.
    x0 += 7
    x1 += 7
    y0 += 6
    y1 += 6
    
    if inset == 0.:
        return (x0,x1,y0,y1)
    return (x0+inset, x1-inset, y0+inset, y1-inset)

def wcs_for_brick(b, W=3600, H=3600, pixscale=0.262):
    '''
    b: row from decals-bricks.fits file
    W,H: size in pixels
    pixscale: pixel scale in arcsec/pixel.

    Returns: Tan wcs object
    '''
    pixscale = pixscale / 3600.
    return Tan(b.ra, b.dec, W/2.+0.5, H/2.+0.5,
               -pixscale, 0., 0., pixscale,
               float(W), float(H))

def bricks_touching_wcs(targetwcs, decals=None, B=None, margin=20):
    # margin: How far outside the image to keep objects
    # FIXME -- should be adaptive to object size!

    from astrometry.libkd.spherematch import match_radec
    if B is None:
        assert(decals is not None)
        B = decals.get_bricks_readonly()

    ra,dec = targetwcs.radec_center()
    radius = targetwcs.radius()
        
    # MAGIC 0.4 degree search radius =
    # DECam hypot(1024,2048)*0.27/3600 + Brick hypot(0.25, 0.25) ~= 0.35 + margin
    I,J,d = match_radec(B.ra, B.dec, ra, dec,
                        radius + np.hypot(0.25,0.25)/2. + 0.05)
    print(len(I), 'bricks nearby')
    keep = []
    for i in I:
        b = B[i]
        brickwcs = wcs_for_brick(b)
        clip = clip_wcs(targetwcs, brickwcs)
        if len(clip) == 0:
            print('No overlap with brick', b.brickname)
            continue
        keep.append(i)
    return B[np.array(keep)]

        
def ccds_touching_wcs(targetwcs, ccds, ccdrad=0.17, polygons=True):
    '''
    targetwcs: wcs object describing region of interest
    ccds: fits_table object of CCDs

    ccdrad: radius of CCDs, in degrees.  Default 0.17 is for DECam.
    #If None, computed from T.

    Returns: index array I of CCDs within range.
    '''
    trad = targetwcs.radius()
    if ccdrad is None:
        ccdrad = max(np.sqrt(np.abs(ccds.cd1_1 * ccds.cd2_2 -
                                    ccds.cd1_2 * ccds.cd2_1)) *
                     np.hypot(ccds.width, ccds.height) / 2.)

    rad = trad + ccdrad
    r,d = targetwcs.radec_center()
    I, = np.nonzero(np.abs(ccds.dec - d) < rad)
    I = I[np.atleast_1d(degrees_between(ccds.ra[I], ccds.dec[I], r, d) < rad)]

    if not polygons:
        return I
    # now check actual polygon intersection
    tw,th = targetwcs.imagew, targetwcs.imageh
    targetpoly = [(0.5,0.5),(tw+0.5,0.5),(tw+0.5,th+0.5),(0.5,th+0.5)]
    cd = targetwcs.get_cd()
    tdet = cd[0]*cd[3] - cd[1]*cd[2]
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)

    keep = []
    for i in I:
        W,H = ccds.width[i],ccds.height[i]
        wcs = Tan(*[float(x) for x in
                    [ccds.crval1[i], ccds.crval2[i], ccds.crpix1[i], ccds.crpix2[i],
                     ccds.cd1_1[i], ccds.cd1_2[i], ccds.cd2_1[i], ccds.cd2_2[i], W, H]])
        cd = wcs.get_cd()
        wdet = cd[0]*cd[3] - cd[1]*cd[2]
        poly = []
        for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5)]:
            rr,dd = wcs.pixelxy2radec(x,y)
            ok,xx,yy = targetwcs.radec2pixelxy(rr,dd)
            poly.append((xx,yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    return I

def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    os.unlink(fn)
    return fn


class Decals(object):
    def __init__(self, decals_dir=None):
        if decals_dir is None:
            decals_dir = os.environ.get('DECALS_DIR')
            if decals_dir is None:
                print('''Warning: you should set the $DECALS_DIR environment variable.
On NERSC, you can do:
  module use /project/projectdirs/cosmo/work/decam/versions/modules
  module load decals
Using the current directory as DECALS_DIR, but this is likely to fail.
''')
                decals_dir = os.getcwd()
                
        self.decals_dir = decals_dir

        self.ZP = None
        self.bricks = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25

        self.image_typemap = {
            'decam': DecamImage,
            '90prime': BokImage,
            }

    def get_calib_dir(self):
        return os.path.join(self.decals_dir, 'calib')

    def get_image_dir(self):
        return os.path.join(self.decals_dir, 'images')

    def get_decals_dir(self):
        return self.decals_dir

    def get_se_dir(self):
        return os.path.join(self.decals_dir, 'calib', 'se-config')

    def get_bricks(self):
        return fits_table(os.path.join(self.decals_dir, 'decals-bricks.fits'))

    ### HACK...
    def get_bricks_readonly(self):
        if self.bricks is None:
            self.bricks = self.get_bricks()
            # Assert that bricks are the sizes we think they are.
            # ... except for the two poles, which are half-sized
            assert(np.all(np.abs((self.bricks.dec2 - self.bricks.dec1)[1:-1] -
                                 self.bricksize) < 1e-8))
        return self.bricks

    def get_brick(self, brickid):
        B = self.get_bricks_readonly()
        I, = np.nonzero(B.brickid == brickid)
        if len(I) == 0:
            return None
        return B[I[0]]

    def get_brick_by_name(self, brickname):
        B = self.get_bricks_readonly()
        I, = np.nonzero(np.array([n == brickname for n in B.brickname]))
        if len(I) == 0:
            return None
        return B[I[0]]

    def bricks_touching_radec_box(self, bricks,
                                  ralo, rahi, declo, dechi):
        '''
        Returns an index vector of the bricks that touch the given RA,Dec box.
        '''
        if bricks is None:
            bricks = self.get_bricks_readonly()
        if self.cache_tree and bricks == self.bricks:
            from astrometry.libkd.spherematch import tree_build_radec, tree_search_radec
            # Use kdtree
            if self.bricktree is None:
                self.bricktree = tree_build_radec(bricks.ra, bricks.dec)
            # brick size
            radius = np.sqrt(2.)/2. * self.bricksize
            # + RA,Dec box size
            radius = radius + degrees_between(ralo, declo, rahi, dechi) / 2.
            dec = (dechi + declo) / 2.
            c = (np.cos(np.deg2rad(rahi)) + np.cos(np.deg2rad(ralo))) / 2.
            s = (np.sin(np.deg2rad(rahi)) + np.sin(np.deg2rad(ralo))) / 2.
            ra  = np.rad2deg(np.arctan2(s, c))
            J = tree_search_radec(self.bricktree, ra, dec, radius)
            I = J[np.nonzero((bricks.ra1[J]  <= rahi ) * (bricks.ra2[J]  >= ralo) *
                             (bricks.dec1[J] <= dechi) * (bricks.dec2[J] >= declo))[0]]
            return I

        if rahi < ralo:
            # Wrap-around
            print('In Dec slice:', len(np.flatnonzero((bricks.dec1 <= dechi) *
                                                      (bricks.dec2 >= declo))))
            print('Above RAlo=', ralo, ':', len(np.flatnonzero(bricks.ra2 >= ralo)))
            print('Below RAhi=', rahi, ':', len(np.flatnonzero(bricks.ra1 <= rahi)))
            print('In RA slice:', len(np.nonzero(np.logical_or(bricks.ra2 >= ralo,
                                                               bricks.ra1 <= rahi))))
                    
            I, = np.nonzero(np.logical_or(bricks.ra2 >= ralo, bricks.ra1 <= rahi) *
                            (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
            print('In RA&Dec slice', len(I))
        else:
            I, = np.nonzero((bricks.ra1  <= rahi ) * (bricks.ra2  >= ralo) *
                            (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
        return I
    
    def get_ccds(self):
        fn = os.path.join(self.decals_dir, 'decals-ccds.fits')
        if not os.path.exists(fn):
            fn += '.gz'
        print('Reading CCDs from', fn)
        T = fits_table(fn)
        print('Got', len(T), 'CCDs')
        # "N4 " -> "N4"
        T.ccdname = np.array([s.strip() for s in T.ccdname])
        return T

    def ccds_touching_wcs(self, wcs, **kwargs):
        T = self.get_ccds()
        I = ccds_touching_wcs(wcs, T, **kwargs)
        if len(I) == 0:
            return None
        T.cut(I)
        return T

    def get_image_object(self, t):
        '''
        Returns a DecamImage or similar object for one row of the CCDs table.
        '''
        imageType = self.image_typemap[t.camera.strip()]
        return imageType(self, t)
    
    def tims_touching_wcs(self, targetwcs, mp, bands=None,
                          **kwargs):
        '''
        mp: multiprocessing object

        kwargs are passed to LegacySurveyImage.get_tractor_image() and may include:

        * gaussPsf
        * const2psf
        * pixPsf
        * splinesky
        
        '''
        # Read images
        C = self.ccds_touching_wcs(targetwcs)
        # Sort by band
        if bands is not None:
            C.cut(np.hstack([np.nonzero(C.filter == band)[0]
                             for band in bands]))
        ims = []
        for t in C:
            print()
            print('Image file', t.image_filename, 'hdu', t.image_hdu)
            im = DecamImage(self, t)
            ims.append(im)
        # Read images, clip to ROI
        W,H = targetwcs.get_width(), targetwcs.get_height()
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        args = [(im, targetrd, kwargs) for im in ims]
        tims = mp.map(read_one_tim, args)
        return tims
    
    def find_ccds(self, expnum=None, ccdname=None):
        T = self.get_ccds()
        if expnum is not None:
            T.cut(T.expnum == expnum)
        if ccdname is not None:
            T.cut(T.ccdname == ccdname)
        return T

    def photometric_ccds(self, CCD):
        '''
        Returns an index array for the members of the table "CCD" that
        are photometric.
        '''
        
        '''
        Recipe in [decam-data 1314], 2015-07-15:
        
        * CCDNMATCH >= 20  (At least 20 stars to determine zero-pt)
        * abs(ZPT - CCDZPT) < 0.10  (Agreement with full-frame zero-pt)
        * CCDPHRMS < 0.2  (Uniform photometry across the CCD)
        * ZPT within 0.50 mag of 25.08 for g-band
        * ZPT within 0.50 mag of 25.29 for r-band
        * ZPT within 0.50 mag of 24.92 for z-band

        * DEC > -20 (in DESI footprint) 
        * CCDNUM = 31 (S7) is OK, but only for the region
          [1:1023,1:4094] (mask region [1024:2046,1:4094] in CCD s7)

        Slightly revised by DJS in Re: [decam-data 828] 2015-07-31:
        
        * CCDNMATCH >= 20 (At least 20 stars to determine zero-pt)
        * abs(ZPT - CCDZPT) < 0.10  (Loose agreement with full-frame zero-pt)
        * ZPT within [25.08-0.50, 25.08+0.25] for g-band
        * ZPT within [25.29-0.50, 25.29+0.25] for r-band
        * ZPT within [24.92-0.50, 24.92+0.25] for z-band
        * DEC > -20 (in DESI footprint)
        * EXPTIME >= 30
        * CCDNUM = 31 (S7) should mask outside the region [1:1023,1:4094]
        '''

        # We assume that the zeropoints are present in the
        # CCDs file (starting in DR2)
        z0 = dict(g = 25.08,
                  r = 25.29,
                  z = 24.92,)
        z0 = np.array([z0[f[0]] for f in CCD.filter])

        good = np.ones(len(CCD), bool)
        n0 = sum(good)
        # This is our list of cuts to remove non-photometric CCD images
        for name,crit in [
            ('exptime < 30 s', (CCD.exptime < 30)),
            ('ccdnmatch < 20', (CCD.ccdnmatch < 20)),
            ('abs(zpt - ccdzpt) > 0.1',
             (np.abs(CCD.zpt - CCD.ccdzpt) > 0.1)),
            ('zpt < 0.5 mag of nominal (for DECam)',
             ((CCD.camera == 'decam') * (CCD.zpt < (z0 - 0.5)))),
            ('zpt > 0.25 mag of nominal (for DECam)',
             ((CCD.camera == 'decam') * (CCD.zpt > (z0 + 0.25)))),
             ]:
            good[crit] = False
            n = sum(good)
            print('Flagged', n0-n, 'more non-photometric using criterion:', name)
            n0 = n

        return np.flatnonzero(good)

    def _get_zeropoints_table(self):
        if self.ZP is not None:
            return self.ZP
        # Hooray, DRY
        self.ZP = self.get_ccds()
        return self.ZP

    def get_zeropoint_row_for(self, im):
        ZP = self._get_zeropoints_table()
        I, = np.nonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I, = np.nonzero((ZP.expnum == im.expnum) *
                            (ZP.ccdname == im.ccdname))
        if len(I) == 0:
            return None
        assert(len(I) == 1)
        return ZP[I[0]]
            
    def get_zeropoint_for(self, im):
        zp = self.get_zeropoint_row_for(im)
        # No updated zeropoint -- use header MAGZERO from primary HDU.
        if zp is None:
            print('WARNING: using header zeropoints for', im)
            hdr = im.read_image_primary_header()
            # DES Year1 Stripe82 images:
            magzero = hdr['MAGZERO']
            return magzero

        magzp = zp.ccdzpt
        magzp += 2.5 * np.log10(zp.exptime)
        return magzp

    def get_astrometric_zeropoint_for(self, im):
        zp = self.get_zeropoint_row_for(im)
        if zp is None:
            print('WARNING: no astrometric zeropoints found for', im)
            return 0.,0.
        dra, ddec = zp.ccdraoff, zp.ccddecoff
        return dra / 3600., ddec / 3600.

def exposure_metadata(filenames, hdus=None, trim=None):
    nan = np.nan
    primkeys = [('FILTER',''),
                ('RA', nan),
                ('DEC', nan),
                ('AIRMASS', nan),
                ('DATE-OBS', ''),
                ('EXPTIME', nan),
                ('EXPNUM', 0),
                ('MJD-OBS', 0),
                ('PROPID', ''),
                ]
    hdrkeys = [('AVSKY', nan),
               ('ARAWGAIN', nan),
               ('FWHM', nan),
               ('CRPIX1',nan),
               ('CRPIX2',nan),
               ('CRVAL1',nan),
               ('CRVAL2',nan),
               ('CD1_1',nan),
               ('CD1_2',nan),
               ('CD2_1',nan),
               ('CD2_2',nan),
               ('EXTNAME',''),
               ('CCDNUM',''),
               ]

    otherkeys = [('IMAGE_FILENAME',''), ('IMAGE_HDU',0),
                 ('HEIGHT',0),('WIDTH',0),
                 ]

    allkeys = primkeys + hdrkeys + otherkeys

    vals = dict([(k,[]) for k,d in allkeys])

    for i,fn in enumerate(filenames):
        print('Reading', (i+1), 'of', len(filenames), ':', fn)
        F = fitsio.FITS(fn)
        primhdr = F[0].read_header()
        expstr = '%08i' % primhdr.get('EXPNUM')

        # # Parse date with format: 2014-08-09T04:20:50.812543
        # date = datetime.datetime.strptime(primhdr.get('DATE-OBS'),
        #                                   '%Y-%m-%dT%H:%M:%S.%f')
        # # Subract 12 hours to get the date used by the CP to label the night;
        # # CP20140818 includes observations with date 2014-08-18 evening and
        # # 2014-08-19 early AM.
        # cpdate = date - datetime.timedelta(0.5)
        # #cpdatestr = '%04i%02i%02i' % (cpdate.year, cpdate.month, cpdate.day)
        # #print 'Date', date, '-> CP', cpdatestr
        # cpdateval = cpdate.year * 10000 + cpdate.month * 100 + cpdate.day
        # print 'Date', date, '-> CP', cpdateval

        cpfn = fn
        if trim is not None:
            cpfn = cpfn.replace(trim, '')
        print('CP fn', cpfn)

        if hdus is not None:
            hdulist = hdus
        else:
            hdulist = range(1, len(F))

        for hdu in hdulist:
            hdr = F[hdu].read_header()

            info = F[hdu].get_info()
            #'extname': 'S1', 'dims': [4146L, 2160L]
            H,W = info['dims']

            for k,d in primkeys:
                vals[k].append(primhdr.get(k, d))
            for k,d in hdrkeys:
                vals[k].append(hdr.get(k, d))

            vals['IMAGE_FILENAME'].append(cpfn)
            vals['IMAGE_HDU'].append(hdu)
            vals['WIDTH'].append(int(W))
            vals['HEIGHT'].append(int(H))

    T = fits_table()
    for k,d in allkeys:
        T.set(k.lower().replace('-','_'), np.array(vals[k]))
    #T.about()

    #T.rename('extname', 'ccdname')
    T.ccdname = np.array([t.strip() for t in T.extname])
    
    T.filter = np.array([s.split()[0] for s in T.filter])
    T.ra_bore  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec_bore = np.array([dmsstring2dec(s) for s in T.dec])

    T.ra  = np.zeros(len(T))
    T.dec = np.zeros(len(T))
    for i in range(len(T)):
        W,H = T.width[i], T.height[i]

        wcs = Tan(T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i],
                  T.cd1_1[i], T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], float(W), float(H))
        
        xc,yc = W/2.+0.5, H/2.+0.5
        rc,dc = wcs.pixelxy2radec(xc,yc)
        T.ra [i] = rc
        T.dec[i] = dc

    return T

            
def run_calibs(X):
    im = X[0]
    kwargs = X[1]
    print('run_calibs for image', im)
    return im.run_calibs(**kwargs)

def read_one_tim(X):
    (im, targetrd, kwargs) = X
    print('Reading', im)
    tim = im.get_tractor_image(radecpoly=targetrd, **kwargs)
    return tim


from tractor.psfex import PsfExModel

class SchlegelPsfModel(PsfExModel):
    def __init__(self, fn=None, ext=1):
        '''
        `ext` is ignored.
        '''
        if fn is not None:
            from astrometry.util.fits import fits_table
            T = fits_table(fn)
            T.about()

            ims = fitsio.read(fn, ext=2)
            print('Eigen-images', ims.shape)
            nsch,h,w = ims.shape

            hdr = fitsio.read_header(fn)
            x0 = 0.
            y0 = 0.
            xscale = 1. / hdr['XSCALE']
            yscale = 1. / hdr['YSCALE']
            degree = (T.xexp + T.yexp).max()
            self.sampling = 1.

            # Reorder the 'ims' to match the way PsfEx sorts its polynomial terms

            # number of terms in polynomial
            ne = (degree + 1) * (degree + 2) / 2
            print('Number of eigen-PSFs required for degree=', degree, 'is', ne)

            self.psfbases = np.zeros((ne, h,w))

            for d in range(degree + 1):
                # x polynomial degree = j
                # y polynomial degree = k
                for j in range(d+1):
                    k = d - j
                    ii = j + (degree+1) * k - (k * (k-1))/ 2

                    jj = np.flatnonzero((T.xexp == j) * (T.yexp == k))
                    if len(jj) == 0:
                        print('Schlegel image for power', j,k, 'not found')
                        continue
                    im = ims[jj,:,:]
                    print('Schlegel image for power', j,k, 'has range', im.min(), im.max(), 'sum', im.sum())
                    self.psfbases[ii,:,:] = ims[jj,:,:]

            self.xscale, self.yscale = xscale, yscale
            self.x0,self.y0 = x0,y0
            self.degree = degree
            print('SchlegelPsfEx degree:', self.degree)
            bh,bw = self.psfbases[0].shape
            self.radius = (bh+1)/2.



from .sdss  import get_sdss_sources
from .image import LegacySurveyImage
from .bok   import BokImage
from .decam import DecamImage
from .detection import (detection_maps, sed_matched_filters, 
                        run_sed_matched_filters)
            
