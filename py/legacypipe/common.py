from __future__ import print_function

import os
import tempfile
import time
from glob import glob

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.util import Tan, Sip, anwcs_t
from astrometry.util.starutil_numpy import degrees_between, hmsstring2ra, dmsstring2dec
from astrometry.util.miscutils import polygons_intersect, estimate_mode, clip_polygon, clip_wcs
from astrometry.util.resample import resample_with_wcs,OverlapError

from tractor.basics import ConstantSky, NanoMaggies, ConstantFitsWcs, LinearPhotoCal, PointSource, RaDecPos
from tractor.engine import Image, Catalog, Patch
from tractor.galaxy import enable_galaxy_cache, disable_galaxy_cache
from tractor.utils import get_class_from_name
from tractor.ellipses import EllipseESoft
from tractor.sfd import SFDMap

from legacypipe.utils import EllipseWithPriors

# search order: $TMPDIR, $TEMP, $TMP, then /tmp, /var/tmp, /usr/tmp
tempdir = tempfile.gettempdir()

# The apertures we use in aperture photometry, in ARCSEC radius
apertures_arcsec = np.array([0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.])

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

class SimpleGalaxy(ExpGalaxy):
    '''This defines the 'SIMP' galaxy profile -- an exponential profile
    with a fixed shape of a 0.45 arcsec effective radius and spherical
    shape.  It is used to detect marginally-resolved galaxies.
    '''
    shape = EllipseE(0.45, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)
        self.shape = SimpleGalaxy.shape

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness))
    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) + ')')
        
    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'SimpleGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(SimpleGalaxy, self).isParamFrozen(pname)
    
class BrickDuck(object):
    '''A little duck-typing class when running on a custom RA,Dec center
    rather than a brick center.
    '''
    pass

def get_git_version(dir=None):
    '''
    Runs 'git describe' in the current directory (or given dir) and
    returns the result as a string.

    Parameters
    ----------
    dir : string
        If non-None, "cd" to the given directory before running 'git describe'

    Returns
    -------
    Git version string
    '''
    from astrometry.util.run_command import run_command
    cmd = ''
    if dir is not None:
        cmd = "cd '%s' && " % dir
    cmd += 'git describe'
    rtn,version,err = run_command(cmd)
    if rtn:
        raise RuntimeError('Failed to get version string (%s): ' % cmd +
                           version + err)
    version = version.strip()
    return version

def get_version_header(program_name, survey_dir, git_version=None):
    '''
    Creates a fitsio header describing a DECaLS data product.
    '''
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
    hdr.add_record(dict(name='SURVEYV', value=survey_dir,
                        comment='Legacy Survey directory'))
    hdr.add_record(dict(name='DECALSDR', value='DR3',
                        comment='DECaLS release name'))
    surveydir_ver = get_git_version(survey_dir)
    hdr.add_record(dict(name='SURVEYDV', value=surveydir_ver,
                        comment='legacypipe-dir git version'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS',
                        comment='DECam Legacy Survey'))

    # Requested by NOAO
    hdr.add_record(dict(name='SURVEYID', value='DECam Legacy Survey (DECaLS)',
                        comment='Survey name'))
    hdr.add_record(dict(name='DRVERSIO', value='DR3',
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
    def clean(self, **kwargs):
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
    *mnmx*  = (min,max), values that will become black/white *after* scaling.
        Default is (-3,10)
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
    '''
    Converts our softened-ellipticity EllipseESoft parameters into
    normal EllipseE ellipses.
    
    *cat*: an iterable of tractor Sources, which will be modified
     in-place.

    '''
    from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
    from tractor.ellipses import EllipseESoft
    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseESoft.fromEllipseE(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeDev = EllipseESoft.fromEllipseE(src.shapeDev)
            src.shapeExp = EllipseESoft.fromEllipseE(src.shapeExp)

def on_bricks_dependencies(brick, survey, bricks=None):
    # Find nearby bricks from earlier brick phases
    if bricks is None:
        bricks = survey.get_bricks_readonly()
    print(len(bricks), 'bricks')
    bricks = bricks[bricks.brickq < brick.brickq]
    print(len(bricks), 'from phases before this brickq:', brick.brickq)
    if len(bricks) == 0:
        return []
    from astrometry.libkd.spherematch import match_radec

    radius = survey.bricksize * np.sqrt(2.) * 1.01
    bricks.cut(np.abs(brick.dec - bricks.dec) < radius)
    #print(len(bricks), 'within %.2f degree of Dec' % radius)
    I,J,d = match_radec(brick.ra, brick.dec, bricks.ra, bricks.dec, radius)
    bricks.cut(J)
    print(len(bricks), 'within', radius, 'degrees')
    return bricks

def brick_catalog_for_radec_box(ralo, rahi, declo, dechi,
                                survey, catpattern, bricks=None):
    '''
    Merges multiple Tractor brick catalogs to cover an RA,Dec
    bounding-box.

    No cleverness with RA wrap-around; assumes ralo < rahi.

    survey: LegacySurveyData object
    
    bricks: table of bricks, eg from LegacySurveyData.get_bricks()

    catpattern: filename pattern of catalog files to read,
        eg "pipebrick-cats/tractor-phot-%06i.its"
    
    '''
    assert(ralo < rahi)
    assert(declo < dechi)

    if bricks is None:
        bricks = survey.get_bricks_readonly()
    I = survey.bricks_touching_radec_box(bricks, ralo, rahi, declo, dechi)
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
    '''valmap: { 'N7' : 1., 'N8' : 17.8 }

    Returns: a numpy image (shape (12,14)) with values mapped to their
    CCD locations.

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

    # Shift from being (0,0)-centered to being aligned with the
    # ccd_map_image() image.
    x0 += 7
    x1 += 7
    y0 += 6
    y1 += 6
    
    if inset == 0.:
        return (x0,x1,y0,y1)
    return (x0+inset, x1-inset, y0+inset, y1-inset)

def wcs_for_brick(b, W=3600, H=3600, pixscale=0.262):
    '''
    Returns an astrometry.net style Tan WCS object for a given brick object.

    b: row from survey-bricks.fits file
    W,H: size in pixels
    pixscale: pixel scale in arcsec/pixel.

    Returns: Tan wcs object
    '''
    pixscale = pixscale / 3600.
    return Tan(b.ra, b.dec, W/2.+0.5, H/2.+0.5,
               -pixscale, 0., 0., pixscale,
               float(W), float(H))

def bricks_touching_wcs(targetwcs, survey=None, B=None, margin=20):
    '''
    Finds LegacySurvey bricks touching a given WCS header object.

    Parameters
    ----------
    targetwcs : astrometry.util.Tan object or similar
        The region of sky to search
    survey : legacypipe.common.LegacySurveyData object
        From which the brick table will be retrieved
    B : FITS table
        The table of brick objects to search
    margin : int
        Margin in pixels around the outside of the WCS

    Returns
    -------
    A table (subset of B, if given) containing the bricks touching the
    given WCS region + margin.
    '''
    from astrometry.libkd.spherematch import match_radec
    if B is None:
        assert(survey is not None)
        B = survey.get_bricks_readonly()

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

def imsave_jpeg(jpegfn, img, **kwargs):
    '''Saves a image in JPEG format.  Some matplotlib installations
    (notably at NERSC) don't support jpeg, so we write to PNG and then
    convert to JPEG using the venerable netpbm tools.
    
    *jpegfn*: JPEG filename
    *img*: image, in the typical matplotlib formats (see plt.imsave)
    '''

    import pylab as plt
    tmpfn = create_temp(suffix='.png')
    plt.imsave(tmpfn, img, **kwargs)
    cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn))
    os.system(cmd)
    os.unlink(tmpfn)

class LegacySurveyData(object):
    '''
    A class describing the contents of a LEGACY_SURVEY_DIR directory --
    tables of CCDs and of bricks, and calibration data.  Methods for
    dealing with the CCDs and bricks tables.

    This class is also responsible for creating LegacySurveyImage
    objects (eg, DecamImage objects), which then allow data to be read
    from disk.
    '''
    def __init__(self, survey_dir=None, output_dir=None, version=None,
                 ccds=None):
        '''Create a LegacySurveyData object using data from the given
        *survey_dir* directory, or from the $LEGACY_SURVEY_DIR environment
        variable.

        Parameters
        ----------
        survey_dir : string
            Defaults to $LEGACY_SURVEY_DIR environment variable.  Where to look for
            files including calibration files, tables of CCDs and bricks, image data,
            etc.
        output_dir : string
            Base directory for output files; default ".".
        '''
        from .decam  import DecamImage
        from .mosaic import MosaicImage
        from .bok    import BokImage
        from .ptf    import PtfImage

        if survey_dir is None:
            survey_dir = os.environ.get('LEGACY_SURVEY_DIR')
            if survey_dir is None:
                print('''Warning: you should set the $LEGACY_SURVEY_DIR environment variable.
On NERSC, you can do:
  module use /project/projectdirs/cosmo/work/decam/versions/modules
  module load legacysurvey

Now using the current directory as LEGACY_SURVEY_DIR, but this is likely to fail.
''')
                survey_dir = os.getcwd()
                
        self.survey_dir = survey_dir

        if output_dir is None:
            self.output_dir = '.'
        else:
            self.output_dir = output_dir

        self.output_files = []

        self.ccds = ccds
        self.bricks = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25

        self.image_typemap = {
            'decam'  : DecamImage,
            'mosaic' : MosaicImage,
            'mosaic3': MosaicImage,
            '90prime': BokImage,
            'ptf'    : PtfImage,
            }

        self.allbands = 'ugrizY'

        assert(version in [None, 'dr2', 'dr1'])
        self.version = version

    def image_class_for_camera(self, camera):
        # Assert that we have correctly removed trailing spaces
        assert(camera == camera.strip())
        return self.image_typemap[camera]
        
    def index_of_band(self, b):
        return self.allbands.index(b)
        
    def find_file(self, filetype, brick=None, brickpre=None, band='%(band)s',
                  output=False):
        '''
        Returns the filename of a Legacy Survey file.

        *filetype* : string, type of file to find, including:
             "tractor" -- Tractor catalogs
             "depth"   -- PSF depth maps
             "galdepth" -- Canonical galaxy depth maps
             "nexp" -- number-of-exposure maps
             
        *brick* : string, brick name such as "0001p000"

        *output*: True if we are about to write this file; will use self.outdir as
        the base directory rather than self.survey_dir.

        Returns: path to the specified file (whether or not it exists).
        '''
        if brick is None:
            brick = '%(brick)s'
            brickpre = '%(brick).3s'
        else:
            brickpre = brick[:3]

        if output:
            basedir = self.output_dir
        else:
            basedir = self.survey_dir

        if brick is not None:
            codir = os.path.join(basedir, 'coadd', brickpre, brick)

        sname = 'legacysurvey'
        if self.version in ['dr1','dr2']:
            sname = 'decals'
            
        if filetype == 'bricks':
            fn = 'survey-bricks.fits.gz'
            if self.version in ['dr1','dr2']:
                fn = 'decals-bricks.fits'
            return os.path.join(basedir, fn)

        elif filetype == 'ccds':
            if self.version in ['dr1','dr2']:
                return [os.path.join(basedir, 'decals-ccds.fits')]
            else:
                return glob(os.path.join(basedir, 'survey-ccds-*.fits.gz'))

        elif filetype == 'tycho2':
            return os.path.join(basedir, 'tycho2.fits.gz')
            
        elif filetype == 'annotated-ccds':
            return glob(os.path.join(basedir, 'ccds-annotated-*.fits.gz'))

        elif filetype == 'tractor':
            return os.path.join(basedir, 'tractor', brickpre,
                                'tractor-%s.fits' % brick)
        
        elif filetype == 'galaxy-sims':
            return os.path.join(basedir, 'tractor', brickpre,
                                'galaxy-sims-%s.fits' % brick)

        elif filetype in ['ccds-table', 'depth-table']:
            ty = filetype.split('-')[0]
            return os.path.join(codir, 'legacysurvey-%s-%s.fits' % (brick, ty))

        elif filetype in ['image-jpeg', 'model-jpeg', 'resid-jpeg',
                          'imageblob-jpeg', 'simscoadd-jpeg','imagecoadd-jpeg']: #KJB
            ty = filetype.split('-')[0]
            return os.path.join(codir, 'legacysurvey-%s-%s.jpg' % (brick, ty))

        elif filetype in ['depth', 'galdepth', 'nexp', 'model']:
            return os.path.join(codir,
                                '%s-%s-%s-%s.fits.gz' % (sname, brick, filetype, band))

        elif filetype in ['invvar', 'chi2', 'image']:
            if self.version in ['dr1','dr2']:
                prefix = 'decals'
            else:
                prefix = 'legacysurvey'
            return os.path.join(codir, '%s-%s-%s-%s.fits' %
                                (prefix, brick, filetype,band))

        elif filetype in ['blobmap']:
            return os.path.join(basedir, 'metrics', brickpre, brick,
                                'blobs-%s.fits.gz' % (brick))
        elif filetype in ['all-models']:
            return os.path.join(basedir, 'metrics', brickpre, brick,
                                'all-models-%s.fits' % (brick))

        elif filetype == 'sha1sum-brick':
            return os.path.join(basedir, 'tractor', brickpre,
                                'brick-%s.sha1sum' % brick)
        
        assert(False)

    def write_output(self, filetype, **kwargs):
        '''
        Returns a context manager for writing an output file; use like:

        with survey.write_output('ccds', brick=brickname) as out:
            ccds.writeto(out.fn, primheader=primhdr)

        Does the following on entry:
        - calls self.find_file() to determine which filename to write to
        - ensures the output directory exists
        - appends a ".tmp" to the filename

        Does the following on exit:
        - moves the ".tmp" to the final filename (to make it atomic)
        - records the filename for later SHA1 computation
        '''
        class OutputFileContext(object):
            def __init__(self, fn, survey):
                self.real_fn = fn
                self.survey = survey
                self.fn = os.path.join(os.path.dirname(fn), 'tmp-'+os.path.basename(fn))

            def __enter__(self):
                dirnm = os.path.dirname(self.fn)
                if not os.path.exists(dirnm):
                    try:
                        os.makedirs(dirnm)
                    except:
                        pass
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                # If no exception was thrown...
                if exc_type is None:
                    os.rename(self.fn, self.real_fn)
                    self.survey.add_output_file(self.real_fn)

        fn = self.find_file(filetype, output=True, **kwargs)
        out = OutputFileContext(fn, self)
        return out

    def add_output_file(self, fn):
        '''
        Intended as a callback to be called in the *write_output* routine.
        Adds the given filename to the list of files written by the
        pipeline on this run.
        '''
        self.output_files.append(fn)

    def __getstate__(self):
        '''
        For pickling: clear the cached ZP and other tables.
        '''
        d = self.__dict__.copy()
        d['ccds'] = None
        d['bricks'] = None
        d['bricktree'] = None
        return d

    def drop_cache(self):
        '''
        Clears all cached data contained in this object.  Useful for
        pickling / multiprocessing.
        '''
        self.ccds = None
        self.bricks = None
        if self.bricktree is not None:
            from astrometry.libkd.spherematch import tree_free
            tree_free(self.bricktree)
        self.bricktree = None

    def get_calib_dir(self):
        '''
        Returns the directory containing calibration data.
        '''
        return os.path.join(self.survey_dir, 'calib')

    def get_image_dir(self):
        '''
        Returns the directory containing image data.
        '''
        return os.path.join(self.survey_dir, 'images')

    def get_survey_dir(self):
        '''
        Returns the base LEGACY_SURVEY_DIR directory.
        '''
        return self.survey_dir

    def get_se_dir(self):
        '''
        Returns the directory containing SourceExtractor config files,
        used during calibration.
        '''
        return os.path.join(self.survey_dir, 'calib', 'se-config')

    def get_bricks_dr2(self):
        '''
        Returns a table of bricks with DR2 stats.  The caller owns the table.
        '''
        return fits_table(os.path.join(self.survey_dir, 'decals-bricks-dr2.fits'))

    def get_bricks(self):
        '''
        Returns a table of bricks.  The caller owns the table.

        For read-only purposes, see *get_bricks_readonly()*, which
        uses a cached version.
        '''
        return fits_table(self.find_file('bricks'))

    def get_bricks_readonly(self):
        '''
        Returns a read-only (shared) copy of the table of bricks.
        '''
        if self.bricks is None:
            self.bricks = self.get_bricks()
            # Assert that bricks are the sizes we think they are.
            # ... except for the two poles, which are half-sized
            assert(np.all(np.abs((self.bricks.dec2 - self.bricks.dec1)[1:-1] -
                                 self.bricksize) < 1e-8))
        return self.bricks

    def get_brick(self, brickid):
        '''
        Returns a brick (as one row in a table) by *brickid* (integer).
        '''
        B = self.get_bricks_readonly()
        I, = np.nonzero(B.brickid == brickid)
        if len(I) == 0:
            return None
        return B[I[0]]

    def get_brick_by_name(self, brickname):
        '''
        Returns a brick (as one row in a table) by name (string).
        '''
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

    def get_ccds_readonly(self):
        '''
        Returns a shared copy of the table of CCDs.
        '''
        if self.ccds is None:
            self.ccds = self.get_ccds()
        return self.ccds

    def get_ccds(self):
        '''
        Returns the table of CCDs.
        '''
        from glob import glob

        fns = self.find_file('ccds')
        TT = []
        for fn in fns:
            print('Reading CCDs from', fn)
            # cols = (
            #     'exptime filter propid crpix1 crpix2 crval1 crval2 ' +
            #     'cd1_1 cd1_2 cd2_1 cd2_2 ccdname ccdzpt ccdraoff ccddecoff ' +
            #     'ccdnmatch camera image_hdu image_filename width height ' +
            #     'ra dec zpt expnum fwhm mjd_obs').split()
            #T = fits_table(fn, columns=cols)
            T = fits_table(fn)
            print('Got', len(T), 'CCDs')
            TT.append(T)
        T = merge_tables(TT, columns='fillzero')
        print('Total of', len(T), 'CCDs')
        del TT

        cols = T.columns()
        # Make DR1 CCDs table somewhat compatible with DR2
        if 'extname' in cols and not 'ccdname' in cols:
            T.ccdname = T.extname
        if not 'camera' in cols:
            T.camera = np.array(['decam'] * len(T))
        if 'cpimage' in cols and not 'image_filename' in cols:
            T.image_filename = T.cpimage
        if 'cpimage_hdu' in cols and not 'image_hdu' in cols:
            T.image_hdu = T.cpimage_hdu

        # Remove trailing spaces from 'ccdname' column
        if 'ccdname' in T.columns():
            # "N4 " -> "N4"
            T.ccdname = np.array([s.strip() for s in T.ccdname])
        # Remove trailing spaces from 'camera' column.
        T.camera = np.array([c.strip() for c in T.camera])
        return T

    def get_annotated_ccds(self):
        '''
        Returns the annotated table of CCDs.
        '''
        from glob import glob

        fns = self.find_file('annotated-ccds')
        TT = []
        for fn in fns:
            print('Reading annotated CCDs from', fn)
            T = fits_table(fn)
            print('Got', len(T), 'CCDs')
            TT.append(T)
        T = merge_tables(TT, columns='fillzero')
        print('Total of', len(T), 'CCDs')
        del TT
        # Remove trailing spaces from 'ccdname' column
        if 'ccdname' in T.columns():
            # "N4 " -> "N4"
            T.ccdname = np.array([s.strip() for s in T.ccdname])
        # Remove trailing spaces from 'camera' column.
        T.camera = np.array([c.strip() for c in T.camera])
        return T
    
    def ccds_touching_wcs(self, wcs, **kwargs):
        '''
        Returns a table of the CCDs touching the given *wcs* region.
        '''
        T = self.get_ccds_readonly()
        I = ccds_touching_wcs(wcs, T, **kwargs)
        if len(I) == 0:
            return None
        T = T[I]
        return T

    def get_image_object(self, t):
        '''
        Returns a DecamImage or similar object for one row of the CCDs table.
        '''
        # get Image subclass
        imageType = self.image_class_for_camera(t.camera)
        # call Image subclass constructor
        return imageType(self, t)

    def get_approx_wcs(self, ccd):
        W,H = ccd.width,ccd.height
        wcs = Tan(*[float(x) for x in
                    [ccd.crval1, ccd.crval2, ccd.crpix1, ccd.crpix2,
                     ccd.cd1_1,  ccd.cd1_2,  ccd.cd2_1, ccd.cd2_2, W, H]])
        return wcs
    
    def tims_touching_wcs(self, targetwcs, mp, bands=None,
                          **kwargs):
        '''Creates tractor.Image objects for CCDs touching the given
        *targetwcs* region.
        
        mp: multiprocessing object

        kwargs are passed to LegacySurveyImage.get_tractor_image() and
        may include:

        * gaussPsf
        * pixPsf

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
        '''
        Returns a table of CCDs matching the given *expnum* (exposure
        number, integer) and *ccdname* (string).
        '''
        T = self.get_ccds_readonly()
        if expnum is not None:
            T = T[T.expnum == expnum]
        if ccdname is not None:
            T = T[T.ccdname == ccdname]
        return T

    def photometric_ccds(self, ccds):
        '''
        Returns an index array for the members of the table "ccds" that
        are photometric.
        '''
        # Make the is-photometric check camera-specific, handled by the
        # Image subclass.
        cameras = np.unique(ccds.camera)
        print('Finding photometric CCDs.  Cameras:', cameras)
        good = np.zeros(len(ccds), bool)
        for cam in cameras:
            imclass = self.image_class_for_camera(cam)
            Icam = np.flatnonzero(ccds.camera == cam)
            print('Checking', len(Icam), 'images from camera', cam)
            Igood = imclass.photometric_ccds(self, ccds[Icam])
            print('Keeping', len(Igood), 'photometric CCDs from camera', cam)
            if len(Igood):
                good[Icam[Igood]] = True
        return np.flatnonzero(good)

    def apply_blacklist(self, ccds):
        '''
        Returns an index array of CCDs to KEEP; ie, do
        ccds = survey.get_ccds()
        I = survey.apply_blacklist(ccds)
        ccds.cut(I)
        '''
        # Make the blacklist check camera-specific, handled by the
        # Image subclass.
        cameras = np.unique(ccds.camera)
        print('Finding blacklisted CCDs.  Cameras:', cameras)
        good = np.zeros(len(ccds), bool)
        for cam in cameras:
            imclass = self.image_class_for_camera(cam)
            Icam = np.flatnonzero(ccds.camera == cam)
            print('Checking', len(Icam), 'images from camera', cam)
            Igood = imclass.apply_blacklist(self, ccds[Icam])
            print('Keeping', len(Igood), 'non-blacklisted CCDs from camera',
                  cam)
            if len(Igood):
                good[Icam[Igood]] = True
        return np.flatnonzero(good)

def exposure_metadata(filenames, hdus=None, trim=None):
    '''
    Creates a CCD table row object by reading metadata from a FITS
    file header.

    Parameters
    ----------
    filenames : list of strings
        Filenames to read
    hdus : list of integers; None to read all HDUs
        List of FITS extensions (HDUs) to read
    trim : string
        String to trim off the start of the *filenames* for the
        *image_filename* table entry

    Returns
    -------
    A table that looks like the CCDs table.
    '''
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
                ('INSTRUME', ''),
                ('SEEING', nan),
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

    # DECam: INSTRUME = 'DECam'
    T.rename('instrume', 'camera')
    T.camera = np.array([t.lower().strip() for t in T.camera])
    
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
    try:
        return im.run_calibs(**kwargs)
    except:
        print('Exception in run_calibs:', im, kwargs)
        import traceback
        traceback.print_exc()
        raise

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

