from __future__ import print_function

import os
import tempfile

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs

from tractor.ellipses import EllipseESoft, EllipseE
from tractor.galaxy import ExpGalaxy
from tractor import PointSource, ParamList, ConstantFitsWcs

from legacypipe.utils import EllipseWithPriors, galaxy_min_re

import logging
logger = logging.getLogger('legacypipe.survey')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

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

def year_to_mjd(year):
    # year_to_mjd(2015.5) -> 57205.875
    from tractor.tractortime import TAITime
    return (year - 2000.) * TAITime.daysperyear + TAITime.mjd2k

def mjd_to_year(mjd):
    # mjd_to_year(57205.875) -> 2015.5
    from tractor.tractortime import TAITime
    return (mjd - TAITime.mjd2k) / TAITime.daysperyear + 2000.

def tai_to_mjd(tai):
    return tai / (24. * 3600.)

def radec_at_mjd(ra, dec, ref_year, pmra, pmdec, parallax, mjd):
    '''
    Units:
    - matches Gaia DR1/DR2
    - pmra,pmdec are in mas/yr.  pmra is in angular speed (ie, has a cos(dec) factor)
    - parallax is in mas.


    NOTE: does not broadcast completely correctly -- all params
    vectors or all motion params vector + scalar mjd work fine.  Other
    combos: not certain.

    Returns RA,Dec

    '''

    from tractor.tractortime import TAITime
    from astrometry.util.starutil_numpy import radectoxyz, arcsecperrad, axistilt, xyztoradec

    dt = mjd_to_year(mjd) - ref_year
    cosdec = np.cos(np.deg2rad(dec))

    dec = dec +  dt * pmdec / (3600. * 1000.)
    ra  = ra  + (dt * pmra  / (3600. * 1000.)) / cosdec

    parallax = np.atleast_1d(parallax)
    I = np.flatnonzero(parallax)
    if len(I):
        scalar = np.isscalar(ra) and np.isscalar(dec)

        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)

        suntheta = 2.*np.pi * np.fmod((mjd - TAITime.equinox) / TAITime.daysperyear, 1.0)
        # Finite differences on the unit sphere -- xyztoradec handles
        # points that are not exactly on the surface of the sphere.
        axis = np.deg2rad(axistilt)
        scale = parallax[I] / 1000. / arcsecperrad
        xyz = radectoxyz(ra[I], dec[I])
        xyz[:,0] += scale * np.cos(suntheta)
        xyz[:,1] += scale * np.sin(suntheta) * np.cos(axis)
        xyz[:,2] += scale * np.sin(suntheta) * np.sin(axis)
        r,d = xyztoradec(xyz)
        ra [I] = r
        dec[I] = d

        # radectoxyz / xyztoradec do weird broadcasting
        if scalar:
            ra  = ra[0]
            dec = dec[0]

    return ra,dec

# Gaia measures positions better than we will, we assume, so the
# GaiaPosition class pretends that it does not have any parameters
# that can be optimized; therefore they stay fixed.

class GaiaPosition(ParamList):
    def __init__(self, ra, dec, ref_epoch, pmra, pmdec, parallax):
        '''
        Units:
        - matches Gaia DR1
        - pmra,pmdec are in mas/yr.  pmra is in angular speed (ie, has a cos(dec) factor)
        - parallax is in mas.
        - ref_epoch: year (eg 2015.5)
        '''
        self.ra = ra
        self.dec = dec
        self.ref_epoch = float(ref_epoch)
        self.pmra = pmra
        self.pmdec = pmdec
        self.parallax = parallax
        super(GaiaPosition, self).__init__()
        self.cached_positions = {}

    def copy(self):
        return GaiaPosition(self.ra, self.dec, self.ref_epoch, self.pmra, self.pmdec,
                            self.parallax)

    def getPositionAtTime(self, mjd):
        from tractor import RaDecPos

        try:
            return self.cached_positions[mjd]
        except KeyError:
            # not cached
            pass
        if self.pmra == 0. and self.pmdec == 0. and self.parallax == 0.:
            pos = RaDecPos(self.ra, self.dec)
            self.cached_positions[mjd] = pos
            return pos

        ra,dec = radec_at_mjd(self.ra, self.dec, self.ref_epoch,
                              self.pmra, self.pmdec, self.parallax, mjd)
        pos = RaDecPos(ra, dec)
        self.cached_positions[mjd] = pos
        return pos

    @staticmethod
    def getName():
        return 'GaiaPosition'

    def __str__(self):
        return ('%s: RA, Dec = (%.5f, %.5f), pm (%.1f, %.1f), parallax %.3f' %
                (self.getName(), self.ra, self.dec, self.pmra, self.pmdec, self.parallax))


class GaiaSource(PointSource):
    @staticmethod
    def getName():
        return 'GaiaSource'

    def getSourceType(self):
        return 'GaiaSource'

    @classmethod
    def from_catalog(cls, g, bands):
        from tractor import NanoMaggies
        # Gaia has NaN entries when no proper motion or parallax is measured.
        # Convert to zeros.
        def nantozero(x):
            if not np.isfinite(x):
                return 0.
            return x
        pos = GaiaPosition(g.ra, g.dec, g.ref_epoch,
                           nantozero(g.pmra),
                           nantozero(g.pmdec),
                           nantozero(g.parallax))

        # initialize from decam_mag_B if available, otherwise Gaia G.
        fluxes = {}
        for band in bands:
            try:
                mag = g.get('decam_mag_%s' % band)
            except KeyError:
                mag = g.phot_g_mean_mag
            fluxes[band] = NanoMaggies.magToNanomaggies(mag)
        bright = NanoMaggies(order=bands, **fluxes)
        src = cls(pos, bright)
        src.forced_point_source = g.pointsource
        src.reference_star = getattr(g, 'isgaia', False) or getattr(g, 'isbright', False)
        return src

#
# We need a subclass of the standand WCS class to handle moving sources.
#

class LegacySurveyWcs(ConstantFitsWcs):
    def __init__(self, wcs, tai):
        super(LegacySurveyWcs, self).__init__(wcs)
        self.tai = tai

    def copy(self):
        return LegacySurveyWcs(self.wcs, self.tai)

    def positionToPixel(self, pos, src=None):
        if isinstance(pos, GaiaPosition):
            pos = pos.getPositionAtTime(tai_to_mjd(self.tai.getValue()))
        return super(LegacySurveyWcs, self).positionToPixel(pos, src=src)


class LegacyEllipseWithPriors(EllipseWithPriors):
    # Prior on (softened) ellipticity: Gaussian with this standard deviation
    ellipticityStd = 0.25

from tractor.sersic import SersicIndex

class LegacySersicIndex(SersicIndex):
    def __init__(self, val=0):
        super(LegacySersicIndex, self).__init__(val=val)
        self.lower = 0.5
        self.upper = 6.0

class LogRadius(EllipseESoft):
    ''' Class used during fitting of the RexGalaxy type -- an ellipse
    type where only the radius is variable, and is represented in log
    space.'''
    def __init__(self, *args, **kwargs):
        super(LogRadius, self).__init__(*args, **kwargs)
        self.lowers = [None]
        # MAGIC -- 10" default max r_e!
        # SEE ALSO utils.py : class(EllipseWithPriors)!
        self.uppers = [np.log(10.)]
        self.lowers = [np.log(galaxy_min_re)]

    def isLegal(self):
        return ((self.logre <= self.uppers[0]) and
                (self.logre >= self.lowers[0]))

    def setMaxLogRadius(self, rmax):
        self.uppers[0] = rmax

    def getMaxLogRadius(self):
        return self.uppers[0]

    @staticmethod
    def getName():
        return 'LogRadius'

    @staticmethod
    def getNamedParams():
        # log r: log of effective radius in arcsec
        return dict(logre=0)

    def __repr__(self):
        return 'log r_e=%g' % (self.logre)

    @property
    def theta(self):
        return 0.

    @property
    def e(self):
        return 0.

class RexGalaxy(ExpGalaxy):
    '''This defines the 'REX' galaxy profile -- an exponential profile
    that is round (zero ellipticity) with variable radius.
    It is used to measure marginally-resolved galaxies.

    The real action (what makes it a Rex) happens when it is constructed,
    via, eg,

        rex = RexGalaxy(position, brightness, LogRadius(0.))

    (which happens in oneblob.py)
    '''
    def __init__(self, *args):
        super(RexGalaxy, self).__init__(*args)

    def getName(self):
        return 'RexGalaxy'

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
    def __init__(self, ra, dec, brickname):
        self.ra  = ra
        self.dec = dec
        self.brickname = brickname
        self.brickid = -1


def get_git_version(dirnm=None):
    '''
    Runs 'git describe' in the current directory (or given dir) and
    returns the result as a string.

    Parameters
    ----------
    dirnm : string
        If non-None, "cd" to the given directory before running 'git describe'

    Returns
    -------
    Git version string
    '''
    from astrometry.util.run_command import run_command
    cmd = ''
    if dirnm is None:
        # Get the git version of the legacypipe product
        import legacypipe
        dirnm = os.path.dirname(legacypipe.__file__)

    cmd = "cd '%s' && git describe" % dirnm
    rtn,version,err = run_command(cmd)
    if rtn:
        raise RuntimeError('Failed to get version string (%s): ' % cmd +
                           version + err)
    version = version.strip()
    return version

def get_version_header(program_name, survey_dir, release, git_version=None):

    '''
    Creates a fitsio header describing a DECaLS data product.
    '''
    import datetime

    if program_name is None:
        import sys
        program_name = sys.argv[0]

    if git_version is None:
        git_version = get_git_version()

    hdr = fitsio.FITSHDR()
    #lsdir_prefix1 = '/global/project/projectdirs'
    #lsdir_prefix2 = '/global/projecta/projectdirs'
    for s in [
        'Data product of the DESI Imaging Legacy Survey (DECaLS)',
        'Full documentation at http://legacysurvey.org',
        ]:
        hdr.add_record(dict(name='COMMENT', value=s, comment=s))
    hdr.add_record(dict(name='LEGPIPEV', value=git_version,
                        comment='legacypipe git version'))
    #hdr.add_record(dict(name='LSDIRPFX', value=lsdir_prefix1,
    #                    comment='LegacySurveys Directory Prefix'))
    hdr.add_record(dict(name='LSDIR', value=survey_dir,
                        comment='$LEGACY_SURVEY_DIR directory'))
    hdr.add_record(dict(name='LSDR', value='DR9',
                        comment='Data release number'))
    hdr.add_record(dict(name='RUNDATE', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS+BASS+MzLS',
                        comment='The LegacySurveys'))
    # Requested by NOAO
    hdr.add_record(dict(name='SURVEYID', value='DECaLS BASS MzLS',
                        comment='Survey names'))
    #hdr.add_record(dict(name='SURVEYID', value='DECam Legacy Survey (DECaLS)',
    #hdr.add_record(dict(name='SURVEYID', value='BASS MzLS',
    hdr.add_record(dict(name='DRVERSIO', value=release,
                        comment='LegacySurveys Data Release number'))
    hdr.add_record(dict(name='OBSTYPE', value='object',
                        comment='Observation type'))
    hdr.add_record(dict(name='PROCTYPE', value='tile',
                        comment='Processing type'))

    import socket
    hdr.add_record(dict(name='NODENAME', value=socket.gethostname(),
                        comment='Machine where script was run'))
    #hdr.add_record(dict(name='HOSTFQDN', value=socket.getfqdn(),comment='Machine where script was run'))
    hdr.add_record(dict(name='HOSTNAME', value=os.environ.get('NERSC_HOST', 'none'),
                        comment='NERSC machine where script was run'))
    hdr.add_record(dict(name='JOB_ID', value=os.environ.get('SLURM_JOB_ID', 'none'),
                        comment='SLURM job id'))
    hdr.add_record(dict(name='ARRAY_ID', value=os.environ.get('ARRAY_TASK_ID', 'none'),
                        comment='SLURM job array id'))

    return hdr

def get_dependency_versions(unwise_dir, unwise_tr_dir, unwise_modelsky_dir):
    import astrometry
    import astropy
    import matplotlib
    try:
        import mkl_fft
    except:
        mkl_fft = None
    import photutils
    import tractor
    import scipy
    import unwise_psf

    depvers = []
    headers = []
    default_ver = 'UNAVAILABLE'
    for name,pkg in [('astrometry', astrometry),
                     ('astropy', astropy),
                     ('fitsio', fitsio),
                     ('matplotlib', matplotlib),
                     ('mkl_fft', mkl_fft),
                     ('numpy', np),
                     ('photutils', photutils),
                     ('scipy', scipy),
                     ('tractor', tractor),
                     ('unwise_psf', unwise_psf),
                     ]:
        if pkg is None:
            depvers.append((name, 'none'))
            continue
        #depvers.append((name + '-path', os.path.dirname(pkg.__file__)))
        try:
            depvers.append((name, pkg.__version__))
        except:
            try:
                depvers.append((name,
                                get_git_version(os.path.dirname(pkg.__file__))))
            except:
                pass

    # Get additional paths from environment variables
    dep = 'LARGEGALAXIES_CAT'
    value = os.environ.get(dep, default_ver)
    if value == default_ver:
        print('Warning: failed to get version string for "%s"' % dep)
    else:
        depvers.append((dep, value))
        if os.path.exists(value):
            from legacypipe.reference import get_large_galaxy_version
            ver,preburn = get_large_galaxy_version(value)
            depvers.append(('LARGEGALAXIES_VER', ver))
            depvers.append(('LARGEGALAXIES_PREBURN', preburn))

    for dep in ['TYCHO2_KD', 'GAIA_CAT']:
        value = os.environ.get('%s_DIR' % dep, default_ver)
        if value == default_ver:
            print('Warning: failed to get version string for "%s"' % dep)
        else:
            depvers.append((dep, value))

    if unwise_dir is not None:
        dirs = unwise_dir.split(':')
        depvers.append(('unwise', unwise_dir))
        for i,d in enumerate(dirs):
            headers.append(('UNWISD%i' % (i+1), d, ''))

    if unwise_tr_dir is not None:
        depvers.append(('unwise_tr', unwise_tr_dir))
        # this is assumed to be only a single directory
        headers.append(('UNWISTD', unwise_tr_dir, ''))

    if unwise_modelsky_dir is not None:
        depvers.append(('unwise_modelsky', unwise_modelsky_dir))
        # this is assumed to be only a single directory
        headers.append(('UNWISSKY', unwise_modelsky_dir, ''))

    for i,(name,value) in enumerate(depvers):
        headers.append(('DEPNAM%02i' % i, name, ''))
        headers.append(('DEPVER%02i' % i, value, ''))
    return headers

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

def tim_get_resamp(tim, targetwcs):
    from astrometry.util.resample import resample_with_wcs,OverlapError

    if hasattr(tim, 'resamp'):
        return tim.resamp
    try:
        Yo,Xo,Yi,Xi,_ = resample_with_wcs(targetwcs, tim.subwcs, intType=np.int16)
    except OverlapError:
        print('No overlap')
        return None
    if len(Yo) == 0:
        return None
    return Yo,Xo,Yi,Xi


def sdss_rgb(imgs, bands, scales=None, m=0.03, Q=20, mnmx=None):
    rgbscales=dict(g=(2, 6.0),
                   r=(1, 3.4),
                   i=(0, 3.0),
                   z=(0, 2.2))
    # rgbscales = {'u': 1.5, #1.0,
    #              'g': 2.5,
    #              'r': 1.5,
    #              'i': 1.0,
    #              'z': 0.4, #0.3
    #              }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    if Q is not None:
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        I = fI / I
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        if mnmx is None:
            rgb[:,:,plane] = np.clip((img * scale + m) * I, 0, 1)
        else:
            mn,mx = mnmx
            rgb[:,:,plane] = np.clip(((img * scale + m) - mn) / (mx - mn), 0, 1)
    return rgb

def get_rgb(imgs, bands,
            resids=False, mnmx=None, arcsinh=None):
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
    # (ignore arcsinh...)
    if resids:
        mnmx = (-0.1, 0.1)
    if mnmx is not None:
        return sdss_rgb(imgs, bands, m=0., Q=None, mnmx=mnmx)
    return sdss_rgb(imgs, bands)

def wcs_for_brick(b, W=3600, H=3600, pixscale=0.262):
    '''
    Returns an astrometry.net style Tan WCS object for a given brick object.

    b: row from survey-bricks.fits file
    W,H: size in pixels
    pixscale: pixel scale in arcsec/pixel.

    Returns: Tan wcs object
    '''
    from astrometry.util.util import Tan
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
    survey : legacypipe.survey.LegacySurveyData object
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
    from astrometry.util.miscutils import clip_wcs

    if B is None:
        assert(survey is not None)
        B = survey.get_bricks_readonly()

    ra,dec = targetwcs.radec_center()
    radius = targetwcs.radius()

    # MAGIC 0.4 degree search radius =
    # DECam hypot(1024,2048)*0.27/3600 + Brick hypot(0.25, 0.25) ~= 0.35 + margin
    I,_,_ = match_radec(B.ra, B.dec, ra, dec,
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

def ccds_touching_wcs(targetwcs, ccds, ccdrad=None, polygons=True):
    '''
    targetwcs: wcs object describing region of interest
    ccds: fits_table object of CCDs

    ccdrad: radius of CCDs, in degrees.
    If None (the default), compute from the CCDs table.
    (0.17 for DECam)

    Returns: index array I of CCDs within range.
    '''
    from astrometry.util.util import Tan
    from astrometry.util.miscutils import polygons_intersect
    from astrometry.util.starutil_numpy import degrees_between

    trad = targetwcs.radius()
    if ccdrad is None:
        ccdrad = max(np.sqrt(np.abs(ccds.cd1_1 * ccds.cd2_2 -
                                    ccds.cd1_2 * ccds.cd2_1)) *
                     np.hypot(ccds.width, ccds.height) / 2.)

    rad = trad + ccdrad
    r,d = targetwcs.radec_center()
    I, = np.where(np.abs(ccds.dec - d) < rad)
    I = I[np.where(degrees_between(r, d, ccds.ra[I], ccds.dec[I]) < rad)[0]]
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
    don't support jpeg, so we optionally write to PNG and then convert
    to JPEG using the venerable netpbm tools.

    *jpegfn*: JPEG filename
    *img*: image, in the typical matplotlib formats (see plt.imsave)
    '''
    import pylab as plt
    if True:
        kwargs.update(format='jpg')
        plt.imsave(jpegfn, img, **kwargs)
    else:
        tmpfn = create_temp(suffix='.png')
        plt.imsave(tmpfn, img, **kwargs)
        cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn))
        rtn = os.system(cmd)
        print(cmd, '->', rtn)
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

    def __init__(self, survey_dir=None, cache_dir=None, output_dir=None,
                 allbands='grz'):
        '''Create a LegacySurveyData object using data from the given
        *survey_dir* directory, or from the $LEGACY_SURVEY_DIR environment
        variable.

        Parameters
        ----------
        survey_dir : string
            Defaults to $LEGACY_SURVEY_DIR environment variable.  Where to look for
            files including calibration files, tables of CCDs and bricks, image data,
            etc.

        cache_dir : string
            Directory to search for input files before looking in survey_dir.  Useful
            for, eg, Burst Buffer.

        output_dir : string
            Base directory for output files; default ".".
        '''
        from legacypipe.decam  import DecamImage
        from legacypipe.mosaic import MosaicImage
        from legacypipe.bok    import BokImage
        from legacypipe.ptf    import PtfImage
        from legacypipe.cfht   import MegaPrimeImage
        from collections import OrderedDict

        if survey_dir is None:
            survey_dir = os.environ.get('LEGACY_SURVEY_DIR')
            if survey_dir is None:
                print('Warning: you should set the $LEGACY_SURVEY_DIR environment variable.')
                print('Using the current directory as LEGACY_SURVEY_DIR.')
                survey_dir = os.getcwd()

        self.survey_dir = survey_dir
        self.cache_dir = cache_dir

        if output_dir is None:
            self.output_dir = '.'
        else:
            self.output_dir = output_dir

        self.output_file_hashes = OrderedDict()
        self.ccds = None
        self.bricks = None
        self.ccds_index = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25

        # Cached CCD kd-tree --
        # - initially None, then a list of (fn, kd)
        self.ccd_kdtrees = None

        self.image_typemap = {
            'decam'  : DecamImage,
            'decam+noise'  : DecamImage,
            'mosaic' : MosaicImage,
            'mosaic3': MosaicImage,
            '90prime': BokImage,
            'ptf'    : PtfImage,
            'megaprime': MegaPrimeImage,
            }

        self.allbands = allbands

        # Filename prefix for coadd files
        self.file_prefix = 'legacysurvey'

    def __str__(self):
        return ('%s: dir %s, out %s' %
                (type(self).__name__, self.survey_dir, self.output_dir))

    def get_default_release(self):
        return None

    def ccds_for_fitting(self, brick, ccds):
        # By default, use all.
        return None

    def image_class_for_camera(self, camera):
        # Assert that we have correctly removed trailing spaces
        assert(camera == camera.strip())
        return self.image_typemap[camera]

    def sed_matched_filters(self, bands):
        from legacypipe.detection import sed_matched_filters
        return sed_matched_filters(bands)

    def index_of_band(self, b):
        return self.allbands.index(b)

    def read_intermediate_catalog(self, brick, **kwargs):
        '''
        Reads the intermediate tractor catalog for the given brickname.

        *kwargs*: passed to self.find_file()

        Returns (T, hdr, primhdr)
        '''
        fn = self.find_file('tractor-intermediate', brick=brick, **kwargs)
        T = fits_table(fn)
        hdr = T.get_header()
        primhdr = fitsio.read_header(fn)

        in_flux_prefix = ''
        # Ensure flux arrays are 2d (N x 1)
        keys = ['flux', 'flux_ivar', 'rchi2', 'fracflux', 'fracmasked',
                'fracin', 'nobs', 'anymask', 'allmask', 'psfsize', 'depth',
                'galdepth']
        for k in keys:
            incol = '%s%s' % (in_flux_prefix, k)
            X = T.get(incol)
            # Hmm, if we need to reshape one of these arrays, we will
            # need to do all of them.
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
                T.set(incol, X)
        return T, hdr, primhdr

    def find_file(self, filetype, brick=None, brickpre=None, band='%(band)s',
                  camera=None, expnum=None, ccdname=None,
                  output=False, **kwargs):
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
        from glob import glob

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

        # Swap in files in the self.cache_dir, if they exist.
        def swap(fn):
            if output:
                return fn
            return self.check_cache(fn)
        def swaplist(fns):
            if output or self.cache_dir is None:
                return fns
            return [self.check_cache(fn) for fn in fns]

        sname = self.file_prefix

        if filetype == 'bricks':
            return swap(os.path.join(basedir, 'survey-bricks.fits.gz'))

        elif filetype == 'ccds':
            return swaplist(
                glob(os.path.join(basedir, 'survey-ccds*.fits.gz')))

        elif filetype == 'ccd-kds':
            return swaplist(
                glob(os.path.join(basedir, 'survey-ccds*.kd.fits')))

        elif filetype == 'tycho2':
            dirnm = os.environ.get('TYCHO2_KD_DIR')
            if dirnm is not None:
                fn = os.path.join(dirnm, 'tycho2.kd.fits')
                if os.path.exists(fn):
                    return fn
            return swap(os.path.join(basedir, 'tycho2.kd.fits'))

        elif filetype == 'large-galaxies':
            fn = os.environ.get('LARGEGALAXIES_CAT')
            if fn is not None:
                if os.path.isfile(fn):
                    return fn
            pat = os.path.join(basedir, 'LSLGA-*.kd.fits')
            fns = glob(pat)
            if len(fns) == 0:
                return None
            if len(fns) > 2:
                print('More than one filename matched large-galaxy pattern', pat,
                      '; using the first one,', fns[0])
            return fns[0]

        elif filetype == 'annotated-ccds':
            return swaplist(
                glob(os.path.join(basedir, 'ccds-annotated-*.fits.gz')))

        elif filetype == 'tractor':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'tractor-%s.fits' % brick))

        elif filetype == 'tractor-intermediate':
            return swap(os.path.join(basedir, 'tractor-i', brickpre,
                                     'tractor-%s.fits' % brick))

        elif filetype == 'galaxy-sims':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'galaxy-sims-%s.fits' % brick))

        elif filetype in ['ccds-table', 'depth-table']:
            ty = filetype.split('-')[0]
            return swap(
                os.path.join(codir, '%s-%s-%s.fits' % (sname, brick, ty)))

        elif filetype in ['image-jpeg', 'model-jpeg', 'resid-jpeg',
                          'blobmodel-jpeg',
                          'imageblob-jpeg', 'simscoadd-jpeg','imagecoadd-jpeg',
                          'wise-jpeg', 'wisemodel-jpeg']:
            ty = filetype.split('-')[0]
            return swap(
                os.path.join(codir, '%s-%s-%s.jpg' % (sname, brick, ty)))

        elif filetype in ['outliers-pre', 'outliers-post',
                          'outliers-masked-pos', 'outliers-masked-neg']:
            return swap(
                os.path.join(basedir, 'metrics', brickpre,
                             '%s-%s.jpg' % (filetype, brick)))

        elif filetype in ['invvar', 'chi2', 'image', 'model', 'blobmodel',
                          'depth', 'galdepth', 'nexp', 'psfsize',
                          'copsf']:
            return swap(os.path.join(codir, '%s-%s-%s-%s.fits.fz' %
                                     (sname, brick, filetype, band)))

        elif filetype in ['blobmap']:
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'blobs-%s.fits.gz' % (brick)))

        elif filetype in ['maskbits']:
            return swap(os.path.join(codir,
                                     '%s-%s-%s.fits.fz' % (sname, brick, filetype)))

        elif filetype in ['all-models']:
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'all-models-%s.fits' % (brick)))

        elif filetype == 'ref-sources':
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'reference-%s.fits' % (brick)))

        elif filetype == 'checksums':
            return swap(os.path.join(basedir, 'tractor', brickpre,
                                     'brick-%s.sha256sum' % brick))
        elif filetype == 'outliers_mask':
            return swap(os.path.join(basedir, 'metrics', brickpre,
                                     'outlier-mask-%s.fits.fz' % (brick)))

        print('Unknown filetype "%s"' % filetype)
        assert(False)

    def check_cache(self, fn):
        if self.cache_dir is None:
            return fn
        cfn = fn.replace(self.survey_dir, self.cache_dir)
        print('checking for cache fn', cfn)
        if os.path.exists(cfn):
            debug('Cached file hit:', fn, '->', cfn)
            return cfn
        debug('Cached file miss:', fn, '-/->', cfn)
        return fn

    def get_compression_string(self, filetype, shape=None, **kwargs):
        pat = dict(# g: sigma ~ 0.002.  qz -1e-3: 6 MB, -1e-4: 10 MB
            image = '[compress R %(tilew)i,%(tileh)i; qz -1e-4]',
            # g: qz -1e-3: 2 MB, -1e-4: 2.75 MB
            model = '[compress R %(tilew)i,%(tileh)i; qz -1e-4]',
            chi2  = '[compress R %(tilew)i,%(tileh)i; qz -0.1]',
            # qz +8: 9 MB, qz +16: 10.5 MB
            invvar = '[compress R %(tilew)i,%(tileh)i; qz 16]',
            nexp   = '[compress H %(tilew)i,%(tileh)i]',
            maskbits = '[compress H %(tilew)i,%(tileh)i]',
            depth  = '[compress G %(tilew)i,%(tileh)i; qz 0]',
            galdepth = '[compress G %(tilew)i,%(tileh)i; qz 0]',
            psfsize = '[compress G %(tilew)i,%(tileh)i; qz 0]',
            outliers_mask = '[compress G]',
        ).get(filetype)
        #outliers_mask = '[compress H %i,%i]',
        if pat is None:
            return pat
        # Tile compression size
        tilew,tileh = 100,100
        if shape is not None:
            H,W = shape
            # CFITSIO's fpack compression can't handle partial tile
            # sizes < 4 pix.  Select a tile size that works, or don't
            # compress if we can't find one.
            if W < 4 or H < 4:
                return None
            while tilew <= W:
                remain = W % tilew
                if remain == 0 or remain >= 4:
                    break
                tilew += 1
            while tileh <= H:
                remain = H % tileh
                if remain == 0 or remain >= 4:
                    break
                tileh += 1
        return pat % dict(tilew=tilew,tileh=tileh)

    def write_output(self, filetype, hashsum=True, **kwargs):
        '''
        Returns a context manager for writing an output file.

        Example use: ::

        with survey.write_output('ccds', brick=brickname) as out:
            ccds.writeto(out.fn, primheader=primhdr)

        For FITS output, out.fits is a fitsio.FITS object.  The file
        contents will actually be written in memory, and then a
        sha256sum computed before the file contents are written out to
        the real disk file.  The 'out.fn' member variable is NOT set.

        ::
        
        with survey.write_output('ccds', brick=brickname) as out:
            ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

        Does the following on entry:
        - calls self.find_file() to determine which filename to write to
        - ensures the output directory exists
        - appends a ".tmp" to the filename

        Does the following on exit:
        - moves the ".tmp" to the final filename (to make it atomic)
        - computes the sha256sum
        '''
        class OutputFileContext(object):
            def __init__(self, fn, survey, hashsum=True, relative_fn=None,
                         compression=None):
                '''
                *compression*: a CFITSIO compression specification, eg:
                    "[compress R 100,100; qz -0.05]"
                '''
                self.real_fn = fn
                self.relative_fn = relative_fn
                self.survey = survey
                self.is_fits = (fn.endswith('.fits') or
                                fn.endswith('.fits.gz') or
                                fn.endswith('.fits.fz'))
                self.tmpfn = os.path.join(os.path.dirname(fn),
                                          'tmp-'+os.path.basename(fn))
                if self.is_fits:
                    self.fits = fitsio.FITS('mem://' + (compression or ''),
                                            'rw')
                else:
                    self.fn = self.tmpfn
                self.hashsum = hashsum

            def __enter__(self):
                dirnm = os.path.dirname(self.tmpfn)
                trymakedirs(dirnm)
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                # If an exception was thrown, bail out
                if exc_type is not None:
                    return

                if self.hashsum:
                    import hashlib
                    hashfunc = hashlib.sha256
                    sha = hashfunc()
                if self.is_fits:
                    # Read back the data written into memory by the
                    # fitsio library
                    rawdata = self.fits.read_raw()
                    # close the fitsio file
                    self.fits.close()

                    # If gzip, we now have to actually do the
                    # compression to gzip format...
                    if self.tmpfn.endswith('.gz'):
                        from io import BytesIO
                        import gzip
                        #ulength = len(rawdata)
                        # We gzip to a memory file (BytesIO) so we can compute
                        # the hashcode before writing to disk for real
                        gzipped = BytesIO()
                        gzf = gzip.GzipFile(self.real_fn, 'wb', 9, gzipped)
                        gzf.write(rawdata)
                        gzf.close()
                        rawdata = gzipped.getvalue()
                        gzipped.close()
                        del gzipped
                        #clength = len(rawdata)
                        #print('Gzipped', ulength, 'to', clength)

                    if self.hashsum:
                        sha.update(rawdata)

                    f = open(self.tmpfn, 'wb')
                    f.write(rawdata)
                    f.close()
                    debug('Wrote', self.tmpfn)
                    del rawdata
                else:
                    f = open(self.tmpfn, 'rb')
                    if self.hashsum:
                        sha.update(f.read())
                    f.close()
                if self.hashsum:
                    hashcode = sha.hexdigest()
                    del sha

                os.rename(self.tmpfn, self.real_fn)
                debug('Renamed to', self.real_fn)
                info('Wrote', self.real_fn)

                if self.hashsum:
                    # List the relative filename (from output dir) in
                    # shasum file.
                    fn = self.relative_fn or self.real_fn
                    self.survey.add_hashcode(fn, hashcode)
            # end of OutputFileContext class


        # Get the output filename for this filetype
        fn = self.find_file(filetype, output=True, **kwargs)

        compress = self.get_compression_string(filetype, **kwargs)

        # Find the relative path (relative to output_dir), which is the string
        # we will put in the shasum file.
        relfn = fn
        if relfn.startswith(self.output_dir):
            relfn = relfn[len(self.output_dir):]
            if relfn.startswith('/'):
                relfn = relfn[1:]

        out = OutputFileContext(fn, self, hashsum=hashsum, relative_fn=relfn,
                                compression=compress)
        return out

    def add_hashcode(self, fn, hashcode):
        '''
        Callback to be called in the *write_output* routine.
        '''
        self.output_file_hashes[fn] = hashcode

    def __getstate__(self):
        '''
        For pickling; we omit cached tables.
        '''
        d = self.__dict__.copy()
        d['ccds'] = None
        d['bricks'] = None
        d['bricktree'] = None
        d['ccd_kdtrees'] = None
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
        from pkg_resources import resource_filename
        return resource_filename('legacypipe', 'config')

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
                                 self.bricksize) < 1e-3))
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

    def get_bricks_near(self, ra, dec, radius):
        '''
        Returns a set of bricks near the given RA,Dec and radius (all in degrees).
        '''
        bricks = self.get_bricks_readonly()
        if self.cache_tree:
            from astrometry.libkd.spherematch import tree_build_radec, tree_search_radec
            # Use kdtree
            if self.bricktree is None:
                self.bricktree = tree_build_radec(bricks.ra, bricks.dec)
            I = tree_search_radec(self.bricktree, ra, dec, radius)
        else:
            from astrometry.util.starutil_numpy import degrees_between
            d = degrees_between(bricks.ra, bricks.dec, ra, dec)
            I, = np.nonzero(d < radius)
        if len(I) == 0:
            return None
        return bricks[I]

    def bricks_touching_radec_box(self, bricks,
                                  ralo, rahi, declo, dechi):
        '''
        Returns an index vector of the bricks that touch the given RA,Dec box.
        '''
        if bricks is None:
            bricks = self.get_bricks_readonly()
        if self.cache_tree and bricks == self.bricks:
            from astrometry.libkd.spherematch import tree_build_radec, tree_search_radec
            from astrometry.util.starutil_numpy import degrees_between
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
            debug('In Dec slice:', len(np.flatnonzero((bricks.dec1 <= dechi) *
                                                      (bricks.dec2 >= declo))))
            debug('Above RAlo=', ralo, ':', len(np.flatnonzero(bricks.ra2 >= ralo)))
            debug('Below RAhi=', rahi, ':', len(np.flatnonzero(bricks.ra1 <= rahi)))
            debug('In RA slice:', len(np.nonzero(np.logical_or(bricks.ra2 >= ralo,
                                                               bricks.ra1 <= rahi))))

            I, = np.nonzero(np.logical_or(bricks.ra2 >= ralo, bricks.ra1 <= rahi) *
                            (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
            debug('In RA&Dec slice', len(I))
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

    def filter_ccds_files(self, fns):
        '''
        When reading the list of CCDs, we find all files named
        survey-ccds-\*.fits.gz, then filter that list using this function.
        '''
        return fns

    def filter_ccd_kd_files(self, fns):
        return fns

    def get_ccds(self, **kwargs):
        '''
        Returns the table of CCDs.
        '''
        fns = self.find_file('ccd-kds')
        fns = self.filter_ccd_kd_files(fns)
        # If 'ccd-kds' files exist, read the CCDs tables from them!
        # Otherwise, fall back to survey-ccds-*.fits.gz files.
        if len(fns) == 0:
            fns = self.find_file('ccds')
            fns.sort()
            fns = self.filter_ccds_files(fns)
            if len(fns) == 0:
                print('Failed to find any valid survey-ccds tables')
                raise RuntimeError('No survey-ccds files')

        TT = []
        for fn in fns:
            debug('Reading CCDs from', fn)
            T = fits_table(fn, **kwargs)
            debug('Got', len(T), 'CCDs')
            TT.append(T)
        if len(TT) > 1:
            T = merge_tables(TT, columns='fillzero')
        else:
            T = TT[0]
        debug('Total of', len(T), 'CCDs')
        del TT
        T = self.cleanup_ccds_table(T)
        return T

    def get_annotated_ccds(self):
        '''
        Returns the annotated table of CCDs.
        '''
        fns = self.find_file('annotated-ccds')
        TT = []
        for fn in fns:
            debug('Reading annotated CCDs from', fn)
            T = fits_table(fn)
            debug('Got', len(T), 'CCDs')
            TT.append(T)
        T = merge_tables(TT, columns='fillzero')
        debug('Total of', len(T), 'CCDs')
        del TT
        T = self.cleanup_ccds_table(T)
        return T

    def cleanup_ccds_table(self, ccds):
        # Remove trailing spaces from 'ccdname' column
        if 'ccdname' in ccds.columns():
            # "N4 " -> "N4"
            ccds.ccdname = np.array([s.strip() for s in ccds.ccdname])
        # Remove trailing spaces from 'camera' column.
        if 'camera' in ccds.columns():
            ccds.camera = np.array([c.strip() for c in ccds.camera])
        # And 'filter' column
        if 'filter' in ccds.columns():
            ccds.filter = np.array([f.strip() for f in ccds.filter])
        return ccds

    def ccds_touching_wcs(self, wcs, **kwargs):
        '''
        Returns a table of the CCDs touching the given *wcs* region.
        '''
        kdfns = self.get_ccd_kdtrees()

        if len(kdfns):
            from astrometry.libkd.spherematch import tree_search_radec
            # MAGIC number: we'll search a 1-degree radius for CCDs
            # roughly in range, then refine using the
            # ccds_touching_wcs() function.
            radius = 1.
            ra,dec = wcs.radec_center()
            TT = []
            for fn,kd in kdfns:
                I = tree_search_radec(kd, ra, dec, radius)
                debug(len(I), 'CCDs within', radius, 'deg of RA,Dec',
                           '(%.3f, %.3f)' % (ra,dec))
                if len(I) == 0:
                    continue
                # Read only the CCD-table rows within range.
                TT.append(fits_table(fn, rows=I))
            if len(TT) == 0:
                return None
            ccds = merge_tables(TT, columns='fillzero')
            ccds = self.cleanup_ccds_table(ccds)
        else:
            ccds = self.get_ccds_readonly()
        I = ccds_touching_wcs(wcs, ccds, **kwargs)
        if len(I) == 0:
            return None
        return ccds[I]

    def get_ccd_kdtrees(self):
        # check cache...
        if self.ccd_kdtrees is not None:
            return self.ccd_kdtrees

        fns = self.find_file('ccd-kds')
        fns = self.filter_ccd_kd_files(fns)

        from astrometry.libkd.spherematch import tree_open
        self.ccd_kdtrees = []
        for fn in fns:
            debug('Opening kd-tree', fn)
            kd = tree_open(fn, 'ccds')
            self.ccd_kdtrees.append((fn, kd))
        return self.ccd_kdtrees

    def get_image_object(self, t, **kwargs):
        '''
        Returns a DecamImage or similar object for one row of the CCDs table.
        '''
        # get Image subclass
        imageType = self.image_class_for_camera(t.camera)
        # call Image subclass constructor
        return imageType(self, t, **kwargs)

    def get_approx_wcs(self, ccd):
        from astrometry.util.util import Tan
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
            C.cut(np.array([b in bands for b in C.filter]))
        ims = []
        for t in C:
            debug('Image file', t.image_filename, 'hdu', t.image_hdu)
            im = self.get_image_object(t)
            ims.append(im)
        # Read images, clip to ROI
        W,H = targetwcs.get_width(), targetwcs.get_height()
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        args = [(im, targetrd, kwargs) for im in ims]
        tims = mp.map(read_one_tim, args)
        return tims

    def find_ccds(self, expnum=None, ccdname=None, camera=None):
        '''
        Returns a table of CCDs matching the given *expnum* (exposure
        number, integer), *ccdname* (string), and *camera* (string),
        if given.
        '''
        if expnum is not None:
            C = self.try_expnum_kdtree(expnum)
            if C is not None:
                if len(C) == 0:
                    return None
                if ccdname is not None:
                    C = C[C.ccdname == ccdname]
                if camera is not None:
                    C = C[C.camera == camera]
                return C

        if expnum is not None and ccdname is not None:
            # use ccds_index
            if self.ccds_index is None:
                if self.ccds is not None:
                    C = self.ccds
                else:
                    C = self.get_ccds(columns=['expnum','ccdname'])
                self.ccds_index = dict([((e,n),i) for i,(e,n) in
                                        enumerate(zip(C.expnum, C.ccdname))])
            row = self.ccds_index[(expnum, ccdname)]
            if self.ccds is not None:
                return self.ccds[row]
            #import numpy as np
            #C = self.get_ccds(rows=np.array([row]))
            #return C[0]
        T = self.get_ccds_readonly()
        if expnum is not None:
            T = T[T.expnum == expnum]
        if ccdname is not None:
            T = T[T.ccdname == ccdname]
        if camera is not None:
            T = T[T.camera == camera]
        return T

    def try_expnum_kdtree(self, expnum):
        '''
        # By creating a kd-tree from the 'expnum' column, search for expnums
        # can be sped up:
        from astrometry.libkd.spherematch import *
        from astrometry.util.fits import fits_table
        T=fits_table('/global/cscratch1/sd/dstn/dr7-depthcut/survey-ccds-dr7.kd.fits',
            columns=['expnum'])
        ekd = tree_build(np.atleast_2d(T.expnum.copy()).T.astype(float),
                         nleaf=60, bbox=False, split=True)
        ekd.set_name('expnum')
        ekd.write('ekd.fits')

        > fitsgetext -i $CSCRATCH/dr7-depthcut/survey-ccds-dr7.kd.fits -o dr7-%02i -a -M
        > fitsgetext -i ekd.fits -o ekd-%02i -a -M
        > cat dr7-0* ekd-0[123456] > $CSCRATCH/dr7-depthcut+/survey-ccds-dr7.kd.fits
        '''
        fns = self.find_file('ccd-kds')
        fns = self.filter_ccd_kd_files(fns)
        if len(fns) == 0:
            return None
        from astrometry.libkd.spherematch import tree_open
        TT = []
        for fn in fns:
            debug('Searching', fn)
            try:
                kd = tree_open(fn, 'expnum')
            except:
                debug('Failed to open', fn, ':')
                import traceback
                traceback.print_exc()
                continue
            if kd is None:
                return None
            I = kd.search(np.array([expnum]), 0.5, 0, 0)
            debug(len(I), 'CCDs with expnum', expnum, 'in', fn)
            if len(I) == 0:
                continue
            # Read only the CCD-table rows within range.
            TT.append(fits_table(fn, rows=I))
        if len(TT) == 0:
            ##??
            return fits_table()
        ccds = merge_tables(TT, columns='fillzero')
        ccds = self.cleanup_ccds_table(ccds)
        return ccds


def run_calibs(X):
    im = X[0]
    kwargs = X[1]
    noraise = kwargs.pop('noraise', False)
    debug('run_calibs for image', im, ':', kwargs)
    try:
        return im.run_calibs(**kwargs)
    except:
        print('Exception in run_calibs:', im, kwargs)
        import traceback
        traceback.print_exc()
        if not noraise:
            raise

def read_one_tim(X):
    (im, targetrd, kwargs) = X
    #print('Reading', im)
    tim = im.get_tractor_image(radecpoly=targetrd, **kwargs)
    return tim

from tractor.psfex import PsfExModel

class SchlegelPsfModel(PsfExModel):
    def __init__(self, fn=None, ext=1):
        '''
        `ext` is ignored.
        '''
        if fn is not None:
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
            ne = (degree + 1) * (degree + 2) // 2
            print('Number of eigen-PSFs required for degree=', degree, 'is', ne)

            self.psfbases = np.zeros((ne, h,w))

            for d in range(degree + 1):
                # x polynomial degree = j
                # y polynomial degree = k
                for j in range(d+1):
                    k = d - j
                    ii = j + (degree+1) * k - (k * (k-1))// 2

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

def get_inblob_map(blobwcs, refs):
    bh,bw = blobwcs.shape
    bh = int(bh)
    bw = int(bw)
    blobmap = np.zeros((bh,bw), np.uint8)
    # circular/elliptical regions:
    for col,bit,ellipse in [('isbright', 'BRIGHT', False),
                            ('ismedium', 'MEDIUM', False),
                            ('iscluster', 'CLUSTER', True),
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
                debug('Object: PA', ref.pa, 'BA', ref.ba, 'Radius', ref.radius, 'pix', ref.radius_pix)
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
