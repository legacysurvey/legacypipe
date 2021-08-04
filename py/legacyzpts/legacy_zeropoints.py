from __future__ import division, print_function

# Turn off plotting imports for production
if False:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('Agg')

import os
import argparse
import sys

import numpy as np
from scipy.stats import sigmaclip

import fitsio
from astropy.table import Table, vstack

from astrometry.util.file import trymakedirs
from astrometry.util.ttime import Time
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec

import legacypipe
from legacypipe.ps1cat import ps1cat, sdsscat
from legacypipe.gaiacat import GaiaCatalog
from legacypipe.survey import radec_at_mjd, get_git_version
from legacypipe.image import validate_version
from legacypipe.survey import LegacySurveyData

import logging
logger = logging.getLogger('legacyzpts.legacy_zeropoints')
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

CAMERAS=['decam','mosaic','90prime','megaprime', 'hsc']

MAGLIM=dict(
    u=[16, 20],
    g=[16, 20],
    r=[16, 19.5],
    i=[16, 19.5],
    z=[16.5, 19],
    Y=[16.5, 19],
    N501=[16,20],
    N673=[16,19.5],
    )

def ptime(text,t0):
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return list(np.char.strip(lines))

def astropy_to_astrometry_table(t):
    T = fits_table()
    for c in t.colnames:
        T.set(c, t[c])
    return T

def _ccds_table(camera='decam', overrides=None):
    '''Initialize the CCDs table.

    Description and Units at:
    https://github.com/legacysurvey/legacyzpts/blob/master/DESCRIPTION_OF_OUTPUTS.md
    '''
    max_camera_length = max([len(c) for c in CAMERAS])
    if max_camera_length > 9:
        print('Warning! Increase camera length header card to S{}'.format(max_camera_length))

    cols = [
        ('err_message', 'S30'),
        ('image_filename', 'S120'),
        ('image_hdu', 'i2'),
        ('camera', 'S9'),
        ('expnum', 'i8'),
        ('plver', 'S8'),
        ('procdate', 'S19'),
        ('plprocid', 'S7'),
        ('ccdname', 'S4'),
        ('ccdnum', 'i2'),
        ('expid', 'S17'),
        ('object', 'S35'),
        ('propid', 'S10'),
        ('filter', 'S4'),
        ('exptime', 'f4'),
        ('mjd_obs', 'f8'),
        ('airmass', 'f4'),
        ('fwhm', 'f4'),
        ('width', 'i2'),
        ('height', 'i2'),
        ('ra_bore', 'f8'),
        ('dec_bore', 'f8'),
        ('crpix1', 'f4'),
        ('crpix2', 'f4'),
        ('crval1', 'f8'),
        ('crval2', 'f8'),
        ('cd1_1', 'f4'),
        ('cd1_2', 'f4'),
        ('cd2_1', 'f4'),
        ('cd2_2', 'f4'),
        ('pixscale', 'f4'),
        ('zptavg', 'f4'),
        ('yshift', 'bool'),
        # -- CCD-level quantities --
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('skysb', 'f4'),
        ('skycounts', 'f4'),
        ('skyrms', 'f4'),
        ('sig1', 'f4'),
        ('nstars_photom', 'i2'),
        ('nstars_astrom', 'i2'),
        ('phoff', 'f4'),
        ('phrms', 'f4'),
        ('phrmsavg', 'f4'),
        ('zpt', 'f4'),
        ('raoff',  'f4'),
        ('decoff', 'f4'),
        ('rarms',  'f4'),
        ('decrms', 'f4'),
        ('rastddev',  'f4'),
        ('decstddev', 'f4'),
        ]

    if overrides is not None:
        ovr = []
        for k,v in cols:
            ovr.append((k, overrides.get(k, v)))
        cols = ovr

    ccds = Table(np.zeros(1, dtype=cols))
    return ccds

def _stars_table(nstars=1):
    '''Initialize the stars table.

    Description and Units at:
    https://github.com/legacysurvey/legacyzpts/blob/master/DESCRIPTION_OF_OUTPUTS.md
    '''
    cols = [('image_filename', 'S100'),('image_hdu', 'i2'),
            ('expid', 'S16'), ('filter', 'S1'),('nstars', 'i2'),
            ('x', 'f4'), ('y', 'f4'), ('expnum', 'i8'),
            ('plver', 'S8'), ('procdate', 'S19'), ('plprocid', 'S7'),
            ('gain', 'f4'),
            ('ra', 'f8'), ('dec', 'f8'),
            ('apmag', 'f4'),('apflux', 'f4'),('apskyflux', 'f4'),('apskyflux_perpix', 'f4'),
            ('radiff', 'f8'), ('decdiff', 'f8'),
            ('ps1_mag', 'f4'),
            ('gaia_g','f8'),('ps1_g','f8'),('ps1_r','f8'),('ps1_i','f8'),('ps1_z','f8'),
            ('exptime', 'f4')]
    stars = Table(np.zeros(nstars, dtype=cols))
    return stars

def cols_for_survey_table():
    """Return list of -survey.fits table colums
    """
    return ['airmass', 'ccdskysb', 'plver', 'procdate', 'plprocid',
     'ccdnastrom', 'ccdnphotom', 'ra', 'dec', 'ra_bore', 'dec_bore',
     'image_filename', 'image_hdu', 'expnum', 'ccdname', 'object',
     'filter', 'exptime', 'camera', 'width', 'height', 'propid',
     'mjd_obs', 'fwhm', 'zpt', 'ccdzpt', 'ccdraoff', 'ccddecoff',
     'ccdrarms', 'ccddecrms', 'ccdskycounts', 'phrms', 'ccdphrms',
     'cd1_1', 'cd2_2', 'cd1_2', 'cd2_1', 'crval1', 'crval2', 'crpix1',
     'crpix2', 'skyrms', 'sig1', 'yshift']

def prep_survey_table(T, camera=None, bad_expid=None):
    assert(camera in CAMERAS)
    need_keys = cols_for_survey_table()
    # Rename
    rename_keys= [('zpt','ccdzpt'),
                  ('zptavg','zpt'),
                  ('raoff','ccdraoff'),
                  ('decoff','ccddecoff'),
                  ('skycounts', 'ccdskycounts'),
                  ('skysb', 'ccdskysb'),
                  ('rarms',  'ccdrarms'),
                  ('decrms', 'ccddecrms'),
                  ('phrms', 'ccdphrms'),
                  ('phrmsavg', 'phrms'),
                  ('nstars_astrom','ccdnastrom'),
                  ('nstars_photom','ccdnphotom')]
    for old,new in rename_keys:
        T.rename(old,new)
    # Delete
    del_keys= list( set(T.get_columns()).difference(set(need_keys)) )
    for key in del_keys:
        T.delete_column(key)
    # precision
    T.width  = T.width.astype(np.int16)
    T.height = T.height.astype(np.int16)
    T.cd1_1 = T.cd1_1.astype(np.float32)
    T.cd1_2 = T.cd1_2.astype(np.float32)
    T.cd2_1 = T.cd2_1.astype(np.float32)
    T.cd2_2 = T.cd2_2.astype(np.float32)

    # Set placeholder that masks everything until update_ccd_cuts is
    # run.
    from legacyzpts import psfzpt_cuts
    T.ccd_cuts = np.zeros(len(T), np.int16) + psfzpt_cuts.CCD_CUT_BITS['err_legacyzpts']
    return T

def create_annotated_table(T, ann_fn, camera, survey, mp, header=None):
    from legacyzpts.annotate_ccds import annotate, init_annotations
    T = survey.cleanup_ccds_table(T)
    init_annotations(T)
    I, = np.nonzero(T.ccdzpt)
    if len(I):
        annotate(T, survey, camera, mp=mp, normalizePsf=True, carryOn=True)
    writeto_via_temp(ann_fn, T, header=header)
    print('Wrote %s' % ann_fn)

def getrms(x):
    return np.sqrt(np.mean(x**2))

def measure_image(img_fn, mp, image_dir='images',
                  run_calibs_only=False,
                  run_psf_only=False,
                  just_imobj=False,
                  survey=None, psfex=True, camera=None, **measureargs):
    '''Wrapper on the camera-specific classes to measure the CCD-level data on all
    the FITS extensions for a given set of images.
    '''
    t0 = Time()
    quiet = measureargs.get('quiet', False)
    image_hdu = measureargs.get('image_hdu', None)

    img = survey.get_image_object(None, camera=camera,
                                  image_fn=img_fn, image_hdu=image_hdu)
    if just_imobj:
        return img

    print('Got image object', img)
    # Confirm camera field.
    assert(img.camera == camera)

    primhdr = img.read_image_primary_header()
    if (not img.calibration_good(primhdr)) or (img.exptime == 0):
        # FIXME
        # - all-zero weight map
        if run_calibs_only:
            return
        debug('%s: Zero exposure time or low-level calibration flagged as bad; skipping image.'
              % str(img))
        ccds = _ccds_table(camera)
        ccds['image_filename'] = img_fn
        ccds['err_message'] = 'Failed CP calib, or Exptime=0'
        ccds['zpt'] = 0.
        return ccds, None, img

    if measureargs['choose_ccd']:
        extlist = [measureargs['choose_ccd']]
    else:
        extlist = img.get_extension_list(debug=measureargs['debug'])

    print('Extensions to process:', extlist)

    all_ccds = []
    all_photom = []
    splinesky = measureargs['splinesky']

    survey_blob_mask = None
    blobdir = measureargs.pop('blob_mask_dir', None)
    if blobdir is not None:
        survey_blob_mask = LegacySurveyData(survey_dir=blobdir)

    survey_zeropoints = None
    zptdir = measureargs.pop('zeropoints_dir', None)
    if zptdir is not None:
        survey_zeropoints = LegacySurveyData(survey_dir=zptdir)

    plots = measureargs.get('plots', False)

    if run_psf_only:
        splinesky = False

    # Validate the splinesky and psfex merged files, and (re)make them if
    # they're missing.
    if splinesky:
        fn = img.get_splinesky_merged_filename()
        if (fn is None or
            validate_version(fn, 'table', img.expnum, img.plver, img.plprocid, quiet=quiet)):
            splinesky = False
    if psfex:
        fn = img.get_psfex_merged_filename()
        if (fn is None or
            validate_version(fn, 'table', img.expnum, img.plver, img.plprocid, quiet=quiet)):
            psfex = False

    if splinesky or psfex:
        imgs = mp.map(run_one_calib, [(img_fn, camera, survey, ext, psfex, splinesky,
                                       plots, survey_blob_mask, survey_zeropoints)
                                      for ext in extlist])
        from legacyzpts.merge_calibs import merge_splinesky, merge_psfex
        class FakeOpts(object):
            pass
        opts = FakeOpts()
        # Allow some CCDs to be missing, e.g., if the weight map is all zero.
        opts.all_found = False
        if splinesky:
            skyoutfn = img.get_splinesky_merged_filename()
            ccds = None
            err_splinesky = merge_splinesky(survey, img.expnum, ccds, skyoutfn, opts, imgs=imgs)
            if err_splinesky != 1:
                print('Problem writing {}'.format(skyoutfn))
        if psfex:
            psfoutfn = img.get_psfex_merged_filename()
            ccds = None
            err_psfex = merge_psfex(survey, img.expnum, ccds, psfoutfn, opts, imgs=imgs)
            if err_psfex != 1:
                print('Problem writing {}'.format(psfoutfn))

    # Now, if they're still missing it's because the entire exposure is borked
    # (WCS failed, weight maps are all zero, etc.), so exit gracefully.
    if splinesky:
        fn = img.get_splinesky_merged_filename()
        if not os.path.exists(fn):
            print('Merged splinesky file not found {}'.format(fn))
            return []
        if not validate_version(img.get_splinesky_merged_filename(),
                                'table', img.expnum, img.plver, img.plprocid):
            raise RuntimeError('Merged splinesky file did not validate!')
        # At this point the merged file exists and has been validated, so remove
        # the individual splinesky files.
        for img in imgs:
            fn = img.get_splinesky_unmerged_filename()
            if os.path.isfile(fn):
                os.remove(fn)
    if psfex:
        fn = img.get_psfex_merged_filename()
        if not os.path.exists(fn):
            print('Merged psfex file not found {}'.format(fn))
            return []
        if not validate_version(img.get_psfex_merged_filename(),
                                'table', img.expnum, img.plver, img.plprocid):
            raise RuntimeError('Merged psfex file did not validate!')
        # At this point the merged file exists and has been validated, so remove
        # the individual PSFEx and SE files.
        for img in imgs:
            fn = img.get_psfex_unmerged_filename()
            if os.path.isfile(fn):
                os.remove(fn)
            sefn = img.sefn
            if os.path.isfile(sefn):
                os.remove(sefn)

    # FIXME -- remove temporary individual files directory

    if run_calibs_only or run_psf_only:
        return

    rtns = mp.map(run_one_ext, [(img, ext, survey, splinesky,
                                 measureargs['sdss_photom'])
                                for ext in extlist])

    for ccd,photom in rtns:
        if ccd is not None:
            all_ccds.append(ccd)
        if photom is not None:
            all_photom.append(photom)

    # Compute the median zeropoint across all the CCDs.
    all_ccds = vstack(all_ccds)

    if len(all_photom):
        all_photom = merge_tables(all_photom, columns='fillzero')
    else:
        all_photom = None

    zpts = all_ccds['zpt']
    zptgood = np.isfinite(zpts)
    if np.sum(zptgood) > 0:
        all_ccds['zptavg'] = np.median(zpts[zptgood])
    phrms = all_ccds['phrms']
    phrmsgood = np.isfinite(phrms) & (all_ccds['phrms'] > 0)
    if np.sum(phrmsgood) > 0:
        all_ccds['phrmsavg'] = np.median(phrms[phrmsgood])

    t0 = ptime('measure-image-%s' % img_fn,t0)
    return all_ccds, all_photom, img

def run_one_calib(X):
    (img_fn, camera, survey, ext, psfex, splinesky, _, survey_blob_mask,
     survey_zeropoints) = X
    img = survey.get_image_object(None, camera=camera,
                                  image_fn=img_fn, image_hdu=ext)

    do_psf = False
    do_sky = False
    if psfex:
        do_psf = True
        try:
            psf = img.read_psf_model(0., 0., pixPsf=True)
            if psf is not None:
                do_psf = False
        except:
            pass
    if splinesky:
        do_sky = True
        try:
            sky = img.read_sky_model()
            if sky is not None:
                do_sky = False
        except:
            pass

    if (not do_psf) and (not do_sky):
        # Nothing to do!
        return img

    # Only do stellar halo subtraction if we have a zeropoint (via --zeropoint-dir)
    # Note that this is the only place we use survey_zeropoints; it does not get
    # passed to image.run_calibs().
    have_zpt = False
    if survey_zeropoints is not None:
        ccds = survey_zeropoints.find_ccds(
            expnum=img.expnum, ccdname=img.ccdname, camera=img.camera)
        if len(ccds) != 1:
            print('WARNING, did not find a zeropoint for', img_fn,
                  'by camera', camera, 'expnum', img.expnum,
                  'ext', ext)
        else:
            img.ccdzpt = ccds[0].ccdzpt
            img.ccdraoff = ccds[0].ccdraoff
            img.ccddecoff = ccds[0].ccddecoff
            if img.ccdzpt == 0.:
                print('WARNING, found zeropoint=0 for', img_fn,
                      'by camera', camera, 'expnum', img.expnum,
                      'ext', ext)
            else:
                have_zpt = True

    git_version = get_git_version(dirnm=os.path.dirname(legacypipe.__file__))
    ps = None
    img.run_calibs(psfex=do_psf, sky=do_sky, splinesky=True,
                   git_version=git_version, survey=survey, ps=ps,
                   survey_blob_mask=survey_blob_mask,
                   halos=have_zpt,
                   subtract_largegalaxies=have_zpt)
    return img

def run_one_ext(X):
    img, ext, survey, splinesky, sdss_photom = X

    img = survey.get_image_object(None, camera=img.camera,
                                  image_fn=img.image_filename, image_hdu=ext)
    return run_zeropoints(img, splinesky=splinesky, sdss_photom=sdss_photom)

class outputFns(object):
    def __init__(self, imgfn, outdir, camera, image_dir='images', debug=False):
        """
        Assigns filename, makes needed dirs.

        Args:
            imgfn: abs path to image, should be a ooi or oki file
            outdir: root dir for outptus
            debug: 4 ccds only if true

        Attributes:
            imgfn: image that will be read
            zptfn: zeropoints file
            starfn: stars file

        Example:
            outdir/decam/DECam_CP/CP20151226/img_fn.fits.fz
            outdir/decam/DECam_CP/CP20151226/img_fn-zpt%s.fits
            outdir/decam/DECam_CP/CP20151226/img_fn-star%s.fits
        """
        self.imgfn = imgfn
        self.image_dir = image_dir

        """
        # Keep the last directory component
        dirname = os.path.basename(os.path.dirname(self.imgfn))
        basedir = os.path.join(outdir, camera, dirname)
        """
        # Keep same path structure as the images
        dirname = os.path.dirname(self.imgfn)
        basedir = os.path.join(outdir, dirname)
        trymakedirs(basedir)

        basename = os.path.basename(self.imgfn)
        # zpt,star fns
        base = basename
        if base.endswith('.fz'):
            base = base[:-len('.fz')]
        if base.endswith('.fits'):
            base = base[:-len('.fits')]
        if debug:
            base += '-debug'
        self.photomfn = os.path.join(basedir, base + '-photom.fits')
        self.annfn = os.path.join(basedir, base + '-annotated.fits')

def writeto_via_temp(outfn, obj, func_write=False, **kwargs):
    tempfn = os.path.join(os.path.dirname(outfn), 'tmp-' + os.path.basename(outfn))
    if func_write:
        obj.write(tempfn, **kwargs)
    else:
        obj.writeto(tempfn, **kwargs)
    os.rename(tempfn, outfn)

def runit(imgfn, photomfn, annfn, mp, bad_expid=None,
          survey=None, run_calibs_only=False, run_psf_only=False,
          version_header=None, **measureargs):
    '''Generate a legacypipe-compatible (survey) CCDs file for a given image.
    '''
    t0 = Time()

    results = measure_image(imgfn, mp, survey=survey,
                            run_calibs_only=run_calibs_only,
                            run_psf_only=run_psf_only,
                            **measureargs)
    if run_calibs_only or run_psf_only:
        return

    if len(results) == 0:
        print('All CCDs bad, quitting.')
        return

    ccds, photom, img = results
    t0 = ptime('measure_image',t0)

    primhdr = img.read_image_primary_header()

    hdr = fitsio.FITSHDR()
    if version_header is not None:
        for r in version_header.records():
            hdr.add_record(r)
    for key in ['AIRMASS', 'OBJECT', 'TELESCOP', 'INSTRUME', 'EXPTIME', 'DATE-OBS',
                'MJD-OBS', 'PROGRAM', 'OBSERVER', 'PROPID', 'FILTER', 'HA', 'ZD',
                'AZ', 'DOMEAZ', 'HUMIDITY', 'PLVER', ]:
        if not key in primhdr:
            continue
        v = primhdr[key]
        if isinstance(v, str):
            v = v.strip()
        hdr.add_record(dict(name=key, value=v,
                            comment=primhdr.get_comment(key)))

    hdr.add_record(dict(name='EXPNUM', value=img.expnum,
                        comment='Exposure number'))
    hdr.add_record(dict(name='PROCDATE', value=img.procdate,
                        comment='CP processing date'))
    hdr.add_record(dict(name='PLPROCID', value=img.plprocid,
                        comment='CP processing batch'))
    hdr.add_record(dict(name='RA_BORE',  value=ccds['ra_bore'][0],  comment='Boresight RA (deg)'))
    hdr.add_record(dict(name='DEC_BORE', value=ccds['dec_bore'][0], comment='Boresight Dec (deg)'))

    zptgood = np.isfinite(ccds['zpt'])
    if np.sum(zptgood) > 0:
        medzpt = np.median(ccds['zpt'][zptgood])
    else:
        medzpt = 0.0
    hdr.add_record(dict(name='CCD_ZPT', value=medzpt,
                        comment='Exposure median zeropoint'))

    goodfwhm = (ccds['fwhm'] > 0)
    if np.sum(goodfwhm) > 0:
        fwhm = np.median(ccds['fwhm'][goodfwhm])
    else:
        fwhm = 0.0
    pixscale = ccds['pixscale'][0]
    hdr.add_record(dict(name='FWHM', value=fwhm, comment='Exposure median FWHM (CP)'))
    hdr.add_record(dict(name='SEEING', value=fwhm * pixscale,
                        comment='Exposure median seeing (FWHM*pixscale)'))

    base = os.path.basename(imgfn)
    dirnm = os.path.dirname(imgfn)
    firstdir = os.path.basename(dirnm)
    hdr.add_record(dict(name='FILENAME', value=os.path.join(firstdir, base)))

    if photom is not None:
        writeto_via_temp(photomfn, photom, overwrite=True, header=hdr)

    accds = astropy_to_astrometry_table(ccds)

    # survey table
    T = prep_survey_table(accds, camera=measureargs['camera'], bad_expid=bad_expid)
    # survey --> annotated
    create_annotated_table(T, annfn, measureargs['camera'], survey, mp, header=hdr)

    t0 = ptime('write-results-to-fits',t0)

def get_parser():
    '''return parser object, tells it what options to look for
    options can come from a list of strings or command line'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                     description='Generate a legacypipe-compatible (survey) CCDs file \
                                                  from a set of reduced imaging.')

    parser.add_argument('images', metavar='image-filename', nargs='*',
                        help='Image filenames to process (prepend "@" for a text file containing a list of filenames)')

    parser.add_argument('--camera',choices=CAMERAS, action='store',required=True)
    parser.add_argument('--image',action='append',default=[],help='relative path to image starting from [survey_dir]/images/',required=False)
    parser.add_argument('--image_list',action='append',default=[],help='text file listing multiples images like --image',required=False)
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('--outdir', type=str, default=None, help='Where to write photom and annotated files; default [survey_dir]/zpt')
    parser.add_argument('--sdss-photom', default=False, action='store_true',
                        help='Use SDSS rather than PS-1 for photometric cal.')
    parser.add_argument('--debug', action='store_true', default=False, help='Write additional files and plots for debugging')
    parser.add_argument('--choose_ccd', action='store', default=None, help='forced to use only the specified ccd')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--verboseplots', action='store_true', default=False, help='use to plot FWHM Moffat PSF fits to the 20 brightest stars')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use this option when deriving the photometric transformation equations.')
    parser.add_argument('--plots', action='store_true', help='Calib plots?')
    parser.add_argument('--nproc', type=int,action='store',default=1,
                        help='set to > 1 if using legacy-zeropoints-mpiwrapper.py')
    parser.add_argument('--run-calibs-only', default=False, action='store_true',
                        help='Only ensure calib files exist, do not compute zeropoints.')
    parser.add_argument('--run-psf-only', default=False, action='store_true',
                        help='Only create / ensure PSF calib files exist, do not compute zeropoints.')
    parser.add_argument('--no-splinesky', dest='splinesky', default=True, action='store_false',
                        help='Do not use spline sky model for sky subtraction?')
    parser.add_argument('--blob-mask-dir', type=str, default=None,
                        help='The base directory to search for blob masks during sky model construction')
    parser.add_argument('--zeropoints-dir', type=str, default=None,
                        help='The base directory to search for survey-ccds files for subtracting star halos before doing sky calibration.')
    parser.add_argument('--calibdir', default=None,
                        help='if None will use LEGACY_SURVEY_DIR/calib, e.g. /global/cscratch1/sd/desiproc/dr5-new/calib')
    parser.add_argument('--no-check-photom', dest='check_photom', action='store_false',
                        help='Do not check for photom file when deciding if this file is done or not.')
    parser.add_argument('--threads', default=None, type=int,
                        help='Multiprocessing threads (parallel by HDU)')
    parser.add_argument('--quiet', default=False, action='store_true', help='quiet down')
    parser.add_argument('--overhead', type=str, default=None, help='Print python startup time since the given date.')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='More logging')
    return parser


def main(args=None):
    parser= get_parser()
    args = parser.parse_args(args=args)
    if args.overhead is not None:
        t0 = args.overhead
        if t0.endswith('.N'):
            t0 = t0[:-2]
        t0 = float(t0)
        import time
        print('Startup time:', time.time()-t0, 'seconds')

    image_list = []
    # add image specified with --image
    args.images.extend(args.image)
    # add image lists specified with --image_list
    args.images.extend(['@'+fn for fn in args.image_list])
    for fn in args.images:
        if fn.startswith('@') and os.path.exists(fn[1:]):
            ims = read_lines(fn)
            # drop empty lines
            ims = [fn for fn in ims if len(fn)]
            image_list.extend(ims)
            continue
        image_list.append(fn)

    ''' Produce zeropoints for all CP images in image_list
    image_list -- iterable list of image filenames
    args -- parsed argparser objection from get_parser()

    '''
    from pkg_resources import resource_filename

    assert(not args is None)
    assert(not image_list is None)
    t0 = tbegin = Time()

    # Build a dictionary with the optional inputs.
    measureargs = vars(args)
    measureargs.pop('image_list')
    measureargs.pop('image')
    nimage = len(image_list)

    quiet = measureargs.get('quiet', False)

    from astrometry.util.multiproc import multiproc
    threads = measureargs.pop('threads')
    mp = multiproc(nthreads=(threads or 1))

    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    camera = measureargs['camera']

    survey = LegacySurveyData(survey_dir=measureargs['survey_dir'])
    if measureargs.get('calibdir'):
        survey.calib_dir = measureargs['calibdir']
    measureargs.update(survey=survey)

    outdir = measureargs.pop('outdir')
    if outdir is None:
        outdir = os.path.join(survey.survey_dir, 'zpt')

    if camera in ['mosaic', 'decam', '90prime']:
        from legacyzpts.psfzpt_cuts import read_bad_expid
        fn = resource_filename('legacyzpts', 'data/{}-bad_expid.txt'.format(camera))
        if os.path.isfile(fn):
            print('Reading {}'.format(fn))
            measureargs.update(bad_expid=read_bad_expid(fn))
        else:
            print('No bad exposure file found for camera {}'.format(camera))

    from legacypipe.survey import get_version_header, get_git_version, get_dependency_versions
    release = 10000
    gitver = get_git_version()
    version_header = get_version_header('legacy_zeropoints.py', survey.survey_dir, release,
                                        git_version=gitver, proctype='InstCal')
    deps = get_dependency_versions(None, None, None, None)
    for name,value,comment in deps:
        version_header.add_record(dict(name=name, value=value, comment=comment))
    command_line=' '.join(sys.argv)
    version_header.add_record(dict(name='CMDLINE', value=command_line,
                                   comment='runbrick command-line'))
    measureargs['version_header'] = version_header
    check_photom = measureargs.pop('check_photom')

    for ii, imgfn in enumerate(image_list):
        print('Working on image {}/{}: {}'.format(ii+1, nimage, imgfn))

        # Check if the outputs are done and have the correct data model.
        F = outputFns(imgfn, outdir, camera, image_dir=survey.get_image_dir(),
                      debug=measureargs['debug'])

        measure = measure_image(F.imgfn, None, just_imobj=True, image_hdu=None,
                                **measureargs)
        psffn = measure.get_psfex_merged_filename()
        skyfn = measure.get_splinesky_merged_filename()

        ann_ok, psf_ok, sky_ok = [(fn is None) or validate_version(
            fn, 'table', measure.expnum, measure.plver,
            measure.plprocid, quiet=quiet)
            for fn in [F.annfn, psffn, skyfn]]

        if measureargs['run_calibs_only'] and psf_ok and sky_ok:
            print('Already finished {}'.format(psffn))
            print('Already finished {}'.format(skyfn))
            continue

        if measureargs['run_psf_only'] and psf_ok:
            print('Already finished {}'.format(psffn))
            continue

        phot_ok = validate_version(F.photomfn, 'header', measure.expnum,
                                   measure.plver, measure.plprocid,
                                   ext=1, quiet=quiet)

        if ann_ok and (phot_ok or not(check_photom)) and psf_ok and sky_ok:
            print('Already finished: {}'.format(F.annfn))
            continue

        # Create the file
        t0 = ptime('before-run',t0)
        runit(F.imgfn, F.photomfn, F.annfn, mp, **measureargs)
        t0 = ptime('after-run',t0)
    tnow = Time()
    print("TIMING:total %s" % (tnow-tbegin,))

def run_zeropoints(imobj, splinesky=False, sdss_photom=False):
    """Computes photometric and astrometric zeropoints for one CCD.

    Args:

    Returns:
        ccds, stars_photom, stars_astrom
    """
    from tractor.brightness import NanoMaggies
    t0= Time()
    t0= ptime('Measuring CCD=%s from image=%s' % (imobj.ccdname, imobj.image_filename),t0)

    # Initialize CCDs (annotated) table data structure.
    ccds = _ccds_table(imobj.camera, overrides=imobj.override_ccd_table_types())

    # init_ccd():
    namemap = { 'object': 'obj',
                'filter': 'band',
                'image_hdu': 'hdu',
                'mjd_obs': 'mjdobs',
    }
    for key in ['image_filename', 'image_hdu', 'camera', 'expnum', 'plver', 'procdate',
                'plprocid', 'ccdname', 'propid', 'exptime', 'mjd_obs',
                'pixscale', 'width', 'height', 'fwhm', 'filter']:
        val = getattr(imobj, namemap.get(key, key))
        print('Setting', key, '=', val)
        ccds[key] = val

    primhdr = imobj.read_image_primary_header()
    hdr = imobj.read_image_header(ext=imobj.hdu)

    ra_bore, dec_bore = imobj.get_radec_bore(primhdr)
    ccds['ra_bore'],ccds['dec_bore'] = ra_bore, dec_bore
    airmass = imobj.get_airmass(primhdr, hdr, ra_bore, dec_bore)
    ccds['airmass'] = airmass
    ccds['gain'] = imobj.get_gain(primhdr, hdr)
    ccds['object'] = primhdr.get('OBJECT')

    optional = ['avsky']
    for ccd_col in ['avsky', 'crpix1', 'crpix2', 'crval1', 'crval2',
                    'cd1_1','cd1_2', 'cd2_1', 'cd2_2']:
        if ccd_col.upper() in hdr:
            ccds[ccd_col] = hdr[ccd_col]
        elif ccd_col in optional:
            ccds[ccd_col] = np.nan
        else:
            raise KeyError('Could not find %s key in header:' % ccd_col)

    wcs = imobj.get_wcs(hdr=hdr)
    H = imobj.height
    W = imobj.width
    ccdra, ccddec = wcs.pixelxy2radec((W+1) / 2.0, (H+1) / 2.0)
    ccds['ra']  = ccdra
    ccds['dec'] = ccddec
    t0= ptime('header-info',t0)

    # Quick check for PsfEx file
    try:
        psf = imobj.read_psf_model(0., 0., pixPsf=True)
    except RuntimeError as e:
        print('Failed to read PSF model: %s' % e)
        return None, None

    dq,dqhdr = imobj.read_dq(header=True)
    invvar = imobj.read_invvar(dq=dq)
    img = imobj.read_image()
    imobj.fix_saturation(img, dq, invvar, primhdr, hdr, None)
    if dq is not None:
        dq = imobj.remap_dq(dq, dqhdr)
    # Compute sig1 before rescaling (later it gets scaled by zpscale)
    imobj.sig1 = imobj.estimate_sig1(img, invvar, dq, primhdr, hdr)
    ccds['sig1'] = imobj.sig1

    invvar = imobj.remap_invvar(invvar, primhdr, img, dq)

    invvar = imobj.scale_weight(invvar)
    img = imobj.scale_image(img)

    t0= ptime('read image',t0)

    # Measure the sky brightness and (sky) noise level.
    sky_img, skymed, skyrms = imobj.estimate_sky(img, invvar, dq, primhdr, hdr)
    zp0 = imobj.nominal_zeropoint(imobj.band)
    skybr = zp0 - 2.5*np.log10(skymed / imobj.pixscale / imobj.pixscale / imobj.exptime)
    print('Sky level: %.2f count/pix' % skymed)
    print('Sky brightness: %.3f mag/arcsec^2 (assuming nominal zeropoint)' % skybr)
    ccds['skyrms'] = skyrms / imobj.exptime
    ccds['skycounts'] = skymed / imobj.exptime
    ccds['skysb'] = skybr   # [mag/arcsec^2]
    t0= ptime('measure-sky',t0)

    # Does this image already have (photometry and astrometric) zeropoints computed?
    zpt = imobj.get_zeropoint(primhdr, hdr)
    if zpt is not None:
        print('Image', imobj, ': using zeropoint %.3f' % zpt)
        zpscale = NanoMaggies.zeropointToScale(zpt)
        ccds['sig1'] /= zpscale
        ccds['zpt'] = zpt
        return ccds, None

    # Load PS1 & Gaia catalogues

    phot = None
    if sdss_photom:
        try:
            phot = sdsscat(ccdwcs=wcs).get_stars(magrange=None)
        except OSError as e:
            print('No SDSS stars found for this image -- outside the SDSS footprint?', e)
    else:
        try:
            phot = ps1cat(ccdwcs=wcs).get_stars(magrange=None)
        except OSError as e:
            print('No PS1 stars found for this image -- outside the PS1 footprint, or in the Galactic plane?', e)

    if phot is not None and len(phot) == 0:
        phot = None

    if sdss_photom:
        name = 'sdss'
    else:
        name = 'ps1'

    if phot is not None:
        phot.cut(imobj.get_photometric_calibrator_cuts(name, phot))
        if len(phot) == 0:
            phot = None
        else:
            # Convert to Legacy Survey mags
            phot.legacy_survey_mag = imobj.photometric_calibrator_to_observed(name, phot)
            print(len(phot), 'photometric calibrator stars')

    gaia = GaiaCatalog().get_catalog_in_wcs(wcs)
    assert(gaia is not None)
    assert(len(gaia) > 0)
    gaia = GaiaCatalog.catalog_nantozero(gaia)
    assert(gaia is not None)
    print(len(gaia), 'Gaia stars')

    maxgaia = 10000
    if len(gaia) > maxgaia:
        I = np.argsort(gaia.phot_g_mean_mag)
        print('Min mag:', gaia.phot_g_mean_mag[I[0]])
        gaia.cut(I[:maxgaia])
        print('Cut to', len(gaia), 'Gaia stars')

    t0= Time()

    # Now put Gaia stars into the image and re-fit their centroids
    # and fluxes using the tractor with the PsfEx PSF model.

    # assume that the CP WCS has gotten us to within a few pixels
    # of the right answer.  Find Gaia stars, initialize Tractor
    # sources there, optimize them and see how much they want to
    # move.

    if splinesky:
        sky = imobj.read_sky_model()
        print('Instantiating and subtracting sky model')
        skymod = np.zeros_like(img)
        sky.addTo(skymod)
        # Apply the same transformation that was applied to the image...
        skymod = imobj.scale_image(skymod)
        fit_img = img - skymod
    else:
        fit_img = img - sky_img

    # after sky subtraction, apply optional per-amp relative zeropoints.
    imobj.apply_amp_correction(fit_img, invvar, 0, 0)

    with np.errstate(invalid='ignore'):
        # sqrt(0.) can trigger complaints;
        #   https://github.com/numpy/numpy/issues/11448
        ierr = np.sqrt(invvar)

    # Move Gaia stars to the epoch of this image.
    gaia.rename('ra',  'ra_gaia')
    gaia.rename('dec', 'dec_gaia')
    gaia.rename('source_id', 'gaia_sourceid')
    ra,dec = radec_at_mjd(gaia.ra_gaia, gaia.dec_gaia,
                          gaia.ref_epoch.astype(float),
                          gaia.pmra, gaia.pmdec, gaia.parallax, imobj.mjdobs)
    gaia.ra_now = ra
    gaia.dec_now = dec
    for b in ['g', 'bp', 'rp']:
        sn = gaia.get('phot_%s_mean_flux_over_error' % b)
        magerr = np.abs(2.5/np.log(10.) * 1./np.fmax(1., sn))
        gaia.set('phot_%s_mean_mag_error' % b, magerr)
    gaia.flux0 = np.ones(len(gaia), np.float32)
    # we set 'astrom' and omit 'photom'; it will get filled in with zeros.
    gaia.astrom = np.ones(len(gaia), bool)

    refs = [gaia]

    if phot is not None:
        # Photometry
        # Initial flux estimate, from nominal zeropoint
        phot.flux0 = (10.**((zp0 - phot.legacy_survey_mag) / 2.5) * imobj.exptime
                     ).astype(np.float32)

        if sdss_photom:
            phot.ra_sdss  = phot.ra.copy()
            phot.dec_sdss = phot.dec.copy()
            phot.ra_phot = phot.ra_sdss
            phot.dec_phot = phot.dec_sdss
            bands = 'ugriz'
            for band in bands:
                i = sdsscat.sdssband.get(band, None)
                if i is None:
                    print('No band', band, 'in SDSS catalog')
                    continue
                phot.set('sdss_'+band.lower(), phot.psfmag[:,i].astype(np.float32))
            phot_cols = [
                ('ra_sdss', np.double),
                ('dec_sdss', np.double),
                ('sdss_u', np.float32),
                ('sdss_g', np.float32),
                ('sdss_r', np.float32),
                ('sdss_i', np.float32),
                ('sdss_z', np.float32),
            ]
        else:
            # we don't have/use proper motions for PS1 stars
            phot.rename('ra_ok',  'ra_now')
            phot.rename('dec_ok', 'dec_now')
            phot.ra_ps1  = phot.ra_now.copy()
            phot.dec_ps1 = phot.dec_now.copy()
            phot.ra_phot = phot.ra_ps1
            phot.dec_phot = phot.dec_ps1
            phot.ps1_objid  = phot.obj_id
            bands = 'grizY'
            for band in bands:
                i = ps1cat.ps1band.get(band, None)
                if i is None:
                    print('No band', band, 'in PS1 catalog')
                    continue
                phot.set('ps1_'+band.lower(), phot.median[:,i].astype(np.float32))
            phot_cols = [
                ('ps1_objid', np.int64),
                ('ra_ps1', np.double),
                ('dec_ps1', np.double),
                ('ps1_g', np.float32),
                ('ps1_r', np.float32),
                ('ps1_i', np.float32),
                ('ps1_z', np.float32),
                ('ps1_y', np.float32),
            ]

        # we set 'photom' and omit 'astrom'; it will get filled in with zeros.
        phot.photom = np.ones (len(phot), bool)

        # Match photometric stars to Gaia stars within 1".
        I,J,_ = match_radec(gaia.ra_gaia, gaia.dec_gaia,
                            phot.ra_phot, phot.dec_phot, 1./3600.,
                            nearest=True)
        print(len(I), 'of', len(gaia), 'Gaia and', len(phot), 'photometric cal stars matched')

        # Merged = photocal stars + unmatched Gaia
        if len(I):
            # Merge columns for the matched stars
            for c in gaia.get_columns():
                G = gaia.get(c)
                # If column exists in both (eg, ra_now, dec_now), override
                # the PHOT value with the Gaia value; except for "photom".
                if c in phot.get_columns():
                    X = phot.get(c)
                else:
                    X = np.zeros(len(phot), G.dtype)
                X[J] = G[I]
                phot.set(c, X)
            # unmatched Gaia stars
            unmatched = np.ones(len(gaia), bool)
            unmatched[I] = False
            gaia.cut(unmatched)
            del unmatched

        refs.append(phot)
    else:
        phot_cols = []

    if len(refs) == 1:
        refs = refs[0]
    else:
        refs = merge_tables(refs, columns='fillzero')

    cols = ([('ra_gaia', np.double),
             ('dec_gaia', np.double),
             ('gaia_sourceid', np.int64),
             ('phot_g_mean_mag', np.float32),
             ('phot_g_mean_mag_error', np.float32),
             ('phot_bp_mean_mag', np.float32),
             ('phot_bp_mean_mag_error', np.float32),
             ('phot_rp_mean_mag', np.float32),
             ('phot_rp_mean_mag_error', np.float32),
             ('ra_phot', np.double),
             ('dec_phot', np.double),]
            + phot_cols + [
             ('ra_now', np.double),
             ('dec_now', np.double),
             ('flux0', np.float32),
             ('legacy_survey_mag', np.float32),
             ('psfmag', np.float32),
             ('astrom', bool),
             ('photom', bool),
            ])

    refcols = refs.get_columns()
    for c,dt in cols:
        if not c in refcols:
            refs.set(c, np.zeros(len(refs), dt))
    refcols = refs.get_columns()
    wantcols = dict(cols)
    for c in refcols:
        if not c in wantcols:
            refs.delete_column(c)
            continue

    # Run tractor fitting on the ref stars, using the PsfEx model.
    phot = tractor_fit_sources(imobj, wcs, refs.ra_now, refs.dec_now, refs.flux0,
                               fit_img, ierr, psf)
    print('Got photometry results for', len(phot), 'reference stars')
    if len(phot) == 0:
        return None, None

    # Cut to ref stars that were photometered
    refs.cut(phot.iref)
    phot.delete_column('iref')
    refs.delete_column('flux0')

    with np.errstate(divide='ignore'):
        phot.flux_sn = (phot.flux / phot.dflux)
    phot.flux_sn[phot.dflux == 0] = 0.

    phot.raoff  = (refs.ra_now  - phot.ra_fit ) * 3600. * np.cos(np.deg2rad(refs.dec_now))
    phot.decoff = (refs.dec_now - phot.dec_fit) * 3600.

    dra  = phot.raoff [refs.astrom]
    ddec = phot.decoff[refs.astrom]

    raoff  = np.median(dra)
    decoff = np.median(ddec)
    rastd  = np.std(dra)
    decstd = np.std(ddec)
    ra_clip, _, _ = sigmaclip(dra, low=3., high=3.)
    rarms = getrms(ra_clip)
    dec_clip, _, _ = sigmaclip(ddec, low=3., high=3.)
    decrms = getrms(dec_clip)

    # For astrom, since we have Gaia everywhere, count the number
    # that pass the sigma-clip, ie, the number of stars that
    # corresponds to the reported offset and scatter (in RA).
    nastrom = len(ra_clip)

    print('RA, Dec offsets (arcsec): %.4f, %.4f' % (raoff, decoff))
    print('RA, Dec stddev  (arcsec): %.4f, %.4f' % (rastd, decstd))
    print('RA, Dec RMS     (arcsec): %.4f, %.4f' % (rarms, decrms))

    ok, = np.nonzero(phot.flux > 0)
    phot.instpsfmag = np.zeros(len(phot), np.float32)
    phot.instpsfmag[ok] = -2.5*np.log10(phot.flux[ok] / imobj.exptime)
    # Uncertainty on psfmag
    phot.dpsfmag = np.zeros(len(phot), np.float32)
    phot.dpsfmag[ok] = np.abs((-2.5 / np.log(10.)) * phot.dflux[ok] / phot.flux[ok])

    H,W = dq.shape
    phot.bitmask = dq[np.clip(phot.y1, 0, H-1).astype(int),
                      np.clip(phot.x1, 0, W-1).astype(int)]

    phot.psfmag = np.zeros(len(phot), np.float32)

    print('Flux S/N min/median/max: %.1f / %.1f / %.1f' %
          (phot.flux_sn.min(), np.median(phot.flux_sn), phot.flux_sn.max()))
    # Note that this is independent of whether we have reference stars
    # (eg, will work where we don't have PS1)
    nphotom = np.sum(phot.flux_sn > 5.)

    dmag = refs.legacy_survey_mag - phot.instpsfmag
    maglo, maghi = MAGLIM[imobj.band]
    dmag = dmag[refs.photom &
                (refs.legacy_survey_mag > maglo) &
                (refs.legacy_survey_mag < maghi) &
                np.isfinite(dmag)]
    if len(dmag):
        print('Zeropoint: using', len(dmag), 'good stars')
        dmag, _, _ = sigmaclip(dmag, low=2.5, high=2.5)
        print('Zeropoint: using', len(dmag), 'stars after sigma-clipping')

        zptstd = np.std(dmag)
        zptmed = np.median(dmag)
        dzpt = zptmed - zp0
        kext = imobj.extinction(imobj.band)
        transp = 10.**(-0.4 * (-dzpt - kext * (airmass - 1.0)))

        print('Number of stars used for zeropoint median %d' % nphotom)
        print('Zeropoint %.4f' % zptmed)
        print('Offset from nominal: %.4f' % dzpt)
        print('Scatter: %.4f' % zptstd)
        print('Transparency %.4f' % transp)

        ok = (phot.instpsfmag != 0)
        phot.psfmag[ok] = phot.instpsfmag[ok] + zptmed

        zpscale = NanoMaggies.zeropointToScale(zptmed)
        ccds['sig1'] /= zpscale

    else:
        dzpt = 0.
        zptmed = 0.
        zptstd = 0.
        transp = 0.

    for c in ['x0','y0','x1','y1','flux','raoff','decoff', 'psfmag',
              'dflux','dx','dy']:
        phot.set(c, phot.get(c).astype(np.float32))
    phot.rename('x0', 'x_ref')
    phot.rename('y0', 'y_ref')
    phot.rename('x1', 'x_fit')
    phot.rename('y1', 'y_fit')

    phot.add_columns_from(refs)

    # Save CCD-level information in the per-star table.
    phot.ccd_raoff  = np.zeros(len(phot), np.float32) + raoff
    phot.ccd_decoff = np.zeros(len(phot), np.float32) + decoff
    phot.ccd_phoff  = np.zeros(len(phot), np.float32) + dzpt
    phot.ccd_zpt    = np.zeros(len(phot), np.float32) + zptmed
    phot.expnum  = np.zeros(len(phot), np.int64)
    # Can't just add it as in previous lines, because of weird int64 type issues in
    # fitsio (see https://github.com/legacysurvey/legacypipe/issues/478)
    phot.expnum += imobj.expnum
    phot.ccdname = np.array([imobj.ccdname] * len(phot))
    phot.filter  = np.array([imobj.band] * len(phot))
    # pad ccdname to 4 characters (Bok: "CCD1")
    if len(imobj.ccdname) < 4:
        phot.ccdname = phot.ccdname.astype('S4')

    phot.exptime = np.zeros(len(phot), np.float32) + imobj.exptime
    phot.gain    = np.zeros(len(phot), np.float32) + ccds['gain']
    phot.airmass = np.zeros(len(phot), np.float32) + airmass

    import photutils
    apertures_arcsec_diam = [6, 7, 8]
    for arcsec_diam in apertures_arcsec_diam:
        ap = photutils.CircularAperture(np.vstack((phot.x_fit, phot.y_fit)).T,
                                        arcsec_diam / 2. / imobj.pixscale)
        with np.errstate(divide='ignore'):
            err = 1./ierr
        apphot = photutils.aperture_photometry(fit_img, ap,
                                               error=err, mask=(ierr==0))
        phot.set('apflux_%i' % arcsec_diam,
                 apphot.field('aperture_sum').data.astype(np.float32))
        phot.set('apflux_%i_err' % arcsec_diam,
                 apphot.field('aperture_sum_err').data.astype(np.float32))

    # Add to the zeropoints table
    ccds['raoff']  = raoff
    ccds['decoff'] = decoff
    ccds['rastddev']  = rastd
    ccds['decstddev'] = decstd
    ccds['rarms']  = rarms
    ccds['decrms'] = decrms
    ccds['phoff'] = dzpt
    ccds['phrms'] = zptstd
    ccds['zpt'] = zptmed
    ccds['nstars_photom'] = nphotom
    ccds['nstars_astrom'] = nastrom

    # .ra,.dec = Gaia else PS1
    phot.ra  = phot.ra_gaia
    phot.dec = phot.dec_gaia
    I, = np.nonzero(phot.ra == 0)
    phot.ra [I] = phot.ra_phot [I]
    phot.dec[I] = phot.dec_phot[I]

    # Create subset table for Eddie's ubercal
    cols = ([
        'ra', 'dec', 'flux', 'dflux', 'chi2', 'fracmasked', 'instpsfmag',
        'dpsfmag',
        'bitmask', 'x_fit', 'y_fit', 'gaia_sourceid', 'ra_gaia', 'dec_gaia',
        'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
        'phot_g_mean_mag_error', 'phot_bp_mean_mag_error',
        'phot_rp_mean_mag_error',
    ] + [c for c,t in phot_cols] + [
        'legacy_survey_mag', 'psfmag',
        'expnum', 'ccdname', 'exptime', 'gain', 'airmass', 'filter',
        'apflux_6', 'apflux_7', 'apflux_8',
        'apflux_6_err', 'apflux_7_err', 'apflux_8_err',
        'ra_now', 'dec_now', 'ra_fit', 'dec_fit', 'x_ref', 'y_ref'
    ])
    for c in phot.get_columns():
        if not c in cols:
            phot.delete_column(c)

    t0= ptime('all-computations-for-this-ccd',t0)
    # Plots for comparing to Arjuns zeropoints*.ps
    verboseplots = False
    if verboseplots:
        imobj.make_plots(phot,dmag,ccds['zpt'],transp)
        t0= ptime('made-plots',t0)
    return ccds, phot

def tractor_fit_sources(imobj, wcs, ref_ra, ref_dec, ref_flux, img, ierr,
                        psf, normalize_psf=True):
    import tractor
    from tractor import PixelizedPSF
    from tractor.brightness import LinearPhotoCal

    plots = False
    plot_this = False
    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('astromfit')

    print('Fitting positions & fluxes of %i stars' % len(ref_ra))

    cal = fits_table()
    # These x0,y0,x1,y1 are zero-indexed coords.
    cal.x0 = []
    cal.y0 = []
    cal.x1 = []
    cal.y1 = []
    cal.flux = []
    cal.dx = []
    cal.dy = []
    cal.dflux = []
    cal.psfsum = []
    cal.iref = []
    cal.chi2 = []
    cal.fracmasked = []
    nzeroivar = 0
    noffim = 0

    for istar,(ra,dec) in enumerate(zip(ref_ra, ref_dec)):
        _,x,y = wcs.radec2pixelxy(ra, dec)
        x -= 1
        y -= 1
        # Fitting radius
        R = 10
        H,W = img.shape
        xlo = int(x - R)
        ylo = int(y - R)
        if xlo < 0 or ylo < 0:
            noffim += 1
            continue
        xhi = xlo + R*2
        yhi = ylo + R*2
        if xhi >= W or yhi >= H:
            noffim += 1
            continue
        subimg = img[ylo:yhi+1, xlo:xhi+1]
        # FIXME -- check that ierr is correct
        subie = ierr[ylo:yhi+1, xlo:xhi+1]

        if False:
            subpsf = psf.constantPsfAt(x, y)
            psfsum = np.sum(subpsf.img)
            if normalize_psf:
                # print('Normalizing PsfEx model with sum:', s)
                subpsf.img /= psfsum

        psfimg = psf.getImage(x, y)
        ph,pw = psf.img.shape
        psfsum = np.sum(psfimg)
        if normalize_psf:
            psfimg /= psfsum
        sz = R + 5
        psfimg = psfimg[ph//2-sz:ph//2+sz+1, pw//2-sz:pw//2+sz+1]
        subpsf = PixelizedPSF(psfimg)

        if np.all(subie == 0):
            nzeroivar += 1
            # print('Inverse-variance map is all zero')
            continue

        tim = tractor.Image(data=subimg, inverr=subie, psf=subpsf)
        flux0 = ref_flux[istar]
        x0 = x - xlo
        y0 = y - ylo
        src = tractor.PointSource(tractor.PixPos(x0, y0), tractor.Flux(flux0))
        tr = tractor.Tractor([tim], [src])

        tr.freezeParam('images')
        optargs = dict(priors=False, shared_params=False,
                       alphas=[0.1, 0.3, 1.0])

        # Do a quick flux fit first
        src.freezeParam('pos')
        pc = tim.photocal
        tim.photocal = LinearPhotoCal(1.)
        tr.optimize_forced_photometry(**optargs)
        tim.photocal = pc
        src.thawParam('pos')

        if plots and plot_this:
            import pylab as plt
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(subimg, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(2,2,2)
            mod = tr.getModelImage(0)
            plt.imshow(mod, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow((subimg - mod) * subie, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.suptitle('Before')
            ps.savefig()

        # Now the position and flux fit
        for _ in range(50):
            dlnp,_,_ = tr.optimize(**optargs)
            if dlnp == 0:
                break
        variance = tr.optimize(variance=True, just_variance=True, **optargs)
        # Yuck -- if inverse-variance is all zero, weird-shaped result...
        if len(variance) == 4 and variance[3] is None:
            print('No variance estimate available')
            continue

        mod = tr.getModelImage(0)
        chi = (subimg - mod) * subie
        psfimg = mod / mod.sum()
        # profile-weighted chi-squared
        cal.chi2.append(np.sum(chi**2 * psfimg))
        # profile-weighted fraction of masked pixels
        cal.fracmasked.append(np.sum(psfimg * (subie == 0)))

        cal.psfsum.append(psfsum)
        cal.x0.append(x0 + xlo)
        cal.y0.append(y0 + ylo)
        cal.x1.append(src.pos.x + xlo)
        cal.y1.append(src.pos.y + ylo)
        cal.flux.append(src.brightness.getValue())
        cal.iref.append(istar)

        std = np.sqrt(variance)
        cal.dx.append(std[0])
        cal.dy.append(std[1])
        cal.dflux.append(std[2])

        if plots and plot_this:
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(subimg, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(2,2,2)
            mod = tr.getModelImage(0)
            plt.imshow(mod, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow((subimg - mod) * subie, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.suptitle('After')
            ps.savefig()

    if nzeroivar > 0:
        print('Zero ivar for %d stars' % nzeroivar)
    if noffim > 0:
        print('Off image for %d stars' % noffim)
    cal.to_np_arrays()
    cal.ra_fit,cal.dec_fit = wcs.pixelxy2radec(cal.x1 + 1, cal.y1 + 1)
    return cal

if __name__ == "__main__":
    main()
