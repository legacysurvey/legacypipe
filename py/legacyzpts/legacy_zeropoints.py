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
from astropy.io import fits as fits_astropy
from astropy.table import Table, vstack

from astrometry.util.file import trymakedirs
from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec
from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.ttime import Time
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec

from tractor.splinesky import SplineSky

import legacypipe
from legacypipe.ps1cat import ps1cat, sdsscat
from legacypipe.gaiacat import GaiaCatalog
from legacypipe.survey import radec_at_mjd, get_git_version
from legacypipe.image import validate_version

CAMERAS=['decam','mosaic','90prime','megaprime']
MAGLIM=dict(
    u=[16, 20],
    g=[16, 20],
    r=[16, 19.5],
    i=[16, 19.5],
    z=[16.5, 19],
    Y=[16.5, 19])

def ptime(text,t0):
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return np.array( list(np.char.strip(lines)) )

def astropy_to_astrometry_table(t):
    T = fits_table()
    for c in t.colnames:
        T.set(c, t[c])
    return T

def _ccds_table(camera='decam'):
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
        ('filter', 'S1'),
        ('exptime', 'f4'),
        ('date_obs', 'S26'),
        ('mjd_obs', 'f8'),
        ('ut', 'S15'),
        ('ha', 'S13'),
        ('airmass', 'f4'),
        ('fwhm', 'f4'),
        ('fwhm_cp', 'f4'),
        ('gain', 'f4'),
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
        ('goodps1', 'i2'),
        ('goodps1_wbadpix5', 'i2'),
        ('phoff', 'f4'),
        ('phrms', 'f4'),
        ('phrmsavg', 'f4'),
        ('zpt', 'f4'),
        ('zpt_wbadpix5', 'f4'),
        ('transp', 'f4'),
        ('raoff',  'f4'),
        ('decoff', 'f4'),
        ('rarms',  'f4'),
        ('decrms', 'f4'),
        ('rastddev',  'f4'),
        ('decstddev', 'f4'),
        ]
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

def get_pixscale(camera):
  return {'decam':0.262,
          'mosaic':0.262,
          '90prime':0.455,
          'megaprime':0.185}[camera]

def cols_for_survey_table(which='all'):
    """Return list of -survey.fits table colums

    Args:
        which: all, numeric,
        nonzero_diff (numeric and expect non-zero diff with reference
        when compute it)
    """
    assert(which in ['all','numeric','nonzero_diff'])
    martins_keys = ['airmass', 'ccdskysb']
    gods_keys = ['plver', 'procdate', 'plprocid', 'ccdnastrom', 'ccdnphotom']
    if which == 'all':
        need_arjuns_keys= ['ra','dec','ra_bore','dec_bore',
                           'image_filename','image_hdu','expnum','ccdname','object',
                           'filter','exptime','camera','width','height','propid',
                           'mjd_obs',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff',
                           'ccdrarms', 'ccddecrms', 'ccdskycounts',
                           'phrms', 'ccdphrms',
                           'cd1_1','cd2_2','cd1_2','cd2_1',
                           'crval1','crval2','crpix1','crpix2']
        dustins_keys= ['skyrms', 'sig1', 'yshift']
    elif which == 'numeric':
        need_arjuns_keys= ['ra','dec','ra_bore','dec_bore',
                           'expnum',
                           'exptime','width','height',
                           'mjd_obs',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff',
                           'cd1_1','cd2_2','cd1_2','cd2_1',
                           'crval1','crval2','crpix1','crpix2']
        dustins_keys= ['skyrms']
    elif which == 'nonzero_diff':
        need_arjuns_keys= ['ra','dec',
                           'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff']
        dustins_keys= ['skyrms']
    return need_arjuns_keys + dustins_keys + martins_keys + gods_keys

def prep_survey_table(T, camera=None, bad_expid=None):
    assert(camera in CAMERAS)
    need_keys = cols_for_survey_table(which='all')
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

def create_annotated_table(T, ann_fn, camera, survey, mp):
    from legacyzpts.annotate_ccds import annotate, init_annotations
    T = survey.cleanup_ccds_table(T)
    init_annotations(T)
    annotate(T, survey, mp=mp, mzls=(camera == 'mosaic'), bass=(camera == '90prime'),
             normalizePsf=True, carryOn=True)
    writeto_via_temp(ann_fn, T)
    print('Wrote %s' % ann_fn)

def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) )

class Measurer(object):
    """Main image processing functions for all cameras.

    Args:
        match_radius: arcsec matching to gaia/ps1
        sn_min,sn_max: if not None then then {min,max} S/N will be enforced from
            aperture photoemtry, where S/N = apflux/sqrt(skyflux)
    """

    def __init__(self, fn, image_dir='images',
                 calibrate=False, quiet=False,
                 **kwargs):
        self.quiet = quiet
        # Set extra kwargs
        self.zptsfile= kwargs.get('zptsfile')
        self.prefix= kwargs.get('prefix')
        self.verboseplots= kwargs.get('verboseplots')

        self.fn = os.path.join(image_dir, fn)
        self.fn_base = fn
        self.debug= kwargs.get('debug')
        self.outdir= kwargs.get('outdir')
        self.calibdir = kwargs.get('calibdir')

        self.calibrate = calibrate

        self.primhdr = self.read_primary_header()

        self.band = self.get_band()
        # CP WCS succeed?
        self.goodWcs = self.good_wcs(self.primhdr)

        # Camera-agnostic primary header cards
        try:
            self.propid = self.primhdr['PROPID']
        except KeyError:
            self.propid = self.primhdr.get('DTPROPID')
        self.exptime = self.primhdr['EXPTIME']
        self.date_obs = self.primhdr['DATE-OBS']
        self.mjd_obs = self.primhdr['MJD-OBS']
        self.ut = self.get_ut(self.primhdr)
        # Add more attributes.
        namechange = dict(date='procdate')
        for key in ['AIRMASS','HA', 'DATE', 'PLVER', 'PLPROCID']:
            val = self.primhdr.get(key)
            if type(val) == str:
                val = val.strip()
                if len(val) == 0:
                    raise ValueError('Empty header card: %s' % key)
            attrkey = namechange.get(key.lower(), key.lower())
            setattr(self, attrkey, val)

        self.ra_bore,self.dec_bore = self.get_radec_bore(self.primhdr)

        if self.airmass is None or self.camera == 'mosaic':
            # Recompute it
            site = self.get_site()
            if site is None:
                print('AIRMASS missing and site not defined.')
            else:
                print('Recomputing AIRMASS')
                from astropy.time import Time as apyTime
                from astropy.coordinates import SkyCoord, AltAz
                time = apyTime(self.mjd_obs + 0.5*self.exptime/3600./24., format='mjd')
                coords = SkyCoord(self.ra_bore, self.dec_bore, unit='deg')
                altaz = coords.transform_to(AltAz(obstime=time, location=site))
                self.airmass = altaz.secz

        self.expnum = self.get_expnum(self.primhdr)
        if not quiet:
            print('CP Header: EXPNUM', self.expnum, 'PLVER', self.plver,
                  'PLPROCID', self.plprocid)
        self.obj = self.primhdr['OBJECT']

    def get_extension_list(self, fn, debug=False):
        raise RuntimeError('get_extension_list not implemented in type ' + str(type(self)))

    def good_wcs(self, primhdr):
        return primhdr.get('WCSCAL', '').strip().lower().startswith('success')

    def get_site(self):
        return None

    def read_primary_header(self):
        try:
            primhdr = read_primary_header(self.fn)
        except ValueError:
            # astropy can handle it
            tmp= fits_astropy.open(self.fn)
            primhdr= tmp[0].header
            tmp.close()
            del tmp
        return primhdr

    def get_radec_bore(self, primhdr):
        # {RA,DEC}: center of exposure, TEL{RA,DEC}: boresight of telescope
        # In some DECam exposures, RA,DEC are floating-point, but RA is in *decimal hours*.
        # In others, RA does not exist (eg CP/V4.8.2a/CP20160824/c4d_160825_062109_ooi_g_ls9.fits.fz)
        # Fall back to TELRA in that case.
        ra_bore = dec_bore = None
        if 'RA' in primhdr.keys():
            try:
                ra_bore = hmsstring2ra(primhdr['RA'])
                dec_bore = dmsstring2dec(primhdr['DEC'])
            except:
                pass
        if dec_bore is None and 'TELRA' in primhdr.keys():
            ra_bore = hmsstring2ra(primhdr['TELRA'])
            dec_bore = dmsstring2dec(primhdr['TELDEC'])
        if dec_bore is None:
            raise ValueError('Failed to parse RA or TELRA in primary header to get telescope boresight')
        return ra_bore, dec_bore

    def get_good_image_subregion(self):
        '''
        Returns x0,x1,y0,y1 of the good region of this chip,
        or None if no cut should be applied to that edge; returns
        (None,None,None,None) if the whole chip is good.

        This cut is applied in addition to any masking in the mask or
        invvar map.
        '''
        return None,None,None,None

    def get_expnum(self, primhdr):
        return primhdr['EXPNUM']

    def get_ut(self, primhdr):
        return primhdr['TIME-OBS']

    def zeropoint(self, band):
        return self.zp0[band]

    def extinction(self, band):
        return self.k_ext[band]

    def read_header(self, ext):
        return fitsio.read_header(self.fn, ext=ext)

    def set_hdu(self,ext):
        self.ext = ext.strip()
        self.ccdname= ext.strip()
        self.expid = '{:08d}-{}'.format(self.expnum, self.ccdname)
        hdulist= fitsio.FITS(self.fn)
        self.image_hdu= hdulist[ext].get_extnum() #NOT ccdnum in header!
        # use header
        self.hdr = self.read_header(ext)
        # Sanity check
        assert(self.ccdname.upper() == self.hdr['EXTNAME'].strip().upper())
        self.ccdnum = np.int(self.hdr.get('CCDNUM', 0))
        self.gain= self.get_gain(self.hdr)
        # WCS
        self.wcs = self.get_wcs()
        # Pixscale is assumed CONSTANT! per camera

        self.height, self.width = self.get_hdu_shape(self.hdr, hdulist[ext])
        self.fwhm_cp = self.get_fwhm(self.hdr, hdulist[ext])

        x0,x1,y0,y1 = self.get_good_image_subregion()
        if x0 is None and x1 is None and y0 is None and y1 is None:
            slc = None
        else:
            x0 = x0 or 0
            x1 = x1 or self.width
            y0 = y0 or 0
            y1 = y1 or self.height
            slc = slice(y0,y1),slice(x0,x1)
        self.slc = slc

    def get_hdu_shape(self, hdr, hdu):
        h,w = hdu.get_info()['dims']
        return int(h), int(w)

    def get_fwhm(self, hdr, hdu):
        return hdr['FWHM']

    def read_bitmask(self):
        dqfn = self.get_bitmask_fn(self.fn)
        if self.slc is not None:
            mask = fitsio.FITS(dqfn)[self.ext][self.slc]
        else:
            mask = fitsio.read(dqfn, ext=self.ext)
        mask = self.remap_bitmask(mask)
        return mask

    def get_bitmask_fn(self, imgfn):
        if 'ooi' in imgfn:
            fn= imgfn.replace('ooi','ood')
        elif 'oki' in imgfn:
            fn= imgfn.replace('oki','ood')
        else:
            raise ValueError('bad imgfn? no ooi or oki: %s' % imgfn)
        return fn

    def get_weight_fn(self, imgfn):
        if 'ooi' in imgfn:
            fn= imgfn.replace('ooi','oow')
        elif 'oki' in imgfn:
            fn= imgfn.replace('oki','oow')
        else:
            raise ValueError('bad imgfn? no ooi or oki: %s' % imgfn)
        return fn

    def remap_bitmask(self, mask):
        return mask

    def read_weight(self, clip=True, clipThresh=0.1, scale=True, bitmask=None):
        fn = self.get_weight_fn(self.fn)

        if self.slc is not None:
            wt = fitsio.FITS(fn)[self.ext][self.slc]
        else:
            wt = fitsio.read(fn, ext=self.ext)

        if bitmask is not None:
            # Set all masked pixels to have weight zero.
            # bitmask value 1 = bad
            wt[bitmask > 0] = 0.

        if clip and np.any(wt > 0):

            fixed = False
            try:
                from legacypipe.image import fix_weight_quantization
                fixed = fix_weight_quantization(wt, fn, self.ext, self.slc)
            except:
                import traceback
                traceback.print_exc()

            if not fixed:
                # Clamp near-zero (incl negative!) weight to zero,
                # which arise due to fpack.
                if clipThresh > 0.:
                    thresh = clipThresh * np.median(wt[wt > 0])
                else:
                    thresh = 0.
                wt[wt < thresh] = 0

        if scale:
            wt = self.scale_weight(wt)

        #assert(np.all(wt >= 0.))
        wt[np.where(wt<0.0)] = 0.0
        assert(np.all(np.isfinite(wt)))

        return wt

    def read_image(self):
        '''Read the image and header; scale the image.'''
        f = fitsio.FITS(self.fn)[self.ext]
        if self.slc is not None:
            img = f[self.slc]
        else:
            img = f.read()
        hdr = f.read_header()
        img = self.scale_image(img)
        return img, hdr

    def scale_image(self, img):
        return img

    def scale_weight(self, img):
        return img

    def remap_invvar(self, invvar, primhdr, img, dq):
        # By default, *do not* remap
        return invvar

    # A function that can be called by a subclasser's remap_invvar() method
    def remap_invvar_shotnoise(self, invvar, primhdr, img, dq):
        #
        # All three cameras scale the image and weight to units of electrons.
        # (actually, not DECam any more! But DECamMeasurer doesn't use this
        #  function.)
        #
        print('Remapping weight map for', self.fn)
        const_sky = primhdr['SKYADU'] # e/s, Recommended sky level keyword from Frank
        expt = primhdr['EXPTIME'] # s
        with np.errstate(divide='ignore'):
            var_SR = 1./invvar # e**2

        print('median img:', np.median(img), 'vs sky estimate * exptime', const_sky*expt)

        var_Astro = np.abs(img - const_sky * expt) # img in electrons; Poisson process so variance = mean
        wt = 1./(var_SR + var_Astro) # 1/(e**2)
        # Zero out NaNs and masked pixels
        wt[np.isfinite(wt) == False] = 0.
        wt[dq != 0] = 0.
        return wt

    def create_zero_one_mask(self,bitmask,good=None):
        """Return zero_one_mask array given a bad pixel map and good pix values
        bitmask: ood image
        good: list of values to treat as good in the bitmask
        """
        # 0 == good, 1 == bad
        if good is None:
            good = []
        zero_one_mask= bitmask.copy()
        for val in good:
            zero_one_mask[zero_one_mask == val]= 0
        zero_one_mask[zero_one_mask > 0]= 1
        return zero_one_mask

    def get_zero_one_mask(self,bitmask,good=None):
        """Convert bitmask into a zero and ones mask, 1 = bad, 0 = good
        bitmask: ood image
        good: (optional) list of values to treat as good in the bitmask
            default is to use appropiate values for the camera
        """
        if good is None:
            good = []
        # Defaults
        if len(good) == 0:
            if self.camera == 'decam':
                # 7 = transient
                good=[7]
            elif self.camera == 'mosaic':
                # 5 is truly a cosmic ray
                good=[]
            elif self.camera == '90prime':
                # 5 can be really bad for a good image because these are subtracted
                # and interpolated stats
                good= []
        return self.create_zero_one_mask(bitmask,good=good)

    def sensible_sigmaclip(self, arr, nsigma = 4.0):
        '''sigmaclip returns unclipped pixels, lo,hi, where lo,hi are the
        mean(goodpix) +- nsigma * sigma
        '''
        goodpix, lo, _ = sigmaclip(arr, low=nsigma, high=nsigma)
        meanval = np.mean(goodpix)
        sigma = (meanval - lo) / nsigma
        return meanval, sigma

    def get_sky_and_sigma(self, img, nsigma=3):
        '''returns 2d sky image and sky rms'''
        splinesky= False
        if splinesky:
            skyobj = SplineSky.BlantonMethod(img, None, 256)
            skyimg = np.zeros_like(img)
            skyobj.addTo(skyimg)
            _, skystd = self.sensible_sigmaclip(img - skyimg,nsigma=nsigma)
            skymed= np.median(skyimg)
        else:
            if self.camera in ['decam', 'megaprime']:
                slc=[slice(1500,2500),slice(500,1500)]
            elif self.camera in ['mosaic','90prime']:
                slc=[slice(500,1500),slice(500,1500)]
            else:
                raise RuntimeError('unknown camera %s' % self.camera)
            clip_vals,_,_ = sigmaclip(img[tuple(slc)],low=nsigma,high=nsigma)
            skymed= np.median(clip_vals)
            skystd= np.std(clip_vals)
            skyimg= np.zeros(img.shape) + skymed
        return skyimg, skymed, skystd

    def get_ps1_cuts(self,ps1):
        """Returns bool of PS1 sources to keep
        ps1: catalogue with ps1 data
        """
        gicolor= ps1.median[:,0] - ps1.median[:,2]
        return ((ps1.nmag_ok[:, 0] > 0) &
                (ps1.nmag_ok[:, 1] > 0) &
                (ps1.nmag_ok[:, 2] > 0) &
                (gicolor > 0.4) &
                (gicolor < 2.7))

    def return_on_error(self,err_message='',
                        ccds=None, stars_photom=None):
        """Sets ccds table err message, zpt to nan, and returns appropriately for self.run()

        Args:
         err_message: length <= 30
         ccds, stars_photom, stars_astrom: (optional) tables partially filled by run()
        """
        assert(len(err_message) > 0 & len(err_message) <= 30)
        if ccds is None:
            ccds= _ccds_table(self.camera)
            ccds['image_filename'] = self.fn_base
        ccds['err_message']= err_message
        ccds['zpt']= 0
        return ccds, stars_photom

    def init_ccd(self, ccds):
        '''
        ccds: 1-row astropy table object from _ccds_table.
        '''
        ccds['image_filename'] = self.fn_base
        ccds['object'] = self.obj
        ccds['filter'] = self.band
        ccds['yshift'] = 'YSHIFT' in self.primhdr

        for key in ['image_hdu', 'ccdnum', 'camera', 'expnum', 'plver', 'procdate',
                    'plprocid', 'ccdname', 'expid', 'propid', 'exptime', 'date_obs',
                    'mjd_obs', 'ut', 'ra_bore', 'dec_bore', 'ha', 'airmass', 'gain',
                    'pixscale', 'width', 'height', 'fwhm_cp']:
            ccds[key] = getattr(self, key)

        # Formerly we grabbed the PsfEx FWHM; instead just use the CP value!
        ccds['fwhm'] = ccds['fwhm_cp']

    def run(self, ext=None, save_xy=False, splinesky=False, survey=None,
            sdss_photom=False):

        """Computes statistics for 1 CCD

        Args:
            ext: ccdname
            save_xy: save daophot x,y and x,y after various cuts to dict and save
                to json

        Returns:
            ccds, stars_photom, stars_astrom
        """
        self.set_hdu(ext)
        #
        t0= Time()
        t0= ptime('Measuring CCD=%s from image=%s' % (self.ccdname,self.fn),t0)

        # Initialize
        ccds = _ccds_table(self.camera)

        self.init_ccd(ccds)

        notneeded_cols= ['avsky']
        for ccd_col in ['avsky', 'crpix1', 'crpix2', 'crval1', 'crval2',
                        'cd1_1','cd1_2', 'cd2_1', 'cd2_2']:
            if ccd_col.upper() in self.hdr.keys():
                #print('CP Header: %s = ' % ccd_col,self.hdr[ccd_col])
                ccds[ccd_col] = self.hdr[ccd_col]
            elif ccd_col in notneeded_cols:
                ccds[ccd_col] = np.nan
            else:
                raise KeyError('Could not find %s key in header:' % ccd_col)

        # WCS: 1-indexed so pixel pixelxy2radec(1,1) corresponds to img[0,0]
        H = self.height
        W = self.width
        ccdra, ccddec = self.wcs.pixelxy2radec((W+1) / 2.0, (H+1) / 2.0)
        ccds['ra']  = ccdra  # [degree]
        ccds['dec'] = ccddec # [degree]
        t0= ptime('header-info',t0)

        if not self.goodWcs:
            print('WCS Failed on CCD {}'.format(self.ccdname))
            return self.return_on_error(err_message='WCS Failed', ccds=ccds)
        if self.exptime == 0:
            print('Exptime = 0 on CCD {}'.format(self.ccdname))
            return self.return_on_error(err_message='Exptime = 0', ccds=ccds)

        self.bitmask = self.read_bitmask()
        weight = self.read_weight(bitmask=self.bitmask, scale=False)
        if np.all(weight == 0):
            txt = 'All weight-map pixels are zero on CCD {}'.format(self.ccdname)
            print(txt)
            return self.return_on_error(txt,ccds=ccds)
        # bizarro image CP20151119/k4m_151120_040715_oow_zd_v1.fits.fz
        if np.all(np.logical_or(weight == 0, weight == 1)):
            txt = 'All weight-map pixels are zero or one'
            print(txt)
            return self.return_on_error(txt,ccds=ccds)
        weight = self.scale_weight(weight)

        # Quick check for PsfEx file
        psf = self.get_psfex_model()
        if psf.psfex.sampling == 0.:
            print('PsfEx model has SAMPLING=0')
            nacc = psf.header.get('ACCEPTED')
            print('PsfEx model number of stars accepted:', nacc)
            return self.return_on_error(err_message='Bad PSF model', ccds=ccds)

        self.img,_ = self.read_image()

        # Per-pixel error
        medweight = np.median(weight[(weight > 0) * (self.bitmask == 0)])
        # We scale the image & weight into ADU, but we want to report
        # sig1 in nanomaggie-ish units, ie, in per-second units.
        ccds['sig1'] = (1. / np.sqrt(medweight)) / self.exptime

        self.invvar = self.remap_invvar(weight, self.primhdr, self.img, self.bitmask)

        t0= ptime('read image',t0)

        # Measure the sky brightness and (sky) noise level.
        zp0 = self.zeropoint(self.band)
        #print('Computing the sky background.')
        sky_img, skymed, skyrms = self.get_sky_and_sigma(self.img)

        # Bunch of sky estimates
        # Median of absolute deviation (MAD), std dev = 1.4826 * MAD
        print('sky from median of image = %.2f' % skymed)
        skybr = zp0 - 2.5*np.log10(skymed / self.pixscale / self.pixscale / self.exptime)
        print('Sky brightness: {:.3f} mag/arcsec^2 (assuming nominal zeropoint)'.format(skybr))

        ccds['skyrms'] = skyrms / self.exptime # e/sec
        ccds['skycounts'] = skymed / self.exptime # [electron/pix]
        ccds['skysb'] = skybr   # [mag/arcsec^2]
        t0= ptime('measure-sky',t0)

        # Load PS1 & Gaia catalogues
        # We will only used detected sources that have PS1 or Gaia matches
        # So cut to this super set immediately

        phot = None
        if sdss_photom:
            try:
                phot = sdsscat(ccdwcs=self.wcs).get_stars(magrange=None)
            except OSError as e:
                print('No SDSS stars found for this image -- outside the SDSS footprint?', e)
        else:
            try:
                phot = ps1cat(ccdwcs=self.wcs).get_stars(magrange=None)
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
                phot.legacy_survey_mag = self.photometric_calibrator_to_observed(name, phot)
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

        return self.run_psfphot(ccds, ps1, gaia, zp0, sky_img, splinesky, survey,
                                sdss_photom=sdss_photom)

    def run_psfphot(self, ccds, ps1, gaia, zp0, sky_img, splinesky, survey,
                    sdss_photom=False):
        t0= Time()

        # Now put Gaia stars into the image and re-fit their centroids
        # and fluxes using the tractor with the PsfEx PSF model.

        # assume that the CP WCS has gotten us to within a few pixels
        # of the right answer.  Find Gaia stars, initialize Tractor
        # sources there, optimize them and see how much they want to
        # move.
        psf = self.get_psfex_model()

        if splinesky:
            sky = self.get_splinesky()
            print('Instantiating and subtracting sky model')
            skymod = np.zeros_like(self.img)
            sky.addTo(skymod)
            # Apply the same transformation that was applied to the image...
            skymod = self.scale_image(skymod)
            fit_img = self.img - skymod
        else:
            fit_img = self.img - sky_img

        # after sky subtraction, apply optional per-amp relative zeropoints.
        self.apply_amp_correction(fit_img, self.invvar)

        with np.errstate(invalid='ignore'):
            # sqrt(0.) can trigger complaints;
            #   https://github.com/numpy/numpy/issues/11448
            ierr = np.sqrt(self.invvar)

        # Move Gaia stars to the epoch of this image.
        gaia.rename('ra',  'ra_gaia')
        gaia.rename('dec', 'dec_gaia')
        gaia.rename('source_id', 'gaia_sourceid')
        ra,dec = radec_at_mjd(gaia.ra_gaia, gaia.dec_gaia,
                              gaia.ref_epoch.astype(float),
                              gaia.pmra, gaia.pmdec, gaia.parallax, self.mjd_obs)
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

        if ps1 is not None:
            # PS1 for photometry
            # Initial flux estimate, from nominal zeropoint
            ps1.flux0 = (10.**((zp0 - ps1.legacy_survey_mag) / 2.5) * self.exptime
                         ).astype(np.float32)

            if sdss_photom:
                ps1.ra_sdss  = ps1.ra.copy()
                ps1.dec_sdss = ps1.dec.copy()
                ps1.ra_phot = ps1.ra_sdss
                ps1.dec_phot = ps1.dec_sdss
                bands = 'ugriz'
                for band in bands:
                    i = sdsscat.sdssband.get(band, None)
                    if i is None:
                        print('No band', band, 'in SDSS catalog')
                        continue
                    ps1.set('sdss_'+band.lower(), ps1.psfmag[:,i].astype(np.float32))
                phot_cols = [
                    #('ps1_objid', np.int64),
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
                ps1.rename('ra_ok',  'ra_now')
                ps1.rename('dec_ok', 'dec_now')
                ps1.ra_ps1  = ps1.ra_now.copy()
                ps1.dec_ps1 = ps1.dec_now.copy()
                ps1.ra_phot = ps1.ra_ps1
                ps1.dec_phot = ps1.dec_ps1
                ps1.ps1_objid  = ps1.obj_id
                bands = 'grizY'
                for band in bands:
                    i = ps1cat.ps1band.get(band, None)
                    if i is None:
                        print('No band', band, 'in PS1/SDSS catalog')
                        continue
                    ps1.set('ps1_'+band.lower(), ps1.median[:,i].astype(np.float32))
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
            ps1.photom = np.ones (len(ps1), bool)

            # Match PS1 to Gaia stars within 1".
            I,J,_ = match_radec(gaia.ra_gaia, gaia.dec_gaia,
                                ps1.ra_phot, ps1.dec_phot, 1./3600.,
                                nearest=True)
            print(len(I), 'of', len(gaia), 'Gaia and', len(ps1), 'PS1 stars matched')

            # Merged = PS1 + unmatched Gaia
            if len(I):
                # Merge columns for the matched stars
                for c in gaia.get_columns():
                    G = gaia.get(c)
                    # If column exists in both (eg, ra_now, dec_now), override
                    # the PS1 value with the Gaia value; except for "photom".
                    if c in ps1.get_columns():
                        X = ps1.get(c)
                    else:
                        X = np.zeros(len(ps1), G.dtype)
                    X[J] = G[I]
                    ps1.set(c, X)
                # unmatched Gaia stars
                unmatched = np.ones(len(gaia), bool)
                unmatched[I] = False
                gaia.cut(unmatched)
                del unmatched

            refs.append(ps1)

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
        phot = self.tractor_fit_sources(refs.ra_now, refs.dec_now, refs.flux0,
                                        fit_img, ierr, psf)
        print('Got photometry results for', len(phot), 'reference stars')
        if len(phot) == 0:
            return self.return_on_error('No photometry available',ccds=ccds)

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
        phot.instpsfmag[ok] = -2.5*np.log10(phot.flux[ok] / self.exptime)
        # Uncertainty on psfmag
        phot.dpsfmag = np.zeros(len(phot), np.float32)
        phot.dpsfmag[ok] = np.abs((-2.5 / np.log(10.)) * phot.dflux[ok] / phot.flux[ok])

        H,W = self.bitmask.shape
        phot.bitmask = self.bitmask[np.clip(phot.y1, 0, H-1).astype(int),
                                    np.clip(phot.x1, 0, W-1).astype(int)]

        phot.psfmag = np.zeros(len(phot), np.float32)

        print('Flux S/N min/median/max: %.1f / %.1f / %.1f' %
              (phot.flux_sn.min(), np.median(phot.flux_sn), phot.flux_sn.max()))
        # Note that this is independent of whether we have reference stars
        # (eg, will work where we don't have PS1)
        nphotom = np.sum(phot.flux_sn > 5.)

        dmag = refs.legacy_survey_mag - phot.instpsfmag
        maglo, maghi = MAGLIM[self.band]
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
            kext = self.extinction(self.band)
            transp = 10.**(-0.4 * (-dzpt - kext * (self.airmass - 1.0)))

            print('Number of stars used for zeropoint median %d' % nphotom)
            print('Zeropoint %.4f' % zptmed)
            print('Offset from nominal: %.4f' % dzpt)
            print('Scatter: %.4f' % zptstd)
            print('Transparency %.4f' % transp)

            ok = (phot.instpsfmag != 0)
            phot.psfmag[ok] = phot.instpsfmag[ok] + zptmed

            from tractor.brightness import NanoMaggies
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
        phot.expnum += self.expnum
        phot.ccdname = np.array([self.ccdname] * len(phot))
        phot.filter  = np.array([self.band] * len(phot))
        # pad ccdname to 4 characters (Bok: "CCD1")
        if len(self.ccdname) < 4:
            phot.ccdname = phot.ccdname.astype('S4')

        phot.exptime = np.zeros(len(phot), np.float32) + self.exptime
        phot.gain    = np.zeros(len(phot), np.float32) + self.gain
        phot.airmass = np.zeros(len(phot), np.float32) + self.airmass

        import photutils
        apertures_arcsec_diam = [6, 7, 8]
        for arcsec_diam in apertures_arcsec_diam:
            ap = photutils.CircularAperture(np.vstack((phot.x_fit, phot.y_fit)).T,
                                            arcsec_diam / 2. / self.pixscale)
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
        ccds['transp'] = transp
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
            'legacy_survey_mag',
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
        if self.verboseplots:
            self.make_plots(phot,dmag,ccds['zpt'],ccds['transp'])
            t0= ptime('made-plots',t0)
        return ccds, phot

    def apply_amp_correction(self, img, invvar):
        pass

    def apply_amp_correction_northern(self, img, invvar):
        from legacypipe.image import apply_amp_correction_northern
        x0 = y0 = 0
        if self.slc is not None:
            sy,sx = self.slc
            x0 = sx.start
            y0 = sy.start
        apply_amp_correction_northern(self.camera, self.band, self.expnum,
                                      self.ccdname, self.mjd_obs,
                                      img, invvar, x0, y0)

    def ps1_to_observed(self, ps1):
        colorterm = self.colorterm_ps1_to_observed(ps1.median, self.band)
        ps1band = ps1cat.ps1band[self.band]
        return ps1.median[:, ps1band] + np.clip(colorterm, -1., +1.)

    def get_sdss_cuts(self, sdss):
        """Returns bool of SDSS sources to keep
        sdss: catalogue with SDSS data
        """
        return np.ones(len(sdss), bool)

    def sdss_to_observed(self, sdss):
        colorterm = self.colorterm_sdss_to_observed(sdss.psfmag, self.band)
        band = sdsscat.sdssband[self.band]
        return sdss.psfmag[:, band] + np.clip(colorterm, -1., +1.)

    def colorterm_sdss_to_observed(self, sdssstars, band):
        """sdssstars: sdss.psfmag 2D array of mag for each band"""
        from legacypipe.ps1cat import sdss_to_decam
        return sdss_to_decam(sdssstars, band)

    def get_splinesky_merged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-splinesky.fits'
        fn = os.path.join(self.calibdir, 'sky', basedir, fn)
        return fn

    def get_splinesky_unmerged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-' + self.ext + '-splinesky.fits'
        fn = os.path.join(self.calibdir, 'sky-single', basedir, base, fn)
        return fn

    def get_splinesky(self):
        # Find splinesky model file and read it
        from tractor.utils import get_class_from_name

        # Look for merged file
        fn = self.get_splinesky_merged_filename()
        if os.path.exists(fn):
            T = fits_table(fn)
            if validate_version(fn, 'table', self.expnum, self.plver,
                                       self.plprocid, data=T, quiet=self.quiet):
                I, = np.nonzero((T.expnum == self.expnum) *
                                np.array([c.strip() == self.ext for c in T.ccdname]))
                if len(I) == 1:
                    Ti = T[I[0]]
                    # Remove any padding
                    h,w = Ti.gridh, Ti.gridw
                    Ti.gridvals = Ti.gridvals[:h, :w]
                    Ti.xgrid = Ti.xgrid[:w]
                    Ti.ygrid = Ti.ygrid[:h]
                    skyclass = Ti.skyclass.strip()
                    clazz = get_class_from_name(skyclass)
                    fromfits = getattr(clazz, 'from_fits_row')
                    sky = fromfits(Ti)
                    return sky

        # Look for single-CCD file
        fn = self.get_splinesky_unmerged_filename()
        if not os.path.exists(fn):
            return None
        hdr = read_primary_header(fn)
        if not validate_version(fn, 'primaryheader', self.expnum, self.plver,
                                       self.plprocid, data=hdr, quiet=self.quiet):
            return None
        try:
            skyclass = hdr['SKY']
        except NameError:
            raise NameError('SKY not in header: skyfn={}'.format(fn))
        clazz = get_class_from_name(skyclass)
        if getattr(clazz, 'from_fits', None) is not None:
            fromfits = getattr(clazz, 'from_fits')
            sky = fromfits(fn, hdr)
        else:
            fromfits = getattr(clazz, 'fromFitsHeader')
            sky = fromfits(hdr, prefix='SKY_')
        return sky

    def tractor_fit_sources(self, ref_ra, ref_dec, ref_flux, img, ierr,
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
            _,x,y = self.wcs.radec2pixelxy(ra, dec)
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
            tr.optimize_forced_photometry()
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
            for step in range(50):
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
        cal.ra_fit,cal.dec_fit = self.wcs.pixelxy2radec(cal.x1 + 1, cal.y1 + 1)
        return cal

    def get_psfex_merged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-psfex.fits'
        fn = os.path.join(self.calibdir, 'psfex', basedir, fn)
        return fn

    def get_psfex_unmerged_filename(self):
        basefn = os.path.basename(self.fn_base)
        basedir = os.path.dirname(self.fn_base)
        base = basefn.split('.')[0]
        fn = base + '-' + self.ext + '-psfex.fits'
        fn = os.path.join(self.calibdir, 'psfex-single', basedir, base, fn)
        return fn

    def get_psfex_model(self):
        import tractor

        # Look for merged PsfEx file
        fn = self.get_psfex_merged_filename()
        #print('Looking for PsfEx file', fn)
        if os.path.exists(fn):
            #print('Reading psfex-merged {}'.format(fn))
            T = fits_table(fn)
            if validate_version(fn, 'table', self.expnum, self.plver,
                                       self.plprocid, data=T, quiet=self.quiet):
                I, = np.nonzero((T.expnum == self.expnum) *
                                np.array([c.strip() == self.ext for c in T.ccdname]))
                if len(I) == 1:
                    Ti = T[I[0]]
                    # Remove any padding
                    degree = Ti.poldeg1
                    # number of terms in polynomial
                    ne = (degree + 1) * (degree + 2) // 2
                    Ti.psf_mask = Ti.psf_mask[:ne, :Ti.psfaxis1, :Ti.psfaxis2]
                    psfex = tractor.PsfExModel(Ti=Ti)
                    psf = tractor.PixelizedPsfEx(None, psfex=psfex)
                    psf.fwhm = Ti.psf_fwhm
                    psf.header = {}
                    return psf

        # Look for single-CCD PsfEx file
        fn = self.get_psfex_unmerged_filename()
        if not os.path.exists(fn):
            return None
        hdr = read_primary_header(fn)
        if not validate_version(fn, 'primaryheader', self.expnum, self.plver,
                                       self.plprocid, data=hdr, quiet=self.quiet):
            return None

        hdr = fitsio.read_header(fn, ext=1)
        psf = tractor.PixelizedPsfEx(fn)
        psf.header = hdr
        psf.fwhm = hdr['PSF_FWHM']
        return psf

    def make_plots(self,stars,dmag,zpt,transp):
        '''stars -- stars table'''
        import pylab as plt
        stars.radiff  = 3600. * (stars.ra_fit  - stars.ra) * np.cos(np.deg2rad(stars.dec))
        stars.decdiff = 3600. * (stars.dec_fit - stars.dec)
        _,ax=plt.subplots(1,2,figsize=(10,4))
        plt.subplots_adjust(wspace=0.2,bottom=0.2,right=0.8)
        for key in ['astrom_gaia','photom']:
            if key == 'astrom_gaia':
                ax[0].scatter(stars.radiff, stars.decdiff)
                xlab=ax[0].set_xlabel(r'$\Delta Ra$ (Gaia - CCD)')
                ylab=ax[0].set_ylabel(r'$\Delta Dec$ (Gaia - CCD)')
            elif key == 'astrom_ps1':
                raise ValueError('not needed')
            elif key == 'photom':
                ax[1].hist(dmag)
                xlab=ax[1].set_xlabel('PS1 - AP mag (main seq, 2.5 clipping)')
                ylab=ax[1].set_ylabel('Number of Stars')
        # List key numbers
        ax[1].text(1.02, 1.,r'$\Delta$ Ra,Dec',\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=12)
        ax[1].text(1.02, 0.9,r'  Median: %.4f,%.4f' % \
                  (np.median(stars.radiff),np.median(stars.decdiff)),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.80,'  RMS: %.4f,%.4f' % \
                  (getrms(stars.radiff),getrms(stars.decdiff)),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.7,'PS1-CCD Mag',\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=12)
        ax[1].text(1.02, 0.6,'  Median:%.4f,%.4f' % \
                  (np.median(dmag),np.std(dmag)),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.5,'  Stars: %d' % len(dmag),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.4,'  Zpt=%.4f' % zpt,\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.3,'  Transp=%.4f' % transp,\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        # Save
        fn = 'qa_%i_%s.png' % (stars.expnum[0], str(stars.ccdname[0]).strip())
        plt.savefig(fn,bbox_extra_artists=[xlab,ylab])
        plt.close()
        print('Wrote %s' % fn)

    def run_calibs(self, survey, ext, psfex=True, splinesky=True, read_hdu=True,
                   plots=False, survey_blob_mask=None, survey_zeropoints=None):
        '''
        survey_zeropoints: LegacySurveyData object to use for fetching the
        zeropoint for this CCD, which is used for subtracting stellar halos
        and SGA galaxies.
        '''
        # Initialize with some basic data
        self.set_hdu(ext)

        ccd = FakeCCD()
        ccd.image_filename = self.fn_base
        ccd.image_hdu = self.image_hdu
        ccd.expnum = self.expnum
        ccd.ccdname = self.ccdname
        ccd.filter = self.band
        ccd.exptime = self.exptime
        ccd.camera = self.camera
        ccd.ccdzpt = 0. # <-- we update this below if survey_zeropoints is available.
        ccd.ccdraoff = 0.
        ccd.ccddecoff = 0.
        ccd.fwhm = self.fwhm_cp
        ccd.propid = self.propid
        # fake -- the image.py class only uses the CD matrix to compute
        # the pixel scale, so this is ok
        ccd.cd1_1 = ccd.cd2_2 = self.pixscale / 3600.
        ccd.cd1_2 = ccd.cd2_1 = 0.
        ccd.pixscale = self.pixscale
        ccd.mjd_obs = self.mjd_obs
        ccd.width = self.width
        ccd.height = self.height
        ccd.arawgain = self.gain
        ccd.sig1 = 0.
        ccd.plver = self.plver
        ccd.procdate = self.procdate
        ccd.plprocid = self.plprocid

        if not self.goodWcs:
            print('WCS Failed on CCD {}, skipping calibs'.format(self.ccdname))
            return ccd
        if self.exptime == 0:
            print('Exptime = 0 on CCD {}, skipping calibs'.format(self.ccdname))
            return ccd
        do_psf = False
        do_sky = False
        if psfex and self.get_psfex_model() is None:
            do_psf = True
        if splinesky and self.get_splinesky() is None:
            do_sky = True

        if (not do_psf) and (not do_sky):
            # Nothing to do!
            return ccd

        # Check for all-zero weight maps
        bitmask = self.read_bitmask()
        wt = self.read_weight(bitmask=bitmask)
        if np.all(wt == 0):
            print('Weight map is all zero on CCD {} -- skipping'.format(self.ccdname))
            return ccd

        ps = None
        if plots:
            import matplotlib
            matplotlib.use('Agg')
            from astrometry.util.plotutils import PlotSequence
            ps = PlotSequence('%s-%i-%s' % (self.camera, self.expnum, self.ccdname))

        # Only do stellar halo subtraction if we have a zeropoint (via --zeropoint-dir)
        # Note that this is the only place we use survey_zeropoints; it does not get
        # passed to image.run_calibs().
        have_zpt = False
        if survey_zeropoints is not None:
            ccds = survey_zeropoints.find_ccds(
                expnum=self.expnum, ccdname=self.ccdname, camera=self.camera)
            if len(ccds) != 1:
                print('WARNING, did not find a zeropoint for', self.fn_base,
                      'by camera', self.camera, 'expnum', self.expnum,
                      'ccdname', self.ccdname)
            else:
                ccd.ccdzpt = ccds[0].ccdzpt
                ccd.ccdraoff = ccds[0].ccdraoff
                ccd.ccddecoff = ccds[0].ccddecoff
                if ccd.ccdzpt == 0.:
                    print('WARNING, found zeropoint=0 for', self.fn_base,
                          'by camera', self.camera, 'expnum', self.expnum,
                          'ccdname', self.ccdname)
                else:
                    have_zpt = True

        # Set sig1 after (possibly) updating zeropoint!
        from tractor.brightness import NanoMaggies
        zpscale = NanoMaggies.zeropointToScale(ccd.ccdzpt)
        medweight = np.median(wt[(wt > 0) * (bitmask == 0)])
        # note, read_weight() for Mosaic and 90prime scales by 1/exptime**2
        ccd.sig1 = (1. / np.sqrt(medweight)) / self.exptime / zpscale

        im = survey.get_image_object(ccd)
        git_version = get_git_version(dirnm=os.path.dirname(legacypipe.__file__))
        im.run_calibs(psfex=do_psf, sky=do_sky, splinesky=True,
                      git_version=git_version, survey=survey, ps=ps,
                      survey_blob_mask=survey_blob_mask,
                      halos=have_zpt,
                      subtract_largegalaxies=have_zpt)
        return ccd

class FakeCCD(object):
    pass

class DecamMeasurer(Measurer):
    '''DECam CP units: ADU

    Formerly, we converted DECam images to e- (multiplied images by
    gain), but this was annoying for Eddie's ubercal processing; the
    photom file reported fluxes included that gain factor, which
    wanders around over time; removing it resulted in much smaller
    scatter.
    '''
    def __init__(self, *args, **kwargs):
        self.camera = 'decam'
        super(DecamMeasurer, self).__init__(*args, **kwargs)
        self.pixscale = get_pixscale(self.camera)

        # /global/homes/a/arjundey/idl/pro/observing/decstat.pro
        self.zp0 =  dict(g = 26.610,
                         r = 26.818,
                         z = 26.484,
                         # u from Arjun 2021-03-17, based on DECosmos-to-SDSS
                         u = 23.3205,
                         # i,Y from DESY1_Stripe82 95th percentiles
                         i=26.758,
                         Y=25.321,
                         N501=23.812,
                         N673=24.151,
        ) # e/sec
        # extinction per airmass
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06,
                          # From Arjun 2021-03-17 based on DECosmos (calib against SDSS)
                          u = 0.63,
                          #i, Y totally made up
                          i=0.08, Y=0.06)

    def get_extension_list(self, fn, debug=False):
        if debug:
            return ['N4']
        hdu = fitsio.FITS(fn)
        extlist = [hdu[i].get_extname() for i in range(1,len(hdu))]
        return extlist

    def good_wcs(self, primhdr):
        wcscal = primhdr.get('WCSCAL', '').strip().lower()
        if wcscal.startswith('success'):  # success / successful
            return True
        # Frank's work-around for some with incorrect WCSCAL=Failed (DR9 re-reductions)
        return primhdr.get('SCAMPFLG') == 0

    def get_site(self):
        from astropy.coordinates import EarthLocation
        # zomg astropy's caching mechanism is horrific
        # return EarthLocation.of_site('ctio')
        from astropy.units import m
        from astropy.utils import iers
        iers.conf.auto_download = False
        return EarthLocation(1814304. * m, -5214366. * m, -3187341. * m)

    def get_good_image_subregion(self):
        x0,x1,y0,y1 = None,None,None,None

        # Handle 'glowing' edges in DES r-band images
        # aww yeah
        # if self.band == 'r' and (
        #         ('DES' in self.imgfn) or ('COSMOS' in self.imgfn) or
        #         (self.mjdobs < DecamImage.glowmjd)):
        #     # Northern chips: drop 100 pix off the bottom
        #     if 'N' in self.ccdname:
        #         print('Clipping bottom part of northern DES r-band chip')
        #         y0 = 100
        #     else:
        #         # Southern chips: drop 100 pix off the top
        #         print('Clipping top part of southern DES r-band chip')
        #         y1 = self.height - 100

        # Clip the bad half of chip S7.
        # The left half is OK.
        if self.ccdname == 'S7':
            print('Clipping the right half of chip S7')
            x1 = 1023
        return x0,x1,y0,y1

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band

    def get_gain(self, hdr):
        return np.average((hdr['GAINA'],hdr['GAINB']))

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_decam
        return ps1_to_decam(ps1stars, band)

    def scale_image(self, img):
        return img

    def scale_weight(self, img):
        return img

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion

    def remap_bitmask(self, mask):
        from legacypipe.image import remap_dq_cp_codes
        from legacypipe.decam import decam_has_dq_codes
        plver = self.primhdr['PLVER']
        if decam_has_dq_codes(plver):
            mask = remap_dq_cp_codes(mask)
        return mask

class MegaPrimeMeasurer(Measurer):
    def __init__(self, *args, **kwargs):
        self.camera = 'megaprime'
        super(MegaPrimeMeasurer, self).__init__(*args, **kwargs)
        self.pixscale = get_pixscale(self.camera)

        # # /global/homes/a/arjundey/idl/pro/observing/decstat.pro
        ### HACK!!!
        self.zp0 =  dict(g = 26.610,
                         r = 26.818,
                         z = 26.484,
                         # Totally made up
                         u = 26.610,
                         )
                         #                  # i,Y from DESY1_Stripe82 95th percentiles
                         #                  i=26.758, Y=25.321) # e/sec
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06,
                          # Totally made up
                          u = 0.24)
        #                   #i, Y totally made up
        #                   i=0.08, Y=0.06)
        # --> e/sec

        self.primhdr['WCSCAL'] = 'success'
        self.goodWcs = True

    def get_extension_list(self, fn, debug=False):
        if debug:
            return ['ccd03']
        hdu = fitsio.FITS(fn)
        return [hdu[i].get_extname() for i in range(1,len(hdu))]

    def get_ut(self, primhdr):
        return primhdr['UTC-OBS']

    def get_radec_bore(self, primhdr):
        return primhdr['RA_DEG'], primhdr['DEC_DEG']

    def ps1_to_observed(self, ps1):
        # u->g
        ps1band = dict(u='g').get(self.band, self.band)
        ps1band_index = ps1cat.ps1band[ps1band]
        colorterm = self.colorterm_ps1_to_observed(ps1.median, self.band)
        return ps1.median[:, ps1band_index] + np.clip(colorterm, -1., +1.)

    def get_band(self):
        band = self.primhdr['FILTER'][0]
        return band

    def get_gain(self,hdr):
        return hdr['GAIN']

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacypipe.ps1cat import ps1_to_decam
        print('HACK -- using DECam color term for CFHT!!')
        if band == 'u':
            print('HACK -- using g-band color term for u band!')
            band = 'g'
        return ps1_to_decam(ps1stars, band)

    def scale_image(self, img):
        return img.astype(np.float32)

    def scale_weight(self, img):
        return img

    def get_wcs(self):
        ### FIXME -- no distortion solution in here
        # from astrometry.util.util import Tan
        # return Tan(self.hdr)

        # "pitcairn" reductions have PV header cards (CTYPE is still RA---TAN)
        return wcs_pv2sip_hdr(self.hdr)

    def read_weight(self, clip=True, clipThresh=0.01, **kwargs):
        # Just estimate from image...
        img,_ = self.read_image()
        print('Image:', img.shape, img.dtype)

        # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        invvar = (1. / sig1**2)
        return np.zeros_like(img) + invvar

    def read_bitmask(self):
        # from legacypipe/cfht.py
        dqfn = 'cfis/test.mask.0.40.01.fits'
        if self.slc is not None:
            mask = fitsio.FITS(dqfn)[self.ext][self.slc]
        else:
            mask = fitsio.read(dqfn, ext=self.ext)
        # This mask is a 16-bit image but has values 0=bad, 1=good.  Flip.
        return (1 - mask).astype(np.int16)

class Mosaic3Measurer(Measurer):
    '''Class to measure a variety of quantities from a single Mosaic3 CCD.
    UNITS: e-/s'''
    def __init__(self, *args, **kwargs):
        self.camera = 'mosaic' # this has to appear before super to recompute airmass
        super(Mosaic3Measurer, self).__init__(*args, **kwargs)
        self.pixscale = get_pixscale(self.camera)

        self.zp0 = dict(z = 26.552,
                        D51 = 24.351, # from obsbot
        )
        self.k_ext = dict(z = 0.06,
                          D51 = 0.211, # from obsbot
        )

    def apply_amp_correction(self, img, invvar):
        self.apply_amp_correction_northern(img, invvar)

    def get_fwhm(self, hdr, hdu):
        for key in ['SEEINGP1', 'SEEINGP']:
            if key in hdr:
                return hdr[key]
        raise RuntimeError('No FWHM key in header')

    def get_extension_list(self, fn, debug=False):
        if debug:
            return ['CCD2']
        hdu = fitsio.FITS(fn)
        extlist = [hdu[i].get_extname().upper().strip()
                   for i in range(1,len(hdu))]
        return extlist

    def get_expnum(self, primhdr):
        if 'EXPNUM' in primhdr and primhdr['EXPNUM'] is not None:
            return primhdr['EXPNUM']
        # At the beginning of the survey, eg 2016-01-24, the EXPNUM
        # cards are blank.  Fake up an expnum like 160125082555
        # (yymmddhhmmss), same as the CP filename.
        # OBSID   = 'kp4m.20160125T082555' / Observation ID
        obsid = primhdr['OBSID']
        obsid = obsid.strip().split('.')[1]
        obsid = obsid.replace('T', '')
        obsid = int(obsid[2:], 10)
        print('Faked up EXPNUM', obsid)
        return obsid

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        band = {'zd':'z'}.get(band, band) # zd --> z
        return band

    def get_gain(self,hdr):
        return hdr['GAIN']

    def ps1_to_observed(self, ps1):
        if self.band == 'D51':
            ps1band = ps1cat.ps1band['g']
            return ps1.median[:, ps1band]
        return super(Mosaic3Measurer, self).ps1_to_observed(ps1)

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime

    def scale_weight(self, img):
        return img / (self.exptime**2)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion

    def get_site(self):
        from astropy.coordinates import EarthLocation
        from astropy.units import m
        from astropy.utils import iers
        iers.conf.auto_download = False
        return EarthLocation(-1994503. * m, -5037539. * m, 3358105. * m)

    def remap_bitmask(self, mask):
        from legacypipe.image import remap_dq_cp_codes
        return remap_dq_cp_codes(mask)


class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.
    UNITS -- CP e-/s'''
    def __init__(self, *args, **kwargs):
        self.camera = '90prime'
        super(NinetyPrimeMeasurer, self).__init__(*args, **kwargs)
        self.pixscale = get_pixscale(self.camera)

        # Nominal zeropoints, sky brightness, and extinction values (taken from
        # rapala.ninetyprime.boketc.py)
        # /global/homes/a/arjundey/idl/pro/observing/bokstat.pro
        self.zp0 =  dict(g = 26.93,r = 27.01,z = 26.552) # ADU/sec
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06)

    def apply_amp_correction(self, img, invvar):
        self.apply_amp_correction_northern(img, invvar)

    def get_fwhm(self, hdr, hdu):
        for key in ['SEEINGP1', 'SEEINGP']:
            if key in hdr:
                return hdr[key]
        raise RuntimeError('No FWHM key in header')

    def get_extension_list(self, fn, debug=False):
        if debug:
            return ['CCD1']
        hdu = fitsio.FITS(fn)
        extlist = [hdu[i].get_extname().upper().strip()
                   for i in range(1,len(hdu))]
        return extlist

    def get_expnum(self, primhdr):
        """converts 90prime header key DTACQNAM into the unique exposure number"""
        # /descache/bass/20160710/d7580.0144.fits --> 75800144
        import re
        base= (os.path.basename(primhdr['DTACQNAM'])
               .replace('.fits','')
               .replace('.fz',''))
        return int( re.sub(r'([a-z]+|\.+)','',base) )

    def get_gain(self,hdr):
        return 1.4

    def get_ut(self, primhdr):
        return primhdr['UT']

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band.replace('bokr', 'r')

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

    def scale_image(self, img):
        '''Convert image from electrons/sec to electrons.'''
        return img * self.exptime

    def scale_weight(self, img):
        return img / (self.exptime**2)

    def remap_invvar(self, invvar, primhdr, img, dq):
        return self.remap_invvar_shotnoise(invvar, primhdr, img, dq)

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion

    def remap_bitmask(self, mask):
        from legacypipe.image import remap_dq_cp_codes
        return remap_dq_cp_codes(mask)

def measure_image(img_fn, mp, image_dir='images', run_calibs_only=False,
                  just_measure=False,
                  survey=None, psfex=True, camera=None, **measureargs):
    '''Wrapper on the camera-specific classes to measure the CCD-level data on all
    the FITS extensions for a given set of images.
    '''
    t0 = Time()

    quiet = measureargs.get('quiet', False)

    img_fn_full = os.path.join(image_dir, img_fn)

    print('measure_image', img_fn)
    print('survey', survey)
    print('camera', camera)

    imgclass = survey.image_class_for_camera(camera)
    print('Image class:', imgclass)

    image_hdu = measureargs.get('image_hdu', None)
    
    if just_measure:
        #print('measureargs:', measureargs)
        return survey.get_image_object(None, camera=camera, image_fn=img_fn,
                                       image_hdu=image_hdu)
    

    ma = measureargs.copy()
    ma.pop('bad_expids', None)
    print('measureargs:')
    for k,v in ma.items():
        print('  ', k, '=', v)
    
    img = survey.get_image_object(None, camera=camera,
                                  image_fn=img_fn, image_hdu=image_hdu)
    print('Got image object', img)
    # Confirm camera field.
    cammap = {'mosaic3':'mosaic'}
    assert(img.camera == camera)

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

    # Validate the splinesky and psfex merged files, and (re)make them if
    # they're missing.
    if splinesky:
        if validate_version(img.get_splinesky_merged_filename(),
                            'table', img.expnum, img.plver, img.plprocid, quiet=quiet):
            splinesky = False
    if psfex:
        if validate_version(img.get_psfex_merged_filename(),
                            'table', img.expnum, img.plver, img.plprocid, quiet=quiet):
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

    if run_calibs_only:
        return

    rtns = mp.map(run_one_ext, [(img, ext, survey, splinesky, measureargs['debug'],
                                 measureargs['sdss_photom'])
                                for ext in extlist])
    
    for ext,(ccd,photom) in zip(extlist,rtns):
        if ccd is not None:
            all_ccds.append(ccd)
        if photom is not None:
            all_photom.append(photom)

    # Compute the median zeropoint across all the CCDs.
    all_ccds = vstack(all_ccds)

    if len(all_photom):
        all_photom = merge_tables(all_photom)
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
    from tractor.brightness import NanoMaggies

    (img_fn, camera, survey, ext, psfex, splinesky, plots, survey_blob_mask,
     survey_zeropoints) = X
    img = survey.get_image_object(None, camera=camera,
                                  image_fn=img_fn, image_hdu=ext)

    # FIXME
    # - !goodWcs
    # - exptime==0
    # - all-zero weight map

    # bitmask,dqhdr = img.read_dq(header=True) #read_bitmask()
    # if bitmask is not None:
    #     bitmask = img.remap_dq(bitmask, dqhdr)
    # wt = img.read_invvar(dq=bitmask)
    # zpscale = NanoMaggies.zeropointToScale(img.ccdzpt)
    # medweight = np.median(wt[(wt > 0) * (bitmask == 0)])
    # img.sig1 = (1. / np.sqrt(medweight)) / img.exptime / zpscale
    
    do_psf = False
    do_sky = False
    #if psfex and img.get_psfex_model() is None:
    if psfex:
        try:
            psf = img.read_psf_model(0., 0., pixPsf=True)
        except:
            import traceback
            print('Failed trying to read existing PSF model:')
            traceback.print_exc()
        do_psf = True
    #if splinesky and img.get_splinesky() is None:
    if splinesky:
        try:
            sky = img.read_sky_model()
        except:
            import traceback
            print('Failed trying to read existing sky model:')
            traceback.print_exc()
        do_sky = True

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

    # Set sig1 after (possibly) updating zeropoint!
    # from tractor.brightness import NanoMaggies
    # zpscale = NanoMaggies.zeropointToScale(img.ccdzpt)
    # medweight = np.median(wt[(wt > 0) * (bitmask == 0)])
    # # note, read_weight() for Mosaic and 90prime scales by 1/exptime**2
    # img.sig1 = (1. / np.sqrt(medweight)) / img.exptime / zpscale

    git_version = get_git_version(dirnm=os.path.dirname(legacypipe.__file__))
    ps = None
    
    img.run_calibs(psfex=do_psf, sky=do_sky, splinesky=True,
                   git_version=git_version, survey=survey, ps=ps,
                   survey_blob_mask=survey_blob_mask,
                   halos=have_zpt,
                   subtract_largegalaxies=have_zpt)
    return img
    # return img.run
    # return measure.run_calibs(survey, ext, psfex=psfex, splinesky=splinesky,
    #                           plots=plots,
    #                           survey_blob_mask=survey_blob_mask,
    #                           survey_zeropoints=survey_zeropoints)

def run_one_ext(X):
    img, ext, survey, splinesky, debug, sdss_photom = X

    img = survey.get_image_object(None, camera=img.camera,
                                  image_fn=img.image_filename, image_hdu=ext)

    rtns = run_zeropoints(img, splinesky=splinesky, sdss_photom=sdss_photom)
    #rtns = measure.run(ext, splinesky=splinesky, survey=survey, save_xy=debug,
    #                   sdss_photom=sdss_photom)
    return rtns

class outputFns(object):
    def __init__(self, imgfn, outdir, camera, image_dir='images', debug=False):
        """Assigns filename, makes needed dirs

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
          survey=None, run_calibs_only=False, **measureargs):
    '''Generate a legacypipe-compatible (survey) CCDs file for a given image.
    '''
    t0 = Time()

    results = measure_image(imgfn, mp, survey=survey, run_calibs_only=run_calibs_only,
                            **measureargs)
    if run_calibs_only:
        return

    if len(results) == 0:
        print('All CCDs bad, quitting.')
        return

    ccds, photom, img = results
    t0 = ptime('measure_image',t0)

    primhdr = img.read_image_primary_header()

    hdr = fitsio.FITSHDR()
    for key in ['AIRMASS', 'OBJECT', 'TELESCOP', 'INSTRUME', 'EXPTIME', 'DATE-OBS',
                'MJD-OBS', 'PROGRAM', 'OBSERVER', 'PROPID', 'FILTER', 'HA', 'ZD',
                'AZ', 'DOMEAZ', 'HUMIDITY', 'PLVER', ]:
        if not key in primhdr:
            continue
        v = primhdr[key]
        if type(v) == str:
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
    create_annotated_table(T, annfn, measureargs['camera'], survey, mp)

    t0 = ptime('write-results-to-fits',t0)

def parse_coords(s):
    '''stackoverflow:
    https://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples'''
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")

def get_parser():
    '''return parser object, tells it what options to look for
    options can come from a list of strings or command line'''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                     description='Generate a legacypipe-compatible (survey) CCDs file \
                                                  from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime','megaprime'],action='store',required=True)
    parser.add_argument('--image',action='store',default=None,help='relative path to image starting from decam,bok,mosaicz dir',required=False)
    parser.add_argument('--image_list',action='store',default=None,help='text file listing multiples images in same was as --image',required=False)
    parser.add_argument('--image_dir', type=str, default='images', help='Directory containing the imaging data (analogous to legacypipe.LegacySurveyData.image_dir).')
    parser.add_argument('--outdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--sdss-photom', default=False, action='store_true',
                        help='Use SDSS rather than PS-1 for photometric cal.')
    parser.add_argument('--debug', action='store_true', default=False, help='Write additional files and plots for debugging')
    parser.add_argument('--choose_ccd', action='store', default=None, help='forced to use only the specified ccd')
    parser.add_argument('--logdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--verboseplots', action='store_true', default=False, help='use to plot FWHM Moffat PSF fits to the 20 brightest stars')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use this option when deriving the photometric transformation equations.')
    parser.add_argument('--plots', action='store_true', help='Calib plots?')
    parser.add_argument('--nproc', type=int,action='store',default=1,
                        help='set to > 1 if using legacy-zeropoints-mpiwrapper.py')
    parser.add_argument('--run-calibs-only', default=False, action='store_true',
                        help='Only ensure calib files exist, do not compute zeropoints.')
    parser.add_argument('--no-splinesky', dest='splinesky', default=True, action='store_false',
                        help='Do not use spline sky model for sky subtraction?')
    parser.add_argument('--blob-mask-dir', type=str, default=None,
                        help='The base directory to search for blob masks during sky model construction')
    parser.add_argument('--zeropoints-dir', type=str, default=None,
                        help='The base directory to search for survey-ccds files for subtracting star halos before doing sky calibration.')
    parser.add_argument('--calibdir', default=None, action='store',
                        help='if None will use LEGACY_SURVEY_DIR/calib, e.g. /global/cscratch1/sd/desiproc/dr5-new/calib')
    parser.add_argument('--threads', default=None, type=int,
                        help='Multiprocessing threads (parallel by HDU)')
    parser.add_argument('--quiet', default=False, action='store_true', help='quiet down')
    parser.add_argument('--overhead', type=str, default=None, help='Print python startup time since the given date.')
    return parser


def main(image_list=None,args=None):
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

    import logging
    #if quiet:
    lvl = logging.INFO
    #else:
    #    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if measureargs['calibdir'] is None:
        cal = os.getenv('LEGACY_SURVEY_DIR',None)
        if cal is not None:
            measureargs['calibdir'] = os.path.join(cal, 'calib')

    camera = measureargs['camera']
    image_dir = measureargs['image_dir']

    # survey = FakeLegacySurveyData()
    survey = LegacySurveyData()
    survey.imagedir = image_dir
    survey.calibdir = measureargs.get('calibdir')
    print('Survey image dir:', survey.imagedir)
    print('Survey calib dir:', survey.calibdir)
    measureargs.update(survey=survey)

    if camera in ['mosaic', 'decam', 'megaprime', '90prime']:
        if camera in ['mosaic', 'decam', '90prime']:
            from legacyzpts.psfzpt_cuts import read_bad_expid

            fn = resource_filename('legacyzpts', 'data/{}-bad_expid.txt'.format(camera))
            if os.path.isfile(fn):
                print('Reading {}'.format(fn))
                measureargs.update(bad_expid=read_bad_expid(fn))
            else:
                print('No bad exposure file found for camera {}'.format(camera))

        cal = measureargs.get('calibdir')
        if cal is not None:
            survey.calibdir = cal

        try:
            from legacypipe.cfht import MegaPrimeImage
            survey.image_typemap['megaprime'] = MegaPrimeImage
        except:
            print('MegaPrimeImage class not found')
            raise IOError

    outdir = measureargs.pop('outdir')
    for ii, imgfn in enumerate(image_list):
        print('Working on image {}/{}: {}'.format(ii+1, nimage, imgfn))

        # Check if the outputs are done and have the correct data model.
        F = outputFns(imgfn, outdir, camera, image_dir=image_dir,
                      debug=measureargs['debug'])

        measure = measure_image(F.imgfn, None, just_measure=True, image_hdu=None,
                                **measureargs)
        psffn = measure.get_psfex_merged_filename()
        skyfn = measure.get_splinesky_merged_filename()

        ann_ok, psf_ok, sky_ok = [validate_version(
            fn, 'table', measure.expnum, measure.plver,
            measure.plprocid, quiet=quiet)
            for fn in [F.annfn, psffn, skyfn]]

        if measureargs['run_calibs_only'] and psf_ok and sky_ok:
            print('Already finished {}'.format(psffn))
            print('Already finished {}'.format(skyfn))
            continue

        phot_ok = validate_version(F.photomfn, 'header', measure.expnum,
                                   measure.plver, measure.plprocid,
                                   ext=1, quiet=quiet)

        if ann_ok and phot_ok and psf_ok and sky_ok:
            print('Already finished: {}'.format(F.annfn))
            continue

        # Create the file
        t0 = ptime('before-run',t0)
        runit(F.imgfn, F.photomfn, F.annfn, mp, **measureargs)
        t0 = ptime('after-run',t0)
    tnow = Time()
    print("TIMING:total %s" % (tnow-tbegin,))

from legacypipe.survey import LegacySurveyData
class FakeLegacySurveyData(LegacySurveyData):
    def get_calib_dir(self):
        return self.calibdir
    def get_image_dir(self):
        return self.imagedir

def read_primary_header(fn):
    '''
    Reads the FITS primary header (HDU 0) from the given filename.
    This is just a faster version of fitsio.read_header(fn).
    '''
    if fn.endswith('.gz'):
        return fitsio.read_header(fn)

    # Weirdly, this can be MUCH faster than letting fitsio do it...
    hdr = fitsio.FITSHDR()
    foundEnd = False
    ff = open(fn, 'rb')
    h = b''
    while True:
        hnew = ff.read(32768)
        if len(hnew) == 0:
            # EOF
            ff.close()
            raise RuntimeError('Reached end-of-file in "%s" before finding end of FITS header.' % fn)
        h = h + hnew
        while True:
            line = h[:80]
            h = h[80:]
            #print('Header line "%s"' % line)
            # HACK -- fitsio apparently can't handle CONTINUE.
            # It also has issues with slightly malformed cards, like
            # KEYWORD  =      / no value
            if line[:8] != b'CONTINUE':
                try:
                    hdr.add_record(line.decode())
                except:
                    print('Warning: failed to parse FITS header line: ' +
                          ('"%s"; skipped' % line.strip()))
                    #import traceback
                    #traceback.print_exc()

            if line == (b'END' + b' '*77):
                foundEnd = True
                break
            if len(h) < 80:
                break
        if foundEnd:
            break
    ff.close()
    return hdr


def estimate_sky_from_pixels(img):
    nsigma = 3.
    clip_vals,_,_ = sigmaclip(img, low=nsigma, high=nsigma)
    skymed= np.median(clip_vals)
    skystd= np.std(clip_vals)
    skyimg= np.zeros(img.shape) + skymed
    return skyimg, skymed, skystd

def run_zeropoints(imobj, splinesky=False, sdss_photom=False):
    """Computes photometric and astrometric zeropoints for one CCD.

    Args:

    Returns:
        ccds, stars_photom, stars_astrom
    """
    #
    t0= Time()
    t0= ptime('Measuring CCD=%s from image=%s' % (imobj.ccdname, imobj.image_filename),t0)

    # Initialize
    ccds = _ccds_table(imobj.camera)

    # init_ccd():
    namemap = { 'object': 'obj',
                'filter': 'band',
                'image_hdu': 'hdu',
                'mjd_obs': 'mjdobs',
    }
    #ccds['yshift'] = 'YSHIFT' in self.primhdr
    # 'date_obs',
    # 'ut',
    # 'ccdnum',
    # 'expid', 
    for key in ['image_filename', 'image_hdu', 'camera', 'expnum', 'plver', 'procdate',
                'plprocid', 'ccdname', 'propid', 'exptime',
                'mjd_obs', 'ha',
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

    # if not imobj.goodWcs:
    #     print('WCS Failed on CCD {}'.format(imobj.ccdname))
    #     return imobj.return_on_error(err_message='WCS Failed', ccds=ccds)
    # if imobj.exptime == 0:
    #     print('Exptime = 0 on CCD {}'.format(imobj.ccdname))
    #     return imobj.return_on_error(err_message='Exptime = 0', ccds=ccds)

    # FIXME -- factor out computing sig1 to image.py ?

    # imobj.bitmask = imobj.read_bitmask()
    # weight = imobj.read_weight(bitmask=imobj.bitmask, scale=False)
    # if np.all(weight == 0):
    #     txt = 'All weight-map pixels are zero on CCD {}'.format(imobj.ccdname)
    #     print(txt)
    #     return imobj.return_on_error(txt,ccds=ccds)
    # # bizarro image CP20151119/k4m_151120_040715_oow_zd_v1.fits.fz
    # if np.all(np.logical_or(weight == 0, weight == 1)):
    #     txt = 'All weight-map pixels are zero or one'
    #     print(txt)
    #     return imobj.return_on_error(txt,ccds=ccds)
    # weight = imobj.scale_weight(weight)

    # Quick check for PsfEx file
    psf = imobj.read_psf_model(0., 0., pixPsf=True)
    if psf.psfex.sampling == 0.:
        print('PsfEx model has SAMPLING=0')
        nacc = psf.header.get('ACCEPTED')
        print('PsfEx model number of stars accepted:', nacc)
        return imobj.return_on_error(err_message='Bad PSF model', ccds=ccds)

    bitmask,dqhdr = imobj.read_dq(header=True) #read_bitmask()
    if bitmask is not None:
        bitmask = imobj.remap_dq(bitmask, dqhdr)
    invvar = imobj.read_invvar(dq=bitmask)
    # Compute sig1 before rescaling... well, not quite
    mediv = np.median(invvar[(invvar > 0) * (bitmask == 0)])
    mediv = imobj.scale_weight(mediv)
    imobj.sig1 = (1. / np.sqrt(mediv)) / imobj.exptime
    ccds['sig1'] = imobj.sig1

    img = imobj.read_image()
    invvar = imobj.remap_invvar(invvar, primhdr, img, bitmask)

    invvar = imobj.scale_weight(invvar)
    img = imobj.scale_image(img)

    # # Per-pixel error
    # medweight = np.median(weight[(weight > 0) * (imobj.bitmask == 0)])
    # # We scale the image & weight into ADU, but we want to report
    # # sig1 in nanomaggie-ish units, ie, in per-second units.
    # ccds['sig1'] = (1. / np.sqrt(medweight)) / imobj.exptime
    
    
    t0= ptime('read image',t0)

    # Measure the sky brightness and (sky) noise level.
    zp0 = imobj.nominal_zeropoint(imobj.band)

    # Bunch of sky estimates
    #print('Computing the sky background.')
    sky_img, skymed, skyrms = estimate_sky_from_pixels(img)
    print('sky from median of image = %.2f' % skymed)
    skybr = zp0 - 2.5*np.log10(skymed / imobj.pixscale / imobj.pixscale / imobj.exptime)
    print('Sky brightness: {:.3f} mag/arcsec^2 (assuming nominal zeropoint)'.format(skybr))
    ccds['skyrms'] = skyrms / imobj.exptime # e/sec
    ccds['skycounts'] = skymed / imobj.exptime # [electron/pix]
    ccds['skysb'] = skybr   # [mag/arcsec^2]
    t0= ptime('measure-sky',t0)

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
                #('ps1_objid', np.int64),
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
        return imobj.return_on_error('No photometry available',ccds=ccds)

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

    H,W = bitmask.shape
    phot.bitmask = bitmask[np.clip(phot.y1, 0, H-1).astype(int),
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

        from tractor.brightness import NanoMaggies
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
    ccds['transp'] = transp
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
        imobj.make_plots(phot,dmag,ccds['zpt'],ccds['transp'])
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
        tr.optimize_forced_photometry()
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
        for step in range(50):
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
    parser= get_parser()
    args = parser.parse_args()

    if args.overhead is not None:
        t0 = args.overhead
        if t0.endswith('.N'):
            t0 = t0[:-2]
        t0 = float(t0)
        import time
        print('Startup time:', time.time()-t0, 'seconds')

    if args.image_list:
        images= read_lines(args.image_list)
    elif args.image:
        images= [args.image]

    main(image_list=images,args=args)


