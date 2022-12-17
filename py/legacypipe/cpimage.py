import os
import warnings
import numpy as np
import fitsio
from legacypipe.image import LegacySurveyImage
from astrometry.util.fits import fits_table

import logging
logger = logging.getLogger('legacypipe.cpimage')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

class CPImage(LegacySurveyImage):

    def calibration_good(self, primhdr):
        '''Did the CP processing succeed for this image?  If not, no need to process further.
        '''
        return primhdr.get('WCSCAL', '').strip().lower().startswith('success')

    def validate_version(self, *args, **kwargs):
        return validate_version(*args, **kwargs)

    # A function that can be called by subclassers to apply a per-amp
    # zeropoint correction.
    def apply_amp_correction_northern(self, img, invvar, x0, y0):
        apply_amp_correction_northern(self.camera, self.band, self.expnum,
                                      self.ccdname, self.mjdobs,
                                      img, invvar, x0, y0)

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        return remap_dq_cp_codes(dq, dtype=self.dq_type)

def remap_dq_cp_codes(dq, ignore_codes=None, dtype=np.uint16):
    '''
    Some versions of the CP use integer codes, not bit masks.
    This converts them.

    1 = bad
    2 = no value (for remapped and stacked data)
    3 = saturated
    4 = bleed mask
    5 = cosmic ray
    6 = low weight
    7 = diff detect (multi-exposure difference detection from median)
    8 = long streak (e.g. satellite trail)
    '''
    if ignore_codes is None:
        ignore_codes = []
    dqbits = np.zeros(dq.shape, dtype)

    # Some images (eg, 90prime//CP20160403/ksb_160404_103333_ood_g_v1-CCD1.fits)
    # around saturated stars have the core with value 3 (satur), surrounded by one
    # pixel of value 1 (bad), and then more pixels with value 4 (bleed).
    # Set the BAD ones to SATUR.
    from scipy.ndimage.morphology import binary_dilation
    dq[np.logical_and(dq == 1, binary_dilation(dq == 3))] = 3

    from legacypipe.bits import DQ_BITS
    for code,bitname in [(1, 'badpix'),
                         (2, 'badpix'),
                         (3, 'satur'),
                         (4, 'bleed'),
                         (5, 'cr'),
                         (6, 'badpix'),
                         (7, 'trans'),
                         (8, 'trans'),
                         ]:
        if code in ignore_codes:
            continue
        dqbits[dq == code] |= DQ_BITS[bitname]
    return dqbits

def validate_version(fn, filetype, expnum, plver, plprocid,
                     data=None, ext=1, cpheader=False,
                     old_calibs_ok=False, truncated_ok=True, quiet=False):
    '''
    truncated_ok: the target *plver* or *plprocid* may be truncated, so only
    demand a match up to the length of those variables.  This can happen if, eg,
    the survey-ccds table has the PLVER or PLPROCID columns too short.
    '''
    if not os.path.exists(fn):
        if not quiet:
            info('File not found {}'.format(fn))
        return False
    # Check the data model
    if filetype == 'table':
        if data is None:
            T = fits_table(fn)
        else:
            T = data
        cols = T.get_columns()
        for key,targetval,strip in (('plver', plver, True),
                                    ('plprocid', plprocid, True),
                                    ('expnum', expnum, False)):
            if targetval is None:
                # Skip this check
                debug('Skipping check of', key, 'for', fn)
                continue
            if key not in cols:
                if old_calibs_ok:
                    warnings.warn('Validation: table {} is missing {} but old_calibs_ok=True'.format(fn, key))
                    continue
                else:
                    debug('WARNING: {} missing {}'.format(fn, key))
                    return False
            val = T.get(key)
            if strip:
                val = np.array([str(v).strip() for v in val])
            ok = np.all(val == targetval)
            if (not ok) and truncated_ok:
                N = len(targetval)
                val = np.array([v[:min(len(v),N)] for v in val])
                ok = np.all(val == targetval)
                if ok:
                    warnings.warn('Validation: {}={} validated only after truncating for {}'.format(key, targetval, fn))
            if not ok:
                if old_calibs_ok:
                    warnings.warn('Validation: {} {}!={} in {} table but old_calibs_ok=True'.format(key, val, targetval, fn))
                    continue
                else:
                    debug('WARNING: {} {}!={} in {} table'.format(key, val, targetval, fn))
                    return False
        return True
    elif filetype in ['primaryheader', 'header']:
        if data is None:
            if filetype == 'primaryheader':
                hdr = fitsio.read_header(fn)
            else:
                hdr = fitsio.FITS(fn)[ext].read_header()
        else:
            hdr = data

        cpexpnum = None
        if cpheader:
            # Special handling for EXPNUM in some cases
            if 'EXPNUM' in hdr and hdr['EXPNUM'] is not None:
                cpexpnum = hdr['EXPNUM']
            elif 'OBSID' in hdr:
                # At the beginning of the MzLS survey, eg 2016-01-24, the EXPNUM
                # cards are blank.  Fake up an expnum like 160125082555
                # (yymmddhhmmss), same as the CP filename.
                # OBSID   = 'kp4m.20160125T082555' / Observation ID
                # MzLS:
                obsid = hdr['OBSID']
                if obsid.startswith('kp4m.'):
                    obsid = obsid.strip().split('.')[1]
                    obsid = obsid.replace('T', '')
                    obsid = int(obsid[2:], 10)
                    cpexpnum = obsid
                    if not quiet:
                        debug('Faked up EXPNUM', cpexpnum)
                elif obsid.startswith('ksb'):
                    import re
                    # DTACQNAM are like /descache/bass/20160504/d7513.0033.fits
                    base= (os.path.basename(hdr['DTACQNAM'])
                           .replace('.fits','')
                           .replace('.fz',''))
                    cpexpnum = int(re.sub(r'([a-z]+|\.+)','',base), 10)
                    if not quiet:
                        debug('Faked up EXPNUM', cpexpnum)
            else:
                if not quiet:
                    info('Missing EXPNUM and OBSID in header')

        for key,spval,targetval,stringtype in (('PLVER', None, plver, True),
                                          ('PLPROCID', None, plprocid, True),
                                          ('EXPNUM', cpexpnum, expnum, False)):
            if spval is not None:
                val = spval
            else:
                if key not in hdr:
                    if old_calibs_ok:
                        warnings.warn('Validation: {} header missing {} but old_calibs_ok=True'.format(fn, key))
                        continue
                    else:
                        debug('WARNING: {} header missing {}'.format(fn, key))
                        return False
                val = hdr[key]

            if stringtype:
                # PLPROCID can get parsed as an int by fitsio, ugh
                val = str(val)
                val = val.strip()
            else:
                # EXPNUM is stored as a string in some DECam exposures -- eg
                # decam/CP/V4.8.2a/CP20200224/c4d_200225_072059_ooi_i_v1.fits.fz
                val = int(val)

            # For cases where the CCDs table was truncated...
            if val != targetval and truncated_ok:
                info(key, 'value', val, type(val), 'vs target', targetval, type(targetval))
                origval = val
                val = val[:len(targetval)]
                if val == targetval:
                    warnings.warn('Validation: {} validated only after truncating {} to {} for {}'.format(key, origval, val, fn))
            if val != targetval:
                if old_calibs_ok:
                    warnings.warn('Validation: {} {}!={} in {} header but old_calibs_ok=True'.format(key, val, targetval, fn))
                    continue
                else:
                    debug('WARNING: {} {}!={} in {} header'.format(key, val, targetval, fn))
                    return False
        return True

    else:
        raise ValueError('incorrect filetype')

def apply_amp_correction_northern(camera, band, expnum, ccdname, mjdobs,
                                  img, invvar, x0, y0):
    from pkg_resources import resource_filename
    dirname = resource_filename('legacypipe', 'data')
    fn = os.path.join(dirname, 'ampcorrections.fits')
    A = fits_table(fn)
    # Find relevant row -- camera, filter, ccdname, mjd_start, mjd_end,
    # And then multiple rows of:
    #   xlo, xhi, ylo, yhi -> dzp
    # that might overlap this image.
    I = np.flatnonzero([(cam.strip() == camera) and
                        (f.strip() == band) and
                        (ccd.strip() == ccdname) and
                        (not(np.isfinite(mjdstart)) or (mjdobs >= mjdstart)) and
                        (not(np.isfinite(mjdend  )) or (mjdobs <= mjdend))
                        for cam,f,ccd,mjdstart,mjdend
                        in zip(A.camera, A.filter, A.ccdname,
                               A.mjd_start, A.mjd_end)])
    info('Found', len(I), 'relevant rows in amp-corrections file.')
    if len(I) == 0:
        return
    if img is not None:
        H,W = img.shape
    else:
        H,W = invvar.shape
    # x0,y0 are integer pixel coords
    # x1,y1 are INCLUSIVE integer pixel coords
    x1 = x0 + W - 1
    y1 = y0 + H - 1

    debug_corr = False
    if debug_corr:
        count_corr = np.zeros((H,W), np.uint8)
        corr_map = np.zeros((H,W), np.float32)
        fitsio.write('amp-corr-image-before-%s-%s-%s.fits' % (camera, expnum, ccdname), img, clobber=True)

    for a in A[I]:
        # In the file, xhi,yhi are NON-inclusive.
        if a.xlo > x1 or a.xhi <= x0:
            continue
        if a.ylo > y1 or a.yhi <= y0:
            continue
        # Overlap!
        info('Found overlap: image x', x0, x1, 'and amp range', a.xlo, a.xhi-1,
              'and image y', y0, y1, 'and amp range', a.ylo, a.yhi-1)
        xstart = max(0, a.xlo - x0)
        xend   = min(W, a.xhi - x0)
        ystart = max(0, a.ylo - y0)
        yend   = min(H, a.yhi - y0)
        info('Range in image: x', xstart, xend, ', y', ystart, yend, '(with image size %i x %i)' % (W,H))
        scale = 10.**(0.4 * a.dzp)
        info('dzp', a.dzp, '-> scaling image by', scale)
        if img is not None:
            img   [ystart:yend, xstart:xend] *= scale
        if invvar is not None:
            invvar[ystart:yend, xstart:xend] /= scale**2

        if debug_corr:
            count_corr[ystart:yend, xstart:xend] += 1
            corr_map[ystart:yend, xstart:xend] = scale

    if debug_corr:
        assert(np.all(count_corr == 1))
        fitsio.write('amp-corr-image-after-%s-%s-%s.fits' % (camera, expnum, ccdname), img, clobber=True)
        fitsio.write('amp-corr-map-%s-%s-%s.fits' % (camera, expnum, ccdname), corr_map, clobber=True)
