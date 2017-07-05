#!/usr/bin/env python

"""
RUN
=== 
1) generate file list of cpimages, e.g. for everything mzls
find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/CP*v2/k4m*ooi*.fits.fz > mosaic_allcp.txt
2) use batch script "submit_zpts.sh" to run legacy-zeropoints.py
-- scaling is good with 10,000+ cores
   -- key is Yu Feng's bcast, we made tar.gz files for all the NERSC HPCPorts modules and Yu's bcast efficiently copies them to ram on every compute node for fast python startup
   -- the directory containing the tar.gz files is /global/homes/k/kaylanb/repos/yu-bcase
      also on kaylans tape: bcast_hpcp.tar
-- debug queue gets all zeropoints done < 30 min
-- set SBATCH -N to be as many nodes as will give you mpi tasks = nodes*cores_per_nodes ~ number of cp images
-- ouput ALL plots with --verboseplots, ie Moffat PSF fits to 20 brightest stars for FWHM
3) Make a file (e.g. zpt_files.txt) listing all the zeropoint files you just made (not includeing the -star.fits ones), then 
    a) compare legacy zeropoints to Arjuns 
        run from loggin node: 
        python legacy-zeropoints.py --image_list zpt_files.txt --compare2arjun
    b) gather all zeropoint files into one fits table
        run from loggin node
            python legacy-zeropoints-gather.py --file_list zpt_files.txt --nproc 1 --outname gathered_zpts.fits
        OR if that takes too long, runs out of memory, etc, run with mpi tasks
            uncomment relavent lines of submit_zpts.sh, comment out running legacy-zeropoints.py
            <= 50 mpi tasks is fine
            sbatch submit_zpts.sh 

UNITS
=====
match Arjun's decstat,moststat,bokstat.pro 

DOC
===
Generate a legacypipe-compatible CCD-level zeropoints file for a given set of
(reduced) BASS, MzLS, or DECaLS imaging.

This script borrows liberally from code written by Ian, Kaylan, Dustin, David
S. and Arjun, including rapala.survey.bass_ccds, legacypipe.simple-bok-ccds,
obsbot.measure_raw, and the IDL codes decstat and mosstat.

Although the script was developed to run on the temporarily repackaged BASS data
created by the script legacyccds/repackage-bass.py (which writes out
multi-extension FITS files with a different naming convention relative to what
NAOC delivers), it is largely camera-agnostic, and should therefore eventually
be able to be used to derive zeropoints for all the Legacy Survey imaging.

On edison the repackaged BASS data are located in
/scratch2/scratchdirs/ioannis/bok-reduced with the correct permissions.

Proposed changes to the -ccds.fits file used by legacypipe:
 * Rename arawgain --> gain to be camera-agnostic.
 * The quantities ccdzpta and ccdzptb are specific to DECam, while for 90prime
   these quantities are ccdzpt1, ccdzpt2, ccdzpt3, and ccdzpt4.  These columns
   can be kept in the -zeropoints.fits file but should be removed from the final
   -ccds.fits file.
 * The pipeline uses the SE-measured FWHM (FWHM, pixels) to do source detection
   and to estimate the depth, instead of SEEING (FWHM, arcsec), which is
   measured by decstat in the case of DECam.  We should remove our dependence on
   SExtractor and simply use the seeing/fwhm estimate measured by us (e.g., this
   code).
 * The pixel scale should be added to the output file, although it can be gotten
   from the CD matrix.
 * AVSKY should be converted to electron or electron/s, to account for
   the varying gain of the amplifiers.  It's actually not even clear
   we need this header keyword.
 * We probably shouldn't cross-match against the tiles file in this code (but
   instead in something like merge-zeropoints), but what else from the annotated
   CCDs file should be directly calculated and stored here?
 * Are ccdnum and image_hdu redundant?

REVISION HISTORY:
    13-Sept-2016 J. Moustakas
    15-Dec-2016  K. Burleigh
"""
from __future__ import division, print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pdb
import argparse

import numpy as np
from glob import glob
from pickle import dump
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
from scipy.ndimage.filters import median_filter

import fitsio
from astropy.io import fits as fits_astropy
from astropy.table import Table, vstack
from astropy import units
from astropy.coordinates import SkyCoord
from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec
from astrometry.util.ttime import Time
import datetime
import sys

from photutils import (CircularAperture, CircularAnnulus,
                       aperture_photometry, DAOStarFinder)

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.libkd.spherematch import match_radec
from astrometry.libkd.spherematch import match_xy

from tractor.splinesky import SplineSky

from legacyanalysis.ps1cat import ps1cat

######## 
# stdouterr_redirected() is from Ted Kisner
# Every mpi task (zeropoint file) gets its own stdout file
import time
from contextlib import contextmanager

@contextmanager
def stdouterr_redirected(to=os.devnull, comm=None):
    '''
    Based on http://stackoverflow.com/questions/5081657
    import os
    with stdouterr_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    sys.stdout.flush()
    sys.stderr.flush()
    fd = sys.stdout.fileno()
    fde = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fde) # fd writes to 'to' file
        sys.stderr = os.fdopen(fde, 'w') # Python writes to fd
        
    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        if (comm is None) or (comm.rank == 0):
            print("Begin log redirection to {} at {}".format(to, time.asctime()))
        sys.stdout.flush()
        sys.stderr.flush()
        pto = to
        if comm is None:
            if not os.path.exists(os.path.dirname(pto)):
                os.makedirs(os.path.dirname(pto))
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        else:
            pto = "{}_{}".format(to, comm.rank)
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            if comm is not None:
                # concatenate per-process files
                comm.barrier()
                if comm.rank == 0:
                    with open(to, 'w') as outfile:
                        for p in range(comm.size):
                            outfile.write("================= Process {} =================\n".format(p))
                            fname = "{}_{}".format(to, p)
                            with open(fname) as infile:
                                outfile.write(infile.read())
                            os.remove(fname)
                comm.barrier()

            if (comm is None) or (comm.rank == 0):
                print("End log redirection to {} at {}".format(to, time.asctime()))
            sys.stdout.flush()
            sys.stderr.flush()
            
    return

def try_mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass # Already exists, thats fine

def image_to_fits(img,fn,header=None,extname=None):
    fitsio.write(fn,img,header=header,extname=extname)
    print('Wrote %s' % fn)


# From image.py
# imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn, self.hdu, todelete)
#for fn in todelete:
#   os.unlink(fn)
def funpack_files(imgfn, maskfn, hdu, todelete):
    from legacypipe.survey import create_temp

    tmpimgfn = None
    tmpmaskfn = None
    # For FITS files that are not actually fpack'ed, funpack -E
    # fails.  Check whether actually fpacked.
    fcopy = False
    hdr = fitsio.read_header(imgfn, ext=hdu)
    if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
        print('Image %s, HDU %i is not fpacked; just imcopying.' %
              (imgfn,  hdu))
        fcopy = True

    tmpimgfn  = create_temp(suffix='.fits')
    tmpmaskfn = create_temp(suffix='.fits')
    todelete.append(tmpimgfn)
    todelete.append(tmpmaskfn)

    if fcopy:
        cmd = 'imcopy %s"+%i" %s' % (imgfn, hdu, tmpimgfn)
    else:
        cmd = 'funpack -E %s -O %s %s' % (hdu, tmpimgfn, imgfn)
    print(cmd)
    if os.system(cmd):
        raise RuntimeError('Command failed: ' + cmd)

    if fcopy:
        cmd = 'imcopy %s"+%i" %s' % (maskfn, hdu, tmpmaskfn)
    else:
        cmd = 'funpack -E %s -O %s %s' % (hdu, tmpmaskfn, maskfn)
    print(cmd)
    if os.system(cmd):
        print('Command failed: ' + cmd)
        M,hdr = self._read_fits(maskfn, hdu, header=True)
        print('Read', M.dtype, M.shape)
        fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)

    return tmpimgfn,tmpmaskfn


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

def dobash(cmd):
    print('UNIX cmd: %s' % cmd)
    if os.system(cmd): raise ValueError


def extra_ccd_keys(camera='decam'):
    '''Returns list of camera-specific keywords for the ccd table'''
    if camera == 'decam':
        keys= [('ccdzpta', '>f4'), ('ccdzptb','>f4'), ('ccdnmatcha', '>i2'), ('ccdnmatchb', '>i2'),\
               ('temp', '>f4')]
    elif camera == 'mosaic':
        keys=[]
    elif camera == '90prime':
        keys=[('ccdzpt1', '>f4'), ('ccdzpt2','>f4'), ('ccdzpt3', '>f4'), ('ccdzpt4','>f4'),\
              ('ccdnmatcha', '>i2'), ('ccdnmatch2', '>i2'), ('ccdnmatch3', '>i2'), ('ccdnmatch4', '>i2')]
    return keys 

def get_units():
    return dict(
        ra='deg',dec='deg',exptime='sec',pixscale='arcsec/pix',
        fwhm='pix',seeing='arcsec',
        sky0='mag/arcsec^2',skymag='mag/arcsec^2/sec',
        skycounts='electron/pix/sec',skyrms='electron/pix/sec',
        apflux='electron/7 arcsec aperture',apskyflux='electron/7 arcsec aperture',
        apskyflux_perpix='electron/pix',
        apmags='-2.5log10(electron/sec) + zpt0',
        raoff='arcsec',decoff='arcsec',rarms='arcsec',decrms='arcsec',
        phoff='electron/sec',phrms='electron/sec',
        zpt0='electron/sec',zpt='electron/sec',transp='electron/sec')
 

def _ccds_table(camera='decam'):
    '''Initialize the output CCDs table.  See decstat.pro and merge-zeropoints.py
    for details.

    '''
    cols = [
        ('image_filename', 'S65'), # image filename, including the subdirectory
        ('image_hdu', '>i2'),      # integer extension number
        ('camera', 'S7'),          # camera name
        ('expnum', '>i4'),         # unique exposure number
        ('ccdname', 'S4'),         # FITS extension name
        ('ccdnum', '>i2'),        # CCD number 
        ('expid', 'S16'),          # combination of EXPNUM and CCDNAME
        ('object', 'S35'),         # object (field) name
        ('propid', 'S10'),         # proposal ID
        ('filter', 'S1'),          # filter name / bandpass
        ('exptime', '>f4'),        # exposure time (s)
        ('date_obs', 'S10'),       # date of observation (from header)
        ('mjd_obs', '>f8'),        # MJD of observation (from header)
        ('ut', 'S15'),             # UT time (from header)
        ('ha', 'S13'),             # hour angle (from header)
        ('airmass', '>f4'),        # airmass (from header)
        #('seeing', '>f4'),        # seeing estimate (from header, arcsec)
        ('fwhm', '>f4'),          # FWHM (pixels)
        ('fwhm_cp', '>f4'),          # FWHM (pixels)
        #('arawgain', '>f4'),       
        ('gain', '>f4'),           # average gain (camera-specific, e/ADU) -- remove?
        #('avsky', '>f4'),         # average sky value from CP (from header, ADU) -- remove?
        ('width', '>i2'),          # image width (pixels, NAXIS1, from header)
        ('height', '>i2'),         # image height (pixels, NAXIS2, from header)
        ('ra_bore', '>f8'),        # telescope RA (deg, from header)
        ('dec_bore', '>f8'),       # telescope Dec (deg, from header)
        ('crpix1', '>f4'),         # astrometric solution (no distortion terms)
        ('crpix2', '>f4'),
        ('crval1', '>f8'),
        ('crval2', '>f8'),
        ('cd1_1', '>f4'),
        ('cd1_2', '>f4'),
        ('cd2_1', '>f4'),
        ('cd2_2', '>f4'),
        ('pixscale', 'f4'),   # mean pixel scale [arcsec/pix]
        ('zptavg', '>f4'),    # zeropoint averaged over all CCDs [=zpt in decstat]
        # -- CCD-level quantities --
        ('ra', '>f8'),        # ra at center of the CCD
        ('dec', '>f8'),       # dec at the center of the CCD
        ('skymag', '>f4'),    # average sky surface brightness [mag/arcsec^2] [=ccdskymag in decstat]
        ('skycounts', '>f4'), # median sky level [electron/pix]               [=ccdskycounts in decstat]
        ('skycounts_a', '>f4'), # median sky level [electron/pix]               [=ccdskycounts in decstat]
        ('skyrms', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_a', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_b', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_c', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_d', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_sm', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_clip', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_clip_sm', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('skyrms_sigma', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        #('medskysub', '>f4'),    # sky variance [electron/pix]                   [=ccdskyrms in decstat]
        ('nstarfind', '>i2'),    # number of PS1-matched stars                   [=ccdnmatch in decstat]
        ('nstar', '>i2'),     # number of detected stars                      [=ccdnstar in decstat]
        ('nmatch', '>i2'),    # number of PS1-matched stars                   [=ccdnmatch in decstat]
        ('mdncol', '>f4'),    # median g-i color of PS1-matched main-sequence stars [=ccdmdncol in decstat]
        ('phoff', '>f4'),     # photometric offset relative to PS1 (mag)      [=ccdphoff in decstat]
        ('phrms', '>f4'),     # photometric rms relative to PS1 (mag)         [=ccdphrms in decstat]
        ('zpt', '>f4'),       # median/mean zeropoint (mag)                   [=ccdzpt in decstat]
        ('transp', '>f4'),    # transparency                                  [=ccdtransp in decstat]
        ('raoff', '>f4'),     # median RA offset (arcsec)                     [=ccdraoff in decstat]
        ('decoff', '>f4'),    # median Dec offset (arcsec)                    [=ccddecoff in decstat]
        ('rarms', '>f4'),     # rms RA offset (arcsec)                        [=ccdrarms in decstat]
        ('decrms', '>f4'),     # rms Dec offset (arcsec)                       [=ccddecrms in decstat]
        ('rastddev', '>f4'),     # std RA offset (arcsec)                        [=ccdrarms in decstat]
        ('decstddev', '>f4')     # std Dec offset (arcsec)                       [=ccddecrms in decstat]
        ]

    # Add camera-specific keywords to the output table.
    #cols.extend( extra_ccd_keys(camera=camera) )
    
    ccds = Table(np.zeros(1, dtype=cols))
    return ccds

     
def _stars_table(nstars=1):
    '''Initialize the stars table, which will contain information on all the stars
       detected on the CCD, including the PS1 photometry.

    '''
    cols = [('image_filename', 'S65'),('image_hdu', '>i2'),
            ('expid', 'S16'), ('filter', 'S1'),('nmatch', '>i2'), 
            ('amplifier', 'i2'), ('x', 'f4'), ('y', 'f4'),('expnum', '>i4'),
            ('gain', 'f4'),
            ('ra', 'f8'), ('dec', 'f8'), ('apmag', 'f4'),('apflux', 'f4'),('apskyflux', 'f4'),('apskyflux_perpix', 'f4'),
            ('radiff', 'f8'), ('decdiff', 'f8'),('radiff_ps1', 'f8'), ('decdiff_ps1', 'f8'),
            ('gaia_ra', 'f8'), ('gaia_dec', 'f8'), ('ps1_mag', 'f4'), ('ps1_gicolor', 'f4'),
            ('gaia_g','f8'),('ps1_g','f8'),('ps1_r','f8'),('ps1_i','f8'),('ps1_z','f8'),
            ('daofind_x', 'f4'), ('daofind_y', 'f4'),
            ('exptime', '>f4'),
            ('mycuts_x', 'f4'), ('mycuts_y', 'f4')]
    stars = Table(np.zeros(nstars, dtype=cols))
    return stars

def reduce_survey_ccd_cols(survey_fn,legacy_fn):
    survey=fits_table(survey_fn)
    legacy=fits_table(legacy_fn)
    for col in survey.get_columns():
        if not col in legacy.get_columns():
            survey.delete_column(col)
    assert(len(legacy.get_columns()) == len(survey.get_columns()))
    for col in survey.get_columns():
        assert(col in legacy.get_columns())
    fn=survey_fn.replace('.fits.gz','_reduced.fits.gz')
    survey.writeto(fn) 
    print('Wrote %s' % fn)

def cuts_for_brick_2016p122(legacy_fn,survey_fn):
    survey=fits_table(survey_fn)
    legacy=fits_table(legacy_fn)
    # cut to same data as survey_fn
    keep= np.zeros(len(legacy),bool)
    for sur in survey:
        ind= (np.char.strip(legacy.image_filename) == sur.image_filename.strip() ) *\
             (np.char.strip(legacy.ccdname) == sur.ccdname.strip() )
        keep[ind]= True
    legacy.cut(keep)
    print('size legacy=%d' % (len(legacy),))
    # save
    fn=legacy_fn.replace('.fits','_wcuts.fits')
    legacy.writeto(fn) 
    print('Wrote %s' % fn)
     
        

def run_create_legacypipe_table(zpt_list):
    fns= np.loadtxt(zpt_list,dtype=str)
    assert(len(fns) > 1)
    for fn in fns:
        create_legacypipe_table(fn)
        

def create_legacypipe_table(ccds_fn):
    '''input _ccds_table fn
    output a table formatted for legacypipe/runbrick'''
    # HACK! need func to put in appropriate units e.g. compare to survey-ccds file for decam,mosaic, and bass
    need_arjuns_keys= ['ra','dec','ra_bore','dec_bore',
                       'image_filename','image_hdu','expnum','ccdname','object',
                       'filter','exptime','camera','width','height','propid',
                       'mjd_obs','ccdnmatch',
                       'fwhm','zpt','ccdzpt','ccdraoff','ccddecoff',
                       'cd1_1','cd2_2','cd1_2','cd2_1',
                       'crval1','crval2','crpix1','crpix2']
    dustins_keys= ['skyrms']
    # Load full zpt table
    assert('-zpt.fits' in ccds_fn)
    T = fits_table(ccds_fn)
    #hdr = T.get_header()
    #primhdr = fitsio.read_header(ccds_fn)
    #units= get_units()

    #primhdr.add_record(dict(name='ALLBANDS', value=allbands,
    #                        comment='Band order in array values'))
    #has_zpt = 'zpt' in T.columns()
    # Units
    # DECAM only
    T.set('zpt',T.zpt - 2.5*np.log10(T.gain))
    T.set('zptavg',T.zptavg - 2.5*np.log10(T.gain))
    # Rename
    rename_keys= [('zpt','ccdzpt'),('zptavg','zpt'),
                  ('raoff','ccdraoff'),('decoff','ccddecoff'),
                  ('nmatch','ccdnmatch')]
    for old,new in rename_keys:
        T.rename(old,new)
        #units[new]= units.pop(old)
    # Delete 
    del_keys= list( set(T.get_columns()).difference(set(need_arjuns_keys+dustins_keys)) )
    for key in del_keys:
        T.delete_column(key)
        #if key in units.keys():
        #    _= units.pop(key)
    # legacypipe/merge-zeropoints.py
    T.set('width', np.zeros(len(T), np.int16) + 2046)
    T.set('height', np.zeros(len(T), np.int16) + 4094)
    # precision
    T.width  = T.width.astype(np.int16)
    T.height = T.height.astype(np.int16)
    #T.ccdnum = T.ccdnum.astype(np.int16) #number doesn't follow hdu, not using if possible
    T.cd1_1 = T.cd1_1.astype(np.float32)
    T.cd1_2 = T.cd1_2.astype(np.float32)
    T.cd2_1 = T.cd2_1.astype(np.float32)
    T.cd2_2 = T.cd2_2.astype(np.float32)
    # Align units with 'cols'
    #cols = T.get_columns()
    #units = [units.get(c, '') for c in cols]
    # Column ordering...
    #cols = []
    #if dr4:
    #    cols.append('release')
    #    T.release = np.zeros(len(T), np.int32) + 4000
    outfn=ccds_fn.replace('-zpt.fits','-legacypipe.fits')
    T.writeto(outfn) #, columns=cols, header=hdr, primheader=primhdr, units=units)
    print('Wrote %s' % outfn)


def create_matches_table(stars_fn,**kwargs):
    '''Arjun's "matches-*.fits" stars table
    input _stars_table fn
    output Arjun's matches table, same column names but units can be different'''
    # HACK! need func to put in appropriate units e.g. compare to survey-ccds file for decam,mosaic, and bass
    need_arjuns_keys= ['filename','expnum','extname',
                       'ccd_x','ccd_y','ccd_ra','ccd_dec',
                       'ccd_mag','ccd_sky',
                       'raoff','decoff',
                       'magoff',
                       'nmatch',
                       'gmag','ps1_g','ps1_r','ps1_i','ps1_z']
    extra_keys= ['image_hdu','filter'] # Check for hdu and band depenent trends
    # Load full zpt table
    assert('-star.fits' in stars_fn)
    # HACK: need magoff
    T = fits_table(stars_fn)
    extname=[ccdname for _,ccdname in np.char.split(T.expid,'-')]
    T.set('extname', np.array(extname))
    # AB mag of stars using fiducial ZP to convert
    T.set('ccd_mag',-2.5 * np.log10(T.apflux / T.exptime) +  \
                        kwargs['zp_fid'])
    # ADU per pixel from sky aperture 
    area= np.pi*3.5**2/kwargs.get('pixscale')**2
    T.set('ccd_sky', T.apskyflux / area / T.gain)
    # Arjuns ccd_sky is ADUs in 7-10 arcsec sky aperture
    # e.g. sky (total e/pix/sec)= ccd_sky (ADU) * gain / exptime
    # Rename
    rename_keys= [('ra','ccd_ra'),('dec','ccd_dec'),('x','ccd_x'),('y','ccd_y'),
                  ('radiff','raoff'),('decdiff','decoff'),
                  ('dmagall','magoff'),
                  ('image_filename','filename'),
                  ('gaia_g','gmag')]
    for old,new in rename_keys:
        T.rename(old,new)
        #units[new]= units.pop(old)
    # Delete unneeded keys
    del_keys= list( set(T.get_columns()).difference(set(need_arjuns_keys + extra_keys)) )
    for key in del_keys:
        T.delete_column(key)
        #if key in units.keys():
        #    _= units.pop(key)
    # legacypipe/merge-zeropoints.py
    outfn=stars_fn.replace('-star.fits','-matches.fits')
    T.writeto(outfn) #, columns=cols, header=hdr, primheader=primhdr, units=units)
    print('Wrote %s' % outfn)

def create_zeropoints_table(zpt_fn):
    '''Arjun's "zeropoint-*.fits" zpts table
    input _ccds_table fn
    output same thing but with Arjun's column names and units'''
    # HACK! need func to put in appropriate units e.g. compare to survey-ccds file for decam,mosaic, and bass
    need_arjuns_keys= \
        ['filename', 'object', 'expnum', 'exptime', 'filter', 'seeing', 'ra', 'dec', 
         'date_obs', 'mjd_obs', 'ut', 'ha', 'airmass', 'propid', 'zpt', 'avsky', 
         'arawgain', 'fwhm', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 
         'naxis1', 'naxis2', 'ccdhdunum', 'ccdnum', 'ccdname', 'ccdra', 'ccddec', 
         'ccdzpt', 'ccdzpta', 'ccdzptb', 'ccdphoff', 'ccdphrms', 'ccdskyrms', 'ccdskymag', 
         'ccdskycounts', 'ccdraoff', 'ccddecoff', 'ccdrarms', 'ccddecrms', 'ccdtransp', 
         'ccdnstarfind', 'ccdnstar', 'ccdnmatch', 'ccdnmatcha', 'ccdnmatchb', 'ccdmdncol', 
         'temp']
    ignoring_these= \
        ['arawgain','ccdhdunum','ccdzpta', 'ccdzptb','ccdnstarfind', 'ccdnstar',
         'ccdnmatcha', 'ccdnmatchb', 'ccdmdncol','temp']
    # Load full zpt table
    assert('-zpt.fits' in zpt_fn)
    # HACK: need magoff
    T = fits_table(zpt_fn)
    # Change units
    pix= 0.262
    T.set('fwhm',T.fwhm * pix)
    T.set('skycounts', T.skycounts * T.exptime / T.gain)
    T.set('skyrms', T.skycounts * T.exptime / T.gain)
    T.set('zpt',T.zpt - 2.5*np.log10(T.gain))
    T.set('zptavg',T.zptavg - 2.5*np.log10(T.gain))
    # Rename
    # Append 'ccd' to name
    app_ccd= ['skycounts','skyrms','skymag',
              'phoff','raoff','decoff',
              'phrms','rarms','decrms',
              'nmatch',
              'transp'] 
    for ad_ccd in app_ccd:
        T.rename(ad_ccd,'ccd'+ad_ccd)
    # Other
    rename_keys= [('ra','ccdra'),('dec','ccddec'),
                  ('ra_bore','ra'),('dec_bore','dec'),
                  ('fwhm','seeing'),('fwhm_cp','fwhm'),
                  ('zpt','ccdzpt'),('zptavg','zpt'),
                  ('width','naxis1'),('height','naxis2'),
                  ('image_filename','filename')]
    for old,new in rename_keys:
        T.rename(old,new)
    # New columns
    T.set('avsky', np.zeros(len(T)) + np.mean(T.ccdskycounts))
    # Delete unneeded keys
    needed= set(need_arjuns_keys).difference(set(ignoring_these))
    del_keys= list( set(T.get_columns()).difference(needed) )
    for key in del_keys:
        T.delete_column(key)
    # Write
    outfn=zpt_fn.replace('-zpt.fits','-zeropoint.fits')
    T.writeto(outfn) #, columns=cols, header=hdr, primheader=primhdr, units=units)
    print('Wrote %s' % outfn)



#class NativeTable(object):
#    def __init__(self,fn,camera='decam',ccd_or_stars='ccds'):
#        '''zpt,stars tables have same units by default (e.g. electron/sec for zpt)
#        This func takes either the ccds or stars table and converts the relavent columns
#        into native units for given camera 
#        e.g. ADU for DECam,  electron/sec for Mosaic/BASS'''
#        assert(camera in ['decam','mosaic','90prime'])
#        assert(ccds_or_stars in ['ccds','stars'])
#        if camera in 'decam':
#            self.Decam(fn,ccds_or_stars=ccds_or_stars)
#        if camera in ['mosaic','90prime']:
#            self.Mosaic90Prime(fn,ccds_or_stars=ccds_or_stars)
#
#    def Decam(self,fn,ccds_or_stars):
#        T = fits_table(fn)
#        hdr = T.get_header()
#        primhdr = fitsio.read_header(ccds_fn)
#        units= get_units()
#        # Convert units
#        #T.set('zpt',T.zpt +- 2.5*np.log10(T.gain * T.exptime)) 
#        # Write
#        outfn=fn.replace('.fits','native.fits')
#        T.writeto(outfn, columns=cols, header=hdr, primheader=primhdr, units=units)

def getrms(x):
    return np.sqrt( np.mean( np.power(x,2) ) )

def get_bitmask_fn(imgfn):
    if 'ooi' in imgfn: 
        fn= imgfn.replace('ooi','ood')
    elif 'oki' in imgfn: 
        fn= imgfn.replace('oki','ood')
    else:
        raise ValueError('bad imgfn? no ooi or oki: %s' % imgfn)
    return fn

class Measurer(object):
    def __init__(self, fn, aprad=3.5, skyrad_inner=7.0, skyrad_outer=10.0,
                 det_thresh=8., match_radius=3.,sn_min=None,sn_max=None,
                 aper_sky_sub=False, calibrate=False, **kwargs):
        '''This is the work-horse class which operates on a given image regardless of
        its origin (decam, mosaic, 90prime).

        Args:

        aprad: float
        Aperture photometry radius in arcsec

        skyrad_{inner,outer}: floats
        Sky annulus radius in arcsec

        det_thresh: minimum S/N for matched filter, 8 gives daofind agreendment with IDL daofind of 10
        match_radius: arcsec matching to gaia/ps1, 3 arcsec is IDL codes

        sn_{min,max}: if not None then then {min,max} S/N will be enforced from 
                      aperture photoemtry, where S/N = apflux/sqrt(skyflux)

        aper_sky_sub: do aperture sky subtraction instead of splinesky
        '''
        # Set extra kwargs
        self.zptsfile= kwargs.get('zptsfile')
        self.prefix= kwargs.get('prefix')
        self.verboseplots= kwargs.get('verboseplots')
        
        self.fn = fn
        self.debug= kwargs.get('debug')

        self.aper_sky_sub = aper_sky_sub
        self.calibrate = calibrate
        
        self.aprad = aprad
        self.skyrad = (skyrad_inner, skyrad_outer)

        self.det_thresh = det_thresh    # [S/N] 
        self.match_radius = match_radius 
        self.sn_min = sn_min 
        self.sn_max = sn_max 
        
        # Tractor fitting of final star sample
        self.stampradius= 4. # [arcsec] Should be a bit bigger than radius=3.5'' aperture
        self.tractor_nstars= 30 # Tractorize at most this many stars, saves CPU time

        # Set the nominal detection FWHM (in pixels) and detection threshold.
        # Read the primary header and the header for this extension.
        self.nominal_fwhm = 5.0 # [pixels]
        
        try:
            self.primhdr = fitsio.read_header(fn, ext=0)
        except ValueError:
            # astropy can handle it
            tmp= fits_astropy.open(fn)
            self.primhdr= tmp[0].header
            tmp.close()
            del tmp

        # Camera-agnostic primary header cards
        self.propid = self.primhdr['PROPID']
        self.exptime = self.primhdr['EXPTIME']
        self.date_obs = self.primhdr['DATE-OBS']
        self.mjd_obs = self.primhdr['MJD-OBS']
        try:
            self.airmass = self.primhdr['AIRMASS']
        except KeyError:
            self.airmass = -1
            print('WARNING! no airmass fn=%s' % self.fn) 
        self.ha = self.primhdr['HA']
        
        # FIX ME!, gets unique id for mosaic but not 90prime
        if 'EXPNUM' in self.primhdr: 
            self.expnum = self.primhdr['EXPNUM']
        else:
            print('WARNING! no EXPNUM in %s' % self.fn)
            self.expnum = np.int32(os.path.basename(self.fn)[11:17])

        self.obj = self.primhdr['OBJECT']

    def zeropoint(self, band):
        return self.zp0[band]

    def sky(self, band):
        return self.sky0[band]

    def extinction(self, band):
        return self.k_ext[band]

    def set_hdu(self,ext):
        self.ext = ext
        self.hdr = fitsio.read_header(self.fn, ext=ext)
        self.ccdname = self.hdr['EXTNAME'].strip()
        assert(self.ext.upper() == self.ccdname.upper())
        self.ccdnum = np.int(self.hdr['CCDNUM']) 
        self.image_hdu = self.ccdnum #header: extnum <--> hduname
        self.expid = '{:08d}-{}'.format(self.expnum, self.ccdname)
        self.gain= self.get_gain(self.hdr)
        # WCS
        self.wcs = self.get_wcs()
        # Pixscale is assumed CONSTANT! per camera
        #self.pixscale = self.wcs.pixel_scale()

    def read_bitmask(self):
        dqfn= get_bitmask_fn(self.fn)
        mask, junk = fitsio.read(dqfn, ext=self.ext, header=True)
        return mask

    def get_image_mask(self,img,bitmask):
        '''img -- oki or ooi
        bitmask -- ood'''
        mask = np.zeros(img.shape).astype(np.int8) 
        # Any flagged pixel
        mask[bitmask > 0]= 1
        # Old way of doing things was saturation threshold
        #if saturated_bitmask:
        #    if self.camera == 'decam':
        #        sat_level = 160000. # e-
        #    else:
        #        sat_level= 50000. # e- 
        #    mask[img > sat_level]= 1
        #    return mask
        return mask

    def sensible_sigmaclip(self, arr, nsigma = 4.0):
        '''sigmaclip returns unclipped pixels, lo,hi, where lo,hi are the
        mean(goodpix) +- nsigma * sigma

        '''
        goodpix, lo, hi = sigmaclip(arr, low=nsigma, high=nsigma)
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
            mnsky, skystd = self.sensible_sigmaclip(img - skyimg,nsigma=nsigma)
            skymed= np.median(skyimg)
        else:
            #sky, sig1 = self.sensible_sigmaclip(img[1500:2500, 500:1000])
            if self.camera == 'decam':
                slc=[slice(1500,2500),slice(500,1500)]
            elif self.camera in ['mosaic','90prime']:
                slc=[slice(500,1500),slice(500,1500)]
            clip_vals,_,_ = sigmaclip(img[slc],low=nsigma,high=nsigma)
            # from astropy.stats import sigma_clip as sigmaclip_astropy
            #sky_masked= sigmaclip_astropy(img[slc],sigma=nsigma,iters=20)
            #use= sky1_masked.mask == False
            #skymed= np.median(sky_masked[use])
            #sky1std= np.std(sky_masked[use])
            skymed= np.median(clip_vals) 
            skystd= np.std(clip_vals) 
            skyimg= np.zeros(img.shape) + skymed
            # MAD gives 10% larger value
            # sig1= 1.4826 * np.median(np.abs(clip_vals))
        return skyimg, skymed, skystd

    def remove_sky_gradients(self, img):
        # Ugly removal of sky gradients by subtracting median in first x and then y
        H,W = img.shape
        meds = np.array([np.median(img[:,i]) for i in range(W)])
        meds = median_filter(meds, size=5)
        img -= meds[np.newaxis,:]
        meds = np.array([np.median(img[i,:]) for i in range(H)])
        meds = median_filter(meds, size=5)
        img -= meds[:,np.newaxis]

    def match_ps1_stars(self, px, py, fullx, fully, radius, stars):
        #print('Matching', len(px), 'PS1 and', len(fullx), 'detected stars with radius', radius)
        I,J,d = match_xy(px, py, fullx, fully, radius)
        #print(len(I), 'matches')
        dx = px[I] - fullx[J]
        dy = py[I] - fully[J]
        return I,J,dx,dy

    def fitstars(self, img, ierr, xstar, ystar, fluxstar):
        '''Fit each star using a Tractor model.'''
        import tractor

        H, W = img.shape

        fwhms = []
        radius_pix = self.stampradius / self.pixscale
                
        for ii, (xi, yi, fluxi) in enumerate(zip(xstar, ystar, fluxstar)):
            #print('Fitting source', i, 'of', len(Jf))
            ix = int(np.round(xi))
            iy = int(np.round(yi))
            xlo = max(0, ix-radius_pix)
            xhi = min(W, ix+radius_pix+1)
            ylo = max(0, iy-radius_pix)
            yhi = min(H, iy+radius_pix+1)
            xx, yy = np.meshgrid(np.arange(xlo, xhi), np.arange(ylo, yhi))
            r2 = (xx - xi)**2 + (yy - yi)**2
            keep = (r2 < radius_pix**2)
            pix = img[ylo:yhi, xlo:xhi].copy()
            ie = ierr[ylo:yhi, xlo:xhi].copy()
            #print('fitting source at', ix,iy)
            #print('number of active pixels:', np.sum(ie > 0), 'shape', ie.shape)

            psf = tractor.NCircularGaussianPSF([4.0], [1.0])
            tim = tractor.Image(data=pix, inverr=ie, psf=psf)
            src = tractor.PointSource(tractor.PixPos(xi-xlo, yi-ylo),
                                      tractor.Flux(fluxi))
            tr = tractor.Tractor([tim], [src])
        
            #print('Posterior before prior:', tr.getLogProb())
            src.pos.addGaussianPrior('x', 0.0, 1.0)
            #print('Posterior after prior:', tr.getLogProb())
                
            tim.freezeAllBut('psf')
            psf.freezeAllBut('sigmas')
        
            # print('Optimizing params:')
            # tr.printThawedParams()
        
            #print('Parameter step sizes:', tr.getStepSizes())
            optargs = dict(priors=False, shared_params=False)
            for step in range(50):
                dlnp, x, alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break
                
            # Now fit only the PSF size
            tr.freezeParam('catalog')
            # print('Optimizing params:')
            # tr.printThawedParams()
        
            for step in range(50):
                dlnp, x, alpha = tr.optimize(**optargs)
                #print('dlnp', dlnp)
                #print('src', src)
                #print('psf', psf)
                if dlnp == 0:
                    break

            fwhms.append(2.35 * psf.sigmas[0]) # [pixels]
            #model = tr.getModelImage(0)
            #pdb.set_trace()
        
        return np.array(fwhms)

    def isolated_radec(self,ra,dec,nn=2,minsep=1./3600):
        '''return indices of ra,dec for which the ra,dec points are 
        AT LEAST a distance minsep away from their nearest neighbor point'''
        cat1 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
        cat2 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
        idx, d2d, d3d = cat1.match_to_catalog_3d(cat2,nthneighbor=nn)
        b= np.array(d2d) >= minsep
        return b

    def run(self, ext=None):
        self.set_hdu(ext)
        # 
        t0= Time()
        t0= ptime('Measuring CCD=%s from image=%s' % (self.ccdname,self.fn),t0)

        if self.camera == 'decam':
            # Simultaneous image,bitmask read
            # funpack optional (funpack = slower!)
            hdr, img, bitmask = self.read_image_and_bitmask(funpack=False)
        else:
            img,hdr= self.read_image() 
            bitmask= self.read_bitmask()
        img_mask= self.get_image_mask(img,bitmask)
        t0= ptime('read image, bitmask',t0)
        # Initialize and begin populating the output CCDs table.
        ccds = _ccds_table(self.camera)
        # starts with the decam/ mosaic/ or 90prime/ dir
        ccds['image_filename'] = self.fn[self.fn.rfind('/%s/' % self.camera)+1:]
        ccds['image_hdu'] = self.image_hdu 
        ccds['ccdnum'] = self.ccdnum 
        ccds['camera'] = self.camera
        ccds['expnum'] = self.expnum
        ccds['ccdname'] = self.ccdname
        ccds['expid'] = self.expid
        ccds['object'] = self.obj
        ccds['propid'] = self.propid
        ccds['filter'] = self.band
        ccds['exptime'] = self.exptime
        ccds['date_obs'] = self.date_obs
        ccds['mjd_obs'] = self.mjd_obs
        ccds['ut'] = self.ut
        ccds['ra_bore'] = self.ra_bore
        ccds['dec_bore'] = self.dec_bore
        ccds['ha'] = self.ha
        ccds['airmass'] = self.airmass
        ccds['gain'] = self.gain
        ccds['pixscale'] = self.pixscale
        # FWHM from CP header
        if self.camera in ['mosaic','90prime']:
            hdr_fwhm= hdr['seeingp1'] # pixel seeing so FWHM
        else:
            hdr_fwhm= hdr['fwhm']
        ccds['fwhm_cp']= hdr_fwhm
        # Copy some header cards directly.
        # ZNAXIS[12] not NAXIS
        hdrkey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                  'cd1_2', 'cd2_1', 'cd2_2', 'znaxis1', 'znaxis2')
        ccdskey = ('avsky', 'crpix1', 'crpix2', 'crval1', 'crval2', 'cd1_1',
                   'cd1_2', 'cd2_1', 'cd2_2', 'width', 'height')
        for ckey, hkey in zip(ccdskey, hdrkey):
            try:
                ccds[ckey] = hdr[hkey]
            except KeyError:
                if hkey == 'avsky':
                    print('CP image does not have avsky in hdr: %s' % ccds['image_filename'])
                    ccds[hkey]= -1
                else:
                    raise NameError('key not in header: %s' % hkey)
            
        exptime = ccds['exptime'].data[0]
        airmass = ccds['airmass'].data[0]
        print('Band {}, Exptime {}, Airmass {}'.format(self.band, exptime, airmass))

        # WCS: 1-indexed so pixel pixelxy2radec(1,1) corresponds to img[0,0]
        H, W = img.shape
        ccdra, ccddec = self.wcs.pixelxy2radec((W+1) / 2.0, (H + 1) / 2.0)
        ccds['ra'] = ccdra   # [degree]
        ccds['dec'] = ccddec # [degree]
        t0= ptime('header-info',t0)

        # Test WCS again IDL, WCS is 1-indexed
        #x_pix= [1,img.shape[0]/2,img.shape[0]]
        #y_pix= [1,img.shape[1]/2,img.shape[1]]
        #test_wcs= [(_x,_y)+self.wcs.pixelxy2radec(_x,_y) for _x,_y in zip(x_pix,y_pix)]
        #with open('three_camera_vals.txt','a') as foo:
        #    foo.write('ccdname=%s, hdu=%d, image=%s\n' % (self.ccdname,self.image_hdu,self.fn))
        #    foo.write('image shape: x=%d y=%d\n' % (img.shape[0],img.shape[1]))
        #    for i in test_wcs:
        #        foo.write('x=%d y=%d ra=%.9f dec=%.9f\n' % (i[0],i[1],i[2],i[3]))
        #return ccds, _stars_table()
        
        # Measure the sky brightness and (sky) noise level.  Need to capture
        # negative sky.
        sky0 = self.sky(self.band)
        zp0 = self.zeropoint(self.band)
        kext = self.extinction(self.band)
        print('Computing the sky background.')
        sky_img, skymed, skyrms = self.get_sky_and_sigma(img)
        img_sub_sky= img - sky_img

        #fn= 'N4.fits' 
        #fitsio.write(fn,img_sub_sky,extname='N4')
        #raise ValueError
        

        # Bunch of sky estimates
        # Median of absolute deviation (MAD), std dev = 1.4826 * MAD
        print('sky from median of image= %.2f' % skymed)
        skybr = zp0 - 2.5*np.log10(skymed / self.pixscale / self.pixscale / exptime)
        print('  Sky brightness: {:.3f} mag/arcsec^2'.format(skybr))
        print('  Fiducial:       {:.3f} mag/arcsec^2'.format(sky0))

        ccds['skyrms'] = skyrms / exptime # e/sec
        ccds['skycounts'] = skymed / exptime # [electron/pix]
        ccds['skymag'] = skybr   # [mag/arcsec^2]
        t0= ptime('measure-sky',t0)
        
        if self.debug:
            extra= {}
            extra['proj_fn']= os.path.join('/project/projectdirs/cosmo/staging',
                                           ccds['image_filename'].data[0])
            extra['hdu']= ccds['image_hdu'].data[0]

        # Detect stars on the image.  
        # 10 sigma, sharpness, roundness all same as IDL zeropoints (also the defaults)
        # Exclude_border=True removes the stars with centroid on or out of ccd edge
        # Good, but we want to remove with aperture touching ccd edge too
        print('det_thresh = %d' % self.det_thresh)
        #threshold=self.det_thresh * stddev_mad,
        dao = DAOStarFinder(fwhm= hdr_fwhm,
                            threshold=self.det_thresh * skyrms,
                            sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0,
                            exclude_border=False)
        obj= dao(img)

        if len(obj) < 20:
            dao.threshold /= 2.
            obj= dao(img)
        nobj = len(obj)
        print('{} sources detected with detection threshold {}-sigma'.format(nobj, self.det_thresh))
        ccds['nstarfind']= nobj

        if nobj == 0:
            print('No sources detected!  Giving up.')
            return ccds, _stars_table()
        t0= ptime('detect-stars',t0)

        # 1st round of cuts:  
        # stars too close to CCD edges which can have outlying cnts
        minsep = 1. + self.skyrad[1] #1'' buffer after 10 arcsec, same as Arjuns
        minsep_px = minsep/self.pixscale
        wid,ht= img.shape[1],img.shape[0] #2046,4096 for DECam
        istar =  (obj['xcentroid'] > minsep_px)*\
                 (obj['xcentroid'] < wid - minsep_px)*\
                 (obj['ycentroid'] > minsep_px)*\
                 (obj['ycentroid'] < ht - minsep_px)
        obj = obj[istar]

        if self.debug:
            extra['dao_x']= obj['xcentroid']
            extra['dao_y']= obj['ycentroid']
            extra['dao_ra'], extra['dao_dec'] = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)

        # Do aperture photometry in a fixed aperture but using either local (in
        # an annulus around each star) or global sky-subtraction.
        print('Performing aperture photometry')
        ap = CircularAperture((obj['xcentroid'], obj['ycentroid']), self.aprad / self.pixscale)
        if self.aper_sky_sub:
            print('**WARNING** using sky apertures for local sky subtraction')
            skyap = CircularAnnulus((obj['xcentroid'], obj['ycentroid']),
                                    r_in=self.skyrad[0] / self.pixscale, 
                                    r_out=self.skyrad[1] / self.pixscale)
            # Use skyap to subtractr local sky
            apphot = aperture_photometry(img, ap)
            #skyphot = aperture_photometry(img, skyap)
            skyphot = aperture_photometry(img, skyap, mask= img_mask > 0)
            apskyflux= skyphot['aperture_sum'] / skyap.area() * ap.area()
            apskyflux_perpix= skyphot['aperture_sum'] / skyap.area() 
            apflux = apphot['aperture_sum'] - apskyflux
        else:
            # ON image not sky subtracted image
            apphot = aperture_photometry(img, ap)
            apflux = apphot['aperture_sum']
            # Placeholders
            #apskyflux= apflux.copy()
            #apskyflux.fill(0.)
            #apskyflux_perpix= apskyflux.copy()
        t0= ptime('aperture-photometry',t0)
        # Get close enough sky/pixel in sky annulus
        # Take cutout of size ~ rout x rout, use same pixels in this slice for sky level
        rin,rout= self.skyrad[0]/self.pixscale, self.skyrad[1]/self.pixscale
        rad= int(np.ceil(rout)) #
        box= 2*rad + 1 # Odd integer so source exactly in center
        use_for_sky= np.zeros((box,box),bool)
        x,y= np.meshgrid(range(box),range(box)) # array valus are the indices
        ind_of_center= rad
        r= np.sqrt((x - ind_of_center)**2 + (y - ind_of_center)**2)
        use_for_sky[(r > rin)*(r <= rout)]= True
        # Get cutout around each source
        apskyflux,apskyflux_perpix=[],[]
        for x,y in zip(obj['xcentroid'].data,obj['ycentroid'].data):
            xc,yc= int(x),int(y)
            x_sl= slice(xc-rad,xc+rad+1)
            y_sl= slice(yc-rad,yc+rad+1)
            cutout= img[y_sl,x_sl]
            assert(cutout.shape == use_for_sky.shape)
            from astropy.stats import sigma_clipped_stats
            mean, median, std = sigma_clipped_stats(cutout[use_for_sky], sigma=3.0, iters=5)
            mode_est= 3*median - 2*mean
            apskyflux_perpix.append( mode_est )
        apskyflux_perpix = np.array(apskyflux_perpix) # cnts / pixel
        apskyflux= apskyflux_perpix * ap.area() # cnts / 7'' aperture
        t0= ptime('local-sky-photometry',t0)

        apflux= apflux - apskyflux

        # Remove stars if saturated within 5 pixels of centroid
        ap_for_mask = CircularAperture((obj['xcentroid'], obj['ycentroid']), 5.)
        phot_for_mask = aperture_photometry(img_mask, ap_for_mask)
        flux_for_mask = phot_for_mask['aperture_sum'] 
        # Aperture mags
        apmags= - 2.5 * np.log10(apflux.data) + zp0 + 2.5 * np.log10(exptime)
        # Good stars following IDL codes
        # We are ignoring aperature errors though
        # No stars within our skyrad_outer (10'')
        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        b_isolated= self.isolated_radec(objra,objdec,nn=2,minsep=minsep/3600.)
      
        # 2nd round of cuts:  
        # In order of biggest affect: isolated,apmags, apflux, flux_for_mask
        istar =  (apflux > 0)*\
                 (flux_for_mask == 0)*\
                 (apmags > 12.)*\
                 (apmags < 30.)*\
                 (b_isolated == True)
        print('First round of cuts, nstars=%d' % (np.where(istar)[0].size,))
        if self.debug:
            extra['apflux']= apflux > 0
            extra['flux_for_mask']= flux_for_mask == 0
            extra['apmags']= (apmags > 12.)*(apmags < 30.)
            extra['b_isolated']= b_isolated == True
        obj = obj[istar]
        objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        apflux = apflux[istar]
        apskyflux= apskyflux[istar]
        apskyflux_perpix= apskyflux_perpix[istar]
        if self.debug:
            extra['1st_x']= obj['xcentroid']
            extra['1st_y']= obj['ycentroid']
            extra['1st_ra'], extra['1st_dec'] = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        # 2nd round: isolated
        # If used isolated above thre would be a ton of faint or bad sources that would be...
        # ...close to and remove bright or good sources 
        # No stars within our skyrad_outer (10'')
        #objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        #b_isolated= self.isolated_radec(objra,objdec,nn=2,minsep=minsep/3600.)
        # 
        #istar =  (b_isolated == True)
        #print('Second round of cuts, nstars=%d' % (np.where(istar)[0].size,))
        #extra['b_isolated']= b_isolated == True
        nidl=np.where(istar)[0].size

        #obj = obj[istar]
        #objra, objdec = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)
        #apflux = apflux[istar].data
        #apskyflux= apskyflux[istar].data
        #apskyflux_perpix= apskyflux_perpix[istar].data
        #extra['2nd_x']= obj['xcentroid']
        #extra['2nd_y']= obj['ycentroid']
        #extra['2nd_ra'], extra['2nd_dec'] = self.wcs.pixelxy2radec(obj['xcentroid']+1, obj['ycentroid']+1)

        # 3rd Optional Cut: SN
        if self.sn_min or self.sn_max:
            sn= apflux.data / np.sqrt(apskyflux)
            if self.sn_min:
                above= sn >= self.sn_min
                istar *= (above)
                print('Stars with SN < %d: %d' % (self.sn_min,np.where(above == False)[0].size))
            if self.sn_max:
                below= sn <= self.sn_max
                istar *= (below)
                print('Stars with SN > %d: %d' % (self.sn_max,np.where(below == False)[0].size))
            nmore= nidl - np.where(istar)[0].size
            print('Additional %d removed that were not already flagged' % nmore)
            print('Stars after SN cuts: %d' % (np.where(istar)[0].size,))
        else:
            print('No additional sn_cut')

        ccds['nstar']= np.where(istar)[0].size
        if ccds['nstar'] == 0:
            print('FAIL: All stars have negative aperture photometry AND/OR contain masked pixels!')
            return ccds, _stars_table()
        
        # Now match against (good) PS1 stars 
        try: 
            ps1 = ps1cat(ccdwcs=self.wcs).get_stars() #magrange=(15, 22))
        except IOError:
            # The gaia file does not exist:
            # e.g. /project/projectdirs/cosmo/work/gaia/chunks-ps1-gaia/chunk-*.fits
            with open('zpts_bad_nogaiachunk.txt','a') as foo:
                foo.write('%s %s\n' % (self.fn,self.image_hdu))
            return ccds, _stars_table()
        # Are there Good PS1 on this CCD?
        if len(ps1) == 0:
            with open('zpts_bad_nops1onccd.txt','a') as foo:
                foo.write('%s %s\n' % (self.fn,self.image_hdu))
            return ccds, _stars_table()
        good = (ps1.nmag_ok[:, 0] > 0)*(ps1.nmag_ok[:, 1] > 0)*(ps1.nmag_ok[:, 2] > 0)
        gicolor= ps1.median[:,0] - ps1.median[:,2]
        good*= (gicolor > 0.4)*(gicolor < 2.7)
        # Cut 0.5 deg from CCD center and non star colors
        #gdec=ps1.dec_ok-ps1.ddec/3600000.
        #gra=ps1.ra_ok-ps1.dra/3600000./np.cos(np.deg2rad(gdec))
        #gaia_cat = SkyCoord(ra=gra*units.degree, dec=gdec*units.degree)
        #center_ccd = SkyCoord(ra=ccds['ra']*units.degree, dec=ccds['dec']*units.degree)
        #ang = gaia_cat.separation(center_ccd) 
        # Zeropoint Sample is Main Sequence stars
        #good*= (np.array(ang) < 0.50)*(gicolor > 0.4)*(gicolor < 2.7)
        # final cut
        #good = np.where(good)[0]
        ps1.cut(good)
        gdec=ps1.dec_ok-ps1.ddec/3600000.
        gra=ps1.ra_ok-ps1.dra/3600000./np.cos(np.deg2rad(gdec))
        nps1 = len(ps1)

        if nps1 == 0:
            print('No overlapping PS1 stars in this field!')
            return ccds, _stars_table()
    
        # Match GAIA and Our Data
        m1, m2, d12 = match_radec(objra, objdec, gra, gdec, self.match_radius/3600.0,\
                                  nearest=True)
        ccds['nmatch'] = len(m1)
        print('{} GAIA sources match detected sources within {} arcsec.'.format(ccds['nmatch'], self.match_radius))
        t0= ptime('match-to-gaia-radec',t0)

        # Stars table 
        print('Add the amplifier number!!!')
        stars = _stars_table(ccds['nmatch'])
        stars['image_filename'] =ccds['image_filename']
        stars['image_hdu']= ccds['image_hdu'] 
        stars['expnum'] = self.expnum
        stars['expid'] = self.expid
        stars['filter'] = self.band
        stars['gain'] = self.gain
        stars['exptime'] = exptime
        # Matched quantities
        stars['nmatch'] = ccds['nmatch'] 
        stars['x'] = obj['xcentroid'][m1]
        stars['y'] = obj['ycentroid'][m1]
        #
        stars['ra'] = objra[m1]
        stars['dec'] = objdec[m1]
        stars['radiff'] = (gra[m2] - stars['ra']) * np.cos(np.deg2rad(stars['dec'])) * 3600.0
        stars['decdiff'] = (gdec[m2] - stars['dec']) * 3600.0
        stars['apmag'] = - 2.5 * np.log10(apflux[m1]) + zp0 + 2.5 * np.log10(exptime)
        stars['apflux'] = apflux[m1]
        stars['apskyflux'] = apskyflux[m1]
        stars['apskyflux_perpix'] = apskyflux_perpix[m1]
        # Additional x,y
        #b= np.zeros(len(obj),bool)
        #b[m1]= True
        if self.debug:
            extra['x'] = stars['x']
            extra['y'] = stars['y']
            extra['ra'], extra['dec'] = stars['ra'].data,stars['dec'].data
            extra['apflux'], extra['apskyflux'] = stars['apflux'].data,stars['apskyflux'].data
        
        # FWHM from Tractor
        # SN from sky_img aperture photometry
        ap = CircularAperture((stars['x'], stars['y']), self.aprad / self.pixscale)
        skyphot = aperture_photometry(sky_img, ap)
        skyflux = skyphot['aperture_sum'].data
        t0= ptime('sky_img aperture photometry',t0)
        star_SN= stars['apflux'].data / np.sqrt(stars['apflux'].data + skyflux)
 
        # SN cut because interactive iraf gives best FWHM when star not too bright
        sn_cut = (star_SN >= 10.)*(star_SN <= 100.)
        # Only tractoring nstars is approx. random selection of nstars within sn
        sample=dict(x= stars['x'][sn_cut][:self.tractor_nstars],
                    y= stars['y'][sn_cut][:self.tractor_nstars],
                    apflux= stars['apflux'][sn_cut][:self.tractor_nstars],
                    sn= star_SN[sn_cut][:self.tractor_nstars])
        #ivar = np.zeros_like(img) + 1.0/sig1**2
        # Hack! To avoid 1/0 and sqrt(<0) just considering Poisson Stats due to sky
        ierr = 1.0/np.sqrt(sky_img)
        fwhms = self.fitstars(img_sub_sky, ierr, sample['x'], sample['y'], sample['apflux'])
        ccds['fwhm'] = np.median(fwhms) # fwhms= 2.35 * psf.sigmas 
        print('FWHM med=%f, std=%f, std_med=%f' % (np.median(fwhms),np.std(fwhms),np.std(fwhms)/len(sample['x'])))
        #ccds['seeing'] = self.pixscale * np.median(fwhms)
        t0= ptime('Tractor fit FWHM to %d/%d stars' % (len(sample['x']),len(stars)), t0) 

        ## Add ps1 astrometric residuals for comparison
        #ps1_m1, ps1_m2, ps1_d12 = match_radec(objra, objdec, ps1.ra, ps1.dec, self.matchradius/3600.0,\
        #                                      nearest=True)
        ## If different number gaia matches versus ps1 matches, need to handle
        #num_gaia= len(stars['apmag'])
        #stars['radiff_ps1'] = (ps1.ra[ps1_m2][:num_gaia] - objra[ps1_m1][:num_gaia]) * np.cos(np.deg2rad(objdec[ps1_m1][:num_gaia])) * 3600.0
        #stars['decdiff_ps1'] = (ps1.dec[ps1_m2][:num_gaia] - objdec[ps1_m1][:num_gaia]) * 3600.0
        # Photometry
        # Unless we're calibrating the photometric transformation, bring PS1
        # onto the photometric system of this camera (we add the color term
        # below).
        if self.calibrate:
            raise ValueError('not Calibrating PS1 to our Camera, are you sure?')
            colorterm = np.zeros(ccds['nmatch'])
        else:
            colorterm = self.colorterm_ps1_to_observed(ps1.median[m2, :], self.band)
        ps1band = ps1cat.ps1band[self.band]
        # g-band DECAM,MzLS,or BASS = g-band PS1 - poly(gicolor, gcoeff)
        stars['ps1_mag'] = ps1.median[m2, ps1band] + colorterm
        # Additonal mags for comparison with Arjun's star sample
        # PS1 Median PSF mag in [g,r,i,z],  Gaia G-band mean magnitude
        for ps1_band,ps1_index in zip(['g','r','i','z'],[0,1,2,3]):
            stars['ps1_%s' % ps1_band]= ps1.median[m2, ps1_index]
        stars['gaia_g']=ps1.phot_g_mean_mag[m2]
        #print('Computing the photometric zeropoint.')
        #stars['ps1_gicolor'] = ps1.median[m2, 0] - ps1.median[m2, 2]
        #print('Before gicolor cut, len(stars)=%d' % len(stars['ps1_gicolor']))
        #mskeep = np.where((stars['ps1_gicolor'] > 0.4) * (stars['ps1_gicolor'] < 2.7))[0]
        #if len(mskeep) == 0:
        #    print('Not enough PS1 stars with main sequence colors.')
        #    return ccds, stars
        #ccds['mdncol'] = np.median(stars['ps1_gicolor'][mskeep]) # median g-i color
        #print('After gicolor cut, len(stars)=%d' % len(mskeep))
        
        # Compute Zeropoint
        #dmagall = stars['ps1_mag'][mskeep] - stars['apmag'][mskeep]
        assert(len(stars['ps1_mag']) == ccds['nmatch'])
        dmagall = stars['ps1_mag'] - stars['apmag']
        dmag, _, _ = sigmaclip(dmagall, low=2.5, high=2.5)
        dmagmed = np.median(dmag)
        dmagsig = np.std(dmag)  # agrees with IDL codes, they just compute std
        ndmag = len(dmag)

        zptmed = zp0 + dmagmed
        transp = 10.**(-0.4 * (zp0 - zptmed - kext * (airmass - 1.0)))

        stars['dmagall']= dmagall

        t0= ptime('photometry-using-ps1',t0)
        ccds['raoff'] = np.median(stars['radiff'])
        ccds['decoff'] = np.median(stars['decdiff'])
        ccds['rastddev'] = np.std(stars['radiff'])
        ccds['decstddev'] = np.std(stars['decdiff'])
        ra_clip, _, _ = sigmaclip(stars['radiff'], low=3., high=3.)
        ccds['rarms'] = getrms(ra_clip)
        dec_clip, _, _ = sigmaclip(stars['decdiff'], low=3., high=3.)
        ccds['decrms'] = getrms(dec_clip)
        ccds['phoff'] = dmagmed
        ccds['phrms'] = dmagsig
        ccds['zpt'] = zptmed
        ccds['transp'] = transp       
        print('RA, Dec offsets (arcsec) relative to GAIA: %.4f, %.4f' % (ccds['raoff'], ccds['decoff']))
        print('RA, Dec rms (arcsec) relative to GAIA: %.4f, %.4f' % (ccds['rarms'], ccds['decrms']))
        print('RA, Dec stddev (arcsec) relative to GAIA: %.4f, %.4f' % (ccds['rastddev'], ccds['decstddev']))
        print('Mag offset: %.4f' % (ccds['phoff'],))
        print('Scatter: %.4f' % (ccds['phrms'],))
        
        print('Number stars used for zeropoint median %d' % ndmag)
        print('Zeropoint %.4f' % (ccds['zpt'],))
        print('Transparency %.4f' % (ccds['transp'],))

        t0= ptime('all-computations-for-this-ccd',t0)
        # Plots for comparing to Arjuns zeropoints*.ps
        if self.verboseplots:
            self.make_plots(stars,dmag,ccds['zpt'],ccds['transp'])
            t0= ptime('made-plots',t0)

       
        if self.debug:
            # Save extra
            extra_fn= os.path.basename(ccds['image_filename'].data[0]).replace('.fits.fz','') + \
                      '-%s-' % extra['hdu'] + 'extra.pkl'
            extra_fn= os.path.join(os.path.dirname(self.fn),
                                   extra_fn)
            with open(extra_fn,'w') as foo:
                dump(extra,foo)
            print('Wrote %s' % extra_fn)
 
        return ccds, stars
    
    def make_plots(self,stars,dmag,zpt,transp):
        '''stars -- stars table'''
        suffix='_qa_%s.png' % stars['expid'][0][-4:]
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        plt.subplots_adjust(wspace=0.2,bottom=0.2,right=0.8)
        for key in ['astrom_gaia','photom']:
        #for key in ['astrom_gaia','astrom_ps1','photom']:
            if key == 'astrom_gaia':    
                ax[0].scatter(stars['radiff'],stars['decdiff'])
                xlab=ax[0].set_xlabel(r'$\Delta Ra$ (Gaia - CCD)')
                ylab=ax[0].set_ylabel(r'$\Delta Dec$ (Gaia - CCD)')
            elif key == 'astrom_ps1':  
                raise ValueError('not needed')  
                ax.scatter(stars['radiff_ps1'],stars['decdiff_ps1'])
                ax.set_xlabel(r'$\Delta Ra [arcsec]$ (PS1 - CCD)')
                ax.set_ylabel(r'$\Delta Dec [arcsec]$ (PS1 - CCD)')
                ax.text(0.02, 0.95,'Median: %.4f,%.4f' % \
                          (np.median(stars['radiff_ps1']),np.median(stars['decdiff_ps1'])),\
                        va='center',ha='left',transform=ax.transAxes,fontsize=20)
                ax.text(0.02, 0.85,'RMS: %.4f,%.4f' % \
                          (getrms(stars['radiff_ps1']),getrms(stars['decdiff_ps1'])),\
                        va='center',ha='left',transform=ax.transAxes,fontsize=20)
            elif key == 'photom':
                ax[1].hist(dmag)
                xlab=ax[1].set_xlabel('PS1 - AP mag (main seq, 2.5 clipping)')
                ylab=ax[1].set_ylabel('Number of Stars')
        # List key numbers
        ax[1].text(1.02, 1.,r'$\Delta$ Ra,Dec',\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=12)
        ax[1].text(1.02, 0.9,r'  Median: %.4f,%.4f' % \
                  (np.median(stars['radiff']),np.median(stars['decdiff'])),\
                va='center',ha='left',transform=ax[1].transAxes,fontsize=10)
        ax[1].text(1.02, 0.80,'  RMS: %.4f,%.4f' % \
                  (getrms(stars['radiff']),getrms(stars['decdiff'])),\
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
        fn= self.zptsfile.replace('.fits',suffix)
        plt.savefig(fn,bbox_extra_artists=[xlab,ylab])
        plt.close()
        print('Wrote %s' % fn)
   
 
class DecamMeasurer(Measurer):
    '''DECam CP units: ADU
    Class to measure a variety of quantities from a single DECam CCD.

    Image read will be converted to e-
    also zpt to e-
    '''
    def __init__(self, *args, **kwargs):
        super(DecamMeasurer, self).__init__(*args, **kwargs)

        self.pixscale=0.262 
        self.camera = 'decam'
        self.ut = self.primhdr['TIME-OBS']
        self.band = self.get_band()
        # {RA,DEC}: center of exposure, TEL{RA,DEC}: boresight of telescope
        self.ra_bore = hmsstring2ra(self.primhdr['RA']) 
        self.dec_bore = dmsstring2dec(self.primhdr['DEC'])
        #self.gain = self.hdr['ARAWGAIN'] # hack! average gain [electron/sec]

        # /global/homes/a/arjundey/idl/pro/observing/decstat.pro
        self.zp0 =  dict(g = 26.610,r = 26.818,z = 26.484) # e/sec
        self.sky0 = dict(g = 22.04,r = 20.91,z = 18.46) # AB mag/arcsec^2
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain)  
    
    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band

    def get_gain(self,hdr):
        return np.average((hdr['GAINA'],hdr['GAINB']))
        #return hdr['ARAWGAIN']

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_decam
        return ps1_to_decam(ps1stars, band)

    def read_image_and_bitmask(self,funpack=True):
        '''funpack, then read'''
        imgfn= self.fn
        maskfn= get_bitmask_fn(self.fn)
        print('Reading %s %s' % (imgfn,maskfn))
        if funpack:
            todelete=[]
            imgfn,maskfn = funpack_files(imgfn, maskfn, self.ext, todelete)
            # Read
            img, hdr = fitsio.read(imgfn, ext=self.ext, header=True)
            mask, junk = fitsio.read(maskfn, ext=self.ext, header=True)
            for fn in todelete:
               os.unlink(fn)
        else:
            # Read
            img, hdr = fitsio.read(imgfn, ext=self.ext, header=True)
            mask, junk = fitsio.read(maskfn, ext=self.ext, header=True)
        # ADU --> e
        img *= self.gain 
        return hdr,img,mask
    
    def read_image(self):
        '''Read the image and header.  Convert image from ADU to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        #fits=fitsio.FITS(fn,mode='r',clobber=False,lower=True)
        #hdr= fits[0].read_header()
        #img= fits[ext].read()
        #img *= self.gain
        #img *= self.gain / self.exptime
        img *= self.gain 
        return img, hdr
    
    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion
    
class Mosaic3Measurer(Measurer):
    '''Class to measure a variety of quantities from a single Mosaic3 CCD.
    UNITS: e-/s'''
    def __init__(self, *args, **kwargs):
        super(Mosaic3Measurer, self).__init__(*args, **kwargs)

        self.pixscale=0.262 # 0.260 is right, but mosstat.pro has 0.262
        self.camera = 'mosaic'
        self.band= self.get_band()
        self.ut = self.primhdr['TIME-OBS']
        # {RA,DEC}: center of exposure, TEL{RA,DEC}: boresight of telescope
        self.ra_bore = hmsstring2ra(self.primhdr['RA'])
        self.dec_bore = dmsstring2dec(self.primhdr['DEC'])
        # ARAWGAIN does not exist, 1.8 or 1.94 close
        #self.gain = self.hdr['GAIN']

        self.zp0 = dict(z = 26.552)
        self.sky0 = dict(z = 18.46)
        self.k_ext = dict(z = 0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain)  

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0][0] # zd --> z
        return band

    def get_gain(self,hdr):
        return hdr['GAIN']
        #return np.average((hdr['GAINA'],hdr['GAINB']))
        #return hdr['ARAWGAIN']

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_mosaic
        return ps1_to_mosaic(ps1stars, band)

    def read_image(self):
        '''Read the image and header.  Convert image from electrons/sec to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        #fits=fitsio.FITS(fn,mode='r',clobber=False,lower=True)
        #hdr= fits[0].read_header()
        #img= fits[ext].read()
        img *= self.exptime 
        return img, hdr

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion

class NinetyPrimeMeasurer(Measurer):
    '''Class to measure a variety of quantities from a single 90prime CCD.
    UNITS -- CP e-/s'''
    def __init__(self, *args, **kwargs):
        super(NinetyPrimeMeasurer, self).__init__(*args, **kwargs)
        
        self.pixscale= 0.470 # 0.455 is correct, but mosstat.pro has 0.470
        self.camera = '90prime'
        self.band= self.get_band()
        # {RA,DEC}: center of exposure, doesn't have TEL{RA,DEC}
        self.ra_bore = hmsstring2ra(self.primhdr['RA'])
        self.dec_bore = dmsstring2dec(self.primhdr['DEC'])
        self.ut = self.primhdr['UT']

        # Can't find what people are using for this!
        # 1.4 is close to average of hdr['GAIN[1-16]']
        #self.gain= 1.4 
        
        #self.gain = np.average((self.hdr['GAINA'],self.hdr['GAINB'])) 
        # Average (nominal) gain values.  The gain is sort of a hack since this
        # information should be scraped from the headers, plus we're ignoring
        # the gain variations across amplifiers (on a given CCD).
        #gaindict = dict(ccd1 = 1.47, ccd2 = 1.48, ccd3 = 1.42, ccd4 = 1.4275)
        #self.gain = gaindict[self.ccdname.lower()]

        # Nominal zeropoints, sky brightness, and extinction values (taken from
        # rapala.ninetyprime.boketc.py).  The sky and zeropoints are both in
        # ADU, so account for the gain here.
        #corr = 2.5 * np.log10(self.gain)

        # /global/homes/a/arjundey/idl/pro/observing/bokstat.pro
        self.zp0 =  dict(g = 26.93,r = 27.01,z = 26.552) # ADU/sec
        self.sky0 = dict(g = 22.04,r = 20.91,z = 18.46) # AB mag/arcsec^2
        self.k_ext = dict(g = 0.17,r = 0.10,z = 0.06)
        # --> e/sec
        #for b in self.zp0.keys(): 
        #    self.zp0[b] += -2.5*np.log10(self.gain)  
    
    def get_gain(self,hdr):
        self.gain= 1.4 # no GAINA,B

    def get_band(self):
        band = self.primhdr['FILTER']
        band = band.split()[0]
        return band.replace('bokr', 'r')

    def colorterm_ps1_to_observed(self, ps1stars, band):
        from legacyanalysis.ps1cat import ps1_to_90prime
        return ps1_to_90prime(ps1stars, band)

    def read_image(self):
        '''Read the image and header.  Convert image from electrons/sec to electrons.'''
        img, hdr = fitsio.read(self.fn, ext=self.ext, header=True)
        img *= self.exptime
        return img, hdr

    def get_wcs(self):
        return wcs_pv2sip_hdr(self.hdr) # PV distortion


def get_extlist(camera):
    '''
    Returns 'mosaic', 'decam', or '90prime'
    '''
    if camera == '90prime':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
    elif camera == 'mosaic':
        extlist = ['CCD1', 'CCD2', 'CCD3', 'CCD4']
    elif camera == 'decam':
        extlist = ['S29', 'S31', 'S25', 'S26', 'S27', 'S28', 'S20', 'S21', 'S22',
                   'S23', 'S24', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S8',
                   'S9', 'S10', 'S11', 'S12', 'S13', 'S1', 'S2', 'S3', 'S4', 'S5',
                   'S6', 'S7', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9',
                   'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18',
                   'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27',
                   'N28', 'N29', 'N31']
        # Testing only!
        #extlist = ['N4','S4', 'S22','N19']
        #extlist = ['N4']
        #extlist = ['S10', 'S11', 'S12', 'S16', 'S17', 'S4', 'S5', 'S6']
    else:
        print('Camera {} not recognized!'.format(camera))
        pdb.set_trace() 
    return extlist
   
 
#def measure_mosaic3(fn, ext='CCD1', **kwargs):
#    '''Wrapper function to measure quantities from the Mosaic3 camera.'''
#    measure = Mosaic3Measurer(fn, ext, **kwargs)
#    ccds, stars = measure.run()
#    return ccds, stars
#
#def measure_90prime(fn, ext='CCD1', **kwargs):
#    '''Wrapper function to measure quantities from the 90prime camera.'''
#    measure = NinetyPrimeMeasurer(fn, ext, **kwargs)
#    ccds, stars = measure.run()
#    return ccds, stars
#
#def measure_decam(fn, ext='N4', **kwargs):
#    '''Wrapper function to measure quantities from the DECam camera.'''
#    measure = DecamMeasurer(fn, ext, **kwargs)
#    ccds, stars = measure.run()
#    return ccds, stars

def _measure_image(args):
    '''Utility function to wrap measure_image function for multiprocessing map.''' 
    return measure_image(*args)

def measure_image(img_fn, **measureargs): 
    '''Wrapper on the camera-specific classes to measure the CCD-level data on all
    the FITS extensions for a given set of images.
    '''
    t0= Time()

    print('Working on image {}'.format(img_fn))

    # Fitsio can throw error: ValueError: CONTINUE not supported
    try:
        primhdr = fitsio.read_header(img_fn, ext=0)
    except ValueError:
        # astropy can handle it
        tmp= fits_astropy.open(img_fn)
        primhdr= tmp[0].header
        tmp.close()
        del tmp
#        # skip zpt for this image 
#        print('Error reading img_fn=%s, see %s' % \
#                (img_fn,'zpts_bad_headerskipimage.txt')) 
#        with open('zpts_bad_headerskipimage.txt','a') as foo:
#            foo.write('%s\n' % (img_fn,))
#        ccds = []
#        stars = []
#        # FIX ME!! 4 should depend on camera, 60 for decam, 4 for mosaic,bok
#        for cnt in range(4):
#            ccds.append( _ccds_table() )
#            stars.append( _stars_table() )
#        ccds = vstack(ccds)
#        stars = vstack(stars)
#        return ccds,stars
    
    camera= measureargs['camera']
    camera_check = primhdr.get('INSTRUME','').strip().lower()
    # mosaic listed as mosaic3 in hearder, other combos maybe
    assert(camera in camera_check or camera_check in camera)
    
    extlist = get_extlist(camera)
    nnext = len(extlist)

    if camera == 'decam':
        measure = DecamMeasurer(img_fn, **measureargs)
    elif camera == 'mosaic':
        measure = Mosaic3Measurer(img_fn, **measureargs)
    elif camera == '90prime':
        measure = NinetyPrimeMeasurer(img_fn, **measureargs)
    extra_info= dict(zp_fid= measure.zeropoint( measure.band ),
                     sky_fid= measure.sky( measure.band ),
                     ext_fid= measure.extinction( measure.band ),
                     exptime= measure.exptime,
                     pixscale= measure.pixscale)
    
    ccds = []
    stars = []
    for ext in extlist:
        ccds1, stars1 = measure.run(ext)
        t0= ptime('measured-ext-%s' % ext,t0)
        ccds.append(ccds1)
        stars.append(stars1)

    # Compute the median zeropoint across all the CCDs.
    ccds = vstack(ccds)
    stars = vstack(stars)
    ccds['zptavg'] = np.median(ccds['zpt'])

    t0= ptime('measure-image-%s' % img_fn,t0)
        
    return ccds, stars,extra_info


class outputFns(object):
    def __init__(self,imgfn_proj,outdir,prefix='',**kwargs):
        '''
        outdir/decam/DECam_CP/CP20151226/img_fn.fits.fz
        outdir/decam/DECam_CP/CP20151226/img_fn-zpt%s.fits
        outdir/decam/DECam_CP/CP20151226/img_fn-star%s.fits
        '''
        camera= kwargs.get('camera')
        if camera == 'decam':
            proj_name= camera
        elif camera == 'mosaic':
            proj_name= camera+'z'
        elif camera == '90prime': 
            proj_name= 'bok'
        proj_dir= '/project/projectdirs/cosmo/staging/%s/' % proj_name
        root= imgfn_proj.replace(proj_dir,'')
        # Names
        dr= os.path.join(outdir,camera,root)
        # Image fn that will be on SCRATCH
        self.imgfn_scr= dr #os.path.join(dr,'%s.fits.fz' % base)
        # zpt filenames
        base= dr.replace('.fits.fz','')
        self.zptfn= base + '-zpt.fits' #os.path.join(dr,'%s-zpt%s.fits' % (base,prefix))
        self.starfn= base + '-star.fits' #os.path.join(dr,'%s-star%s.fits' % (base,prefix))

def get_output_fns(img_fn,prefix=''):
    zptsfile= os.path.dirname(img_fn).replace('/project/projectdirs','/scratch2/scratchdirs/kaylanb')
    zptsfile= os.path.join(zptsfile,'zpts/','%szeropoint-%s' % (prefix,os.path.basename(img_fn)))
    zptsfile= zptsfile.replace('.fz','')
    zptstarsfile = zptsfile.replace('.fits','-stars.fits')
    return zptsfile,zptstarsfile

def success(ccds, **measureargs):
    num_ccds= dict(decam=60,mosaic=4)
    num_ccds['90prime']=4
    camera= measureargs.get('camera')
    if len(ccds) >= num_ccds.get(camera,0):
        return True
    elif measureargs.get('debug') and len(ccds) >= 1:
        # only 1 ccds needs to be done if debuggin
        return True
    else:
        return False


def runit(imgfn_proj, **measureargs):
    '''Generate a legacypipe-compatible CCDs file for a given image.
    '''
    zptfn= measureargs.get('zptfn')
    starfn= measureargs.get('starfn')
    imgfn_scr= measureargs.get('imgfn_scr')
    # mask fn
    dqfn_proj= get_bitmask_fn(imgfn_proj)
    dqfn_scr= get_bitmask_fn(imgfn_scr)

    t0 = Time()
    for mydir in [os.path.dirname(zptfn),\
                  os.path.dirname(imgfn_scr)]:
        try_mkdir(mydir)

    # Copy to SCRATCH for improved I/O
    if not os.path.exists(imgfn_scr): 
        dobash("cp %s %s" % (imgfn_proj,imgfn_scr))
    if not os.path.exists(dqfn_scr): 
        dobash("cp %s %s" % (dqfn_proj, dqfn_scr))
    t0= ptime('copy-to-scratch',t0)

    ccds, stars, extra_info= measure_image(imgfn_scr, **measureargs)
    t0= ptime('measure_image',t0)

    # Only write if all CCDs are done
    if success(ccds,**measureargs):
        # Write out.
        ccds.write(zptfn)
        # Add header info
        hdulist = fits_astropy.open(zptfn, mode='update')
        prihdr = hdulist[0].header
        for key,val in extra_info.items():
            prihdr[key] = val
        hdulist.close() # Save changes
        print('Wrote {}'.format(zptfn))
        # zpt --> Legacypipe table
        create_legacypipe_table(zptfn)
        # Star table
        stars.write(starfn)
        print('Wrote {}'.format(starfn))
        # star --> Arjun's  matches table
        create_matches_table(starfn,**extra_info)
        # Clean up
        t0= ptime('write-results-to-fits',t0)
    else:
        print('FAILED, only %d CCDs, %s' % (len(ccds),imgfn_proj))
    if os.path.exists(imgfn_scr): 
        # Safegaurd against removing stuff on /project
        assert(not 'project' in imgfn_scr)
        dobash("rm %s" % imgfn_scr)
        dobash("rm %s" % dqfn_scr)
        t0= ptime('removed-cp-from-scratch',t0)
    
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
                                     description='Generate a legacypipe-compatible CCDs file \
                                                  from a set of reduced imaging.')
    parser.add_argument('--camera',choices=['decam','mosaic','90prime'],action='store',required=True)
    parser.add_argument('--image',action='store',default=None,help='if want to run a single image',required=False)
    parser.add_argument('--image_list',action='store',default=None,help='if want to run all images in a text file, Note:if compare2arjun = True then list of legacy zeropoint files',required=False)
    parser.add_argument('--outdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--debug', action='store_true', default=False, help='Write additional files and plots for debugging')
    parser.add_argument('--det_thresh', type=float, default=10., help='source detection, 10x sky sigma')
    parser.add_argument('--match_radius', type=float, default=1., help='arcsec, matching to gaia/ps1, 1 arcsec better astrometry than 3 arcsec as used by IDL codes')
    parser.add_argument('--sn_min', type=float,default=None, help='min S/N, optional cut on apflux/sqrt(skyflux)')
    parser.add_argument('--sn_max', type=float,default=None, help='max S/N, ditto')
    parser.add_argument('--aper_sky_sub', action='store_true',default=False,
                        help='Changes local sky subtraction step. Do aperture sky subraction instead of subtracting legacypipe splinesky')
    parser.add_argument('--logdir', type=str, default='.', help='Where to write zpts/,images/,logs/')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--verboseplots', action='store_true', default=False, help='use to plot FWHM Moffat PSF fits to the 20 brightest stars')
    parser.add_argument('--compare2arjun', action='store_true', default=False, help='turn this on and give --image-list a list of legacy zeropoint files instead of cp images')
    parser.add_argument('--aprad', type=float, default=3.5, help='Aperture photometry radius (arcsec).')
    parser.add_argument('--skyrad_inner', type=float, default=7.0, help='Radius of inner sky annulus (arcsec).')
    parser.add_argument('--skyrad_outer', type=float, default=10.0, help='Radius of outer sky annulus (arcsec).')
    parser.add_argument('--calibrate', action='store_true',
                        help='Use this option when deriving the photometric transformation equations.')
    parser.add_argument('--nproc', type=int,action='store',default=1,
                        help='set to > 1 if using legacy-zeropoints-mpiwrapper.py')
    return parser


def main(image_list=None,args=None): 
    ''' Produce zeropoints for all CP images in image_list
    image_list -- iterable list of image filenames
    args -- parsed argparser objection from get_parser()'''
    assert(not args is None)
    assert(not image_list is None)
    t0 = Time()
    tbegin=t0
    
    if args.compare2arjun:
        comp= Compare2Arjuns(args.image_list)
        sys.exit("Finished compaison to Arjun's zeropoints")

    # Build a dictionary with the optional inputs.
    measureargs = vars(args)
    if not args.compare2arjun:
        measureargs.pop('compare2arjun')
    measureargs.pop('image_list')
    measureargs.pop('image')
    # Add user specified camera, useful check against primhdr
    #measureargs.update(dict(camera= args.camera))

    outdir = measureargs.pop('outdir')
    try_mkdir(outdir)
    t0=ptime('parse-args',t0)
    for imgfn_proj in image_list:
        # Check if zpt already written
        F= outputFns(imgfn_proj,outdir,**measureargs)
        if os.path.exists(F.zptfn) and os.path.exists(F.starfn):
            print('Already finished: %s' % F.zptfn)
            continue
        measureargs.update(dict(zptfn= F.zptfn,\
                                starfn= F.starfn,\
                                imgfn_scr= F.imgfn_scr))
        # Create the file
        t0=ptime('b4-run',t0)
        runit(imgfn_proj, **measureargs)
        #try: 
        #    runit(imgfn_proj, **measureargs)
        #except:
        #    print('zpt failed for %s' % imgfn_proj)
        t0=ptime('after-run',t0)
    tnow= Time()
    print("TIMING:total %s" % (tnow-tbegin,))
    print("Done")
   
if __name__ == "__main__":
    parser= get_parser()  
    args = parser.parse_args()
    if args.image_list:
        images= read_lines(args.image_list) 
    elif args.image:
        images= [args.image]

    main(image_list=images,args=args)


