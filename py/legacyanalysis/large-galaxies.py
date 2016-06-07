#!/usr/bin/env python

"""Redo the Tractor photometry of the "large" galaxies in Legacy Survey imaging.

J. Moustakas
Siena College
2016 June 6

"""
from __future__ import division, print_function

import os
import sys
import pdb
import argparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table, vstack

from astrometry.util.util import Tan
from astrometry.util.fits import merge_tables
from legacypipe.common import ccds_touching_wcs, LegacySurveyData

def _getfiles(ccds):
    '''Figure out the set of images and calibration files we need to transfer, if any.'''

    ccdfile = []
    [ccdfile.append('{}-{}'.format(expnum, ccdname)) for expnum, ccdname in zip(ccds.expnum, ccds.ccdname)]
    _, indx = np.unique(ccdfile, return_index=True)

    ccdinfo = ccds[indx]
    nccd = len(ccdinfo)

    expnum = ccdinfo.expnum
    ccdname = ccdinfo.ccdname

    psffiles = []
    skyfiles = []
    imagefiles = []
    for ii in range(nccd):
        exp = '{0:08d}'.format(expnum[ii])
        rootfile = os.path.join(exp[:5], exp, 'decam-'+exp+'-'+ccdname[ii]+'.fits')
        psffiles.append(os.path.join('calib', 'decam', 'psfex', rootfile))
        skyfiles.append(os.path.join('calib', 'decam', 'splinesky', rootfile))
        imagefiles.append(os.path.join('images', str(np.core.defchararray.strip(ccdinfo.image_filename[ii]))))

    ccdfiles = open('/tmp/ccdfiles.txt', 'w')
    for ii in range(nccd):
        ccdfiles.write(psffiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(skyfiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii]+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii].replace('ooi', 'oow')+'\n')
    for ii in range(nccd):
        ccdfiles.write(imagefiles[ii].replace('ooi', 'ood')+'\n')
    ccdfiles.close()

    cmd = "rsync -avP --files-from='/tmp/ccdfiles.txt' cori:/global/cscratch1/sd/desiproc/dr3/ /global/work/decam/versions/work/"
    print('You should run the following command:')
    print('  {}'.format(cmd))

def _catalog_template(nobj=1):
    from astropy.table import Table
    cols = [
        ('GALAXY', 'S20'), 
        ('RA', 'f8'), 
        ('DEC', 'f8'),
        ('RADIUS', 'f4')
        ]
    catalog = Table(np.zeros(nobj, dtype=cols))
    catalog['RADIUS'].unit = 'arcsec'

    return catalog

def _simplewcs(gal):
    '''Build a simple WCS object for a single galaxy.'''
    pixscale = 0.262 # average pixel scale [arcsec/pix]
    diam = 4*np.ceil(gal['RADIUS']/3600.0) # [deg]
    galwcs = Tan(gal['RA'], gal['DEC'], diam/2+0.5, diam/2+0.5,
                 -pixscale, 0.0, pixscale, 0.0, 
                 float(diam), float(diam))
    return galwcs

def read_rc3():
    """Read the RC3 catalog and put it in a standard format."""
    catdir = os.getenv('CATALOGS_DIR')
    cat = fits.getdata(os.path.join(catdir, 'rc3', 'rc3_catalog.fits'), 1)

    # For testing -- randomly pre-select a subset of galaxies.
    nobj = 500
    seed = 5781
    rand = np.random.RandomState(seed)
    these = rand.randint(0, len(cat)-1, nobj)
    cat = cat[these]

    outcat = _catalog_template(len(cat))
    outcat['RA'] = cat['RA']
    outcat['DEC'] = cat['DEC']
    outcat['RADIUS'] = 0.1*10.0**cat['LOGD_25']*60.0/2.0 # semi-major axis diameter [arcsec]
    fix = np.where(outcat['RADIUS'] == 0.0)[0]
    if len(fix) > 0:
        outcat['RADIUS'][fix] = 30.0
    
    #plt.hist(outcat['RADIUS'], bins=50, range=(1, 300))
    #plt.show()

    for name in ('NAME1', 'NAME2', 'NAME3', 'PGC'):
        need = np.where(outcat['GALAXY'] == '')[0]
        if len(need) > 0:
            outcat['GALAXY'][need] = cat[name][need].replace(' ', '')

    return outcat

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sample', action='store_true', help='Build the sample.')
    args = parser.parse_args()

    # Top-level directory
    key = 'LEGACY_SURVEY_LARGE_GALAXIES'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    largedir = os.getenv(key)

    # --------------------------------------------------
    # Build the sample of large galaxies based on the available imaging.
    if args.build_sample:

        # Read the parent catalog.
        cat = read_rc3()
        
        # Create a simple WCS object for each object and find all the CCDs
        # touching that WCS footprint.
        survey = LegacySurveyData(version='dr2') # hack!
        allccds = survey.get_ccds()
        keep = np.concatenate((survey.apply_blacklist(allccds),
                               survey.photometric_ccds(allccds)))
        allccds.cut(keep)

        ccdlist = []
        outcat = []
        for gal in cat:
            print('Finding CCDs for {}, D(25)={:.4f}'.format(
                gal['GALAXY'], gal['RADIUS']))
            galwcs = _simplewcs(gal)

            ccds1 = allccds[ccds_touching_wcs(galwcs, allccds)]
            if len(ccds1) > 0 and 'g' in ccds1.filter and 'r' in ccds1.filter and 'z' in ccds1.filter:
                ccdlist.append(ccds1)
                if len(outcat) == 0:
                    outcat = gal
                else:
                    outcat = vstack((outcat, gal))
                #if gal['GALAXY'].strip() == 'MCG-1-6-63':
                #    pdb.set_trace()

        # Write out the final catalog.
        outfile = os.path.join(largedir, 'large-galaxies-sample.fits')
        if os.path.isfile(outfile):
            os.remove(outfile)
        print('Writing {}'.format(outfile))
        outcat.write(outfile)
        print(outcat)

        # Do we need to transfer any of the data to nyx?
        _getfiles(merge_tables(ccdlist))

        pdb.set_trace()

        # # Build the url cutout for this galaxy
        # url = 'http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2/?'+\
        #   'ra={:.4f}&dec={:.4f}&pixscale=0.262&size=500'.format(obj.ra, obj.dec)
        # print(url)


        
if __name__ == "__main__":
    main()
