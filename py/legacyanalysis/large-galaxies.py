#!/usr/bin/env python

"""Redo the Tractor photometry of the "large" galaxies in Legacy Survey imaging.

J. Moustakas
Siena College
2016 June 6

"""
from __future__ import division, print_function

import os
import sys
import argparse
import pdb

import numpy as np
import logging as log

import matplotlib.pyplot as plt

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table, merge_tables

from legacypipe.common import bricks_touching_wcs, ccds_touching_wcs, LegacySurveyData

from legacyanalysis.get_brick_files import getbrickfiles

from astropy.io import fits
from astropy.table import Table
import matplotlib as mpl

def _getfiles(ccdinfo):
    '''Construct image file names and the calibration file names.'''

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--build-sample', action='store_true', help='Build the sample.')
    args = parser.parse_args()

    # Top-level directory
    key = 'LEGACY_SURVEY_LARGE_GALAXIES'
    if key not in os.environ:
        log.fatal('Required ${} environment variable not set'.format(key))
        return 0
    largedir = os.getenv(key)

    # --------------------------------------------------
    # Build the sample of "large" galaxies and identify the corresponding CCDs and bricks. 

    if args.build_sample:
        catdir = os.getenv('CATALOGS_DIR')
        cat = fits_table(os.path.join(catdir, 'rc3', 'rc3_catalog.fits'))
        #cat = fits.getdata(os.path.join(catdir, 'rc3', 'rc3_catalog.fits'), 1)

        # Randomly pre-select 10 galaxies.
        nobj = 30
        seed = 5781
        rand = np.random.RandomState(seed)
        these = rand.randint(0, len(cat)-1, nobj)
        cat = cat[these]

        d25_maj = 0.1*10.0**cat.logd_25 # [arcmin]
        #plt.hist(d25_maj, bins=50, range=(0.02,5))
        #plt.show()

        # For each object, create a simple WCS header and then find the unique
        # set of bricks containing the galaxies.  We'll worry about galaxies
        # spanning more than one brick later.

        survey = LegacySurveyData()
        allccds = survey.get_ccds()
        keep = np.concatenate((survey.apply_blacklist(allccds), survey.photometric_ccds(allccds)))
        allccds.cut(keep)

        #dr2bricks = survey.get_bricks_dr2()
        #dr2bricks = dr2bricks[np.where((dr2bricks.nobs_med_g>0)*(dr2bricks.nobs_med_r>0)*
        #                               (dr2bricks.nobs_med_z>0))[0]]
        #print('Searching through {} bricks'.format(len(dr2bricks)))
        
        pixscale = 0.262 # average pixel scale [arcsec/pix]

        ccdlist = []
        #bricklist = []
        for ii, obj in enumerate(cat):
            print('Finding bricks for {}/{}/{}, D(25)={:.4f}'.format(
                cat[ii].name1.replace(' ',''),
                cat[ii].name2.replace(' ',''), 
                cat[ii].name3.replace(' ',''), d25_maj[ii]))
            diam = 2*np.ceil(d25_maj[ii]/60) # [deg]
            objwcs = Tan(obj.ra, obj.dec, diam/2+0.5, diam/2+0.5,
                         -pixscale, 0.0, pixscale, 0.0,
                         float(diam), float(diam))
            ccds1 = allccds[ccds_touching_wcs(objwcs, allccds)]
            #bricks1 = bricks_touching_wcs(objwcs, survey, dr2bricks)
            #if len(bricks1)>0:
            if len(ccds1)>0:
                ccdlist.append(ccds1)
                #bricklist.append(bricks1)

                # Build the url cutout for this galaxy
                url = 'http://legacysurvey.org/viewer/jpeg-cutout-decals-dr2/?'+\
                  'ra={:.4f}&dec={:.4f}&pixscale=0.262&size=500'.format(obj.ra, obj.dec)
                print(url)

        # Merge, get the unique subset, and write out.
        ccds = merge_tables(ccdlist)

        ccdfile = []
        [ccdfile.append('{}-{}'.format(expnum, ccdname)) for expnum, ccdname in zip(ccds.expnum, ccds.ccdname)]
        _, indx = np.unique(ccdfile, return_index=True)
        ccds[indx]
        nccd = len(ccds)
        
        #bricks = merge_tables(bricklist)
        #_, indx = np.unique(bricks.brickname, return_index=True)
        #bricks = bricks[np.sort(indx)]

        # Write out the table!

        # Get the files
        _getfiles(ccds)

        pdb.set_trace()
        
if __name__ == "__main__":
    main()
