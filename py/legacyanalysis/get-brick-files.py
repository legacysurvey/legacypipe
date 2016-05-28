#!/usr/bin/env python

"""List the images and calibration files one would need to pull from NERSC to
process a single brick.  Used by decals_sim.py.

"""
from __future__ import division, print_function

import os
import sys
import argparse

from legacypipe.common import LegacySurveyData, wcs_for_brick, ccds_touching_wcs

import numpy as np
import pdb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brickname', type=str, default='2428p117', help='Brick of interest')
    args = parser.parse_args()

    survey = LegacySurveyData()
    brickinfo = survey.get_brick_by_name(args.brickname)
    brickwcs = wcs_for_brick(brickinfo)
    ccdinfo = survey.ccds_touching_wcs(brickwcs)
    nccd = len(ccdinfo)

    # Construct image file names.
    # Construct calibration file names.
    expnum = ccdinfo.expnum
    ccdname = ccdinfo.ccdname

    psffiles = list()
    skyfiles = list()
    imagefiles = list()
    for ii in range(nccd):
        exp = '{0:08d}'.format(expnum[ii])
        rootfile = os.path.join(exp[:5], exp, 'decam-'+exp+'-'+ccdname[ii]+'.fits')
        #pdb.set_trace()
        psffiles.append(os.path.join('calib', 'decam', 'psfex', rootfile))
        skyfiles.append(os.path.join('calib', 'decam', 'splinesky', rootfile))
        imagefiles.append(os.path.join('images', str(np.core.defchararray.strip(ccdinfo.image_filename[ii]))))

    print(np.array(imagefiles))
    print(np.array(psffiles))
    print(np.array(skyfiles))

    brickfiles = open('/tmp/brickfiles.txt', 'w')
    for ii in range(nccd):
        brickfiles.write(psffiles[ii]+'\n')
    for ii in range(nccd):
        brickfiles.write(skyfiles[ii]+'\n')
    for ii in range(nccd):
        brickfiles.write(imagefiles[ii]+'\n')
    for ii in range(nccd):
        brickfiles.write(imagefiles[ii].replace('ooi', 'oow')+'\n')
    for ii in range(nccd):
        brickfiles.write(imagefiles[ii].replace('ooi', 'ood')+'\n')
    brickfiles.close()

    cmd = "rsync -avP --files-from='/tmp/brickfiles.txt' cori:/global/cscratch1/sd/desiproc/dr3/ /global/work/decam/versions/work/"
    print('You should run the following command:')
    print('  {}'.format(cmd))
    
if __name__ == "__main__":
    main()
