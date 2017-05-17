#!/usr/bin/env python

"""List the images and calibration files one would need to pull from NERSC to
process a single brick.  Used by decals_sim.py.

"""
from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
import pdb

from legacypipe.cpimage import CPImage
from legacypipe.image import LegacySurveyImage
from legacypipe.survey import LegacySurveyData, wcs_for_brick, ccds_touching_wcs

def getbrickfiles(brickname=None):

    survey = LegacySurveyData()
    brickinfo = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    ccdinfo = survey.ccds_touching_wcs(brickwcs)
    nccd = len(ccdinfo)

    calibdir = survey.get_calib_dir()
    imagedir = survey.survey_dir

    # Construct image file names and the calibration file names.
    expnum = ccdinfo.expnum
    ccdname = ccdinfo.ccdname

    psffiles = list()
    skyfiles = list()
    imagefiles = list()
    for ccd in ccdinfo:
        info = survey.get_image_object(ccd)
        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(info, attr).replace(imagedir+'/', '')
            #if '160108_073601' in fn:
            #    pdb.set_trace()
            imagefiles.append(fn)
        psffiles.append(info.psffn.replace(calibdir, 'calib'))
        skyfiles.append(info.splineskyfn.replace(calibdir, 'calib'))
        
    #for ii in range(nccd):
        #exp = '{0:08d}'.format(expnum[ii])
        #rootfile = os.path.join(exp[:5], exp, 'decam-'+exp+'-'+ccdname[ii]+'.fits')
        #psffiles.append(os.path.join('calib', 'decam', 'psfex', rootfile))
        #skyfiles.append(os.path.join('calib', 'decam', 'splinesky', rootfile))
        #imagefiles.append(os.path.join('images', str(np.core.defchararray.strip(ccdinfo.image_filename[ii]))))

    #print(np.array(imagefiles))
    #print(np.array(psffiles))
    #print(np.array(skyfiles))
    return imagefiles, psffiles, skyfiles

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brickname', type=str, default='2428p117', help='Brick of interest')
    args = parser.parse_args()

    imagefiles, psffiles, skyfiles = getbrickfiles(args.brickname)
    imagefiles = np.unique(np.array(imagefiles))

    brickfiles = open('/tmp/brickfiles.txt', 'w')
    for ff in psffiles:
        brickfiles.write(ff+'\n')
    for ff in skyfiles:
        brickfiles.write(ff+'\n')
    for ff in imagefiles:
        brickfiles.write(ff+'\n')
    brickfiles.close()

    cmd = "rsync -avP --files-from='/tmp/brickfiles.txt' cori:/global/cscratch1/sd/desiproc/dr3/ /global/work/decam/versions/work/"
    print('You should run the following command:')
    print('  {}'.format(cmd))
    
if __name__ == "__main__":
    main()
