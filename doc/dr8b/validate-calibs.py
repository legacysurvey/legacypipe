#!/usr/bin/env python

import os, pdb
import argparse
import numpy as np

from glob import glob
import fitsio
from astropy.table import vstack, Table, Column

from legacyzpts.legacy_zeropoints import read_lines
from legacyzpts.legacy_zeropoints import outputFns, Measurer, DecamMeasurer, NinetyPrimeMeasurer, Mosaic3Measurer

dr8dir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8b'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', choices=['decam', '90prime', 'mosaic'],
                        action='store', required=True)
    args = parser.parse_args()

    outdir = 'zpts'
    calibdir = 'calib'
    image_dir = 'images'

    image_list = read_lines('image-lists/image-list-{}-dr8b.txt'.format(args.camera))
    for ii, imfn in enumerate(image_list):
        if ii % 100 == 0:
            print('Checking image {}/{}'.format(ii, len(image_list)))
        
        F = outputFns(imfn, outdir=outdir, camera=args.camera,
                      image_dir=image_dir, debug=False)
        #measure = Measurer(imfn, image_dir=image_dir, camera=args.camera, outdir=outdir,
        #                   ps1_pattern='/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits',
        #                   verboseplots=False, debug=False, calibdir=calibdir,
        #                   quiet=True)

        if args.camera == 'decam':
            measure = DecamMeasurer(imfn, image_dir=image_dir, camera=args.camera, outdir=outdir,
                                    ps1_pattern='/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits',
                                    verboseplots=False, debug=False, calibdir=calibdir,
                                    quiet=True)
        elif args.camera == 'mosaic':
            measure = Mosaic3Measurer(imfn, image_dir=image_dir, camera=args.camera, outdir=outdir,
                                    ps1_pattern='/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits',
                                    verboseplots=False, debug=False, calibdir=calibdir,
                                    quiet=True)
        if args.camera == '90prime':
            measure = NinetyPrimeMeasurer(imfn, image_dir=image_dir, camera=args.camera, outdir=outdir,
                                    ps1_pattern='/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/ps1-%(hp)05d.fits',
                                    verboseplots=False, debug=False, calibdir=calibdir,
                                    quiet=True)
            
            #ff = measure.get_splinesky_merged_filename()
            #if not os.path.isfile(ff) and not 'Failed' in measure.primhdr['WCSCAL']:
            #    pp = imfn.replace('.fits.fz', '.log').split('/')
            #    print(os.path.join('logs-zpts', pp[0], pp[2], pp[3]))
            #    #print('From image {} {} missing: {}'.format(ii, imfn, ff))

            for ff in (F.annfn, F.surveyfn, F.starfn_photom, measure.get_splinesky_merged_filename(),
                       measure.get_psfex_merged_filename()):
                if not os.path.isfile(ff) and not 'Failed' in measure.primhdr['WCSCAL']:
                    #print('From image {} {} missing: {}'.format(ii, imfn, ff))
                    print('From image {} {} missing: {}'.format(ii, imfn, ff))
            
if __name__ == '__main__':
    main()
