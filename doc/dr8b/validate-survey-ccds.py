#!/usr/bin/env python

"""
Validate the number of CCDs in each test region against DR7.

"""
import os, pdb
import numpy as np
from astrometry.util.fits import fits_table

def main():

    print()
    print('Region DR7 DR8-DECam (no cuts) DR8-DECam (after cuts)')
    for region in ('dr8-test-overlap', 'dr8-test-s82', 'dr8-test-hsc-sgc', 'dr8-test-hsc-ngc',
                   'dr8-test-edr', 'dr8-test-hsc-north', 'dr8-test-deep2-egs'):
        if os.path.isfile('dr8b/ccds-{}-decam.fits'.format(region)):
            dr8_decam = fits_table('dr8b/ccds-{}-decam.fits'.format(region))
            dr8_decam_cuts = np.sum(dr8_decam.ccd_cuts == 0)
        else:
            dr8_decam = []
            dr8_decam_cuts = 0
            
        if os.path.isfile('dr7/ccds-{}.fits'.format(region)):
            dr7 = fits_table('dr7/ccds-{}.fits'.format(region))
            dr7_cuts = np.sum(dr7.ccd_cuts == 0)
        else:
            dr7 = []
            dr7_cuts = 0

        print('{:18} {:6d} {:6d} {:6d}'.format(region, len(dr7), len(dr8_decam), dr8_decam_cuts))

    print()
    print('Region DR6 DR8-90prime/mosaic (no cuts) DR8-90prime/mosaic (after cuts) ')
    for region in ('dr8-test-overlap', 'dr8-test-s82', 'dr8-test-hsc-sgc', 'dr8-test-hsc-ngc',
                   'dr8-test-edr', 'dr8-test-hsc-north', 'dr8-test-deep2-egs'):
        if os.path.isfile('dr8b/ccds-{}-90prime-mosaic.fits'.format(region)):
            dr8_mosaic = fits_table('dr8b/ccds-{}-90prime-mosaic.fits'.format(region))
            dr8_mosaic_cuts = np.sum(dr8_mosaic.ccd_cuts == 0)
        else:
            dr8_mosaic = []
            dr8_mosaic_cuts = 0
        
        if os.path.isfile('dr6/ccds-{}.fits'.format(region)):
            dr6 = fits_table('dr6/ccds-{}.fits'.format(region))
            dr6_cuts = np.sum(dr6.ccd_cuts == 0)
        else:
            dr6 = []
            dr6_cuts - 0
            
        print('{:18} {:6d} {:6d} {:6d}'.format(region, len(dr6), len(dr8_mosaic), dr8_mosaic_cuts))
    print()
            
if __name__ == '__main__':
    main()
