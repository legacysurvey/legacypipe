from __future__ import print_function
import os
import sys
import time
import tempfile
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from legacyzpts.legacy_zeropoints import main as lzmain
from legacyzpts.merge_annotated import main as mamain
from legacyzpts.update_ccd_cuts import main as upmain
from legacypipe.runbrick import main as rbmain

def main():
    survey_dir = os.environ['LEGACYPIPE_TEST_DATA']
    lzmain(args=['--survey-dir', survey_dir, '--camera', 'decam', '--image',
            'decam/CP/V4.8.2a/CP20181208/c4d_181209_065355_N23_ooi_g_ls9.fits.fz'])
    lzmain(args=['--survey-dir', survey_dir, '--camera', 'mosaic', '--image',
            'mosaic/CP/V4.4/CP20161016/k4m_161017_115416_CCD3_ooi_zd_ls9.fits.fz'])
    lzmain(args=['--survey-dir', survey_dir, '--camera', '90prime', '--image',
            '90prime/CP/V2.3/CP20170202/ksb_170203_044105_CCD2_ooi_r_ls9.fits.fz'])
    cmd = 'find %s/zpt/decam/ -name "*-ann*" > decam.txt'
    rtn = os.system(cmd)
    assert(rtn == 0)
    cmd = 'find %s/zpt/mosaic/ -name "*-ann*" > mosaic.txt'
    rtn = os.system(cmd)
    assert(rtn == 0)
    cmd = 'find %s/zpt/90prime/ -name "*-ann*" > 90prime.txt'
    rtn = os.system(cmd)
    assert(rtn == 0)

    mamain(args=['--camera', 'decam', '--file_list', 'decam.txt', '--data_release', '100'])
    mamain(args=['--camera', 'mosaic', '--file_list', 'mosaic.txt', '--data_release', '100'])
    mamain(args=['--camera', '90prime', '--file_list', '90prime.txt', '--data_release', '100'])

    upmain(args=['--imlist', 'decam.txt', 'decam',
                 'survey-ccds-decam-dr100.fits', 'ccds-annotated-decam-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-decam-test.fits.gz'), 'ccds-annotated-decam-test.fits',
                 '--tilefile', 'legacyzpts/data/decam-tiles_obstatus.fits', '--depth-cut', '--good-ccd-fraction', '0'])
    upmain(args=['--imlist', 'mosaic.txt', 'mosaic',
                 'survey-ccds-mosaic-dr100.fits', 'ccds-annotated-mosaic-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-mosaic-test.fits.gz'), 'ccds-annotated-mosaic-test.fits',
                 '--tilefile', 'legacyzpts/data/mosaic-tiles_obstatus.fits', '--depth-cut', '--good-ccd-fraction', '0'])
    upmain(args=['--imlist', '90prime.txt', '90prime',
                 'survey-ccds-90prime-dr100.fits', 'ccds-annotated-90prime-dr100.fits',
                 os.path.join(survey_dir, 'survey-ccds-90prime-test.fits.gz'), 'ccds-annotated-90prime-test.fits',
                 '--depth-cut', '--good-ccd-fraction', '0'])

    rbmain(args=['--brick', '1110p322', '--survey-dir', survey_dir,
                 '--zoom', '1550', '1850', '2550', '2850', '--no-outliers',
                 '--no-wise', '--no-large-galaxies'])

if __name__ == '__main__':
    main()
