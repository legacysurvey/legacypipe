from __future__ import print_function
import os
import sys
import time
import tempfile
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from legacyzpts.legacy_zeropoints import main as lzmain

def main():
    survey_dir = os.environ['LEGACYPIPE_TEST_DATA']
    lzmain(['--survey-dir', survey_dir, '--camera', 'decam', '--image',
            'decam/CP/V4.8.2a/CP20181208/c4d_181209_065355_N23_ooi_g_ls9.fits.fz'])

if __name__ == '__main__':
    main()
