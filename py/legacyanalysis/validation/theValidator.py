#!/usr/bin/env python

"""make all validation plots

export VALIDATION_DIR=relative/path/to/write/output
python theValidator
"""
import os
import argparse

from legacyanalysis.validation.pathnames import bash,get_indir

def main():
    # BASS/MzLS vs. DECaLS
    decam_cats= os.path.join(get_indir('bmd'),'decam.txt')
    bm_cats= os.path.join(get_indir('bmd'),'bassmos.txt')
    bash('python legacyanalysis/validation/decals_bass_mzls.py --decals_list %s --bassmos_list %s' % (decam_cats,bm_cats))
    # COSMOS
    #bash('python legacyanalysis/cosmos.py')
    print "done"

if __name__ == '__main__':
    main()
