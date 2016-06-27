#!/usr/bin/env python

"""run everyone's validation codes

export VALIDATION_DIR=relative/path/to/write/output
python theValidator
"""
import os
import argparse

from legacyanalysis.pathnames import bash,get_indir

def main():
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Validation plots')
    parser.add_argument('--val', choices=['all','bmd','cosmos','dr23'], help='which validation to do',required=True)
    args = parser.parse_args()
    
    if args.val == 'all' or args.val == 'bmd':
        # BASS/MzLS vs. DECaLS
        decam_cats= os.path.join(get_indir('bmd'),'decam.txt')
        bm_cats= os.path.join(get_indir('bmd'),'bassmos.txt')
        bash('python legacyanalysis/decals_bass_mzls.py --decals_list %s --bass_list %s' % (decam_cats,bm_cats))
        #bash('python legacyanalysis/compare_tractor_cats.py -fn1 %s -fn2 %s' % (decam_cats,bm_cats))
        #GOAL code
        # obj= bassmos_v_decals(decam_cats, bm_cats)
        # obj.make_plots()
    if args.val == 'all' or args.val == 'cosmos':
        # Johan's cosmos
        #bash('python legacyanalysis/johans_cosmos.py')
    if args.val == 'all' or args.val == 'dr23':
        # Johan's DR2 v DR3
    print "done"

if __name__ == '__main__':
    main()
