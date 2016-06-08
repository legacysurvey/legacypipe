#!/usr/bin/env python

"""run everyone's validation codes
"""

from argparse import ArgumentParser
from legacyanalysis.pathnames import bash,get_indir

parser=ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('-fn1', type=str, help='process this brick (required input)')
parser.add_argument('-fn2', type=str, help='object type (STAR, ELG, LRG, BGS)') 
args = parser.parse_args()

#BASS/MzLS vs. DECaLS
decam_cats= os.path.join(get_indir('bmd'),'decam.txt')
bm_cats= os.path.join(get_indir('bmd'),'bassmos.txt')
bash('python legacyanalysis/compare_tractor_cats.py -fn1 %s -fn2 %s' % (decam_cats,bm_cats))
#Johan's cosmos
bash('python legacyanalysis/johans_cosmos.py')
#Johan's DR2 v DR3
print "done"

