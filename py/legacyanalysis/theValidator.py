#!/usr/bin/env python

"""run everyone's validation codes
"""
import os

from legacyanalysis.pathnames import bash,get_indir

#BASS/MzLS vs. DECaLS
decam_cats= os.path.join(get_indir('bmd'),'decam.txt')
bm_cats= os.path.join(get_indir('bmd'),'bassmos.txt')
bash('python legacyanalysis/compare_tractor_cats.py -fn1 %s -fn2 %s' % (decam_cats,bm_cats))
#Johan's cosmos
bash('python legacyanalysis/johans_cosmos.py')
#Johan's DR2 v DR3
print "done"

