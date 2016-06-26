#!/usr/bin/env python

"""run everyone's validation codes

export VALIDATION_DIR=relative/path/to/write/output
python theValidator
"""
import os

from legacyanalysis.pathnames import bash,get_indir

#BASS/MzLS vs. DECaLS
decam_cats= os.path.join(get_indir('bmd'),'decam.txt')
bm_cats= os.path.join(get_indir('bmd'),'bassmos.txt')
bash('python legacyanalysis/compare_tractor_cats.py -fn1 %s -fn2 %s' % (decam_cats,bm_cats))
#GOAL code
# obj= bassmos_v_decals(decam_cats, bm_cats)
# obj.make_plots()

#Johan's cosmos
#bash('python legacyanalysis/johans_cosmos.py')
#Johan's DR2 v DR3
print "done"

