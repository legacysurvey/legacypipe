#!/usr/bin/env python

"""do cosmos comparison

to run dustins/johans cosmos script:
python legacyanalysis/compare-two-catalogs.py --plot-prefix 4041 /project/projectdirs/desi/imaging/data/validation/cosmos/cosmos-40/ /project/projectdirs/desi/imaging/data/validation/cosmos/cosmos-41/

Input Data
==========
cosmos-1[012] -- hand-chosen to have roughly one image from pass 1, 2, 3 in each of g,r,z. Did not choose the exposures that had big sky artifacts and strong gradients. They have, as before, noise added to bring them so that each image has the same depth for our canonical "simple" galaxy profile, and so that the total depth with the three images is exactly our target depth.
cosmos-2[012] -- same as cosmos-1[012] but without any added noise
cosmos-3[012] -- slightly different set of images (swapped out two images that were a little too shallow), with a slight change to the way the amount of extra noise to add was computed (per-CCD rather than per-exposure).
cosmos-4[012] -- same as cosmos-3[012] but without  noise


Additonal Data
==============
We have two catalogs of the same field that used other observations than what Dustin is using.
 - the DES SV reductions that are public, here :
https://des.ncsa.illinois.edu/releases/sva1/doc/gold
Comparison should be easier, because the broad band filters are the same.
 - the 30 band photoZ catalog that was updated in 2015, also public, here :
http://cosmos.astro.caltech.edu/page/photoz
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import numpy as np

from legacyanalysis.validation.pathnames import get_outdir,get_indir
from legacyanalysis.validation.combine_cats import Matched_DataSet
import legacyanalysis.validation.cosmos_plots as plots


# Find tractor catalogues
indir= get_indir('cosmos')
cosmos10BrickList = glob.glob(os.path.join(indir,"cosmos-10/tractor/1*/tractor*.fits"))
cosmos11BrickList = glob.glob(os.path.join(indir,"cosmos-11/tractor/1*/tractor*.fits"))
cosmos12BrickList = glob.glob(os.path.join(indir,"cosmos-12/tractor/1*/tractor*.fits"))
cosmos20BrickList = glob.glob(os.path.join(indir,"cosmos-20/tractor/1*/tractor*.fits"))
cosmos21BrickList = glob.glob(os.path.join(indir,"cosmos-21/tractor/1*/tractor*.fits"))
cosmos22BrickList = glob.glob(os.path.join(indir,"cosmos-22/tractor/1*/tractor*.fits"))
cosmos30BrickList = glob.glob(os.path.join(indir,"cosmos-30/tractor/1*/tractor*.fits"))
cosmos31BrickList = glob.glob(os.path.join(indir,"cosmos-31/tractor/1*/tractor*.fits"))
cosmos32BrickList = glob.glob(os.path.join(indir,"cosmos-32/tractor/1*/tractor*.fits"))
cosmos40BrickList = glob.glob(os.path.join(indir,"cosmos-40/tractor/1*/tractor*.fits"))
cosmos41BrickList = glob.glob(os.path.join(indir,"cosmos-41/tractor/1*/tractor*.fits"))
cosmos42BrickList = glob.glob(os.path.join(indir,"cosmos-42/tractor/1*/tractor*.fits"))

# Write cat names to text files
list_10= os.path.join(indir,'10.txt') 
list_11= os.path.join(indir,'11.txt') 
list_12= os.path.join(indir,'12.txt') 
list_20= os.path.join(indir,'20.txt') 
list_21= os.path.join(indir,'21.txt') 
list_22= os.path.join(indir,'22.txt') 
list_30= os.path.join(indir,'30.txt') 
list_31= os.path.join(indir,'31.txt') 
list_32= os.path.join(indir,'32.txt') 
list_40= os.path.join(indir,'40.txt') 
list_41= os.path.join(indir,'41.txt') 
list_42= os.path.join(indir,'42.txt') 
np.savetxt(list_10, cosmos10BrickList, fmt='%s')
np.savetxt(list_11, cosmos11BrickList, fmt='%s')
np.savetxt(list_12, cosmos12BrickList, fmt='%s')
np.savetxt(list_20, cosmos20BrickList, fmt='%s')
np.savetxt(list_21, cosmos21BrickList, fmt='%s')
np.savetxt(list_22, cosmos22BrickList, fmt='%s')
np.savetxt(list_30, cosmos30BrickList, fmt='%s')
np.savetxt(list_31, cosmos31BrickList, fmt='%s')
np.savetxt(list_32, cosmos32BrickList, fmt='%s')
np.savetxt(list_40, cosmos40BrickList, fmt='%s')
np.savetxt(list_41, cosmos41BrickList, fmt='%s')
np.savetxt(list_42, cosmos42BrickList, fmt='%s')

combinations = [[list_40, list_41, "_40_41"] ]# , [list_10, list_12, "_10_12"], [list_11, list_12, "_11_12"]]

for comb in combinations:
    list_a, list_b, suffix = comb
    # Compare list_a to list_b
    # List_a is the "reference", e.g. list_b sources will be matched against list_a
    d=Matched_DataSet(list_a, list_b, \
                       comparison='cosmos',debug=True)
    # Cosmos_plots.py does all cuts so hand it all objects for now 
    d.ref_matched.apply_mask_by_names(['all'])
    d.test_matched.apply_mask_by_names(['all'])
    plots.all(d.ref_matched,d.test_matched,d.meta['d_matched'],\
              name1='40',name2='41')

print('finished COSMOS comparison')
