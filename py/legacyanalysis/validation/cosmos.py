#!/usr/bin/env python

"""do cosmos comparison
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

from legacyanalysis.validation.pathnames import get_outdir
from legacyanalysis.validation.combine_cats import Matched_DataSet
import legacyanalysis.validation.common_plots as plots

# Compare list_a to list_b
# list_a is the "reference", e.g. list_b sources will be matched against list_a
list_a= '/project/projectdirs/desi/imaging/data/validation/cosmos/a.txt' 
list_b= '/project/projectdirs/desi/imaging/data/validation/cosmos/b.txt' 
d= Matched_DataSet(list_a, list_b, \
                   comparison='cosmos')
# Plot number of psf,exp,etc for list_a 
# All 
d.ref_matched.apply_mask_by_names(['all'])
plots.hist_types(d.ref_matched, name='All')
# Default, good quality photometry only
d.ref_matched.apply_mask_by_names(['default'])
plots.hist_types(d.ref_matched, name='Defaul')
# psf only
d.ref_matched.apply_mask_by_names(['psf'])
plots.hist_types(d.ref_matched, name='psf')
# Default + psf + exp
d.ref_matched.apply_mask_by_names(['default','psf','exp'])
plots.hist_types(d.ref_matched, name='DefaulPsfExp')

print('finished COSMOS comparison')
