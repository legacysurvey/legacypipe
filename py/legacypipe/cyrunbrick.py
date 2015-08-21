if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

import fitsio
import astrometry.util.fits
import astrometry.util.file
import astrometry.util.plotutils
import astrometry.util.multiproc
import logging

import cytractor

import pyximport; pyximport.install(pyimport=True)

import    cytractor
print dir(cytractor)
print 'cytractor:', cytractor
import    cytractor.mixture_profiles
print dir(cytractor.mixture_profiles)
import    cytractor.patch
print dir(cytractor.patch)
import    cytractor.utils
print dir(cytractor.utils)
import    cytractor.engine
print dir(cytractor.engine)
import    cytractor.basics
print dir(cytractor.basics)
import    cytractor.ellipses
print dir(cytractor.ellipses)
import    cytractor.galaxy
print dir(cytractor.galaxy)
import    cytractor.psfex
print dir(cytractor.psfex)

print dir(cytractor)

tractor = cytractor
print 'tractor', tractor
print dir(tractor)


#import tractor.ellipses
# #-- tractor.engine imports
# 
# from math import ceil, floor, pi, sqrt, exp
# import time
# import logging
# import random
# import os
# import resource
# import gc
# import numpy as np
# from astrometry.util.ttime import *
# 
# #--
# 
# import tractor
# 
# import tractor.cache
# import tractor.patch
# import tractor.utils
# 
# import tractor.engine
# 
# from runbrick import main

def main():
    pass

if __name__ == '__main__':
    main()



