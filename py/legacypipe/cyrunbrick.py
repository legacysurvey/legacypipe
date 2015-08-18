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

import pyximport; pyximport.install(pyimport=True)

#-- tractor.engine imports

from math import ceil, floor, pi, sqrt, exp
import time
import logging
import random
import os
import resource
import gc
import numpy as np
from astrometry.util.ttime import *

#--

import tractor

import tractor.cache
import tractor.patch
import tractor.utils

import tractor.engine

from runbrick import main

if __name__ == '__main__':
    main()



