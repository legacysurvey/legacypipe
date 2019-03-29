#!/usr/bin/env python

import os, shutil
from glob import glob

allold = glob('zpts/*/*/*-legacypipe.fits')
for ii, old in enumerate(allold):
    new = old.replace('-legacypipe.fits', '-survey.fits')
    print(ii, old, new)
    shutil.move(old, new)
