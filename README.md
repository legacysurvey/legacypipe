# pipeline
Our image reduction pipeline, using the Tractor framework

The license is 3-clause BSD.

[![Build Status](https://travis-ci.org/legacysurvey/legacypipe.svg?branch=master)](https://travis-ci.org/legacysurvey/legacypipe)
[![Docs](https://readthedocs.org/projects/legacypipe/badge/?version=latest)](http://legacypipe.readthedocs.org/en/latest/)
[![Coverage](https://coveralls.io/repos/github/legacysurvey/legacypipe/badge.svg?branch=master)](https://coveralls.io/github/legacysurvey/legacypipe)

Code for the analysis of the DECam Legacy Survey (DECaLS).
========================

- legacypipe/runbrick.py -- run the Tractor analysis of one DECaLS brick.
- legacypipe/oneblob.py -- code run for a single "blob" of connected pixels.
- legacypipe/common.py -- used by runbrick.py and others
- legacypipe/merge-zeropoints.py -- create survey-ccds.fits file
- bin/pipebrick.sh -- for running "runbrick.py" in production runs (via qdo & SLURM at NERSC)
- legacypipe/desi_common.py -- an older set of common routines
- legacypipe/queue-calib.py -- find & qdo queue bricks & CCDs
- legacypipe/run-calib.py -- calibrate CCDs
- legacypipe/kick-tires.py -- check out Tractor catalogs
- legacypipe/image.py -- generic routines for reading images
- legacypipe/cpimage.py -- image subclass for images from the NOAO Community Pipeline
- legacypipe/decam.py -- subclass for Dark Energy Camera/Blanco images
- legacypipe/mosaic.py -- subclass for Mosaic3 Camera/Mayall images
- legacypipe/bok.py -- subclass for 90Prime/Bok images
- legacypipe/ptf.py -- subclass for Palomar Transient Factory images
