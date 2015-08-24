# pipeline
Our image reduction pipeline, using the Tractor framework

[![Build Status](https://travis-ci.org/legacysurvey/legacypipe.svg?branch=master)](https://travis-ci.org/legacysurvey/legacypipe)
[![Docs](https://readthedocs.org/projects/legacypipe/badge/?version=latest)](http://legacypipe.readthedocs.org/en/latest/)

Code for the analysis of the DECam Legacy Survey (DECaLS).
========================

- runbrick.py -- run the Tractor analysis of one DECaLS brick.
- common.py -- used by runbrick.py and others
- make-exposure-list.py -- to create the decals-ccds.fits metadata summary
- pipebrick.sh -- for running "runbrick.py" in production runs (via qdo)
- check-psf.py -- investigating PsfEx fits and MoG fits thereof
- desi_common.py -- an older set of common routines
- queue-calib.py -- find & qdo queue bricks & CCDs
- run-calib.py -- calibrate CCDs
- kick-tires.py -- check out Tractor catalogs


How various files are generated:

- zeropoints.fits: from Arjun's FITS tables of zeropoints:

 python -c "import numpy as np; from glob import glob; from astrometry.util.fits import *; TT = [fits_table(x) for x in glob('/global/homes/a/arjundey/ZeroPoints/ZeroPoint_*.fits.gz')]; T = merge_tables(TT, columns='fillzero'); T.expnum = np.array([int(x) for x in T.expnum]); T.writeto('zp.fits')"

- decals-ccds.fits:

For DR1:

```
python -u projects/desi/make-exposure-list.py -o decals-ccds-dr1.fits --trim decals/images/ decals/images/decam/{COSMOS,CP*_v2,CP20141227,CP20150108,DESY1_Stripe82}/*_ooi_* > list.log 2>&1 &
```
