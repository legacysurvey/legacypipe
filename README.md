# legacypipe
DESI Imaging Legacy Surveys data reduction pipeline, using the Tractor framework

The license is 3-clause BSD.

CircleCI: [![Build Status](https://img.shields.io/circleci/project/github/legacysurvey/legacypipe.svg)](https://circleci.com/gh/legacysurvey/legacypipe)
[![Docs](https://readthedocs.org/projects/legacypipe/badge/?version=latest)](http://legacypipe.readthedocs.org/en/latest/)
Coveralls: [![Coverage](https://coveralls.io/repos/github/legacysurvey/legacypipe/badge.svg?branch=master)](https://coveralls.io/github/legacysurvey/legacypipe)
Codecov: [![codecov](https://codecov.io/gh/legacysurvey/legacypipe/branch/master/graph/badge.svg)](https://codecov.io/gh/legacysurvey/legacypipe)
Codacy: [![Codacy Badge](https://app.codacy.com/project/badge/Grade/730586099fd4462d9c5827d58d166f9f)](https://www.codacy.com/gh/legacysurvey/legacypipe/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=legacysurvey/legacypipe&amp;utm_campaign=Badge_Grade)

## Code for the analysis of the Legacy Surveys (DECam/DECaLS, MzLS+BASS)

Some notable contents:
-   legacyzpts/legacy_zeropoints.py -- code for computing photometric and astrometric zeropoints of Community Pipeline-calibrated images
-   bin/runbrick-shifter.sh -- template script for our large-scale runs at NERSC
-   legacypipe/runbrick.py -- the top-level script to reduce one Legacy Surveys brick.
-   docker-nersc -- Dockerfile recipe for an Intel-compiler optimized build of the code, used in production
-   docker -- a generic/public Dockerfile

The Docker containers we use in production are available on Docker Hub:
<https://cloud.docker.com/u/legacysurvey/repository/docker/legacysurvey/legacypipe>

### Requirements include

-   Python 3
-   numpy
-   scipy
-   matplotlib
-   fitsio
-   astropy
-   photutils
-   astrometry.net
-   tractor
-   cython
