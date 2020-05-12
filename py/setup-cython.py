from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CFLAGS'] = '-march=native'

setup(
    ext_modules = cythonize([
        'legacypipe/bits.py',
        'legacypipe/bok.py',
        'legacypipe/catalog.py',
        'legacypipe/coadds.py',
        'legacypipe/decam.py',
        'legacypipe/detection.py',
        'legacypipe/format_catalog.py',
        'legacypipe/gaiacat.py',
        'legacypipe/halos.py',
        'legacypipe/image.py',
        'legacypipe/mosaic.py',
        'legacypipe/oneblob.py',
        'legacypipe/outliers.py',
        'legacypipe/ps1cat.py',
        'legacypipe/reference.py',
        'legacypipe/runbrick.py',
        'legacypipe/runs.py',
        'legacypipe/survey.py',
        'legacypipe/unwise.py',
        'legacypipe/utils.py',
        'legacyzpts/annotate_ccds.py',
        'legacyzpts/legacy_zeropoints.py',
        'legacyzpts/merge_calibs.py',
        'legacyzpts/psfzpt_cuts.py',
    ], annotate=True,
       nthreads=4,
       compiler_directives=dict(language_level=3,
                                infer_types=True,
#profile=True
))
)
