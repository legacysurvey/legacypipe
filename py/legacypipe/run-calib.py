#! /usr/bin/env python
"""This script runs calibration pre-processing steps including WCS, sky, and PSF models.
"""
from __future__ import print_function
import os
from sys import exit
import numpy as np
from astrometry.util.fits import fits_table
import sys

# Argh, no relative imports in runnable scripts
from legacypipe.common import run_calibs, LegacySurveyData

def main():
    """Main program.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                      help='Run calib processes even if files already exist?')
    parser.add_argument('--ccds', help='Set ccds.fits file to load')

    parser.add_argument('--expnum', type=int, help='Cut to a single exposure')
    parser.add_argument('--extname', '--ccdname', help='Cut to a single extension/CCD name')

    parser.add_argument('--no-psf', dest='psfex', action='store_false',
                      help='Do not compute PsfEx calibs')
    parser.add_argument('--no-sky', dest='sky', action='store_false',
                      help='Do not compute sky models')
    parser.add_argument('--run-se', action='store_true', help='Run SourceExtractor')

    parser.add_argument('--splinesky', action='store_true', help='Spline sky, not constant')
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)
    parser.add_argument('args',nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    survey = LegacySurveyData()
    if opt.ccds is not None:
        T = fits_table(opt.ccds)
        print('Read', len(T), 'from', opt.ccds)
    else:
        T = survey.get_ccds()
        #print len(T), 'CCDs'

    if len(opt.args) == 0:
        if opt.expnum is not None:
            T.cut(T.expnum == opt.expnum)
            print('Cut to', len(T), 'with expnum =', opt.expnum)
        if opt.extname is not None:
            T.cut(np.array([(t.strip() == opt.extname) for t in T.ccdname]))
            print('Cut to', len(T), 'with extname =', opt.extname)

        opt.args = range(len(T))

    args = []
    for a in opt.args:
        i = int(a)
        print('Index', i)
        t = T[i]

        im = survey.get_image_object(t)
        print('Running', im.calname)

        kwargs = dict(psfex=opt.psfex, sky=opt.sky)
        if opt.force:
            kwargs.update(force=True)
        if opt.run_se:
            kwargs.update(se=True)
        if opt.splinesky:
            kwargs.update(splinesky=True)

        if opt.threads:
            args.append((im, kwargs))
        else:
            run_calibs((im, kwargs))

    if opt.threads:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)
        mp.map(run_calibs, args)
        
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
