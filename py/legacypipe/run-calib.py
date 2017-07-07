#! /usr/bin/env python
"""This script runs calibration pre-processing steps including WCS, sky, and PSF models.
"""
from __future__ import print_function
import os
import numpy as np
from astrometry.util.fits import fits_table

from legacypipe.survey import run_calibs, LegacySurveyData

def main():
    """Main program.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                      help='Run calib processes even if files already exist?')
    parser.add_argument('--ccds', help='Set ccds.fits file to load')

    parser.add_argument('--expnum', type=str, help='Cut to a single or set of exposures; comma-separated list')
    parser.add_argument('--extname', '--ccdname', help='Cut to a single extension/CCD name')

    parser.add_argument('--no-psf', dest='psfex', action='store_false',
                      help='Do not compute PsfEx calibs')
    parser.add_argument('--no-sky', dest='sky', action='store_false',
                      help='Do not compute sky models')
    parser.add_argument('--run-se', action='store_true', help='Run SourceExtractor')

    parser.add_argument('--splinesky', action='store_true', help='Spline sky, not constant')
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)
    parser.add_argument('--continue', dest='cont', default=False, action='store_true',
                        help='Continue even if one file fails?')

    parser.add_argument('args',nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    survey = LegacySurveyData()
    if opt.ccds is not None:
        T = fits_table(opt.ccds)
        # Remove trailing spaces from 'camera' & 'ccdname' columns.
        T.camera = np.array([c.strip() for c in T.camera])
        T.ccdname = np.array([c.strip() for c in T.ccdname])

        print('Read', len(T), 'from', opt.ccds)
    else:
        T = survey.get_ccds()
        #print len(T), 'CCDs'

    if len(opt.args) == 0:
        if opt.expnum is not None:
            expnums = set([int(e) for e in opt.expnum.split(',')])
            T.cut(np.array([e in expnums for e in T.expnum]))
            print('Cut to', len(T), 'with expnum in', expnums)
        if opt.extname is not None:
            T.cut(np.array([(t.strip() == opt.extname) for t in T.ccdname]))
            print('Cut to', len(T), 'with extname =', opt.extname)

        opt.args = range(len(T))

    args = []
    for a in opt.args:
        # Check for "expnum-ccdname" format.
        if '-' in str(a):
            words = a.split('-')
            assert(len(words) == 2)
            expnum = int(words[0])
            ccdname = words[1]
            I = np.flatnonzero((T.expnum == expnum) * (T.ccdname == ccdname))
            if len(I) != 1:
                print('Found', len(I), 'CCDs for expnum', expnum, 'CCDname', ccdname, ':', I)
                print('WARNING: skipping this expnum,ccdname')
                continue
            assert(len(I) == 1)
            t = T[I[0]]
        else:
            i = int(a)
            print('Index', i)
            t = T[i]

        #print('CCDnmatch', t.ccdnmatch)
        #if t.ccdnmatch < 20 and not opt.force:
        #    print('Skipping ccdnmatch = %i' % t.ccdnmatch)
        #    continue
            
        im = survey.get_image_object(t)
        print('Running', im.calname)
        
        kwargs = dict(psfex=opt.psfex, sky=opt.sky)
        if opt.force:
            kwargs.update(force=True)
        if opt.run_se:
            kwargs.update(se=True)
        if opt.splinesky:
            kwargs.update(splinesky=True)

        if opt.cont:
            kwargs.update(noraise=True)
            
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
