#! /usr/bin/env python

import os
import numpy as np
from astrometry.util.fits import fits_table

# Argh, no relative imports in runnable scripts
from legacypipe.common import run_calibs, DecamImage, Decals

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--force', action='store_true', default=False,
                      help='Run calib processes even if files already exist?')
    parser.add_option('--ccds', help='Set ccds.fits file to load')

    parser.add_option('--expnum', type=int, help='Cut to a single exposure')
    parser.add_option('--extname', '--ccdname', help='Cut to a single extension/CCD name')

    parser.add_option('--no-astrom', dest='astrom', default=True, action='store_false',
                      help='Do not compute astrometric calibs')
    parser.add_option('--no-psf', dest='psfex', default=True, action='store_false',
                      help='Do not compute PsfEx calibs')
    parser.add_option('--no-sky', dest='sky', default=True, action='store_false',
                      help='Do not compute sky models')
    parser.add_option('--run-se', action='store_true', help='Run SourceExtractor')

    opt,args = parser.parse_args()

    D = Decals()
    if opt.ccds is not None:
        T = fits_table(opt.ccds)
        print 'Read', len(T), 'from', opt.ccds
    else:
        T = D.get_ccds()
        print len(T), 'CCDs'
    print len(T), 'CCDs'

    if len(args) == 0:
        if opt.expnum is not None:
            T.cut(T.expnum == opt.expnum)
            print 'Cut to', len(T), 'with expnum =', opt.expnum
        if opt.extname is not None:
            T.cut(np.array([(t.strip() == opt.extname) for t in T.ccdname]))
            print 'Cut to', len(T), 'with extname =', opt.extname

        args = range(len(T))
            
    for a in args:
        i = int(a)
        t = T[i]

        im = DecamImage(D, t)
        print 'Running', im.calname

        kwargs = dict(pvastrom=opt.astrom, psfex=opt.psfex, sky=opt.sky)
        if opt.force:
            kwargs.update(force=True)
        if opt.run_se:
            kwargs.update(se=True)

        run_calibs((im, kwargs))
