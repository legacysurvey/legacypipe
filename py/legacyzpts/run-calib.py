#! /usr/bin/env python
"""This script runs calibration pre-processing steps including sky and PSF models.
"""
from __future__ import print_function

from astrometry.util.fits import merge_tables

from legacypipe.survey import run_calibs, LegacySurveyData

def main():
    """Main program.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                      help='Run calib processes even if files already exist?')
    parser.add_argument('--survey-dir', help='Override LEGACY_SURVEY_DIR')
    parser.add_argument('--expnum', type=str, help='Cut to a single or set of exposures; comma-separated list')
    parser.add_argument('--extname', '--ccdname', help='Cut to a single extension/CCD name')

    parser.add_argument('--no-psf', dest='psfex', action='store_false',
                      help='Do not compute PsfEx calibs')
    parser.add_argument('--no-sky', dest='sky', action='store_false',
                      help='Do not compute sky models')
    parser.add_argument('--run-se', action='store_true', help='Run SourceExtractor')

    parser.add_argument('--no-splinesky', dest='splinesky', default=True, action='store_false',
                        help='Use constant, not splinesky')
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=None)
    parser.add_argument('--continue', dest='cont', default=False, action='store_true',
                        help='Continue even if one file fails?')
    parser.add_argument('--plot-base', help='Make plots with this base filename')

    parser.add_argument('--blob-mask-dir', type=str, default=None,
                        help='The base directory to search for blob masks during sky model construction')

    parser.add_argument('args',nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    survey = LegacySurveyData(survey_dir=opt.survey_dir)
    T = None
    if len(opt.args) == 0:
        if opt.expnum is not None:
            expnums = set([int(e) for e in opt.expnum.split(',')])
            T = merge_tables([survey.find_ccds(expnum=e, ccdname=opt.extname) for e in expnums])
            print('Cut to', len(T), 'with expnum in', expnums, 'and extname', opt.extname)
            opt.args = range(len(T))
        else:
            parser.print_help()
            return 0
    ps = None
    if opt.plot_base is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plot_base)

    survey_blob_mask=None
    if opt.blob_mask_dir is not None:
        survey_blob_mask = LegacySurveyData(opt.blob_mask_dir)

    args = []
    for a in opt.args:
        # Check for "expnum-ccdname" format.
        if '-' in str(a):
            words = a.split('-')
            assert(len(words) == 2)
            expnum = int(words[0])
            ccdname = words[1]

            T = survey.find_ccds(expnum=expnum, ccdname=ccdname)
            if len(T) != 1:
                print('Found', len(I), 'CCDs for expnum', expnum, 'CCDname', ccdname, ':', I)
                print('WARNING: skipping this expnum,ccdname')
                continue
            t = T[0]
        else:
            i = int(a)
            print('Index', i)
            t = T[i]

        im = survey.get_image_object(t)
        print('Running', im.name)

        kwargs = dict(psfex=opt.psfex, sky=opt.sky, ps=ps, survey=survey,
                      survey_blob_mask=survey_blob_mask)
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
        mp.map(time_run_calibs, args)

    return 0

def time_run_calibs(*args):
    from astrometry.util.ttime import Time
    t0 = Time()
    rtn = run_calibs(*args)
    t1 = Time()
    print('Time run_calibs:', t1-t0)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    return rtn

if __name__ == '__main__':
    import sys
    sys.exit(main())
