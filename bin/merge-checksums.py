#! /usr/bin/env python3
import sys

def stage_merge_checksums(
        old_survey=None,
        survey=None,
        brickname=None,
        **kwargs):
    '''
    For debugging / special-case processing, read previous checksums, and update them with
    current checksums values, then write out the result.
    '''
    from collections import OrderedDict

    cfn = old_survey.find_file('checksums', brick=brickname)
    print('Old checksums:', cfn)
    checksums = OrderedDict()
    with open(cfn, 'r') as f:
        for line in f.readlines():
            words = line.split()
            fn = words[1]
            if fn.startswith('*'):
                fn = fn[1:]
            hashcode = words[0]
            checksums[fn] = hashcode

    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Update hashsums
        for fn,hashsum in survey.output_file_hashes.items():
            print('Updating checksum', fn, '=', hashsum)
            checksums[fn] = hashsum
        # Write outputs
        for fn,hashsum in checksums.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-output', required=True,
                        help='"Old" output directory to read old checksum file from.')

    parser.add_argument('-b', '--brick', required=True,
                        help='Brick name to run')
    parser.add_argument(
        '-P', '--pickle', dest='pickle_pat',
        help='Pickle filename pattern, default %(default)s',
        default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')
    parser.add_argument('-n', '--no-write', dest='write', default=True,
                        action='store_false')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')

    opt = parser.parse_args()
    optdict = vars(opt)

    old_output_dir = optdict.pop('old_output')

    from legacypipe.runbrick import get_runbrick_kwargs
    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs

    import logging
    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)
    
    from legacypipe.survey import LegacySurveyData
    old_survey = LegacySurveyData(survey_dir=old_output_dir,
                                  output_dir=old_output_dir)
    kwargs.update(old_survey=old_survey)
    brickname = optdict['brick']

    from astrometry.util.stages import CallGlobalTime, runstage

    prereqs = {
        'outliers': None,
    }
    prereqs.update({
        'merge_checksums': 'outliers'
    })

    pickle_pat = optdict['pickle_pat']
    pickle_pat = pickle_pat % dict(brick=brickname)

    stagefunc = CallGlobalTime('stage_%s', globals())
    stage = 'merge_checksums'
    R = runstage(stage, pickle_pat, stagefunc, prereqs=prereqs, force=[stage],
                 write=[], **kwargs)
    
if __name__ == '__main__':
    main()
