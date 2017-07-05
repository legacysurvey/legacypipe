from __future__ import print_function
from legacypipe.survey import LegacySurveyData
import numpy as np
from glob import glob
import os

if __name__ == '__main__':
    survey = LegacySurveyData()
    ccds = survey.get_ccds()
    print('Read', len(ccds), 'CCDs')
    expnums = np.unique(ccds.expnum)
    print(len(expnums), 'unique exposure numbers')

    ###
    #expnums = expnums[expnums < 140000]
    #pat = os.path.join(survey.survey_dir, 'calib', 'decam',
    #                   'psfex-merged', '0013*', '*.fits')

    pat = os.path.join(survey.survey_dir, 'calib', 'decam',
                       #'psfex-merged', '*', '*.fits')
                       'splinesky-merged', '*', '*.fits')
    print('file pattern:', pat)
    got_expnums = set()
    merged_fns = glob(pat)
    merged_fns.sort()
    print(len(merged_fns), 'files found')
    for path in merged_fns:
        fn = os.path.basename(path)
        fn = fn.replace('.fits', '').replace('decam-', '')
        expnum = int(fn, 10)
        print('file', path, '-> expnum', expnum)
        got_expnums.add(expnum)

    need_expnums = list(set(expnums) - got_expnums)
    need_expnums.sort()

    print('Need expnums:', need_expnums)

    for expnum in need_expnums:
        for pat in ['/global/cscratch1/sd/desiproc/dr5/./calib/decam/psfex/%(expnumstr).5s/%(expnumstr)s/decam-%(expnumstr)s-*.fits',
                    '/global/cscratch1/sd/desiproc/dr3/./calib/decam/psfex/%(expnumstr).5s/%(expnumstr)s/decam-%(expnumstr)s-*.fits',
                    '/global/cscratch1/sd/dstn/dr5/./calib/decam/psfex/%(expnumstr).5s/%(expnumstr)s/decam-%(expnumstr)s-*.fits',
                    '/global/cscratch1/sd/dstn/dr3plus/./calib/decam/psfex/%(expnumstr).5s/%(expnumstr)s/decam-%(expnumstr)s-*.fits',
                    ]:
            pat = pat % dict(expnumstr=str('%08i' % expnum))

            pat = pat.replace('psfex', 'splinesky')

            fns = glob(pat)
            print('Expnum', expnum, '->', len(fns), 'files in', pat)
            if len(fns) == 0:
                continue
            #break
            cmd = 'rsync -LRar %s .' % pat
            print(cmd)
            rtn = os.system(cmd)
            #print('rtn', rtn)
            assert(rtn == 0)

