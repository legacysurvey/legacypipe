import fitsio
import numpy as np
from glob import glob
import os
from legacypipe.survey import LegacySurveyData

def main():
    bands = ['g','r','i','z']
    wbands = ['W1', 'W2', 'W3', 'W4']
    fns = glob('/pscratch/sd/d/dstn/sub-blobs/tractor/*/brick-*.sha256sum')
    fns.sort()
    surveys = {}
    for fn in fns:
        print()
        #print(fn)
        brick = fn.replace('.sha256sum', '')[-8:]
        print('Brick', brick)
        basedir = '/'.join(fn.split('/')[:-3])
        #print('basedir', basedir)
        cmd = 'cd %s && sha256sum --quiet -c %s' % (basedir, fn)
        print(cmd)
        rtn = os.system(cmd)
        #print(rtn)
        assert(rtn == 0)
        shas = open(fn).readlines()
        shas = set([s.strip().split()[1].replace('*','') for s in shas])

        if not basedir in surveys:
            surveys[basedir] = LegacySurveyData(survey_dir=basedir)
        survey = surveys[basedir]

        allfns = []
        for filetype in ['tractor', 'tractor-intermediate', 'ccds-table', 'depth-table',
                         'image-jpeg', 'model-jpeg', 'resid-jpeg',
                         'blobmodel-jpeg',
                         'wise-jpeg', 'wisemodel-jpeg', 'wiseresid-jpeg',
                         'outliers-pre', 'outliers-post',
                         'outliers-masked-pos', 'outliers-masked-neg',
                         'outliers_mask',
                         'blobmap', 'maskbits', 'all-models', 'ref-sources',
                         ]:
            fn = survey.find_file(filetype, brick=brick)
            #print(fn)
            assert(os.path.exists(fn))
            allfns.append(fn.replace(basedir+'/', ''))

        for band in bands:
            for i,filetype in enumerate(['invvar', 'chi2', 'image', 'model', 'blobmodel',
                                         'depth', 'galdepth', 'nexp', 'psfsize',]):
                fn = survey.find_file(filetype, brick=brick, band=band)
                #print(fn)
                exists = os.path.exists(fn)
                # Either all products exist for a band, or none!
                if i == 0:
                    has_band = exists
                else:
                    assert(has_band == exists)
                if has_band:
                    allfns.append(fn.replace(basedir+'/', ''))
            print('Band', band, 'exists:', has_band)

        for band in wbands:
            for i,filetype in enumerate(['invvar', 'image', 'model']):
                fn = survey.find_file(filetype, brick=brick, band=band)
                #print(fn)
                assert(os.path.exists(fn))
                allfns.append(fn.replace(basedir+'/', ''))

        print('sha:', len(shas))
        print('files:', len(allfns))
        #print(shas)
        #print(set(allfns))
        assert(set(shas) == set(allfns))
            
        #break

if __name__ == '__main__':
    main()
