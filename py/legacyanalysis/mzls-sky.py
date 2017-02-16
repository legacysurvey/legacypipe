from __future__ import print_function

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence
from astrometry.util.multiproc import multiproc
from legacypipe.survey import *
from legacypipe.runs import get_survey

import sys
import os
from pymongo import MongoClient

# mongodb cache
cache = None
# multiprocessing initializer for mongodb
def init():
    global cache
    client = MongoClient(os.environ['MONGO_HOST'])
    db = client[os.environ['MONGO_DB']]
    db.authenticate(os.environ['MONGO_RW_USER'], os.environ['MONGO_RW_PASS'])
    coll = db.mzls_sky
    cache = coll

def read_sky_val((mjd, key, oldvals, ccd)):
    try:
        im = survey.get_image_object(ccd)
        print('Reading', im)
        tim = im.get_tractor_image(gaussPsf=True, splinesky=False,
                                   #subsky=True,
                                   subsky=False,
                                   nanomaggies=False,
                                   dq=False, invvar=False)

        rawmed = np.median(tim.getImage().ravel())
        skyadu = tim.sky.getValue()
        print('ADU median:', rawmed, 'vs SKYADU', skyadu, 'zeropoint', ccd.ccdzpt)

        oldvals.update(median_adu = float(rawmed),
                       skyadu = float(skyadu),
                       ccdzpt = float(ccd.ccdzpt))


    except:
        import traceback
        print('Failed to read:', key)
        traceback.print_exc()

        oldvals.update(median_adu = None)


    print('Replace_one:', key, oldvals)

    cache.replace_one(key, oldvals, upsert=True)
    print('Cached', oldvals)
    return oldvals
    #cache.update_one

    # key.update(median=float(med))
    # print('Caching:', key)
    # cache.insert_one(key)
    # return med


survey = None

def main():
    global survey

    init()

    ps = PlotSequence('sky')
    # export LEGACY_SURVEY_DIR=/scratch1/scratchdirs/desiproc/DRs/dr4-bootes/legacypipe-dir/
    #survey = LegacySurveyData()
    survey = get_survey('dr4v2')
    ccds = survey.get_ccds_readonly()
    print(len(ccds), 'CCDs')
    ccds = ccds[ccds.camera == 'mosaic']
    print(len(ccds), 'Mosaic CCDs')
    # plt.clf()
    # plt.hist(ccds.mjd_obs % 1.0, bins=50)
    # plt.xlabel('MJD mod 1')
    # ps.savefig()

    ccds.imjd = np.floor(ccds.mjd_obs).astype(int)

    mjds = np.unique(ccds.imjd)
    print(len(mjds), 'unique MJDs')

    mp = multiproc(nthreads=8, init=init)

    medians = []
    medmjds = []

    median_adus = []
    skyadus = []

    args = []

    for kk,mjd in enumerate(mjds):
        I = np.flatnonzero(ccds.imjd == mjd)
        print('MJD', mjd, ' (%i of %i):' % (kk+1, len(mjds)), len(I), 'CCDs')
        if len(I) == 0:
            continue

        # pick one near the middle
        #i = I[len(I)/2]
        for i in [I[len(I)/4], I[len(I)/2], I[3*len(I)/4]]:

            ccd = ccds[i]
    
            key = dict(expnum=ccd.expnum, ccdname=ccd.ccdname)
            oldvals = key

            vals = cache.find(key)
            print('Got', vals.count(), 'cache hits for', key)
            gotone = False
            for val in vals:
                if 'median_adu' in val:
                    print('cache hit:', val)
                    madu =val['median_adu']
                    if madu is not None:
                        medians.append(val['median_adu'] - val['skyadu'])
                        medmjds.append(mjd)

                        median_adus.append(val['median_adu'])
                        skyadus.append(val['skyadu'])
                    
                    
                    gotone = True
                    break
                else:
                    print('partial cache hit:', val)
                    oldvals = val
                    ###!
                    cache.delete_one(val)
            if gotone:
                continue

            print('args: key', key, 'oldvals', oldvals)
            args.append((mjd, key, oldvals, ccd))

            # plt.clf()
            # plt.hist(tim.getImage().ravel(), range=(-0.1, 0.1), bins=50,
            #          histtype='step', color='b')
            # plt.axvline(med, color='b', alpha=0.3, lw=2)
            # plt.axvline(0., color='k', alpha=0.3, lw=2)
            # plt.xlabel('Sky-subtracted pixel values')
            # plt.title('Date ' + ccd.date_obs + ': ' + str(im))
            # ps.savefig()

    if len(args):
        meds = mp.map(read_sky_val, args)
        print('Medians:', meds)
        for (mjd,key,oldvals,ccd),val in zip(args, meds):
            if val is None:
                continue

            madu =val['median_adu']
            if madu is None:
                continue
                
            medians.append(val['median_adu'] - val['skyadu'])
            medmjds.append(mjd)

            median_adus.append(val['median_adu'])
            skyadus.append(val['skyadu'])

    plt.clf()
    plt.plot(medmjds, medians, 'b.')
    plt.xlabel('MJD')
    #plt.ylabel('Image median after sky subtraction (nmgy)')
    plt.ylabel('Image median after sky subtraction (ADU)')
    #plt.ylim(-0.03, 0.03)
    #plt.ylim(-10, 10)
    plt.ylim(-1, 1)

    plt.axhline(0, color='k', lw=2, alpha=0.5)
    ps.savefig()


    plt.clf()
    plt.plot(median_adus, skyadus, 'b.')
    ax = plt.axis()
    lo = min(ax[0], ax[2])
    hi = max(ax[1], ax[3])
    plt.plot([lo,hi],[lo,hi], 'k-', alpha=0.5)
    plt.xlabel('Median image ADU')
    plt.ylabel('SKYADU')
    plt.axis(ax)
    ps.savefig()

    plt.clf()
    plt.plot(medmjds, np.array(skyadus) / np.array(median_adus), 'b.')
    plt.xlabel('MJD')
    plt.ylim(0.98, 1.02)
    plt.axhline(1.0, color='k', alpha=0.5)
    plt.ylabel('SKYADU / median image ADU')
    ps.savefig()




if __name__ == '__main__':
    main()

