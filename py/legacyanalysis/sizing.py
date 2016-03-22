'''
This is a little script to look at the sizes of raw data on disk at NERSC, for
review document purposes.

'''
from __future__ import print_function
from glob import glob
from collections import Counter
import os
import numpy as np
import pylab as plt
from astrometry.util.plotutils import PlotSequence
import fitsio

def dir_sizing(ps, dirnm):
    '''
    Looks in directory 'dirnm' for raw FITS (.fz) images, reporting summary stats on
    file sizes and counts by type.  Works for NERSC directories containing Bok,
    Mosaic, and DECam data.

    *ps*: PlotSequence object for saving plots.
    '''
    dates = glob(dirnm)
    nighttotals = {}
    suffsizes = {}
    yeartotals = {}
    nightsperyear = Counter()

    propids = Counter()

    for date in dates:
        print('\nDate', date)
        day = os.path.basename(date)
        print('Day', day)
        year = datestring_to_fy(day)

        if not year in yeartotals:
            yeartotals[year] = {}
        nightsperyear.update([year])
        
        fns = glob(os.path.join(date, '*.fits*'))
        suffixes = np.unique([fn.split('_')[-1] for fn in fns])
        print('Suffixes:', suffixes)
        for suff in suffixes:
            fns = glob(os.path.join(date, '*'+suff))
            totsize = 0
            for fn in fns:
                st = os.stat(fn)
                totsize += st.st_size
                if not suff in suffsizes:
                    suffsizes[suff] = []
                suffsizes[suff].append(st.st_size)

                #if suff.startswith('ori'):
                #    hdr = fitsio.read_header(fn)
                #    propids.update([hdr['PROPID'].strip()])

            totsize /= (1024*1024)
            print('Suffix', suff, ':', len(fns), 'files, total', totsize, 'MB')
            if not suff in nighttotals:
                nighttotals[suff] = []
            nighttotals[suff].append(totsize)

            if not suff in yeartotals[year]:
                yeartotals[year][suff] = []
            yeartotals[year][suff].append(totsize)

    mx = 0
    for k,v in nighttotals.items():
        mx = max(mx, max(v))

    plt.clf()
    lp,lt = [],[]
    for k,v in nighttotals.items():
        n,b,p = plt.hist(v, histtype='step', range=(0, mx*1.1), bins=20)
        lp.append(p[0])
        lt.append(k)
    plt.legend(lp, lt, loc='upper right')
    plt.title('Total file size per night')
    ps.savefig()

    years = yeartotals.keys()
    years.sort()
    for year in years:
        totals = yeartotals[year]
        plt.clf()
        lp,lt = [],[]
        for k,v in totals.items():
            n,b,p = plt.hist(v, histtype='step', range=(0, mx*1.1), bins=20)
            lp.append(p[0])
            lt.append(k)
        plt.legend(lp, lt, loc='upper right')
        plt.title('Total file size per night: Year %i' % year)
        ps.savefig()

        print()
        print('Fiscal Year', year, ':', nightsperyear[year], 'nights')
        for k,v in yeartotals[year].items():
            print('Total', k, sum(v), 'MB')



    mx = 0
    for k,v in suffsizes.items():
        mx = max(mx, max(v))
    plt.clf()
    lp,lt = [],[]
    for k,v in suffsizes.items():
        n,b,p = plt.hist(v, histtype='step', range=(0, mx*1.1), bins=20, normed=True)
        lp.append(p[0])
        lt.append(k)
    plt.legend(lp, lt, loc='upper right')
    plt.title('File sizes by type')
    ps.savefig()


    print()
    for k,v in suffsizes.items():
        print('File type', k, 'average size', np.mean(v)/(1024*1024), 'MB')

    print()
    for k,v in suffsizes.items():
        print('File type', k, 'number of files:', len(v))

    print()
    for k,v in nighttotals.items():
        print('Total', k, sum(v), 'MB')

    print()
    print('Proposal IDs')
    for k,v in propids.most_common():
        print('  ', k, ':', v)


def datestring_to_fy(day):
    '''
    Converts a string like "20160316" to fiscal year, defined as
    beginning Oct 1 of previous calendar year.
    '''
    # HACK...
    day = int(day)
    if day >= 20111001 and day < 20121001:
        year = 2012
    elif day >= 20121001 and day < 20131001:
        year = 2013
    elif day >= 20131001 and day < 20141001:
        year = 2014
    elif day >= 20141001 and day < 20151001:
        year = 2015
    elif day >= 20151001 and day < 20161001:
        year = 2016
    else:
        raise RuntimeError('Bad year', day)
    return year

def decam_public(ps):
    fns = glob('/project/projectdirs/cosmo/staging/decam-public/20*/*.fits.fz')
    total = 0
    sizes = []
    yearsizes = {}
    
    for i,fn in enumerate(fns):
        if i % 1000 == 0:
            print(i, 'fn', fn)
        #if i == 10000:
        #    break
        st = os.stat(fn)
        total += st.st_size
        sizes.append(st.st_size)
        dirname = os.path.dirname(fn)
        date = os.path.basename(dirname)
        date = date.replace('-','')
        year = datestring_to_fy(date)
        if not year in yearsizes:
            yearsizes[year] = []
        yearsizes[year].append(st.st_size)
    
    
    print(len(fns), 'DECam-public; total size', total/(1024*1024*1024), 'GB')
    
    print('By year:')
    yrs = yearsizes.keys()
    yrs.sort()
    for year in yrs:
        print('Fiscal', year, ': total size', sum(yearsizes[year])/(1024*1024*1024), 'GB',
              'in', len(yearsizes[year]), 'files')
    
    sizes = np.array(sizes)
    plt.clf()
    plt.hist(sizes/(1024*1024), bins=50, histtype='step', color='b')
    plt.xlabel('File size (MB)')
    plt.title('decam-public @ NERSC')
    ps.savefig()



ps = PlotSequence('sizing')
dir_sizing(ps, '/project/projectdirs/cosmo/staging/bok/BOK_Raw/*')
dir_sizing(ps, '/project/projectdirs/cosmo/staging/mosaicz/MZLS_Raw/*')
dir_sizing(ps, '/project/projectdirs/cosmo/staging/decam/DECam_Raw/*')
decam_public(ps)

