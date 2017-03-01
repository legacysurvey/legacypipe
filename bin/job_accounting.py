from __future__ import print_function
import argparse
import os
import numpy as np
from datetime import datetime
#import matplotlib
#import matplotlib.pyplot as plt
from astrometry.util.fits import fits_table, merge_tables

def ccd_cuts_inplace(ccds, use_blacklist=True):
    # coadd/ccd.fits file has extra white space
    ccds.camera= np.char.strip(ccds.camera)
    bands= ['g','r','z']
    # Continue like runbrick
    from legacypipe.runs import get_survey
    survey= get_survey('dr4v2',survey_dir=os.getenv('LEGACY_SURVEY_DIR'),output_dir='./')

    if use_blacklist:
        I = survey.apply_blacklist(ccds)
        ccds.cut(I)

    # Sort images by band -- this also eliminates images whose
    # *filter* string is not in *bands*.
    ccds.cut(np.hstack([np.flatnonzero(ccds.filter==band) for band in bands]))

    I = survey.photometric_ccds(ccds)
    ccds.cut(I)

    I = survey.bad_exposures(ccds)
    ccds.cut(I)


    I = survey.other_bad_things(ccds)
    ccds.cut(I)

    I = survey.ccds_for_fitting(brick, ccds)
    if I is not None:
        ccds.cut(I) 
 

parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
parser.add_argument('--therun', choices=['dr4','obiwan'],action='store', default=True) 
parser.add_argument('--dowhat', choices=['bricks_notdone','time_per_brick','nersc_time','sanity_tractors'],action='store', default=True) 
parser.add_argument('--fn', action='store', default=False) 
args = parser.parse_args() 

if args.dowhat == 'bricks_notdone':
    if args.therun == 'obiwan':
        b=fits_table(os.path.join(os.environ['LEGACY_SURVEY_DIR'],'survey-bricks-eboss-ngc.fits.gz'))
    elif args.therun == 'dr4':
        b=fits_table(os.path.join(os.environ['LEGACY_SURVEY_DIR'],'survey-bricks-dr4.fits.gz'))
    don=np.loadtxt(args.fn,dtype=str)
    fout= args.fn.replace('_done.tmp','_notdone.tmp')
    if os.path.exists(fout):
        os.remove(fout)
    # Bricks not finished
    with open(fout,'w') as fil:
        for brick in list( set(b.brickname).difference( set(don) ) ):
            fil.write('%s\n' % brick)
    print('Wrote %s' % fout)
    # All Bricks
    #fout= args.fn.replace('.tmp','_all.tmp')
    #if os.path.exists(fout):
    #    exit()
    #with open(fout,'w') as fil:
    #    for brick in b.brickname:
    #        fil.write('%s\n' % brick)
    #print('Wrote %s' % fout)
elif args.dowhat == 'time_per_brick':
    fns,start1,start2,end1,end2=np.loadtxt(args.fn,dtype=str,unpack=True)
    # Remove extraneous digits
    # Handle single log file
    if type(fns) == np.string_:
        fns= [fns]
        start1= [start1]
        end1= [end1]
        start2= [start2]
        end2= [end2]
    start2=np.array([val[:11] for val in start2])
    end2=np.array([val[:11] for val in end2])
    sumdt=0
    out=args.fn.replace('startend.txt','dt.txt')
    with open(out,'w') as foo:
        foo.write('# %s\n' % (os.path.dirname(args.fn),) )
        for fn,s1,s2,e1,e2 in zip(fns,start1,start2,end1,end2):
            name=os.path.basename(fn)
            start=datetime.strptime("%s %s" % (s1,s2), "%Y-%m-%d %H:%M:%S.%f")
            end=datetime.strptime("%s %s" % (e1,e2), "%Y-%m-%d %H:%M:%S.%f")
            # Write dt to file
            dt= end - start
            dt= dt.total_seconds() / 60.
            foo.write('%s %s\n' % (name,dt) )
            # Sum
            sumdt+= dt
        foo.write('# Total(min) %.1f\n' % sumdt)
    print('Wrote %s' % out)
elif args.dowhat == 'nersc_time':
    def time_hist(hrs,name='hist.png'):
        print('Making hist')
        bins=np.linspace(0,2,num=100)
        fig,ax= plt.subplots()
        ax.hist(hrs,bins=bins)
        # Median and 90% of everything
        med=np.percentile(hrs,q=50)
        ax.plot([med]*2,ax.get_ylim(),'r--')
        print('med=%f' % med)
        xlab=ax.set_xlabel('Hours (Wall Clock)')
        #ax.set_xlim(xlim)
        plt.savefig(name)
        plt.close()
        print('Wrote %s' % name)

    def nersc_time(hrs):
        tot= np.sum(hrs)
        nersc= 2*2*24*np.sum(hrs) # 24 cores/node, 2x queue factor, 2x machine factor
        print('total time nodes used [hrs]=%f' % tot)
        print('total NERSC time [hrs]=%f' % nersc)

    out='bricks_time_don.txt'
    if not os.path.exists(out):
        bricks,dt=np.loadtxt('bricks_time.txt',dtype=str,unpack=True)
        dt= dt.astype(float)
        don=np.loadtxt('don.txt',dtype=str)
        data={}
        with open(out,'w') as foo:
            print('Looping over don bricks')
            for b in don:
                data[b]= dt[bricks == b]
                if len( data[b] ) == 1:
                    foo.write('%s %s\n' % (b,str(data[b][0])) )
        print('Wrote %s' % out)
    # ALL DR4 bricks
    bricks,dt=np.loadtxt('bricks_time.txt',dtype=str,unpack=True)
    hrs=dt.astype(float)/3600
    time_hist(hrs,name='hist_bricks_all.png')
    print('ALL bricks')
    nersc_time(hrs)
    # Finished bricks
    bricks,dt=np.loadtxt(out,dtype=str,unpack=True)
    hrs= dt.astype(float)/3600.
    time_hist(hrs,name='hist_bricks_done.png')
    print('Done bricks')
    nersc_time(hrs)
elif args.dowhat == 'sanity_tractors':
    # RUN: python job_accounting.py --therun dr4 --dowhat sanity_tractors --fn dr4_tractors_done.tmp
    # Read each finished Tractor Catalogue
    # Append name to file if:
    # -- error reading it
    # -- no wise flux
    fns=np.loadtxt(args.fn,dtype=str)
    assert(len(fns) > 0)
    # Remove file lists for clean slate
    fils= dict(readerr='sanity_tractors_readerr.txt',\
               nowise='sanity_tractors_nowise.txt',\
               nolc='sanity_tractors_nolc.txt',\
               ccds='sanity_tractors_hascutccds.txt')
    for outfn in fils.keys():
        if os.path.exists(outfn):
            os.remove(outfn)
    # Loop over completed Tractor Cats
    for fn in fns:
        try: 
            t=fits_table(fn)
        except:
            print('error reading %s' % fn)
            with open(fils['readerr'],'a') as foo:
                foo.write('%s\n' % fn)
        # No wise info
        if not 'wise_flux' in t.get_columns():
            print('wise_flux not in %s' % fn)
            with open(fils['nowise'],'a') as foo:
                foo.write('%s\n' % fn)
        elif not 'wise_lc_flux' in t.get_columns():
            print('wise LCs not in %s' % fn)
            with open(fils['nolc'],'a') as foo:
                foo.write('%s\n' % fn)
        # No ccds that should have been removed by cuts
        #tractor/120/tractor-1201p715.fits 
        #coadd/120/1201p715/legacysurvey-1201p715-ccds.fits
        brick= os.path.basename(fn).replace('tractor-','').replace('.fits','')
        ccdfn= os.path.join( os.path.dirname(fn).replace('tractor','coadd'),\
                             brick, \
                             'legacysurvey-%s-ccds.fits' % brick )
        ccds= fits_table(ccdfn)
        ccds2= ccds.copy()
        ccd_cuts_inplace(ccds)
        if len(ccds) != len(ccds2):
            print('has ccds that should have been removed: %s' % ccdfn)
            with open(fils['ccds'],'a') as foo:
                foo.write('%s\n' % ccdfn)

