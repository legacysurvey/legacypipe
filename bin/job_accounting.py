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

def parse_coords(s):
    '''stackoverflow: 
    https://stackoverflow.com/questions/9978880/python-argument-parser-list-of-list-or-tuple-of-tuples'''
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y") 

parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
parser.add_argument('--dowhat', choices=['sanity_dr4c','dr4c_vs_dr4b','bricks_notdone','time_per_brick','nersc_time','sanity_tractors','num_grz','badastrom','count_objects'],action='store', default=True) 
parser.add_argument('--fn', action='store', default=False) 
parser.add_argument('--line_start', type=int,default=None, help='first line in fn list to use')
parser.add_argument('--line_end', type=int,default=None, help='last line in fn list to use')
args = parser.parse_args() 

if args.dowhat == 'sanity_dr4c':
    ncols= 165
    # RUN: python job_accounting.py --dowhat sanity_tractors --fn dr4_tractors_done.tmp
    fns=np.loadtxt(args.fn,dtype=str)
    assert(len(fns) > 0)
    print(args)
    if args.line_start and args.line_end:
        fns= fns[args.line_start:args.line_end]
    # Remove file lists for clean slate
    fils= dict(readerr='%s_readerr.txt' % args.dowhat,
               ncolswrong='%s_ncolswrong.txt' % args.dowhat,
               nancols='%s_nancols.txt' % args.dowhat,
               unexpectedcol='%s_unexpectedcol.txt' % args.dowhat)
    for outfn in fils.keys():
        if os.path.exists(outfn):
            os.remove(outfn)
    # Loop over completed Tractor Cats
    for ith,fn in enumerate(fns):
        if ith % 100 == 0: print('%d/%d' % (ith+1,len(fns)))
        try: 
            t=fits_table(fn)
        except:
            # Report any read errors
            print('error reading %s' % fn)
            with open(fils['readerr'],'a') as foo:
                foo.write('%s\n' % fn)
        # Number of columns
        if len(t.get_columns()) != ncols:
            with open(fils['ncolswrong'],'a') as foo:
                foo.write('%s %d\n' % (fn,len(t.get_columns())))
        # Any Nans?
        for col in t.get_columns():
            try:
                ind= np.isfinite(t.get(col)) == False
                # This col has a non-finite value
                if np.any(ind):
                    with open(fils['nancols'],'a') as foo:
                        foo.write('%s %s\n' % (fn,col))
            except TypeError:
                # np.isfinite cannot be applied to these data types
                if col in ['brickname','type','wise_coadd_id']:
                    pass
                # report col if this error occurs for a col not in the above
                else:
                    with open(fils['unexpectedcol'],'a') as foo:
                        foo.write('%s %s\n' % (fn,col))
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
    # RUN: python job_accounting.py --dowhat sanity_tractors --fn dr4_tractors_done.tmp
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
               badastrom='sanity_tractors_badastromccds.txt',\
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
        # CCDs
        #  tractor/120/tractor-1201p715.fits 
        #  coadd/120/1201p715/legacysurvey-1201p715-ccds.fits
        brick= os.path.basename(fn).replace('tractor-','').replace('.fits','')
        ccdfn= os.path.join( os.path.dirname(fn).replace('tractor','coadd'),\
                             brick, \
                             'legacysurvey-%s-ccds.fits' % brick )
        ccds= fits_table(ccdfn)
        # Bad Astrometry
        flag= (np.sqrt(ccds.ccdrarms**2 + ccds.ccddecrms**2) >= 0.1)*(ccds.ccdphrms >= 0.1)
        if np.where(flag)[0].size > 0:
            with open(fils['badastrom'],'a') as foo:
                foo.write('%s\n' % ccdfn)
        #ccds2= ccds.copy()
        #ccd_cuts_inplace(ccds)
        #if len(ccds) != len(ccds2):
        #    print('has ccds that should have been removed: %s' % ccdfn)
        #    with open(fils['ccds'],'a') as foo:
        #        foo.write('%s\n' % ccdfn)
elif args.dowhat == 'badastrom':
    # RUN: python job_accounting.py --dowhat sanity_tractors --fn dr4_tractors_done.tmp
    # Read each finished Tractor Catalogue
    # Append name to file if:
    # -- error reading it
    # -- no wise flux
    bricks=np.loadtxt(args.fn,dtype=str)
    assert(len(bricks) > 0)
    # Remove file lists for clean slate
    fils= dict(readerr='sanity_ccds_readerr.txt',\
               badastrom='sanity_ccds_have_badastrom.txt')
    for outfn in fils.keys():
        if os.path.exists(outfn):
            os.remove(outfn)
    for cnt,brick in enumerate(bricks):
        if cnt % 100 == 0: print('reading %d/%d' % (cnt,len(bricks)))
        bri= brick[:3]
        fn= '/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4_fixes/coadd/%s/%s/legacysurvey-%s-ccds.fits' % (bri,brick,brick)
        try: 
            ccds=fits_table(fn)
            # Bad Astrometry
            flag= np.any((np.sqrt(ccds.ccdrarms**2 + ccds.ccddecrms**2) > 0.1,
                         ccds.ccdphrms > 0.2), axis=0)
            if np.where(flag)[0].size > 0:
                with open(fils['badastrom'],'a') as foo:
                    foo.write('%s %s\n' % (brick,fn))
        except:
            print('error reading %s' % fn)
            with open(fils['readerr'],'a') as foo:
                foo.write('%s\n' % fn)
elif args.dowhat == 'count_objects':
    # RUN: python job_accounting.py --dowhat sanity_tractors --fn dr4_tractors_done.tmp
    # Read each finished Tractor Catalogue
    # Append name to file if:
    # -- error reading it
    # -- no wise flux
    fns=np.loadtxt(args.fn,dtype=str)
    assert(len(fns) > 0)
    # Remove file lists for clean slate
    fils= dict(readerr='sanity_tractors_readerr.txt',\
               counts='sanity_tractors_counts.txt')
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
        with open(fils['counts'],'a') as foo:
            foo.write('%d %s\n' % (len(t),fn))
elif args.dowhat == 'num_grz':
    # record number of grz for a list of bricks
    bricklist= np.loadtxt= np.loadtxt(args.fn,dtype=str)
    nccds= []
    d={}
    for i in range(1,10,2):
        d['%d <= grz' % i]=[]
    for cnt,brick in enumerate(bricklist):
        if cnt % 100 == 0:
            print('Reading %d/%d' % (cnt+1,len(bricklist)))
        # Try reading
        try:
            fn= "/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4/coadd/%s/%s/legacysurvey-%s-ccds.fits" % (brick[:3],brick,brick)
            #fn= "/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4_fixes/coadd/%s/%s/legacysurvey-%s-ccds.fits" % (brick[:3],brick,brick)
            a=fits_table(fn)
        except IOError:
            print('Cannot find %s' % fn) 
            continue
        # Save info
        nccds.append( len(a) )
        for i in range(1,10,2):
            g= np.where(a.filter == 'g')[0].size
            r= np.where(a.filter == 'r')[0].size
            z= np.where(a.filter == 'z')[0].size
            if (g >= i) & (r >= i) & (z >= i):
                d['%d <= grz' % i].append( brick )
    # Print
    for i in range(1,10,2):
        key= '%d <= grz' % i
        print('%d bricks with %s' % (len(d[key]),key))
    nccds= np.array(nccds)
    inds= np.argsort(nccds)[::-1]
    for brick,ccd in zip(bricklist[inds][:10],nccds[inds][:10]):
        print('%d ccds in %s' % (ccd,brick))
    with open('grz_ge_1.txt','w') as foo:
        for brick in d['1 <= grz']:
            foo.write('%s\n' % brick)
    print('wrote %s' % 'grz_ge_1.txt')
