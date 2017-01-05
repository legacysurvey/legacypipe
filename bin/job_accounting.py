from __future__ import print_function
import argparse
import os
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from astrometry.util.fits import fits_table, merge_tables

parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
parser.add_argument('--get_bricks_notdone', action='store_true', default=False) 
parser.add_argument('--time_per_brick', action='store_true', default=False) 
parser.add_argument('--nersc_time', action='store_true', default=False) 
args = parser.parse_args() 

if args.get_bricks_notdone:
    b=fits_table(os.path.join(os.environ['LEGACY_SURVEY_DIR'],'survey-bricks-dr4.fits.gz'))
    don=np.loadtxt('bricks_done.tmp',dtype=str)
    fout= 'bricks_notdone.tmp'
    if os.path.exists(fout):
        os.remove(fout)
    # Bricks not finished
    with open(fout,'w') as fil:
        for brick in list( set(b.brickname).difference( set(don) ) ):
            fil.write('%s\n' % brick)
    print('Wrote %s' % fout)
    # All Bricks
    fout= 'bricks_all.tmp'
    if os.path.exists(fout):
        exit()
    with open(fout,'w') as fil:
        for brick in b.brickname:
            fil.write('%s\n' % brick)
    print('Wrote %s' % fout)
elif args.time_per_brick:
    fns,start1,start2,end1,end2=np.loadtxt('logs_time.txt',dtype=str,unpack=True)
    # Remove extraneous digits
    start2=np.array([val[:11] for val in start2])
    end2=np.array([val[:11] for val in end2])
    tot={}
    for fn,s1,s2,e1,e2 in zip(fns,start1,start2,end1,end2):
        start=datetime.strptime("%s %s" % (s1,s2), "%Y-%m-%d %H:%M:%S.%f")
        end=datetime.strptime("%s %s" % (e1,e2), "%Y-%m-%d %H:%M:%S.%f")
        dt= end - start
        # Unique brickname, total time
        brick=os.path.basename( os.path.dirname(fn) )
        if tot.has_key(brick):
            tot[brick]+= dt.total_seconds()
        else:
            tot[brick]= dt.total_seconds()
    # Unique Brick list
    out="bricks_time.txt"
    with open(out,'w') as foo:
        for brick in tot.keys():
            foo.write('%s %s\n' % (brick,tot[brick]) )
    print('Wrote %s' % out)
elif args.nersc_time:
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

