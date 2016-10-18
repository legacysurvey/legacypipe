'''
 PURPOSE:
   Lookup information in all raw images, cp images, or tractor catalogues and collect the results into a single fits file

 CALLING SEQUENCE:
   python digger.py -h

 DATA FILES:
   /global/project/projectdirs/cosmo/data/legacysurvey/dr3/
   /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP
   /project/projectdirs/cosmo/staging/bok/BOK_CP

 OUTPUT FILES:
   data_mine/

 REVISION HISTORY:
   18-Oct-2016  K. Burleigh
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import vstack, Table, Column
from astropy import units
from astropy.coordinates import SkyCoord
from argparse import ArgumentParser
import numpy as np
import os
import sys
import glob
import pickle
import traceback
#from functools import partial
#from subprocess import check_output

#def catcherr_decorator(some_function):
#    def wrapper(*args, **kwargs):
#        try: some_function(*args, **kwargs)
#        except:
#            exc_type, exc_value, exc_traceback = sys.exc_info()
#            print("Error:")
#            #traceback.print_tb(exc_type, file=sys.stdout)
#            #traceback.print_tb(exc_value, file=sys.stdout)
#            traceback.print_tb(exc_traceback, file=sys.stdout)
#            sys.stdout.flush()
#    return wrapper

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return np.sort(np.array( list(np.char.strip(lines)) ))

def rem_if_exists(name):
    if os.path.exists(name):
        if os.system(' '.join(['rm','%s' % name]) ): raise ValueError

#@catcherr_decorator
def mine_data(args=None,usempi=True):
    if usempi:
        from mpi4py import MPI
    parser = ArgumentParser(description="test")
    parser.add_argument("--file_list",action="store",help='text file listing absolute paths to FITS files want to dig through',required=True)
    parser.add_argument("--type",choices=['cpimages_mosaic','cpimages_bok','cpimages_decam','tractor'],action="store",help='what type of files are they? cp images, tractor catalogues?',required=True)
    parser.add_argument("--outdir",default='data_mine',action="store",help='where save rank ouputs',required=False)
    args = parser.parse_args(args=args)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    fits_files= read_lines(args.file_list) 
    cnt=0
    if usempi:
        comm = MPI.COMM_WORLD
        rank= comm.rank
        nodes= comm.size
        i=rank+cnt*nodes
        out_fn=os.path.join(args.outdir,"rank%d_%s.txt" % (rank,args.type))
    else:
        i=0
        out_fn=os.path.join(args.outdir,"serial_%s.txt" % args.type)
    
    rem_if_exists(out_fn)
    while i < len(fits_files):
        # Read catalogue
        cat_fn=fits_files[i]
        if usempi == False:
            print('reading %d/%d: %s' % (i+1,len(fits_files),cat_fn))
        # Save info to node specific file
        fobj=open(out_fn,'a')
        try:
            if args.type == 'tractor':
                data = Table(fits.getdata(cat_fn, 1))
                fobj.write("%s %.2f\n" % (cat_fn,tractor['cpu_source'].sum()))
            elif args.type == 'cpimages_mosaic':
                data = fits.open(cat_fn)
                for iccd in range(1,5):
                    fobj.write("%s %s %d %.5f %.5f\n" % (\
                                  cat_fn,data[0].header['DATE-OBS'],\
                                  data[iccd].header['CCDNUM'],\
                                  data[iccd].header['CENRA1'],data[iccd].header['CENDEC1']))
            elif args.type == 'cpimages_bok':
                data = fits.open(cat_fn)
                for iccd in range(1,5):
                    fobj.write("%s %s %d %.5f %.5f\n" % (\
                                  cat_fn,data[0].header['DATE-OBS'],\
                                  data[iccd].header['CCDNUM'],\
                                  data[iccd].header['CENRA1'],data[iccd].header['CENDEC1']))
        except KeyError as err:
            print("skipping {0} has error: {1}".format(cat_fn,err)) 
        fobj.close() 
        # Read next catalogue
        cnt+=1
        if usempi:
            i=comm.rank+cnt*comm.size
        else:
            i+=1
    if usempi:
        print("rank %d finished" % rank)
    else:
        print("finished serial run")

def gather_results(args=None):
    parser = ArgumentParser(description="test")
    parser.add_argument("--search",action="store",default='rank*.txt',help='wildcard string to search for',required=False)
    parser.add_argument("--type",choices=['mosaic','bok','decam','tractor'],action="store",help='what type of files are they? cp images, tractor catalogues?',required=True)
    parser.add_argument("--outdir",default='data_mine',action="store",help='where save rank ouputs',required=False)
    args = parser.parse_args(args=args)
    
    fns=glob.glob(args.search)
    if len(fns) < 1: 
        print("fns=",fns)
        raise ValueErrror
    for cnt,fn in enumerate(fns):
        if cnt == 0:
            if args.type == 'tractor':
                allcats,alltime= np.loadtxt(fn, dtype=str, delimiter=' ',unpack=True)
                alltime=alltime.astype(float)
            else:
                all={}
                all['fns'],all['obs'],all['ccdnum'],all['ra'],all['dec']= np.loadtxt(fn, dtype=str, delimiter=' ',unpack=True)
                #test= np.loadtxt(fn, dtype=str, delimiter=' ')
                #print(test)
                #print(test[0])
                #print(test.shape)
                #sys.exit('early')
                for key in all.keys():
                    if key == 'fns':
                        rt= '/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/'
                        if args.type == 'bok':
                            rt.replace('mosaicz/MZLS_CP','bok/BOK_CP')
                        all[key]= np.char.replace(all[key],rt,'')
                    elif key == 'obs':
                        all[key]= np.array([x[:10] for x in all[key]])
                    elif key == 'ccdnum':
                        all[key]= all[key].astype(int)
                    else:
                        all[key]= all[key].astype(float)
        else:
            if args.type == 'tractor':
                cats,time= np.loadtxt(fn, dtype=str, delimiter=' ',unpack=True)
                time=time.astype(float)
                allcats=np.concatenate((cats,allcats), axis=0)
                alltime=np.concatenate((time,alltime), axis=0)
            else:
                d={}
                d['fns'],d['obs'],d['ccdnum'],d['ra'],d['dec']= np.loadtxt(fn, dtype=str, delimiter=' ',unpack=True)
                for key in d.keys():
                    if key == 'fns':
                        rt= '/project/projectdirs/cosmo/staging/mosaicz/MZLS_CP/'
                        if args.type == 'bok':
                            rt.replace('mosaicz/MZLS_CP','bok/BOK_CP')
                        d[key]= np.char.replace(d[key],rt,'')
                    elif key == 'obs':
                        d[key]= np.array([x[:10] for x in d[key]])
                    elif key == 'ccdnum':
                        d[key]= d[key].astype(int)
                    else:
                        d[key]= d[key].astype(float)
                for key in all.keys():
                    all[key]= np.concatenate((d[key],all[key]), axis=0)
    if args.type == 'tractor':
        # extract brick names
        bricks=np.zeros(allcats.size).astype(str)
        for i in range(allcats.size):
            bricks[i]= allcats[i][allcats[i].find('tractor-')+8:-5]
    print('finished gather')
    # Save to fits
    if args.type == 'tractor':
        savefn= os.path.join(args.outdir,'results.pickle')
        if os.path.exists(savefn): 
            os.remove(savefn)
        fobj=open(savefn,'w')
        pickle.dump((allcats,bricks,alltime),fobj)
        fobj.close()    
    else:
        savefn= os.path.join(args.outdir,'survey-ccds-basic-info-{0}.fits'.format(args.type))
        if os.path.exists(savefn): 
            os.remove(savefn)
        fobj=open(savefn,'w')
        data_l=[]
        for key in all.keys():
            if key == 'fns':
                data_l+= [fits.Column(name=key, format='50A', array=all[key])]
            elif key == 'obs':
                data_l+= [fits.Column(name=key, format='10A', array=all[key])]
            elif key == 'ccdnum':
                data_l+= [fits.Column(name=key, format='I', array=all[key])]
            else:
                data_l+= [fits.Column(name=key, format='D', array=all[key])]
        tbhdu = fits.BinTableHDU.from_columns(data_l)
        tbhdu.writeto(savefn)
    print('wrote {0}'.format(savefn))

def sphere_nn(ref_ra,ref_dec,ra,dec):
    '''return index of spherical goemetry NN in ra,dec arrays to reference arrays ref_ra,ref_dec'''
    # ref is brick ra,dec and test is ccd 1-4 ra,dec
    ref_cat = SkyCoord(ra=ref_ra*units.degree, dec=ref_dec*units.degree)
    cat1 = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
    iref, d2d, d3d = cat1.match_to_catalog_sky(ref_cat,nthneighbor=1) 
    return np.array(list(set(iref)))

def get_bricks_touching_ccds(args=None):
    '''get bricks touched by these ccds'''
    parser = ArgumentParser(description="test")
    parser.add_argument("--type",choices=['mosaic','bok','decam'],action="store",help='what type of files are they?',required=True)
    parser.add_argument("--outdir",default='data_mine',action="store",help='where save rank ouputs',required=False)
    args = parser.parse_args(args=args)
    
    ccd_fn= os.path.join(args.outdir,'survey-ccds-basic-info-{0}.fits'.format(args.type))
    ccds= Table(fits.getdata(ccd_fn,1))
    bricks= Table(fits.getdata('/global/project/projectdirs/cosmo/data/legacysurvey/dr3/survey-bricks.fits.gz', 1)) 
    # Cut to footprint 
    if args.type in ['mosaic','bok']: 
        i= np.all((bricks['DEC'] >= 32.,\
                   bricks['DEC'] <= 75.,\
                   bricks['RA'] >= 88,\
                   bricks['RA'] <= 301),axis=0)
    else:
        i= np.all((bricks['DEC'] >= -20,\
                   bricks['DEC'] <= 34.),axis=0)
    print("%d/%d bricks in footprint" % (len(bricks[i]),len(bricks)))
    bricks= bricks[i]
    # CCD nearest neighbor 
    print("Finding neareast neighbors")
    keep= sphere_nn(bricks['RA'].data,bricks['DEC'].data,\
                    ccds['ra'],ccds['dec'])
    bricks= bricks[keep]
    print("%d bricks touching ccds" % len(bricks))
    fname= os.path.join(args.outdir,'survey-bricks-dr4-%s.fits.gz' % args.type) 
    if os.path.exists(fname): 
        os.remove(fname)
    bricks.write(fname,format='fits') 
    print("wrote {0}".format(fname))


def add_scatter(ax,x,y,c='b',m='o',lab='hello',s=80,drawln=False):
    ax.scatter(x,y, s=s, lw=2.,facecolors='none',edgecolors=c, marker=m,label=lab)
    if drawln: ax.plot(x,y, c=c,ls='-')

def plot_medt_grz(cpu,append='',max_exp=9):
    name='medt_grz_%s.png' % append 
    fig,ax=plt.subplots()
    add_scatter(ax,np.arange(cpu.size), cpu, c='b',m='o',lab='',drawln=True)
    plt.legend(loc='lower right',scatterpoints=1)
    #ax.set_yscale('log')
    #ax.set_ylim([1e-3,1e2])
    xlab=ax.set_ylabel('MPP hours (Median over Bricks)')
    ylab=ax.set_xlabel('nexp g+r+z (g,r,z <= %d)' % max_exp)
    plt.savefig(name, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close() 
 
def plot_medt_grz_errbars(cpu,cdf,max_exp=9):
    name='medt_grz_errbars.png'
    yerr=np.vstack((cpu['med']-cpu['min'],cpu['max']-cpu['med'])) 
    plt.errorbar(range(cpu['med'].size), cpu['med'],yerr=yerr,\
                 marker='o',fmt='o',c='b')
    # nexp= 3,6,9 are special
    imp=np.arange(max_exp*3)[::3][1:]
    yerr=yerr[:,imp] 
    plt.errorbar(np.arange(cpu['med'].size)[imp], cpu['med'][imp],yerr=yerr,\
                 marker='o',fmt='o',c='r')
    for i in imp:
        plt.text(i,cpu['med'][i],'%.2f' % (cpu['med'][i],),ha='left',va='top',fontsize='small')
        plt.text(i,cpu['max'][i],'%.2f' % (cpu['max'][i],),ha='left',va='top',fontsize='small')
        plt.text(i,cpu['min'][i],'%.2f' % (cpu['min'][i],),ha='left',va='top',fontsize='small')
    # cumulative %
    for i in range(max_exp*3):
        plt.text(i,cpu['max'][i],'%d%%' % (cdf[i],),ha='center',va='bottom',fontsize='small')
    #plt.legend(loc='lower right',scatterpoints=1)
    #ax.set_yscale('log')
    #ax.set_ylim([1e-3,1e2])
    xlab=plt.ylabel('MPP hours (Median over Bricks)')
    ylab=plt.xlabel('nexp g+r+z (g,r,z <= %d)' % max_exp)
    plt.savefig(name, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close() 
 
def analyze_results(args=None,max_exp=6):
    '''max_band_exp = 9 b/c at most have 3 exposures overlapping same time, 3 passes each'''
    parser = ArgumentParser(description="test")
    parser.add_argument("--savefn",action="store",default='results.pickle',required=False)
    args = parser.parse_args(args=args)

    fobj=open(args.savefn,'r')
    cats,bricks,times=pickle.load(fobj)
    fobj.close()

    fn='/project/projectdirs/cosmo/work/legacysurvey/dr3/survey-bricks-dr3.fits.gz'
    info=Table(fits.getdata(fn, 1))
    # sort both sets of data by brickname so they are rank aligned
    i=np.argsort(bricks)
    bricks,times=bricks[i],times[i]
    i=np.argsort(info['brickname'])
    info=info[i]
    assert( np.all(info['brickname'] == bricks) )
    # indices
    b={}
    for band in ['g','r','z']:
        b['%s' % band]=info['nexp_%s' % band] <= max_exp
    grz= info['nexp_g']+info['nexp_r']+info['nexp_z']
    for i in range(max_exp*3):
        b['grz%d' % i]= grz == i
    # data
    cdf=np.zeros(max_exp*3).astype(float)
    cpu={}
    for nam in ['min','max','med']: cpu[nam]= np.zeros(max_exp*3)-1
    for i in range(max_exp*3):
        cut= b['g']*b['r']*b['z']*b['grz%d' % i]
        cpu['med'][i]= np.median( times[cut] )
        cpu['min'][i]= np.min( times[cut] )
        cpu['max'][i]= np.max( times[cut] )
        cdf[i]= times[cut].size
    cdf=np.cumsum(cdf)
    cdf= cdf/bricks.size*100
    cdf.astype(int)
    # Convert to MPP hours, this is what nersc charges
    # MPP hours = wall[hrs] * nodes * cores * machine factor (2/2.5 edison/cori) * queue factor
    for nam in ['min','max','med']: 
        cpu[nam]= cpu[nam]/6./3600. * 1. * 6. * 2.5 * 2. #sum(cpu_source in cat)/6 is ~ wall time
    # plot
    for nam in ['min','max','med']: plot_medt_grz(cpu[nam],append=nam,max_exp=max_exp)
    plot_medt_grz_errbars(cpu,cdf,max_exp=max_exp)
    # Print MPP hours 
    print("total MPP hours dr3 'fitblobs'= %.2f" % (times.sum()/6./3600. * 1. * 6. * 2.5 * 2.,))
    for passes in [1,2,3]:
        b=np.ones(info['nexp_g'].size).astype(bool)
        for band in ['g','r','z']: b*= (info['nexp_%s' % band] >= passes)
        print("fraction of bricks with median g,r,z >= %d is %.2f" % \
                (passes,float(info['brickname'][b].size)/info['brickname'].size))

def main(args=None,program='analyze'):
    if program == 'analyze':
        analyze_results()
    elif program == 'get_bricks':
        get_bricks_touching_ccds()
    elif program == 'gather':
        gather_results()
    else: 
        mine_data(usempi=False) #set usempi=False to run in serial so can debug

if __name__ == '__main__':
    # run like this, do 3 times
    # find /project/projectdirs/cosmo/staging/mosaicz/MZLS_CP -type f -name "k4m_*_ooi_zd_v1.fits.fz" > all_mosaic.txt
    # python digger.py --file_list all_mosaic.txt --type cpimages_mosaic
    #main(program='other')
    # python digger.py --search 'data_mine/rank*mosaic.txt' --type mosaic
    #main(program='gather')
    # python digger.py --type mosaic
    main(program='get_bricks')

