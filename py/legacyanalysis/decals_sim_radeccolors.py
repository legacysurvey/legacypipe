#!/usr/bin/env python

"""
"""
from __future__ import division, print_function

import os
import argparse
import numpy as np
from glob import glob
from astrometry.util.ttime import Time
import datetime
import sys
import pickle

from astrometry.util.fits import fits_table, merge_tables
from theValidator.catalogues import CatalogueFuncs

######## 
## Ted's
import time
from contextlib import contextmanager

@contextmanager
def stdouterr_redirected(to=os.devnull, comm=None):
    '''
    Based on http://stackoverflow.com/questions/5081657
    import os
    with stdouterr_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    sys.stdout.flush()
    sys.stderr.flush()
    fd = sys.stdout.fileno()
    fde = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fde) # fd writes to 'to' file
        sys.stderr = os.fdopen(fde, 'w') # Python writes to fd
        
    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        if (comm is None) or (comm.rank == 0):
            print("Begin log redirection to {} at {}".format(to, time.asctime()))
        sys.stdout.flush()
        sys.stderr.flush()
        pto = to
        if comm is None:
            if not os.path.exists(os.path.dirname(pto)):
                os.makedirs(os.path.dirname(pto))
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        else:
            pto = "{}_{}".format(to, comm.rank)
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            if comm is not None:
                # concatenate per-process files
                comm.barrier()
                if comm.rank == 0:
                    with open(to, 'w') as outfile:
                        for p in range(comm.size):
                            outfile.write("================= Process {} =================\n".format(p))
                            fname = "{}_{}".format(to, p)
                            with open(fname) as infile:
                                outfile.write(infile.read())
                            os.remove(fname)
                comm.barrier()

            if (comm is None) or (comm.rank == 0):
                print("End log redirection to {} at {}".format(to, time.asctime()))
            sys.stdout.flush()
            sys.stderr.flush()
            
    return
##############

def ptime(text,t0):
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    if len(lines) < 1: raise ValueError('lines not read properly from %s' % fn)
    return np.array( list(np.char.strip(lines)) )

def dobash(cmd):
    print('UNIX cmd: %s' % cmd)
    if os.system(cmd): raise ValueError

def get_area(radec):
    '''returns area on sphere between ra1,ra2,dec2,dec1
    https://github.com/desihub/imaginglss/model/brick.py#L64, self.area=...
    '''
    deg = np.pi / 180.
    # Wrap around
    if radec['ra2'] < radec['ra1']:
        ra2=radec['ra2']+360.
    else:
        ra2=radec['ra2']
    
    area= (np.sin(radec['dec2']*deg)- np.sin(radec['dec1']*deg)) * \
          (ra2 - radec['ra1']) * \
          deg* 129600 / np.pi / (4*np.pi)
    approx_area= (radec['dec2']-radec['dec1'])*(ra2-radec['ra1'])
    print('approx area=%.2f deg2, actual area=%.2f deg2' % (approx_area,area))
    return area

def get_radec(radec,\
              ndraws=1,random_state=np.random.RandomState()):
    '''https://github.com/desihub/imaginglss/blob/master/scripts/imglss-mpi-make-random.py#L55'''
    ramin,ramax= radec['ra1'],radec['ra2']
    dcmin,dcmax= radec['dec1'],radec['dec2']
    u1,u2= random_state.uniform(size=(2, ndraws) )
    #
    cmin = np.sin(dcmin*np.pi/180)
    cmax = np.sin(dcmax*np.pi/180)
    #
    RA   = ramin + u1*(ramax-ramin)
    DEC  = 90-np.arccos(cmin+u2*(cmax-cmin))*180./np.pi
    return RA,DEC

class KDEColors(object):
    def __init__(self,objtype='star',pickle_dir='./'):
        self.objtype= objtype
        self.kdefn=os.path.join(pickle_dir,'%s-kde.pickle' % self.objtype)
        self.kde= self.get_kde()

    def get_kde(self):
        fout=open(self.kdefn,'r')
        kde= pickle.load(fout)
        fout.close()
        return kde

    def get_colors(self,ndraws=1,random_state=np.random.RandomState()):
        samp= self.kde.sample(n_samples=ndraws,random_state=random_state)
        if self.objtype == 'star':
            #labels=['r wdust','r-z','g-r']
            r= samp[:,0]
            z= r- samp[:,1]
            g= r+ samp[:,2]
            return g,r,z
        elif self.objtype == 'qso':
            #labels=['r wdust','r-z','g-r']
            r= samp[:,0]
            z= r- samp[:,1]
            g= r+ samp[:,2]
            redshift= samp[:,3]
            return g,r,z,redshift
        elif self.objtype == 'elg':
            #labels=['r wdust','r-z','g-r'] 
            r= samp[:,0]
            z= r- samp[:,1]
            g= r+ samp[:,2]
            redshift= samp[:,3]
            return g,r,z,redshift
        elif self.objtype == 'lrg':
            #labels=['z wdust','r-z','r-W1','g wdust']
            z= samp[:,0]
            r= z+ samp[:,1]
            redshift= samp[:,3]
            g= samp[:,4]
            return g,r,z,redshift
        else: 
            raise ValueError('objecttype= %s, not supported' % self.objtype)

class KDEshapes(object):
    def __init__(self,objtype='elg',pickle_dir='./'):
        assert(objtype in ['lrg','elg'])
        self.objtype= objtype
        self.kdefn=os.path.join(pickle_dir,'%s-shapes-kde.pickle' % self.objtype)
        self.kde= self.get_kde()

    def get_kde(self):
        fout=open(self.kdefn,'r')
        kde= pickle.load(fout)
        fout.close()
        return kde

    def get_shapes(self,ndraws=1,random_state=np.random.RandomState()):
        samp= self.kde.sample(n_samples=ndraws,random_state=random_state)
        # Same for elg,lrg
        re= samp[:,0]
        n=  samp[:,1]
        ba= samp[:,2]
        pa= samp[:,3]
        # ba can be [1,1.2] due to KDE algorithm, make these 1
        ba[ ba > 1 ]= 1.
        return re,n,ba,pa
 
            
def get_fn(outdir,seed):
    return os.path.join(outdir,'sample_%d.fits' % seed)        
                    
def draw_points(radec,ndraws=1,seed=1,outdir='./'):
    '''writes ra,dec,grz qso,lrg,elg,star to fits file
    for given seed'''
    random_state= np.random.RandomState(seed)
    ra,dec= get_radec(radec,ndraws=ndraws,random_state=random_state)
    # Mags
    mags={}
    for typ in ['star','lrg','elg','qso']:
        kde_obj= KDEColors(objtype=typ,pickle_dir=outdir)
        if typ == 'star':
            mags['%s_g'%typ],mags['%s_r'%typ],mags['%s_z'%typ]= \
                        kde_obj.get_colors(ndraws=ndraws,random_state=random_state)
        else:
            mags['%s_g'%typ],mags['%s_r'%typ],mags['%s_z'%typ],mags['redshift']= \
                        kde_obj.get_colors(ndraws=ndraws,random_state=random_state)
    # Shapes
    gfit={}
    for typ in ['lrg','elg']:
        kde_obj= KDEshapes(objtype=typ,pickle_dir=outdir)
        gfit['%s_re'%typ],gfit['%s_n'%typ],gfit['%s_ba'%typ],gfit['%s_pa'%typ]= \
                    kde_obj.get_shapes(ndraws=ndraws,random_state=random_state)
    # Write fits table
    T=fits_table()
    T.set('id',np.range(ndraws))
    T.set('seed',np.zeros(ndraws).astype(int)+seed)
    T.set('ra',ra)
    T.set('dec',dec)
    for key in mags.keys():
        T.set(key,mags[key])
    for key in gfit.keys():
        T.set(key,gfit[key])
    # Galaxy Properties
    #T.set('sersicn', random_state.uniform(0.5,0.5, ndraws))
    #T.set('rhalf', random_state.uniform(0.5,0.5, ndraws)) #arcsec
    #T.set('ba', random_state.uniform(0.2,1.0, ndraws)) #minor to major axis ratio
    #T.set('phi', random_state.uniform(0.0, 180.0, ndraws)) #position angle
    T.writeto( get_fn(outdir,seed) )

def merge_draws(outdir='./'):
    '''merges all fits tables created by draw_points()'''
    fns=glob(os.path.join(outdir,"sample_*.fits"))
    if not len(fns) > 0: raise ValueError('no fns found')
    T= CatalogueFuncs().stack(fns,textfile=False)
    # Add unique id column
    T.set('id',np.arange(len(T))+1)
    # Save
    name=os.path.join(outdir,'sample-merged.fits')
    if os.path.exists(name):
        os.remove(name)
        print('Making new %s' % name)
    T.writeto(name)
    print('wrote %s' % name)
       

if __name__ == "__main__":
    t0 = Time()
    tbegin=t0
    print('TIMING:after-imports ',datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser.add_argument('--ra1',type=float,action='store',help='bigbox',required=True)
    parser.add_argument('--ra2',type=float,action='store',help='bigbox',required=True)
    parser.add_argument('--dec1',type=float,action='store',help='bigbox',required=True)
    parser.add_argument('--dec2',type=float,action='store',help='bigbox',required=True)
    parser.add_argument('--spacing',type=float,action='store',default=10.,help='choosing N radec pionts so points have spacingxspacing arcsec spacing',required=False)
    parser.add_argument('--ndraws',type=int,action='store',help='default space by 5x5'', number of draws for all mpi tasks',required=False)
    parser.add_argument('--jobid',action='store',help='slurm jobid',default='001',required=False)
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--outdir', type=str, default='./radec_points_dir', help='Output directory.')
    parser.add_argument('--nproc', type=int, default=1, help='Number of CPUs to use.')
    args = parser.parse_args()

    radec={}
    radec['ra1']=args.ra1
    radec['ra2']=args.ra2
    radec['dec1']=args.dec1
    radec['dec2']=args.dec2
    if args.ndraws is None:
        # Number that could fill a grid with 5x5 arcsec spacing
        ndraws= int( get_area(radec)/args.spacing**2 * 3600.**2 ) + 1
    else:
        ndraws= args.ndraws
    print('ndraws= %d' % ndraws)

    # Draws per mpi task
    if args.nproc > 1:
        from mpi4py.MPI import COMM_WORLD as comm
        nper= int(ndraws/float(comm.size))
    else: 
        nper= ndraws
    t0=ptime('parse-args',t0)

    if args.nproc > 1:
        if comm.rank == 0:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
        seed = comm.rank
        cnt=0
        while os.path.exists(get_fn(args.outdir,seed)):
            print('skipping, exists: %s' % get_fn(args.outdir,seed))
            cnt+=1
            seed= comm.rank+ comm.size*cnt
        draw_points(radec,ndraws=nper, seed=seed,outdir=args.outdir)
        # Gather
        junk=[comm.rank]
        junks = comm.gather(junk, root=0 )
        if comm.rank == 0:
            merge_draws(outdir=args.outdir)
        #images_split= np.array_split(images, comm.size)
        # HACK, not sure if need to wait for all proc to finish 
        #confirm_files = comm.gather( images_split[comm.rank], root=0 )
        #if comm.rank == 0:
        #    print('Rank 0 gathered the results:')
        #    print('len(images)=%d, len(gathered)=%d' % (len(images),len(confirm_files)))
        #    tnow= Time()
        #    print("TIMING:total %s" % (tnow-tbegin,))
        #    print("Done")
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        seed= 1
        cnt=1
        while os.path.exists(get_fn(args.outdir,seed)):
            print('skipping, exists: %s' % get_fn(args.outdir,seed))
            cnt+=1
            seed= cnt
        print('working on: %s' % get_fn(args.outdir,seed))
        draw_points(radec,ndraws=nper, seed=seed,outdir=args.outdir)
        # Gather equivalent
        merge_draws(outdir=args.outdir)
        ## Create the file
        #t0=ptime('b4-run',t0)
        #runit(image_fn, measureargs,\
        #      zptsfile=zptsfile,zptstarsfile=zptstarsfile)
        #t0=ptime('after-run',t0)
        #tnow= Time()
        #print("TIMING:total %s" % (tnow-tbegin,))
        #print("Done")

