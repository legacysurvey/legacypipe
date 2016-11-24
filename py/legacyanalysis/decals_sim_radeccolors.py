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
    def __init__(self,objtype='star'):
        self.objtype= objtype
        self.kdefn='%s-kde.pickle' % self.objtype
        self.kde= self.get_kde()

    def get_kde(self):
        fout=open(self.kdefn,'r')
        kde= pickle.load(fout)
        fout.close()
        return kde

    def get_colors(self,ndraws=1,random_state=np.random.RandomState()):
        xyz= self.kde.sample(n_samples=ndraws,random_state=random_state)
        if self.objtype == 'star':
            #labels=['r wdust','r-z','g-r']
            r= xyz[:,0]
            z= r- xyz[:,1]
            g= r+ xyz[:,2]
        elif self.objtype == 'qso':
            #labels=['r wdust','r-z','g-r']
            r= xyz[:,0]
            z= r- xyz[:,1]
            g= r+ xyz[:,2]
        elif self.objtype == 'elg':
            #labels=['r wdust','r-z','g-r'] 
            r= xyz[:,0]
            z= r- xyz[:,1]
            g= r+ xyz[:,2]
        elif self.objtype == 'lrg':
            #labels=['z wdust','r-z','r-W1','g wdust']
            z= xyz[:,0]
            r= z+ xyz[:,1]
            g= xyz[:,3]
        else: 
            raise ValueError('objecttype= %s, not supported' % self.objtype)
        return g,r,z
            
def get_fn(outdir,seed):
    return os.path.join(outdir,'sample_%d.fits' % seed)        
                    
def draw_points(radec,ndraws=1,seed=1,outdir='./'):
    random_state= np.random.RandomState(seed)
    ra,dec= get_radec(radec,ndraws=ndraws,random_state=random_state)
    kde_obj= KDEColors(objtype='star')
    g,r,z= kde_obj.get_colors(ndraws=ndraws,random_state=random_state)
    T=fits_table()
    for key,arr in zip(['ra','dec','g','r','z'],\
                       [ra,dec,g,r,z]):
        T.set(key,arr)
    T.set('seed',np.zeros(ndraws).astype(int)+seed)
    T.writeto(get_fn)

def merge_draws(outdir='./'):
    '''merges all fits tables created by draw_points()'''
    fns=glob.glob(os.path.join(outdir,"sample_*.fits"))
    if not len(fns) > 0: raise ValueError('no fns found')
    from theValidator.catalogues import CatalogueFuncs
    T= CatalogueFuncs().stack(fns,textfile=False)
    name=os.path.join(outdir,'sample_merged.fits')
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
    parser.add_argument('--Nper',type=int,action='store',help='number of draws per mpi task',required=True)
    parser.add_argument('--jobid',action='store',help='slurm jobid',default='001',required=False)
    parser.add_argument('--prefix', type=str, default='', help='Prefix to prepend to the output files.')
    parser.add_argument('--outdir', type=str, default='./legacy_zpt_outdir', help='Output directory.')
    parser.add_argument('--nproc', type=int, default=1, help='Number of CPUs to use.')
    args = parser.parse_args()

    radec={}
    radec['ra1']=args.ra1
    radec['ra2']=args.ra2
    radec['dec1']=args.dec1
    radec['dec2']=args.dec2
 
    if args.nproc > 1:
        from mpi4py.MPI import COMM_WORLD as comm
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
        draw_points(radec,ndraws=args.Nper, seed=seed,outdir=args.outdiri)
        # Gather
        all_cats = comm.gather( cats, root=0 )
        if comm.rank == 0:
            all_cats= merge_tables(all_cats, columns='fillzero')
            if os.path.exists(opt.outname):
                os.remove(opt.outname)
            all_cats.writeto(opt.outname)
            print('Wrote %s' % opt.outname)
            print("Done")
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
        draw_points(radec,ndraws=args.Nper, seed=seed,outdir=args.outdir)
        ## Create the file
        #t0=ptime('b4-run',t0)
        #runit(image_fn, measureargs,\
        #      zptsfile=zptsfile,zptstarsfile=zptstarsfile)
        #t0=ptime('after-run',t0)
        #tnow= Time()
        #print("TIMING:total %s" % (tnow-tbegin,))
        #print("Done")

