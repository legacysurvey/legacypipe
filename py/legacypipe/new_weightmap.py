from __future__ import division, print_function

import os
import numpy as np
from glob import glob
import datetime
import sys
import argparse

from astropy.io import fits
from astrometry.util.ttime import Time

from legacypipe.cpimage import newWeightMap

######## 
# stdouterr_redirected() is from Ted Kisner
# Every mpi task (zeropoint file) gets its own stdout file
import time
from contextlib import contextmanager

def ptime(text,t0):
    '''Timer'''    
    tnow=Time()
    print('TIMING:%s ' % text,tnow-t0)
    return tnow



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
            

if __name__ == "__main__":
    t0= Time()
    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.') 
    parser.add_argument('--image_list', action='store',required=True) 
    parser.add_argument('--nproc', type=int,action='store',default=1,required=False) 
    args = parser.parse_args() 

    image_list= np.loadtxt(args.image_list,dtype=str)

    rootdir= os.path.join(os.getenv('LEGACY_SURVEY_DIR'),'images')
    if os.getenv('NERSC_HOST') == 'edison':
        outdir='/scratch1/scratchdirs/desiproc/DRs/data-releases/dr4'
    elif os.getenv('NERSC_HOST') == 'cori':
        outdir='/global/cscratch1/sd/desiproc/dr4/data_release/dr4'
    else: 
        raise ValueError('NERSC_HOST is not edison or cori --> %s' % os.getenv('NERSC_HOST'))
    outdir= os.path.join(outdir,'logs/new_wtmaps')

    if args.nproc > 1:
        from mpi4py.MPI import COMM_WORLD as comm
    t0=ptime('parse-args',t0)

    if args.nproc > 1:
        fns= np.array_split(image_list, comm.size)[comm.rank]
        # Log to unique file
        outfn=os.path.join(outdir,"log.rank%d_%s" % \
                    (comm.rank,\
                     datetime.datetime.now().strftime("hr%H_min%M"))) 
    else:
        fns= np.array_split(image_list, 1)[0]
    for fn in fns:
        imgfn= os.path.join(rootdir,fn)
        wtfn= imgfn.replace('_ooi_','_oow_')
        dqfn= imgfn.replace('_ooi_','_ood_')
        if args.nproc > 1:
            # Log to unique file
            with stdouterr_redirected(to=outfn, comm=None):  
                t0=ptime('before: %s' % os.path.basename(imgfn).replace('.fits.fz',''), t0)
                _= newWeightMap(wtfn=wtfn,imgfn=imgfn,dqfn=dqfn) 
                t0=ptime('after',t0)
        else:
            _= newWeightMap(wtfn=wtfn,imgfn=imgfn,dqfn=dqfn) 
             
    if args.nproc > 1:
        # Wait for all mpi tasks to finish 
        confirm_files = comm.gather( fns[comm.rank], root=0 )
        if comm.rank == 0:
            tnow= Time()
            print("Done, total time = %s" % (tnow-tbegin,))
    else:
        tnow= Time()
        print("Done, total time = %s" % (tnow-tbegin,))



