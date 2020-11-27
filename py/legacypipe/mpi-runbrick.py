import sys

def hello(i):
    import socket
    import os
    import time
    print('Hello', i, 'from', socket.gethostname(), 'pid', os.getpid())
    time.sleep(2)
    return i

def global_init(loglevel):
    '''
    Global initialization routine called by mpi4py when each worker is started.
    '''
    import socket
    import os
    from mpi4py import MPI
    print('MPI4PY process starting on', socket.gethostname(), 'pid', os.getpid(),
          'MPI rank', MPI.COMM_WORLD.Get_rank())

    import logging
    logging.basicConfig(level=loglevel, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(loglevel + 10)

    from astrometry.util.ttime import Time, MemMeas
    Time.add_measurement(MemMeas)

# mpi4py to multiprocssing.Pool adapters:

# This is an adapter class that provides an iterator over
# imap_unordered results with a next(timeout) function, required
# for checkpointing.
from concurrent.futures import wait as cfwait, FIRST_COMPLETED
class result_iter(object):
    def __init__(self, futures):
        self.futures = futures
    def next(self, timeout=None):
        if len(self.futures) == 0:
            raise StopIteration()
        # Have any of the elements completed?
        for f in self.futures:
            if f.done():
                self.futures.remove(f)
                return f.result()
        # Wait for the first one to complete, with timeout
        done,_ = cfwait(self.futures, timeout=timeout,
                        return_when=FIRST_COMPLETED)
        if len(done):
            f = done.pop()
            self.futures.remove(f)
            return f.result()
        raise TimeoutError()

# Wrapper over MPIPoolExecutor to make it look like a multiprocessing.Pool
# (actually, an astrometry.util.timingpool!)
class MyMPIPool(object):
    def __init__(self, **kwargs):
        from mpi4py.futures import MPIPoolExecutor
        self.real = MPIPoolExecutor(**kwargs)
    def map(self, func, args, chunksize=1):
        return list(self.real.map(func, args, chunksize=chunksize))
    def imap_unordered(self, func, args, chunksize=1):
        return result_iter([self.real.submit(func, a) for a in args])
    def bootup(self, **kwargs):
        return self.real.bootup(**kwargs)
    def shutdown(self, **kwargs):
        return self.real.shutdown(**kwargs)

    def close(self):
        self.shutdown()
    def join(self):
        pass

    def apply_async(self, *args, **kwargs):
        raise RuntimeError('APPLY_ASYNC NOT IMPLEMENTED IN MyMPIPool')
    def get_worker_cpu(self):
        return 0.
    def get_worker_wall(self):
        return 0.
    def get_pickle_traffic(self):
        return None
    def get_pickle_traffic_string(self):
        return 'nope'
    
def main(args=None):
    import os
    import datetime
    import logging
    import numpy as np
    from legacypipe.survey import get_git_version
    from legacypipe.runbrick import (get_parser, get_runbrick_kwargs, run_brick,
                                     NothingToDoError, RunbrickError)

    print()
    print('mpi-runbrick.py starting at', datetime.datetime.now().isoformat())
    print('legacypipe git version:', get_git_version())
    if args is None:
        print('Command-line args:', sys.argv)
        cmd = 'python'
        for vv in sys.argv:
            cmd += ' {}'.format(vv)
        print(cmd)
    else:
        print('Args:', args)
    print()

    parser = get_parser()
    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    verbose = optdict.pop('verbose')

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    global_init(lvl)

    # matplotlib config directory, when running in shifter...
    for tag in ['CACHE', 'CONFIG']:
        if not 'XDG_%s_HOME'%tag in os.environ:
            src = os.path.expanduser('~/.%s' % (tag.lower()))
            # Read-only?
            if os.path.isdir(src) and not(os.access(src, os.W_OK)):
                import shutil
                import tempfile
                tempdir = tempfile.mkdtemp(prefix='%s-' % tag.lower())
                os.rmdir(tempdir)
                shutil.copytree(src, tempdir, symlinks=True) #dirs_exist_ok=True)
                # astropy config file: if XDG_CONFIG_HOME is set,
                # expect to find $XDG_CONFIG_HOME/astropy, not
                # ~/.astropy.
                if tag == 'CONFIG':
                    shutil.copytree(os.path.expanduser('~/.astropy'),
                                    os.path.join(tempdir, 'astropy'),
                                    symlinks=True) #dirs_exist_ok=True)
                os.environ['XDG_%s_HOME' % tag] = tempdir

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    # The "initializer" arg is only available in mpi4py master
    pool = MyMPIPool(initializer=global_init, initargs=(lvl,))

    u = int(os.environ.get('OMPI_UNIVERSE_SIZE', '0'))
    if u == 0:
        u = int(os.environ.get('MPICH_UNIVERSE_SIZE', '0'))
    if u == 0:
        from mpi4py import MPI
        u = MPI.COMM_WORLD.Get_size()

    print('Booting up MPI pool with', u, 'workers...')
    pool.bootup()
    print('Booted up MPI pool.')
    pool._processes = u
    kwargs.update(pool=pool)

    #pool.map(hello, np.arange(128))

    rtn = -1
    try:
        run_brick(opt.brick, survey, **kwargs)
        rtn = 0
    except NothingToDoError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = 0
    except RunbrickError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = -1

    print('Shutting down MPI pool...')
    pool.shutdown()
    print('Shut down MPI pool')
        
    return rtn

if __name__ == '__main__':
    from astrometry.util.ttime import Time,MemMeas
    Time.add_measurement(MemMeas)
    sys.exit(main())


# salloc -N 2 -C haswell -q interactive -t 04:00:00 --ntasks-per-node=32 --cpus-per-task=2
# module unload cray-mpich
# module load openmpi
# mpirun -n 64 --map-by core --rank-by node python -m mpi4py.futures legacypipe/mpi-runbrick.py --no-wise-ceres --brick 0715m657 --zoom 100 300 100 300 --run south --outdir $CSCRATCH/mpi --stage wise_forced


# cray-mpich version:
# srun -n 64 --distribution cyclic:cyclic python -m mpi4py.futures legacypipe/mpi-runbrick.py --no-wise-ceres --brick 0715m657 --zoom 100 300 100 300 --run south --outdir $CSCRATCH/mpi --stage wise_forced
# 
#     
# mpi4py setup:
# cray-mpich module, PrgEnv-intel/6.0.5
# 
# mpi.cfg:
# [mpi]
# # $CRAY_MPICH2_DIR
# mpi_dir = /opt/cray/pe/mpt/7.7.10/gni/mpich-intel/16.0
# include_dirs         = %(mpi_dir)s/include
# libraries            = mpich
# library_dirs         = %(mpi_dir)s/lib
# runtime_library_dirs = %(mpi_dir)s/lib
