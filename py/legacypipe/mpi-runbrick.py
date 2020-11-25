import sys
from legacypipe.runbrick import (get_parser, get_runbrick_kwargs, run_brick,
                                 NothingToDoError, RunbrickError)
import numpy as np

def hello(i):
    import socket
    import os
    import time
    print('Hello', i, 'from', socket.gethostname(), 'pid', os.getpid())
    time.sleep(2)


def main(args=None):
    import os
    import datetime
    import logging
    from legacypipe.survey import get_git_version

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
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
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    from mpi4py.futures import MPIPoolExecutor
    # Wrapper
    class MyMPIPool(object):
        def __init__(self, **kwargs):
            self.real = MPIPoolExecutor(**kwargs)
        def map(self, func, args, chunksize=1):
            return list(self.real.map(func, args, chunksize=chunksize))
        def imap_unordered(self, func, args, chunksize=1):
            return self.real.map(func, args, chunksize=chunksize, unordered=True)
        def bootup(self, **kwargs):
            return self.real.bootup(**kwargs)
        def shutdown(self, **kwargs):
            return self.real.shutdown(**kwargs)

        def close(self):
            self.shutdown()
        def join(self):
            pass

        def apply_async(self, *args, **kwargs):
            raise RuntimeError('APPLY_ASNCY NOT IMPLEMENTED IN MyMPIPool')
        def get_worker_cpu(self):
            return 0.
        def get_worker_wall(self):
            return 0.
        def get_pickle_traffic(self):
            return None
        def get_pickle_traffic_string(self):
            return 'nope'

    # print('ENV:')
    # keys = list(os.environ.keys())
    # keys.sort()
    # for k in keys:
    #     print('  ',k, '=', os.environ[k])

    pool = MyMPIPool()
    #n = MPI.COMM_WORLD.size
    u = int(os.environ.get('OMPI_UNIVERSE_SIZE', '0'))
    if u == 0:
        u = int(os.environ.get('MPICH_UNIVERSE_SIZE', '0'))
    print('Booting up MPI pool with', u, 'workers...')
    pool.bootup()
    print('Booted up MPI pool.')
    pool._processes = u
    kwargs.update(pool=pool)

    pool.map(hello, np.arange(128))

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


    
