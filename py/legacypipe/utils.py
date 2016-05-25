from __future__ import print_function
import os
import numpy as np

from tractor.ellipses import EllipseESoft
from tractor.utils import _GaussianPriors

from astrometry.util.timingpool import TimingPoolTimestamp
from astrometry.util.multiproc import multiproc
from astrometry.util.ttime import Time, CpuMeas
from astrometry.util.fits import fits_table, merge_tables

class EllipseWithPriors(EllipseESoft):
    '''
    An ellipse (used to represent galaxy shapes) with Gaussian priors
    over softened ellipticity parameters.  This class is used during
    fitting.
    
    We ALSO place a prior on log-radius, forcing it to be < +5 (r_e =
    148").

    To use this class, subclass it and set the 'ellipticityStd' class
    member.

    '''
    ellipticityStd = 0.
    ellipsePriors = None

    # EllipseESoft extends EllipseE extends ParamList, has
    # GaussianPriorsMixin.  GaussianPriorsMixin sets a "gpriors"
    # member variable to a _GaussianPriors
    def __init__(self, *args, **kwargs):
        super(EllipseWithPriors, self).__init__(*args, **kwargs)
        if self.ellipsePriors is None:
            ellipsePriors = _GaussianPriors(None)
            ellipsePriors.add('ee1', 0., self.ellipticityStd,
                              param=EllipseESoft(1.,0.,0.))
            ellipsePriors.add('ee2', 0., self.ellipticityStd,
                              param=EllipseESoft(1.,0.,0.))
            self.__class__.ellipsePriors = ellipsePriors
        self.gpriors = self.ellipsePriors
        self.uppers[0] = 5.

    @classmethod
    def fromRAbPhi(cls, r, ba, phi):
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(r, ba, phi)
        return cls(logr, ee1, ee2)

    def isLegal(self):
        return self.logre < +5.

    @classmethod
    def getName(cls):
        return "EllipseWithPriors(%g)" % cls.ellipticityStd

class RunbrickError(RuntimeError):
    pass

class NothingToDoError(RunbrickError):
    pass

class ImapTracker(object):
    def __init__(self, real, mymp, tstart):
        self.real = real
        self.mymp = mymp
        self.tstart = tstart

    def next(self, *args, **kwargs):
        try:
            return self.real.next(*args, **kwargs)
        except StopIteration:
            self.mymp._imap_finished(self.tstart)
            raise
        except:
            import traceback
            print('ImapTracker:')
            traceback.print_exc()
            raise

class MyMultiproc(multiproc):
    def __init__(self, *args, **kwargs):
        super(MyMultiproc, self).__init__(*args, **kwargs)
        self.t0 = Time()
        self.serial = []
        self.parallel = []
        self.phases = []

    def start_subphase(self, name):
        # push current state to stack
        tstart = Time()
        self.serial.append((self.t0, tstart))
        self.t0 = tstart
        self.phases.append((name, self.serial, self.parallel, self.t0))

    def finish_subphase(self):
        # pop
        (name, serial, parallel, t0) = self.phases.pop()
        print('Popping subphase', name)
        serial.extend(self.serial)
        self.serial = serial
        parallel.extend(self.parallel)
        self.parallel = parallel
        #self.t0 = t0
        
    def is_multiproc(self):
        return self.pool is not None
        
    def map(self, *args, **kwargs):
        tstart = Time()
        res = super(MyMultiproc, self).map(*args, **kwargs)
        tend = Time()
        self.serial.append((self.t0, tstart))
        self.parallel.append((tstart, tend))
        self.t0 = tend
        return res

    def _imap_finished(self, tstart):
        tend = Time()
        self.parallel.append((tstart, tend))
        self.t0 = tend

    def imap_unordered(self, func, iterable, chunksize=None, wrap=False):
        # So, this is a bit strange, tracking parallel vs serial time
        # for an async object, via the ImapTracker & callback to
        # _imap_finished.
        tstart = Time()
        self.serial.append((self.t0, tstart))

        #res = super(MyMultiproc, self).imap_unordered(*args, **kwargs)
        cs = chunksize
        if cs is None:
            cs = self.map_chunksize
        if self.pool is None:
            import itertools
            return itertools.imap(func, iterable)
        if wrap or self.wrap_all:
            func = funcwrapper(func)
        res = self.pool.imap_unordered(func, iterable, chunksize=cs)

        return ImapTracker(res, self, tstart)

    def report(self, nthreads):
        # Tally the serial time up to now
        tend = Time()
        self.serial.append((self.t0, tend))
        self.t0 = tend

        # Nasty... peek into Time members
        scpu = 0.
        swall = 0.
        print('Serial:')
        for t0,t1 in self.serial:
            print(t1-t0)
            for m0,m1 in zip(t0.meas, t1.meas):
                if isinstance(m0, CpuMeas):
                    scpu  += m1.cpu_seconds_since(m0)
                    swall += m1.wall_seconds_since(m0)
                    #print '  total cpu', scpu, 'wall', swall
        pworkercpu = 0.
        pworkerwall = 0.
        pwall = 0.
        pcpu = 0.
        print('Parallel:')
        for t0,t1 in self.parallel:
            print(t1-t0)
            for m0,m1 in zip(t0.meas, t1.meas):
                if isinstance(m0, TimingPoolTimestamp):
                    mt0 = m0.t0
                    mt1 = m1.t0
                    pworkercpu  += mt1['worker_cpu' ] - mt0['worker_cpu' ]
                    pworkerwall += mt1['worker_wall'] - mt0['worker_wall']
                elif isinstance(m0, CpuMeas):
                    pwall += m1.wall_seconds_since(m0)
                    pcpu  += m1.cpu_seconds_since(m0)
        print()
        print('Total serial CPU   ', scpu)
        print('Total serial Wall  ', swall)
        print('Total worker CPU   ', pworkercpu)
        print('Total worker Wall  ', pworkerwall)
        print('Total parallel Wall', pwall)
        print('Total parallel CPU ', pcpu)
        print()
        tcpu = scpu + pworkercpu + pcpu
        twall = swall + pwall
        if nthreads is None:
            nthreads = 1
        print('Grand total CPU:              %.1f sec' % tcpu)
        print('Grand total Wall:             %.1f sec' % twall)
        print('Grand total CPU utilization:  %.2f cores' % (tcpu / twall))
        print('Grand total efficiency:       %.1f %%' % (100. * tcpu / (twall * nthreads)))
        print()

class iterwrapper(object):
    def __init__(self, y, n):
        self.n = n
        self.y = y
    def __str__(self):
        return 'iterwrapper: n=%i; ' % self.n + str(self.y)
    def __iter__(self):
        return self
    def next(self):
        try:
            return self.y.next()
        except StopIteration:
            raise
        except:
            import traceback
            print(str(self), 'next()')
            traceback.print_exc()
            raise

    def __len__(self):
        return self.n


def run_ps_thread(pid, ppid, fn):
    from astrometry.util.run_command import run_command
    import time
    import re
    import fitsio
    
    #print('run_ps_thread starting:', pid, ppid, fn)
    #print('My pid:', os.getpid())
    TT = []
    step = 0

    trex = re.compile('(((?P<days>\d*)-)?(?P<hours>\d*):)?(?P<minutes>\d*):(?P<seconds>[\d\.]*)')
    def parse_time_strings(ss):
        etime = []
        any_failed = None
        for s in ss:
            m = trex.match(s)
            if m is None:
                any_failed = s
                break
            days,hours,mins,secs = m.group('days', 'hours', 'minutes',
                                           'seconds')
            # print('Elapsed time', s, 'parsed to', days,hours,mins,secs)
            days = int(days, 10) if days is not None else 0
            hours = int(hours, 10) if hours is not None else 0
            mins = int(mins, 10)
            if secs.startswith('0'):
                secs = secs[1:]
            secs = float(secs)
            tt = days * 24 * 3600 + hours * 3600 + mins * 60 + secs
            #print('->', tt, 'seconds')
            etime.append(tt)
        return any_failed, etime

    fitshdr = fitsio.FITSHDR()
    fitshdr['PPID'] = pid
    
    while True:
        time.sleep(5)
        step += 1
        #cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
        #       'psr rss session vsize args"')
        # OSX-compatible
        cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
               'rss vsize command"')
        #print('Command:', cmd)
        rtn,out,err = run_command(cmd)
        if rtn:
            print('FAILED to run ps:', rtn, out, err)
            time.sleep(1)
            break
        #print('Got PS output')
        #print(out)
        #print('Err')
        #print(err)
        if len(err):
            print('Error string from ps:', err)
        lines = out.split('\n')
        hdr = lines.pop(0)
        cols = hdr.split()
        cols = [c.replace('%','P') for c in cols]
        cols = [c.lower() for c in cols]
        #print('Columns:', cols)
        vals = [[] for c in cols]

        # maximum length for 'command', command-line args field
        maxlen = 128
        for line in lines:
            words = line.split()
            # "command" column can contain spaces; it is last
            if len(words) == 0:
                continue
            words = (words[:len(cols)-1] +
                     [' '.join(words[len(cols)-1:])[:maxlen]])
            assert(len(words) == len(cols))

            for v,w in zip(vals, words):
                v.append(w)

        parsetypes = dict(pcpu = np.float32,
                          pmem = np.float32,
                          pgid = np.int32,
                          pid = np.int32,
                          ppid = np.int32,
                          rs = np.float32,
                          vsz = np.float32,
                          )
        T = fits_table()
        for c,v in zip(cols, vals):
            # print('Col', c, 'Values:', v[:3], '...')
            v = np.array(v)
            tt = parsetypes.get(c, None)
            if tt is not None:
                v = v.astype(tt)
            T.set(c, v)

        # Apply cuts!
        T.cut(reduce(np.logical_or, [
            T.pcpu > 5, T.pmem > 5,
            (T.ppid == pid) * [not c.startswith('ps ax') for c in T.command]]))
        #print('Cut to', len(T), 'with significant CPU/MEM use or my PPID')
        if len(T) == 0:
            continue
        
        T.unixtime = np.zeros(len(T), np.float64) + time.time()
        T.step = np.zeros(len(T), np.int16) + step
        
        any_failed,etime = parse_time_strings(T.elapsed)
        if any_failed is not None:
            print('Failed to parse elapsed time string:', any_failed)
        else:
            T.elapsed = np.array(etime)

        any_failed,ctime = parse_time_strings(T.time)
        if any_failed is not None:
            print('Failed to parse elapsed time string:', any_failed)
        else:
            T.time = np.array(ctime)
        T.rename('time', 'cputime')
        
        TT.append(T)

        if step % 12 == 0:
            # Write out results every ~ minute.
            T = merge_tables(TT, columns='fillzero')
            tmpfn = 'tmp-' + fn
            T.writeto(tmpfn, header=fitshdr)
            os.rename(tmpfn, fn)
            print('Wrote', fn)
            TT = [T]

if __name__ == '__main__':
    ep1 = ellipse_with_priors_factory(0.25)
    ep2 = ellipse_with_priors_factory(0.1)
    print(ep1)
    print(ep2)
    print(ep1.getName())
    print(ep2.getName())
    e1 = ep1.fromRAbPhi(1., 0.5, 45.)
    e2 = ep2.fromRAbPhi(1., 0.5, 45.)
    print(e1)
    print(e2)
    print(e1.getLogPrior())
    print(e2.getLogPrior())
    
    
    
    
