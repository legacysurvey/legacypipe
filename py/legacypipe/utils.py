from __future__ import print_function
import os
import numpy as np

from tractor.ellipses import EllipseESoft
from tractor.utils import _GaussianPriors

def log_info(logger, args):
    msg = ' '.join(map(str, args))
    logger.info(msg)

def log_debug(logger, args):
    msg = ' '.join(map(str, args))
    logger.debug(msg)

class EllipseWithPriors(EllipseESoft):
    '''An ellipse (used to represent galaxy shapes) with Gaussian priors
    over softened ellipticity parameters.  This class is used during
    fitting.

    We ALSO place a prior on log-radius, forcing it to be < +5 (in
    log-arcsec); though this gets dynamically adjusted in the oneblob.py code.

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
        # MAGIC -- 30" default max r_e!
        # SEE ALSO survey.py : class(LogRadius)!
        self.uppers[0] = np.log(30.)

    def setMaxLogRadius(self, rmax):
        self.uppers[0] = rmax

    def getMaxLogRadius(self):
        return self.uppers[0]

    @classmethod
    def fromRAbPhi(cls, r, ba, phi):
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(r, ba, phi)
        return cls(logr, ee1, ee2)

    def isLegal(self):
        return self.logre <= self.uppers[0]

    @classmethod
    def getName(cls):
        return "EllipseWithPriors(%g)" % cls.ellipticityStd

class RunbrickError(RuntimeError):
    pass

class NothingToDoError(RunbrickError):
    pass

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
    # py3
    def __next__(self):
        try:
            return self.y.__next__()
        except StopIteration:
            raise
        except:
            import traceback
            print(str(self), '__next__()')
            traceback.print_exc()
            raise
    def __len__(self):
        return self.n

def _ring_unique(wcs, W, H, i, unique, ra1,ra2,dec1,dec2):
    lo, hix, hiy = i, W-i-1, H-i-1
    # one slice per side; we double-count the last pix of each side.
    sidex = slice(lo,hix+1)
    sidey = slice(lo,hiy+1)
    top = (lo, sidex)
    bot = (hiy, sidex)
    left  = (sidey, lo)
    right = (sidey, hix)
    xx = np.arange(W)
    yy = np.arange(H)
    nu,ntot = 0,0
    for slc in [top, bot, left, right]:
        #print('xx,yy', xx[slc], yy[slc])
        (yslc,xslc) = slc
        rr,dd = wcs.pixelxy2radec(xx[xslc]+1, yy[yslc]+1)
        U = (rr >= ra1 ) * (rr < ra2 ) * (dd >= dec1) * (dd < dec2)
        #print('Pixel', i, ':', np.sum(U), 'of', len(U), 'pixels are unique')
        unique[slc] = U
        nu += np.sum(U)
        ntot += len(U)
    #if allin:
    #    print('Scanned to pixel', i)
    #    break
    return nu,ntot

def find_unique_pixels(wcs, W, H, unique, ra1,ra2,dec1,dec2):
    if unique is None:
        unique = np.ones((H,W), bool)
    # scan the outer annulus of pixels, and shrink in until all pixels
    # are unique.
    step = 10
    for i in range(0, W//2, step):
        nu,ntot = _ring_unique(wcs, W, H, i, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', i, ': nu/ntot', nu, ntot)
        if nu > 0:
            i -= step
            break
        unique[:i,:] = False
        unique[H-1-i:,:] = False
        unique[:,:i] = False
        unique[:,W-1-i:] = False

    for j in range(max(i+1, 0), W//2):
        nu,ntot = _ring_unique(wcs, W, H, j, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', j, ': nu/ntot', nu, ntot)
        if nu == ntot:
            break
    return unique

def read_primary_header(fn):
    '''
    Reads the FITS primary header (HDU 0) from the given filename.
    This is just a faster version of fitsio.read_header(fn).
    '''
    import fitsio
    if fn.endswith('.gz'):
        return fitsio.read_header(fn)

    # Weirdly, this can be MUCH faster than letting fitsio do it...
    hdr = fitsio.FITSHDR()
    foundEnd = False
    ff = open(fn, 'rb')
    h = b''
    while True:
        hnew = ff.read(32768)
        if len(hnew) == 0:
            # EOF
            ff.close()
            raise RuntimeError('Reached end-of-file in "%s" before finding end of FITS header.' % fn)
        h = h + hnew
        while True:
            line = h[:80]
            h = h[80:]
            #print('Header line "%s"' % line)
            # HACK -- fitsio apparently can't handle CONTINUE.
            # It also has issues with slightly malformed cards, like
            # KEYWORD  =      / no value
            if line[:8] != b'CONTINUE':
                try:
                    hdr.add_record(line.decode())
                except OSError as err:
                    print('Warning: failed to parse FITS header line: ' +
                          ('"%s"; error "%s"; skipped' % (line.strip(), str(err))))

            if line == (b'END' + b' '*77):
                foundEnd = True
                break
            if len(h) < 80:
                break
        if foundEnd:
            break
    ff.close()
    return hdr

def run_ps_thread(parent_pid, parent_ppid, fn, shutdown, event_queue):
    from astrometry.util.run_command import run_command
    from astrometry.util.fits import fits_table, merge_tables
    import time
    import re
    import fitsio
    from functools import reduce

    # my pid = parent pid -- this is a thread.
    print('run_ps_thread starting: parent PID', parent_pid, ', my PID', os.getpid(), fn)
    TT = []
    step = 0

    events = []

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
            #print('Elapsed time', s, 'parsed to', days,hours,mins,secs)
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

    def write_results(fn, T, events, hdr):
        T.mine = np.logical_or(T.pid == parent_pid, T.ppid == parent_pid)
        T.main = (T.pid == parent_pid)
        tmpfn = os.path.join(os.path.dirname(fn), 'tmp-' + os.path.basename(fn))
        T.writeto(tmpfn, header=hdr)
        if len(events):
            E = fits_table()
            E.unixtime = np.array([e[0] for e in events])
            E.event = np.array([e[1] for e in events])
            E.step = np.array([e[2] for e in events])
            E.writeto(tmpfn, append=True)
        os.rename(tmpfn, fn)
        print('Wrote', fn)

    fitshdr = fitsio.FITSHDR()
    fitshdr['PPID'] = parent_pid

    last_time = {}
    last_proc_time = {}

    clock_ticks = os.sysconf('SC_CLK_TCK')
    #print('Clock times:', clock_ticks)
    if clock_ticks == -1:
        #print('Failed to get clock times per second; assuming 100')
        clock_ticks = 100

    while True:
        shutdown.wait(5.0)
        if shutdown.is_set():
            print('ps shutdown flag set.  Quitting.')
            break

        if event_queue is not None:
            while True:
                try:
                    (t,msg) = event_queue.popleft()
                    events.append((t,msg,step))
                    #print('Popped event', t,msg)
                except IndexError:
                    # no events
                    break

        step += 1
        #cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
        #       'psr rss session vsize args"')
        # OSX-compatible
        cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
               'rss vsize wchan command"')
        #print('Command:', cmd)
        rtn,out,err = run_command(cmd)
        if rtn:
            print('FAILED to run ps:', rtn, out, err)
            time.sleep(1)
            break
        # print('Got PS output')
        # print(out)
        # print('Err')
        # print(err)
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

        # Compute 'instantaneous' (5-sec averaged) %cpu
        # BUT this only counts whole seconds in the 'ps' output.
        T.icpu = np.zeros(len(T), np.float32)
        icpu = T.icpu
        for i,(p,etime,ctime) in enumerate(zip(T.pid, T.elapsed, T.cputime)):
            try:
                elast,clast = last_time[p]
                # new process with an existing PID?
                if etime > elast:
                    icpu[i] = 100. * (ctime - clast) / (etime - elast)
            except:
                pass
            last_time[p] = (etime, ctime)

        # print('Processes:')
        # J = np.argsort(-T.icpu)
        # for j in J:
        #     p = T.pid[j]
        #     pp = T.ppid[j]
        #     print('  PID', p, '(main)' if p == parent_pid else '',
        #           '(worker)' if pp == parent_pid else '',
        #           'pcpu', T.pcpu[j], 'pmem', T.pmem[j], 'icpu', T.icpu[j],
        #           T.command[j][:20])

        # Apply cuts!
        T.cut(reduce(np.logical_or, [
            T.pcpu > 5,
            T.pmem > 5,
            T.icpu > 5,
            T.pid == parent_pid,
            (T.ppid == parent_pid) * np.array([not c.startswith('ps ax') for c in T.command])]))
        #print('Cut to', len(T), 'with significant CPU/MEM use or my PPID')

        # print('Kept:')
        # J = np.argsort(-T.icpu)
        # for j in J:
        #     p = T.pid[j]
        #     pp = T.ppid[j]
        #     print('  PID', p, '(main)' if p == parent_pid else '',
        #           '(worker)' if pp == parent_pid else '',
        #           'pcpu', T.pcpu[j], 'pmem', T.pmem[j], 'icpu', T.icpu[j],
        #           T.command[j][:20])


        if len(T) == 0:
            continue

        timenow = time.time()
        T.unixtime = np.zeros(len(T), np.float64) + timenow
        T.step = np.zeros(len(T), np.int16) + step

        if os.path.exists('/proc'):
            # Try to grab higher-precision CPU timing info from /proc/PID/stat
            T.proc_utime = np.zeros(len(T), np.float32)
            T.proc_stime = np.zeros(len(T), np.float32)
            T.processor  = np.zeros(len(T), np.int16)
            T.proc_icpu  = np.zeros(len(T), np.float32)
            for i,p in enumerate(T.pid):
                try:
                    # See:
                    # http://man7.org/linux/man-pages/man5/proc.5.html
                    procfn = '/proc/%i/stat' % p
                    txt = open(procfn).read()
                    #print('Read', procfn, ':', txt)
                    words = txt.split()
                    utime = int(words[13]) / float(clock_ticks)
                    stime = int(words[14]) / float(clock_ticks)
                    proc  = int(words[38])
                    #print('utime', utime, 'stime', stime, 'processor', proc)
                    ctime = utime + stime
                    try:
                        tlast,clast = last_proc_time[p]
                        #print('pid', p, 'Tnow,Cnow', timenow, ctime, 'Tlast,Clast', tlast,clast)
                        if ctime >= clast:
                            T.proc_icpu[i] = 100. * (ctime - clast) / float(timenow - tlast)
                    except:
                        pass
                    last_proc_time[p] = (timenow, ctime)

                    T.proc_utime[i] = utime
                    T.proc_stime[i] = stime
                    T.processor [i] = proc
                except:
                    pass

        TT.append(T)

        #print('ps -- step', step)
        if (step % 12 == 0) and len(TT) > 0:
            # Write out results every ~ minute.
            print('ps -- writing', fn)
            T = merge_tables(TT, columns='fillzero')
            write_results(fn, T, events, fitshdr)
            TT = [T]
    # Just before returning, write out results.
    if len(TT) > 0:
        print('ps -- writing', fn)
        T = merge_tables(TT, columns='fillzero')
        write_results(fn, T, events, fitshdr)

# Memory Limits
def get_ulimit():
    import resource
    for name, desc in [
        ('RLIMIT_AS', 'VMEM'),
        ('RLIMIT_CORE', 'core file size'),
        ('RLIMIT_CPU',  'CPU time'),
        ('RLIMIT_FSIZE', 'file size'),
        ('RLIMIT_DATA', 'heap size'),
        ('RLIMIT_STACK', 'stack size'),
        ('RLIMIT_RSS', 'resident set size'),
        ('RLIMIT_NPROC', 'number of processes'),
        ('RLIMIT_NOFILE', 'number of open files'),
        ('RLIMIT_MEMLOCK', 'lockable memory address'),
        ]:
        limit_num = getattr(resource, name)
        soft, hard = resource.getrlimit(limit_num)
        print('Maximum %-25s (%-15s) : %20s %20s' % (desc, name, soft, hard))
