import os
import numpy as np

from tractor.ellipses import EllipseESoft
from tractor.utils import _GaussianPriors

def log_info(logger, args):
    msg = ' '.join(map(str, args))
    logger.info(msg)

def log_debug(logger, args):
    import logging
    if logger.isEnabledFor(logging.DEBUG):
        msg = ' '.join(map(str, args))
        logger.debug(msg)

# singleton
cpu_arch = None
def get_cpu_arch():
    global cpu_arch
    import os
    if cpu_arch is not None:
        return cpu_arch
    family = None
    model = None
    modelname = None
    if os.path.exists('/proc/cpuinfo'):
        for line in open('/proc/cpuinfo').readlines():
            words = [w.strip() for w in line.strip().split(':')]
            if words[0] == 'cpu family' and family is None:
                family = int(words[1])
                #print('Set CPU family', family)
            if words[0] == 'model' and model is None:
                model = int(words[1])
                #print('Set CPU model', model)
            if words[0] == 'model name' and modelname is None:
                modelname = words[1]
                #print('CPU model', modelname)
    codenames = {
        # NERSC Cori machines
        (6, 63): 'has',
        (6, 87): 'knl',
        # NERSC Perlmutter CPU partition (AMD EPYC 7763 64-Core Processor)
        # (7713 on the head nodes)
        (25, 1): 'prl',
    }
    cpu_arch = codenames.get((family, model), '')
    return cpu_arch

galaxy_min_re = 0.01

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
        self.lowers[0] = np.log(galaxy_min_re)

    def setMaxLogRadius(self, rmax):
        self.uppers[0] = rmax

    def getMaxLogRadius(self):
        return self.uppers[0]

    @classmethod
    def fromRAbPhi(cls, r, ba, phi):
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(r, ba, phi)
        return cls(logr, ee1, ee2)

    def isLegal(self):
        return ((self.logre <= self.uppers[0]) and
                (self.logre >= self.lowers[0]))

    @classmethod
    def getName(cls):
        return "EllipseWithPriors(%g)" % cls.ellipticityStd

class RunbrickError(RuntimeError):
    pass

class NothingToDoError(RunbrickError):
    pass

class ZeroWeightError(RunbrickError):
    pass

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

def copy_header_with_wcs(source_header, wcs):
    import fitsio
    hdr = fitsio.FITSHDR()
    if source_header is not None:
        for r in source_header.records():
            hdr.add_record(r)
    # Plug the WCS header cards into these images
    wcs.add_to_header(hdr)
    hdr.add_record(dict(name='EQUINOX', value=2000., comment='WCS epoch'))
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    return hdr

def add_bits(hdr, bitmap, description, desc, bitpre):
    hdr.add_record(dict(name='COMMENT', value='%s bits:' % description))
    bits = list(bitmap.values())
    bits.sort()
    revmap = dict((v,k) for k,v in bitmap.items())
    for i in range(16):
        bit = 1<<i
        if not bit in revmap:
            continue
        hdr.add_record(
            dict(name='%s_%s' % (desc, revmap[bit].upper()[:5]), value=bit,
                 comment='%s bit 2**%i' % (description, i)))
    for i in range(16):
        bit = 1<<i
        if not bit in revmap:
            continue
        hdr.add_record(
            dict(name='%sBIT_%i' % (bitpre, i), value=revmap[bit],
                 comment='%s bit 2**%i=%i meaning' % (description, i, bit)))

def run_ps_thread(parent_pid, parent_ppid, fn, shutdown, event_queue):
    from astrometry.util.fits import fits_table, merge_tables
    import time
    import fitsio
    from functools import reduce
    import re

    # my pid = parent pid -- this is a thread.
    print('run_ps_thread starting: parent PID', parent_pid, ', my PID', os.getpid(), fn)
    TT = []
    step = 0
    events = []
    T_last = None

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

        T = run_ps(parent_pid=parent_pid, last=T_last)
        if T is None:
            time.sleep(1)
            break

        # maximum length for 'command', command-line args field
        maxlen = 128
        T.command = np.array([c[:maxlen] for c in T.command])

        # print('Processes:')
        # J = np.argsort(-T.icpu)
        # for j in J:
        #     p = T.pid[j]
        #     pp = T.ppid[j]
        #     print('  PID', p, '(main)' if p == parent_pid else '',
        #           '(worker)' if pp == parent_pid else '',
        #           'pcpu', T.pcpu[j], 'pmem', T.pmem[j], 'icpu', T.icpu[j],
        #           T.command[j][:20])

        T_last = T.copy()

        # Apply cuts (OR)!
        cuts = [T.pcpu > 5,
                T.pmem > 5,]
        if 'icpu' in T.get_columns():
            cuts.append(T.icpu > 5)
        T.cut(reduce(np.logical_or, cuts))
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

        T.step = np.zeros(len(T), np.int16) + step
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

def run_ps(parent_pid=None, pid=None, last=None):
    import re
    import platform
    import time
    # could surely be replaced with subprocess...
    from astrometry.util.run_command import run_command
    from astrometry.util.fits import fits_table

    is_osx = (platform.system() == 'Darwin')

    trex = re.compile(r'(((?P<days>\d*)-)?(?P<hours>\d*):)?(?P<minutes>\d*):(?P<seconds>[\d\.]*)')
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

    #cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
    #       'psr rss session vsize args"')
    if is_osx:
        # OSX-compatible ps
        cmd = ('ps ax -o "user pcpu pmem state cputime etime pgid pid ppid ' +
               'rss vsize wchan command"')
    else:
        # Linux
        if parent_pid or pid:
            cmd = 'ps x'
            if parent_pid:
                cmd += ' --ppid %i' % parent_pid
            if pid:
                cmd += ' --pid %i' % pid
        else:
            cmd = 'ps ax'
        cmd += (' -o "user pcpu pmem state cputimes etimes pgid pid ppid ' +
                'rss vsize wchan command"')

    #print('Command:', cmd)
    timenow = time.time()
    rtn,out,err = run_command(cmd)
    if rtn:
        print('FAILED to run ps:', rtn, out, err)
        return None

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

    for line in lines:
        words = line.split()
        # "command" column can contain spaces; it is last
        if len(words) == 0:
            continue
        words = (words[:len(cols)-1] +
                 [' '.join(words[len(cols)-1:])])
        assert(len(words) == len(cols))
        for v,w in zip(vals, words):
            v.append(w)

    parsetypes = dict(pcpu = np.float32,
                      pmem = np.float32,
                      pgid = np.int64,
                      pid = np.int64,
                      ppid = np.int64,
                      rs = np.float32,
                      vsz = np.float32,
                      )
    if not is_osx:
        parsetypes.update(time = np.int64,
                          elapsed = np.int64)

    T = fits_table()
    for c,v in zip(cols, vals):
        # print('Col', c, 'Values:', v[:3], '...')
        v = np.array(v)
        tt = parsetypes.get(c, None)
        if tt is not None:
            v = v.astype(tt)
        T.set(c, v)

    # Linux has "cputimes" (in seconds) - Mac doesn't, so parse that
    if is_osx:
        # Parse time fields
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

    # Drop the "ps" command itself from the outputs!
    # (HACK - this is dropping all ps commands... we can live with this)
    T.cut(np.array([not(c.startswith('ps ')) for c in T.command]))

    if parent_pid is not None:
        # Cut to processes with the given parent PID
        T.cut(T.ppid == parent_pid)
    if pid is not None:
        # Cut to processes with the given PID
        T.cut(T.pid == pid)

    T.unixtime = np.zeros(len(T), np.float64) + timenow

    if last is not None:
        # Compute instantaneous CPU%
        # This is very rough since "elapsed" and "cputime" are only reported
        # to the nearest SECOND.
        T.icpu = np.zeros(len(T), np.float32)
        last_cpu = {}
        for p,t,cpu in zip(last.pid, last.elapsed, last.cputime):
            last_cpu[p] = (t, cpu)
        for i,(p,t,cpu) in enumerate(zip(T.pid, T.elapsed, T.cputime)):
            if p in last_cpu:
                (t_last, cpu_last) = last_cpu[p]
                if cpu >= cpu_last and t > t_last:
                    T.icpu[i] = 100. * (cpu - cpu_last) / (t - t_last)

    if is_osx:
        return T

    clock_ticks = os.sysconf('SC_CLK_TCK')
    #print('Clock times:', clock_ticks)
    if clock_ticks == -1:
        #print('Failed to get clock times per second; assuming 100')
        clock_ticks = 100

    # Try to grab higher-precision CPU timing info from /proc/PID/stat
    T.proc_utime = np.zeros(len(T), np.float32)
    T.proc_stime = np.zeros(len(T), np.float32)
    T.processor  = np.zeros(len(T), np.int16)
    T.proc_vmpeak = np.zeros(len(T), np.int64)

    if last is not None:
        last_cpu = {}
        T.proc_icpu  = np.zeros(len(T), np.float32)
        for p,t,cpu in zip(last.pid, last.unixtime, last.proc_utime + last.proc_stime):
            last_cpu[p] = (t, cpu)

    for i,p in enumerate(T.pid):
        try:
            # See:
            # http://man7.org/linux/man-pages/man5/proc.5.html
            procfn = '/proc/%i/stat' % p
            if not os.path.exists(procfn):
                continue
            txt = open(procfn).read()
            #print('Read', procfn, ':', txt)
            words = txt.split()
            utime = int(words[13]) / float(clock_ticks)
            stime = int(words[14]) / float(clock_ticks)
            proc  = int(words[38])
            #print('utime', utime, 'stime', stime, 'processor', proc)
            ctime = utime + stime
            if last is not None:
                if p in last_cpu:
                    (t_last, ctime_last) = last_cpu[p]
                    if ctime >= ctime_last:
                        T.proc_icpu[i] = 100. * (ctime - ctime_last) / (timenow - t_last)

            T.proc_utime[i] = utime
            T.proc_stime[i] = stime
            T.processor [i] = proc

            procfn = '/proc/%i/status' % p
            txt = open(procfn).read()
            lines = txt.split('\n')
            for line in lines:
                words = line.split()
                if len(words) != 3:
                    continue
                if words[0] == 'VmPeak:':
                    mem = int(words[1])
                    assert(words[2] == 'kB')
                    T.proc_vmpeak[i] = mem
        except:
            import traceback
            print('failed to read /proc:')
            traceback.print_exc()
            pass

    return T
