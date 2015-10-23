from __future__ import print_function

from tractor.ellipses import EllipseESoft
from tractor.utils import _GaussianPriors

from astrometry.util.timingpool import TimingPoolTimestamp
from astrometry.util.multiproc import multiproc
from astrometry.util.ttime import Time, CpuMeas

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

class MyMultiproc(multiproc):
    def __init__(self, *args, **kwargs):
        super(MyMultiproc, self).__init__(*args, **kwargs)
        self.t0 = Time()
        self.serial = []
        self.parallel = []
    def map(self, *args, **kwargs):
        tstart = Time()
        res = super(MyMultiproc, self).map(*args, **kwargs)
        tend = Time()
        self.serial.append((self.t0, tstart))
        self.parallel.append((tstart, tend))
        self.t0 = tend
        return res

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
    
    
    
    
