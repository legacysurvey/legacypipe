from __future__ import print_function
from astrometry.util.fits import *
import numpy as np
import pylab as plt

for title,tag in [
        ('Cori KNL, 1 thread', 'knl-1'),
        ('Cori KNL, 2 threads', 'knl-2'),
        ('Cori KNL, 4 threads', 'knl-4'),
        ('Cori KNL, 8 threads', 'knl-8'),
        ('Cori KNL, 16 threads', 'knl-16'),
        ('Cori KNL, 32 threads', 'knl-32'),
        ('Cori KNL, 68 threads', 'knl-68'),
        ('Cori KNL, 136 threads', 'knl-136'),
        ('Cori KNL, 272 threads', 'knl-272'),
        ('Cori Haswell, 1 thread', 'has-1'),
        ('Cori Haswell, 2 threads', 'has-2'),
        ('Cori Haswell, 4 threads', 'has-4'),
        ('Cori Haswell, 8 threads', 'has-8'),
        ('Cori Haswell, 16 threads', 'has-16'),
        ('Cori Haswell, 32 threads', 'has-32'),
        ('Cori Haswell, 64 threads', 'has-64'),
        ]:
    #fn = 'timing/ps-knl-1.fits'
    fn = 'timing/ps-%s.fits' % tag
    print()
    print('Reading', fn)
    print()
    
    T = fits_table(fn)
    #T.about()
    events = fits_table(fn, ext=2)

    steps,Istep = np.unique(T.step, return_index=True)
    stepmap = dict([(s,i) for i,s in enumerate(steps)])
    print('Steps:', len(steps))
    unixtimes = T.unixtime[Istep]
    t0 = min(unixtimes)
    xaxis = unixtimes - t0
    
    plt.clf()
    
    parent_pid = T._header['PPID']
    T.mine = np.logical_or(T.pid == parent_pid, T.ppid == parent_pid)
    T.main = (T.pid == parent_pid)
    
    I = np.flatnonzero(T.mine)
    pids = np.unique(T.pid[I])
    print(len(I), 'measurements of my PIDs')
    print(len(pids), 'PIDs')
    ps_icpu = np.zeros(len(steps), np.float32)
    plotted_worker = False
    for pid in pids:
        II = I[T.pid[I] == pid]
        # Fill in the non-empty steps
        J = np.array([stepmap[s] for s in T.step[II]])
    
        cmds = np.unique(T.command[II])
        #print('PID', pid, ':', cmds)
        if len(cmds) == 1 and cmds[0].startswith('ps ax'):
            ps_icpu[J] += T.proc_icpu[II]
        else:
            icpu = np.zeros(len(steps), np.float32)
            icpu[J] = T.proc_icpu[II]
    
            print('PID', pid, ', parent?', (pid==parent_pid))
            procs = np.zeros(1+max(T.processor[II]), np.float32)
            np.add.at(procs, T.processor[II], T.proc_stime[II]+T.proc_utime[II])
            J = np.flatnonzero(procs)
            print('Used CPU #,time:', zip(J, procs[J]))
    
            if pid == parent_pid:
                plt.plot(xaxis, icpu, 'k-', alpha=0.5, label='My main')
            else:
                kwa = {}
                if not plotted_worker:
                    kwa.update(label='My workers')
                    plotted_worker = True
                plt.plot(xaxis, icpu, 'b-', alpha=0.25, **kwa)
                #for ii in II:
                #    plt.text(T.unixtime[ii]-t0, T.proc_icpu[ii], '%i' % T.processor[ii])
                
    if np.any(ps_icpu > 0):
        plt.plot(xaxis, ps_icpu, 'g-', alpha=0.5, label='ps')
            
    I = np.flatnonzero(np.logical_not(T.mine))
    pids = np.unique(T.pid[I])
    icpu = np.zeros(len(steps), np.float32)
    for pid in pids:
        II = I[T.pid[I] == pid]
        # Fill in the non-empty steps
        J = np.array([stepmap[s] for s in T.step[II]])
        icpu[J] = T.proc_icpu[II]
    plt.plot(xaxis, icpu, 'r-', label='Other PIDs')
    
    for e,t in zip(events.event, events.unixtime - t0):
        plt.axvline(t, color='k', alpha=0.1)
        plt.text(t, 100., e, rotation='vertical')
    
    plt.xlabel('Wall time (s)')
    plt.ylabel('CPU %')
    plt.legend()
    plt.title(title)
    plt.savefig('ps-%s.png' % tag)
    
