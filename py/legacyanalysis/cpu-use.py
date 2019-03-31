from __future__ import print_function
from astrometry.util.fits import *
import numpy as np
import pylab as plt

def plot_cpu_usage(fn, title, cpufn, memfn, total=False):
    T = fits_table(fn)
    #T.about()
    events = fits_table(fn, ext=2)

    steps,Istep = np.unique(T.step, return_index=True)
    stepmap = dict([(s,i) for i,s in enumerate(steps)])
    print('Steps:', len(steps))
    unixtimes = T.unixtime[Istep]
    t0 = min(unixtimes)
    xaxis = unixtimes - t0

    parent_pid = T._header['PPID']

    #children_pids = set(T.pid[T.ppid == parent_pid])
    #todo = list(children_pids)
    #family_pids = children_pids.union(set([parent_pid]))

    mypgid = set(T.pgid[T.pid == parent_pid])
    print('My PGID:', mypgid)
    assert(len(mypgid) == 1)
    mypgid = mypgid.pop()

    # todo = [parent_pid]
    # family_pids = set()
    # while len(todo):
    #     pid = todo.pop()
    #     print('Checking PID', pid)
    #     kids = set(T.pid[T.ppid == pid])
    #     print('Found', len(kids), 'child PIDs')
    #     for k in kids:
    #         if (not k in family_pids) and (not k in todo):
    #             todo.append(k)
    #     family_pids.add(pid)
    #     family_pids.update(kids)
    
    #T.mine = np.logical_or(T.pid == parent_pid, T.ppid == parent_pid)
    #T.mine = np.array([p in family_pids for p in T.pid])

    T.mine = (T.pgid == mypgid)

    T.main = (T.pid == parent_pid)

    I = np.flatnonzero(T.mine)
    mypids = np.unique(T.pid[I])
    print(len(I), 'measurements of my PIDs')
    print(len(mypids), 'PIDs')
    
    plt.clf()

    total_cpu = np.zeros(len(steps), np.float32)
    ps_icpu = np.zeros(len(steps), np.float32)
    plotted_worker = False
    for pid in mypids:
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
    
            # print('PID', pid, ', parent?', (pid==parent_pid))
            # procs = np.zeros(1+max(T.processor[II]), np.float32)
            # np.add.at(procs, T.processor[II], T.proc_stime[II]+T.proc_utime[II])
            # J = np.flatnonzero(procs)
            # print('Used CPU #,time:', list(zip(J, procs[J])))

            total_cpu += icpu

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

    if total:
        plt.plot(xaxis, total_cpu, 'g-', label='My Total')
        
    # Add up all other PIDs
    I = np.flatnonzero(np.logical_not(T.mine))
    pids = np.unique(T.pid[I])
    icpu = np.zeros(len(steps), np.float32)
    for pid in pids:
        II = I[T.pid[I] == pid]
        # Fill in the non-empty steps
        J = np.array([stepmap[s] for s in T.step[II]])
        icpu[J] += T.proc_icpu[II]
    plt.plot(xaxis, icpu, 'r-', label='Other PIDs')
    
    for e,t in zip(events.event, events.unixtime - t0):
        plt.axvline(t, color='k', alpha=0.1)
        plt.text(t, 100., e, rotation='vertical')
    
    plt.xlabel('Wall time (s)')
    plt.ylabel('CPU %')
    plt.legend()
    plt.title(title)
    plt.savefig(cpufn)
    
    plt.clf()
    plotted_worker = False
    I = np.flatnonzero(T.mine)
    for pid in mypids:
        II = I[T.pid[I] == pid]
        # Fill in the non-empty steps
        J = np.array([stepmap[s] for s in T.step[II]])
    
        cmds = np.unique(T.command[II])
        #print('PID', pid, ':', cmds)
        if len(cmds) == 1 and cmds[0].startswith('ps ax'):
            continue
        pmem = np.zeros(len(steps), np.float32)
        pmem[J] = T.vsz[II]
        #pmem[J] = T.pmem[II]

        if pid == parent_pid:
            plt.plot(xaxis, pmem, 'k-', alpha=0.5, label='My main')
        else:
            kwa = {}
            if not plotted_worker:
                kwa.update(label='My workers')
                plotted_worker = True
            plt.plot(xaxis, pmem, 'b-', alpha=0.25, **kwa)

    I = np.flatnonzero(np.logical_not(T.mine))
    pids = np.unique(T.pid[I])
    pmem[:] = 0
    for pid in pids:
        II = I[T.pid[I] == pid]
        # Fill in the non-empty steps
        J = np.array([stepmap[s] for s in T.step[II]])
        #pmem[J] += T.pmem[II]
        pmem[J] += T.vsz[II]
    plt.plot(xaxis, pmem, 'r-', label='Other PIDs')

    for e,t in zip(events.event, events.unixtime - t0):
        plt.axvline(t, color='k', alpha=0.1)
        plt.text(t, 100., e, rotation='vertical')

    plt.xlabel('Wall time (s)')
    #plt.ylabel('Memory %')
    plt.ylabel('VSS')
    plt.legend()
    plt.title(title)
    plt.savefig(memfn)

def oldtimey():
    for title,tag in [
            #('Cori Haswell, 64', 'has-64-0001p000'),
            #('Cori Haswell, shared(16)', 'has-16-0001p000'),
            # ('Cori Haswell, 8, BB', 'bb-has-8-2420p070'),
            # ('Cori Haswell, 8 threads', 'has-8-0001p000'),
            # ('Cori KNL, 1 thread', 'knl-1'),
            # ('Cori KNL, 2 threads', 'knl-2'),
            # ('Cori KNL, 4 threads', 'knl-4'),
            # ('Cori KNL, 8 threads', 'knl-8'),
            # ('Cori KNL, 16 threads', 'knl-16'),
            # ('Cori KNL, 32 threads', 'knl-32'),
            # ('Cori KNL, 68 threads', 'knl-68'),
            # ('Cori KNL, 136 threads', 'knl-136'),
            # ('Cori KNL, 272 threads', 'knl-272'),
            # ('Cori Haswell, 1 thread', 'has-1'),
            # ('Cori Haswell, 2 threads', 'has-2'),
            # ('Cori Haswell, 4 threads', 'has-4'),
            # ('Cori Haswell, 8 threads', 'has-8'),
            # ('Cori Haswell, 16 threads', 'has-16'), # ('Cori Haswell, 32 threads', 'has-32'),
            # ('Cori Haswell, 64 threads', 'has-64'),
            # ('KNL vanilla 16', '7a'),
            # ('KNL vanilla 68', '7b'),
            # ('KNL fast 16', '7c'),
            # ('KNL fast 68', '7d'),
            # ('KNL fast2 16', '7e'),
            # ('KNL fast2 68', '7f'),
            # ('KNL fast2 128', '7g'),
            # ('KNL fast2 256', '7h'),
            ('KNL vanilla 16', '7m'),
            ('KNL vanilla 68', '7n'),
            ('KNL fast 16', '7o'),
            ('KNL fast 68', '7p'),
            ('KNL fast 128', '7q'),
            ('KNL fast 256', '7r'),
            ]:
        #fn = 'timing/ps-knl-1.fits'
        #fn = 'timing/ps-%s.fits' % tag
        #fn = 'ps-%s.fits' % tag
        fn = '/global/cscratch1/sd/dstn/out-%s/ps.fits' % tag
        print()
        print('Reading', fn)
        print()
    
        plot_cpu_usage(fn, title, 'ps-%s.png' % tag, 'ps-mem-%s.png' % tag)
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='', help='Plot title')
    parser.add_argument('--total', default=False, action='store_true',
                        help='Plot total?')
    parser.add_argument('ps-file')
    parser.add_argument('cpu-plot')
    parser.add_argument('mem-plot')

    args = parser.parse_args()

    plot_cpu_usage(getattr(args, 'ps-file'), args.title, getattr(args, 'cpu-plot'),
                   getattr(args, 'mem-plot'), total=args.total)

if __name__ == '__main__':
    main()
    
