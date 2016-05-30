from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import qdo
import sys
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--wrap', action='store_true',
                    help='Wrap RA at 180 degrees?')
parser.add_argument('args',nargs=argparse.REMAINDER)
opt = parser.parse_args()
args = opt.args
if len(args) != 1:
    print('Need one arg: qdo queue name')
    sys.exit(-1)

qname = args[0]

q = qdo.connect(qname, create_ok=False)
print('Connected to QDO queue', qname, q)

plt.clf()

cmap = { qdo.Task.WAITING: 'k',
         qdo.Task.PENDING: '0.5',
         qdo.Task.RUNNING: 'b',
         qdo.Task.SUCCEEDED: 'g',
         qdo.Task.FAILED: 'r',
}

lp,lt = [],[]

for state in qdo.Task.VALID_STATES:
    print('State', state)

    ra,dec = [],[]
    tasks = q.tasks(state=state)
    print(len(tasks), 'tasks with state', state)
    
    for task in tasks:
        brick = task.task
        rastr = brick[:4]
        r = int(rastr, 10) / 10.
        decstr = brick[5:]
        d = int(decstr, 10) / 10.
        d *= (-1 if brick[4] == 'm' else 1)
        #print('Brick', brick, '->', r, d)
        if opt.wrap:
            if r > 180:
                r -= 360.
        ra.append(r)
        dec.append(d)

        
    p = plt.plot(ra, dec, '.', color=cmap.get(state, 'y'))
    lp.append(p[0])
    lt.append(state)
plt.legend(lp, lt)
plt.savefig('status.png')
