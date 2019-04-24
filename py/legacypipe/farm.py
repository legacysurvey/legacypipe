import sys
import os
import pickle
import time
from collections import Counter

import zmq

from legacypipe.runbrick import _blob_iter, _write_checkpoint

import logging
logger = logging.getLogger('farm')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('queue', help='QDO queue name to get brick names from')
    parser.add_argument('--pickle', default='pickles/runbrick-%(brick)s-srcs.pickle',
                        help='Pickle pattern for "srcs" (source detection) stage pickles, default %(default)s')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint-%(brick)s.pickle',
                        help='Checkpoint filename pattern')
    parser.add_argument('--checkpoint-period', type=int, default=300,
                        help='Time between writing checkpoints')
    parser.add_argument('--port', default=5555, type=int,
                        help='Network port (TCP)')
    parser.add_argument('--command-port', default=5565, type=int,
                        help='Network port (TCP) for commands')
    parser.add_argument('--big', type=str, default='keep', choices=['keep', 'drop', 'queue'],
                        help='What to do with big blobs: "keep", "drop", "queue"')
    parser.add_argument('--big-pix', type=int, default=250000,
                        help='Define how many pixels are in a "big" blob')
    parser.add_argument('--big-port', default=5556, type=int,
                        help='Network port (TCP) for big blobs, if --big=queue')
    parser.add_argument('--big-command-port', default=5566, type=int,
                        help='Network port (TCP) for big blob commands, if --big=queue')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')
    opt = parser.parse_args()

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # Make quieter
    logging.getLogger('legacypipe.runbrick').setLevel(lvl+10)

    queuename = opt.queue

    import threading
    import queue

    # inqueue: for holding blob-work-packets
    #inqueue = queue.PriorityQueue(maxsize=10000)
    # Effectively, run a brick at a time, prioritizing as usual...
    inqueue = queue.Queue(maxsize=1000)

    # bigqueue: like inqueue, but for big blobs.
    if opt.big == 'queue':
        bigqueue = queue.Queue(maxsize=1000)
    else:
        bigqueue = None

    # outqueue: for holding blob results
    outqueue = queue.Queue()

    # queue directly from the input thread to the output thread, when checkpointed
    # results are read from disk.
    checkpointqueue = queue.Queue()

    blobsizes = ThreadSafeDict()
    outstanding_work = ThreadSafeDict()
    finished_work = ThreadSafeDict()
    reported_cputime = ThreadSafeDict()

    inthread = threading.Thread(target=input_thread,
                                args=(queuename, inqueue, bigqueue, checkpointqueue,
                                      blobsizes, opt),
                                daemon=True)
    inthread.start()

    outthread = threading.Thread(target=output_thread,
                                 args=(queuename, outqueue, checkpointqueue, blobsizes,
                                       outstanding_work, finished_work, reported_cputime, opt),
                                 daemon=True)
    outthread.start()

    ctx = zmq.Context()
    networkthread = threading.Thread(target=network_thread,
                                     args=(ctx, opt.port, opt.command_port, inqueue, outqueue,
                                           outstanding_work, finished_work, reported_cputime,
                                           'main'))
    networkthread.start()

    if opt.big == 'queue':
        bignetworkthread = threading.Thread(target=network_thread,
                                            args=(ctx, opt.big_port, opt.big_command_port,
                                                  bigqueue, outqueue, outstanding_work, None, None,
                                                  'big'))
                                                  
        bignetworkthread.start()
    else:
        bignetworkthread = None

    inthread.join()
    outthread.join()
    networkthread.join()
    if bignetworkthread is not None:
        bignetworkthread.join()


def network_thread(ctx, port, command_port, inqueue, outqueue, outstanding_work, finished_work,
                   reported_cputime, qname):
    # Network thread:
    import socket
    me = socket.gethostname()

    sock = ctx.socket(zmq.REP)
    addr = 'tcp://*:' + str(port)
    sock.bind(addr)
    print('Listening on tcp://%s:%i for work (queue: %s)' % (me, port, qname))

    command_sock = None
    if command_port is not None:
        command_sock = ctx.socket(zmq.REP)
        caddr = 'tcp://*:' + str(command_port)
        command_sock.bind(caddr)
        print('Listening on tcp://%s:%i for commands (queue: %s)' % (me, command_port, qname))

    nowork = pickle.dumps(None, -1)
    havework = False
    work = None
    worksent = 0
    resultsreceived = 0

    last_printout = time.time()

    while True:

        # Check for any messages on the command socket
        if command_sock is not None:
            while True:
                try:
                    msg = command_sock.recv(flags=zmq.NOBLOCK)
                    pymsg = pickle.loads(msg)
                    print('Message on command socket:', pymsg)

                    if pymsg[0] == 'reset':
                        out = outstanding_work.copy()
                        ncan = 0
                        for k,v in out.items():
                            (br,ib) = k
                            (nm,worker,tstart) = v
                            if nm == qname:
                                worksent -= 1
                                print('Network thread: cancelling', br,ib)
                                result = (br, ib, 'cancel', 0, 0)
                                msg = pickle.dumps(result, -1)
                                outqueue.put((None,msg))
                                outstanding_work.delete(k)
                                ncan += 1
                        reply = (True, 'cancelled %i blobs' % ncan)
                    else:
                        print('Unrecognized message on command socket:', pymsg)
                        reply = (False, 'unrecognized command')

                    msg = pickle.dumps(reply, -1)
                    command_sock.send(msg)

                except zmq.ZMQError:
                    # no message waiting
                    break

        debug('Network thread: work queue:', inqueue.qsize(), 'out queue:', outqueue.qsize(), 'work sent:', worksent, ', received:', resultsreceived, 'outstanding:', worksent-resultsreceived)
        # Retrieve the next work assignment
        if not havework:
            try:
                arg = inqueue.get(block=False)
                work = arg.item
                havework = True
            except:
                work = nowork
                havework = False
                print('Work queue is empty.')
            # Peek into work packet, checking for empty (None) work
            # and short-cutting, putting a None result directly on the
            # output queue.
            if havework:
                (br,iblob,isempty,picl) = work
                if isempty:
                    debug('Short-cutting empty work packet for brick', br, 'iblob', iblob)
                    result = pickle.dumps((br, iblob, None, 0, 0), -1)
                    outqueue.put((None,result))
                    havework = False
                    continue
                work = picl
                debug('Next work packet:', len(work), 'bytes')

        tnow = time.time()
        if tnow - last_printout > 15:
            if finished_work is not None:
                fini = finished_work.copy()
                if len(fini):
                    print('Finished bricks:')
                for k,v in fini.items():
                    print('  ', k, ':', v, 'blobs')
                    
            print()
            print('Work queue:', inqueue.qsize(), 'out queue:', outqueue.qsize(), 'work sent:', worksent, ', received:', resultsreceived, 'outstanding:', worksent-resultsreceived)
            out = outstanding_work.copy()
            print('Outstanding work:')
            c = Counter()
            ct = Counter()
            oldest = {}
            workers_telapsed = {}
            for i,(k,v) in enumerate(out.items()):
                (nm,worker,tstart) = v
                (br,ib) = k
                if i < 10:
                    print('  brick,blob', br,ib, 'queue "%s"' % nm, 'worker', worker.decode(),
                          'started %.1f s ago' % (tnow-tstart))
                c[br] += 1
                ct[br] += (tnow-tstart)
                if not br in oldest:
                    oldest[br] = (tnow-tstart)
                if not worker in workers_telapsed:
                    workers_telapsed[worker] = []
                workers_telapsed[worker].append(tnow-tstart)
            if len(out) > 10:
                print('  ....')
            print('Outstanding bricks:')
            for b,n in c.most_common():
                print('  ', b, 'waiting for', n, 'blobs')
            print('Outstanding bricks: total wall time:')
            for b,t in ct.most_common():
                print('  ', b, ': %.1f s' % t)
            print('Oldest blob per brick:')
            for b,t in oldest.items():
                print('  ', b, ': %.1f s' % t)
            print('Workers:')
            kk = list(workers_telapsed.keys())
            kk.sort()
            for k in kk:
                t = workers_telapsed[k]
                print('  ', k.decode(), ':', len(t), 'tasks, started %.1f to %.1f s ago' % (min(t), max(t)))
            cputime = reported_cputime.copy()
            worker_wall = Counter()
            worker_cpu  = Counter()
            worker_nblobs = Counter()
            for k,v in cputime.items():
                (br,ib) = k
                (worker,cpu,wall) = v
                if worker is None:
                    continue
                worker_wall[worker] += wall
                worker_cpu [worker] += cpu
                worker_nblobs[worker] += 1
                reported_cputime.delete(k)
            kk = list(worker_wall.keys())
            kk.sort()
            print('Completed work since last printout:')
            for k in kk:
                if worker_wall[k] == 0:
                    pct = 0
                else:
                    pct = 100. * worker_cpu[k] / worker_wall[k]
                print('   %s: %i blobs, wall time %.1f s, CPU time %.1f --> %.1f %%' %
                      (k.decode(), worker_nblobs[k], worker_wall[k], worker_cpu[k], pct))

            last_printout = tnow

        debug('Waiting for request')
        if not havework or command_sock is not None:
            events = sock.poll(5000)
            if events == 0:
                # recv timed out; check work queue again
                continue
        parts = sock.recv_multipart()
        assert(len(parts) == 2)
        worker = parts[0]
        result = parts[1]
        debug('Request: from', worker, ':', len(result), 'bytes')
        try:
            sock.send(work) #, flags=zmq.NOBLOCK)
            if havework:
                worksent += 1
                tnow = time.time()
                havework = False
                outstanding_work.set((br,iblob), (qname,worker,tnow))
        except:
            print('Network thread: sock.send(work) failed:')
            import traceback
            traceback.print_exc()

        if result == nowork:
            debug('Empty result')
            continue
        else:
            debug('Non-empty result')
            resultsreceived += 1
        #print('Now sent', worksent, 'work packages, received', resultsreceived, 'results; outstanding:', worksent-resultsreceived)
        outqueue.put((worker, result))


def output_thread(queuename, outqueue, checkpointqueue, blobsizes, outstanding_work,
                  finished_work, reported_cputime, opt):
    import qdo
    q = qdo.connect(queuename)

    allresults = {}

    # thread-Local cache of 'blobsizes'
    brick_nblobs = {}
    brick_taskids = {}

    # Local mapping of brickname -> [set of cancelled blob ids]
    brick_cancelled = {}

    def get_brick_nblobs(brick, defnblobs=None):
        nblobs = defnblobs
        taskid = None
        if brick in brick_nblobs:
            nblobs = brick_nblobs[brick]
            taskid = brick_taskids[brick]
        else:
            bs = blobsizes.get(brick)
            if bs is not None:
                (nblobs, taskid) = bs
                brick_nblobs[brick] = nblobs
                brick_taskids[brick] = taskid
        return nblobs,taskid

    def check_brick_done(brick):
        nblobs,taskid = get_brick_nblobs(brick)
        if nblobs is None:
            return
        ncancelled = len(brick_cancelled.get(brick, []))
        if len(allresults[brick]) + ncancelled < nblobs:
            return
        # Done this brick!  Set qdo state=Succeeded
        checkpoint_fn = opt.checkpoint % dict(brick=brick)
        R = [dict(brickname=brick, iblob=iblob, result=res) for
             iblob,res in allresults[brick].items()]
        print('Writing final checkpoint', checkpoint_fn)
        _write_checkpoint(R, checkpoint_fn)
        print('Setting QDO task to Succeeded:', brick)
        q.set_task_state(taskid, qdo.Task.SUCCEEDED)
        del allresults[brick]
        finished_work.set(brick, len(R))

    last_checkpoint = time.time()
    last_checkpoint_size = {}

    while True:
        tnow = time.time()
        dt = tnow - last_checkpoint
        if dt > opt.checkpoint_period:
            for brick,brickresults in allresults.items():
                if brick in last_checkpoint_size:
                    if len(brickresults) == last_checkpoint_size[brick]:
                        print('Brick', brick, 'has not changed since last checkpoint was written')
                        continue
                checkpoint_fn = opt.checkpoint % dict(brick=brick)
                R = [dict(brickname=brick, iblob=iblob, result=res) for
                     iblob,res in brickresults.items()]
                last_checkpoint_size[brick] = len(brickresults)
                nblobs,_ = get_brick_nblobs(brick, '(unknown)')
                print('Writing interim checkpoint', checkpoint_fn, ':', len(brickresults), 'of',
                      nblobs, 'results')
                _write_checkpoint(R, checkpoint_fn)
            last_checkpoint = tnow

        # Read any checkpointed results sent by the input thread
        c = Counter()
        while True:
            try:
                (brick, iblob, res) = checkpointqueue.get(block=False)
            except:
                break
            if not brick in allresults:
                allresults[brick] = {}
            allresults[brick][iblob] = res
            c[brick] += 1
        if len(c):
            print('Read checkpointed results:', c)
        for brick,n in c.most_common():
            nblobs,_ = get_brick_nblobs(brick, '(unknown)')
            print('Brick', brick, ': now', len(allresults[brick]), 'of', nblobs, 'done')
            check_brick_done(brick)

        try:
            worker,msg = outqueue.get(timeout=60)
        except:
            # timeout
            continue
        result = pickle.loads(msg)
        if result is None:
            continue

        # Worker sent a blob result
        # (OR, network thread sent a "cancel")
        (brick, iblob, res, cpu, wall) = result
        if res == 'cancel':
            if not brick in brick_cancelled:
                brick_cancelled[brick] = set()
            brick_cancelled[brick].add(iblob)

            debug('Output thread: got cancel for brick', brick, 'blob', iblob)
            check_brick_done(brick)

        else:
            if not brick in allresults:
                allresults[brick] = {}
            allresults[brick][iblob] = res

            if worker is not None:
                reported_cputime.set((brick,iblob), (worker, cpu, wall))

            try:
                tnow = time.time()
                qname,worker,tstart = outstanding_work.get_and_del((brick,iblob))
                twork = tnow - tstart
            except:
                qname = None
                worker = None
                twork = None
    
            nblobs,_ = get_brick_nblobs(brick, '(unknown)')
            debug('Output thread: got result', len(allresults[brick]), 'of', (nblobs or '(unknown)'), 'for brick', brick, 'iblob', iblob, 'from', worker, 'on queue', qname, 'took', twork, 'seconds')
            check_brick_done(brick)

def get_blob_iter(skipblobs=None,
                  brickname=None,
                  brick=None,
                  blobsrcs=None, blobslices=None, blobs=None,
                  cat=None,
                  targetwcs=None,
                  W=None,H=None,
                  bands=None, ps=None, tims=None,
                  survey=None,
                  plots=False, plots2=False,
                  max_blobsize=None,
                  simul_opt=False, use_ceres=True, mp=None,
                  refstars=None,
                  rex=False,
                  T_clusters=None,
                  custom_brick=False,
                  **kwargs):
    import numpy as np
    if skipblobs is None:
        skipblobs = []
    
    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if refstars:
        from legacypipe.oneblob import get_inblob_map
        refs = refstars[refstars.donotfit == False]
        if T_clusters is not None:
            refs = merge_tables([refs, T_clusters], columns='fillzero')
        refmap = get_inblob_map(targetwcs, refs)
        del refs
    else:
        HH, WW = targetwcs.shape
        refmap = np.zeros((int(HH), int(WW)), np.uint8)

    # Create the iterator over blobs to process
    blobiter = _blob_iter(brickname, blobslices, blobsrcs, blobs, targetwcs, tims,
                          cat, bands, plots, ps, simul_opt, use_ceres,
                          refmap, brick, rex,
                          max_blobsize=max_blobsize, custom_brick=custom_brick,
                          skipblobs=skipblobs)
    return blobiter

class PrioritizedItem(object):
    def __init__(self, priority=0, item=None):
        self.priority = priority
        self.item = item
    def __lt__(self, other):
        return self.priority < other.priority

class ThreadSafeDict(object):
    def __init__(self):
        import threading
        self.d = dict()
        self.lock = threading.Lock()
    def set(self, k, v):
        with self.lock:
            self.d[k] = v
    def get(self, k):
        with self.lock:
            return self.d.get(k)
    def get_and_del(self, k):
        with self.lock:
            v = self.d.get(k)
            del self.d[k]
            return v
    def delete(self, k):
        with self.lock:
            del self.d[k]
    def copy(self):
        with self.lock:
            return self.d.copy()

def queue_work(brickname, inqueue, bigqueue, checkpointqueue, opt):
    '''
    Called from the input thread to generate work packets for the given *brickname*.
    '''
    from astrometry.util.file import unpickle_from_file

    pickle_fn = opt.pickle % dict(brick=brickname, brickpre=brickname[:3])
    print('Looking for', pickle_fn)
    if not os.path.exists(pickle_fn):
        raise RuntimeError('Input pickle does not exist: ' + pickle_fn)
    kwargs = unpickle_from_file(pickle_fn)
    debug('Unpickled:', kwargs.keys())

    # Total blobs includes checkpointed ones.
    nchk = 0

    # Check for and read existing checkpoint file.
    checkpoint_fn = opt.checkpoint % dict(brick=brickname, brickpre=brickname[:3])
    if os.path.exists(checkpoint_fn):
        debug('Reading checkpoint file', checkpoint_fn)
        R = unpickle_from_file(checkpoint_fn)
        print('Read', len(R), 'from checkpoint file')

        skipblobs = []
        for r in R:
            br = r['brickname']
            assert(br == brickname)
            iblob = r['iblob']
            result = r['result']
            checkpointqueue.put((brickname, iblob, result))
            skipblobs.append(iblob)
            nchk += 1
        kwargs.update(skipblobs=skipblobs)

    # (brickname is in the kwargs read from the pickle!)
    assert(kwargs['brickname'] == brickname)
    blobiter = get_blob_iter(**kwargs)

    big_npix = opt.big_pix
    if opt.big == 'keep':
        big_npix = 10000 * 10000

    nq = 0
    for arg in blobiter:
        if arg is None:
            continue

        (br, iblob, args) = arg
        assert(br == brickname)
        if args is None:
            # do these first?
            priority = -1000000000
        else:
            # HACK -- reach into args to get blob size, for priority ordering
            blobw = args[6]
            blobh = args[7]
            priority = -(blobw*blobh)

        if opt.big == 'drop' and blobw*blobh > big_npix:
            print('Dropping a blob of size', blobw, 'x', blobh)
            continue

        dest_queue = inqueue

        if opt.big == 'queue' and blobw*blobh > big_npix:
            print('Blob of size', blobw, 'x', blobh, 'goes on big queue')
            dest_queue = bigqueue

        picl = pickle.dumps(arg, -1)

        qitem = PrioritizedItem(priority=priority, item=(br, iblob, (args is None), picl))
        debug('Queuing blob', (nq+1), 'for brick', brickname, '- queue size ~', inqueue.qsize())
        nq += 1
        dest_queue.put(qitem)

    # Finished queuing all blobs for this brick -- record how many blobs we sent out.
    return nchk + nq

def input_thread(queuename, inqueue, bigqueue, checkpointqueue, blobsizes, opt):
    import qdo
    q = qdo.connect(queuename)

    while True:
        task = q.get(timeout=10)
        if task is None:
            #- no more tasks in queue so break
            break
        try:
            brickname = task.task
            debug('Brick', brickname)
            # WORK
            nblobs = queue_work(brickname, inqueue, bigqueue, checkpointqueue, opt)
            blobsizes.set(brickname, (nblobs, task.id))
            #
            debug('Finished', brickname, 'with', nblobs, 'blobs')
        except:
            print('Oops')
            import traceback
            traceback.print_exc()
            task.set_state(qdo.Task.FAILED, err=1)

if __name__ == '__main__':
    main()
