import pickle
import time
from collections import Counter

import zmq

from legacypipe.runbrick import _blob_iter, _write_checkpoint

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queue', help='QDO queue name to get brick names from')
    parser.add_argument('--pickle', default='pickles/runbrick-%(brick)s-srcs.pickle',
                        help='Pickle pattern for "srcs" (source detection) stage pickles')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint-%(brick)s.pickle',
                        help='Checkpoint filename pattern')
    parser.add_argument('--checkpoint-period', type=int, default=300,
                        help='Time between writing checkpoints')
    opt = parser.parse_args()

    queuename = opt.queue


    import threading
    import queue

    #inqueue = queue.PriorityQueue(maxsize=10000)
    # Effectively, run a brick at a time, prioritizing as usual...
    inqueue = queue.Queue(maxsize=1000)
    outqueue = queue.Queue()
    # queue directly from the input thread to the output thread, when checkpointed
    # results are read from disk.
    checkpointqueue = queue.Queue()
    blobsizes = ThreadSafeDict()

    outstanding_work = ThreadSafeDict()

    inthread = threading.Thread(target=input_thread, args=(queuename, inqueue, checkpointqueue, blobsizes, opt),
                                daemon=True)
    inthread.start()

    outthread = threading.Thread(target=output_thread, args=(queuename, outqueue, checkpointqueue, blobsizes, outstanding_work, opt),
                                 daemon=True)
    outthread.start()

    # Network thread:
    import socket
    me = socket.gethostname()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:5555')
    print('Listening on tcp://%s:5555' % me)

    nowork = pickle.dumps(None, -1)
    iswork = False
    worksent = 0
    resultsreceived = 0
    #  FIXME -- track brick,iblob of outstanding work, for timeout?

    last_printout = time.time()

    while True:

        print('Network thread: approximate size of input queue:', inqueue.qsize(), 'output queue:', outqueue.qsize())
        try:
            arg = inqueue.get(block=False)
            work = arg.item
            iswork = True
        except:
            work = nowork
            iswork = False
            print('Work queue is empty.')

        # Peek into work packet, checking for empty (None) work and short-cutting, putting a None
        # result directly on the output queue.
        if iswork:
            (br,iblob,isempty,picl) = work
            if isempty:
                print('Short-cutting empty work packet for brick', br, 'iblob', iblob)
                result = pickle.dumps((br, iblob, None), -1)
                outqueue.put(result)
                continue
            work = picl
            print('Next work packet:', len(work), 'bytes')

        tnow = time.time()
        if tnow - last_printout > 60:
            out = outstanding_work.copy()
            print('Outstanding work:')
            c = Counter()
            ct = Counter()
            oldest = {}
            for i,(k,v) in enumerate(out.items()):
                (worker,tstart) = v
                (br,ib) = k
                if i < 20:
                    print('Brick,blob', k, 'worker', worker, 'started', tnow-tstart, 's ago')
                c[br] += 1
                ct[br] += (tnow-tstart)
                if not br in oldest:
                    oldest[br] = (tnow-tstart)
            print('Outstanding bricks:', c.most_common())
            print('Outstanding brick total time:', ct.most_common())
            print('Oldest blob per brick:', oldest)
            last_printout = tnow

        print('Waiting for request')
        if not iswork:
            events = sock.poll(5000)
            if events == 0:
                # recv timed out; check work queue again
                continue
        parts = sock.recv_multipart()
        assert(len(parts) == 2)
        sender = parts[0]
        result = parts[1]
        print('Request: from', sender, ':', len(result), 'bytes')
        try:
            sock.send(work, flags=zmq.NOBLOCK)
            if iswork:
                worksent += 1
                tnow = time.time()
                outstanding_work.set((br,iblob), (sender,tnow))
        except:
            import traceback
            traceback.print_exc()

        if result == nowork:
            print('Empty result')
            continue
        else:
            print('Non-empty result')
            resultsreceived += 1
        print('Now sent', worksent, 'work packages, received', resultsreceived, 'results; outstanding:', worksent-resultsreceived)

        outqueue.put(result)


def output_thread(queuename, outqueue, checkpointqueue, blobsizes, outstanding_work, opt):
    import qdo
    q = qdo.connect(queuename)

    allresults = {}

    # thread-Local cache of 'blobsizes'
    brick_nblobs = {}
    brick_taskids = {}

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
        if nblobs is not None:
            if len(allresults[brick]) == nblobs:
                # Done this brick!  Set qdo state=Succeeded
                checkpoint_fn = opt.checkpoint % dict(brick=brick)
                R = [dict(brickname=brick, iblob=iblob, result=res) for
                     iblob,res in allresults[brick].items()]
                print('Writing final checkpoint', checkpoint_fn)
                _write_checkpoint(R, checkpoint_fn)
                print('Setting QDO task to Succeeded')
                q.set_task_state(taskid, qdo.Task.SUCCEEDED)
                del allresults[brick]


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
            msg = outqueue.get(timeout=60)
        except:
            # timeout
            continue
        result = pickle.loads(msg)
        if result is None:
            continue

        # Worker sent a blob result
        (brick, iblob, res) = result
        if not brick in allresults:
            allresults[brick] = {}
        allresults[brick][iblob] = res

        try:
            tnow = time.time()
            worker,tstart = outstanding_work.get_and_del((brick,iblob))
            twork = tnow - tstart
        except:
            worker = None
            twork = None

        nblobs,_ = get_brick_nblobs(brick, '(unknown)')
        print('Output thread: got result', len(allresults[brick]), 'of', (nblobs or '(unknown)'), 'for brick', brick, 'iblob', iblob, 'from', worker, 'took', twork, 'seconds')
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
                  custom_brick=False,
                  **kwargs):
    if skipblobs is None:
        skipblobs = []
    
    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if refstars:
        from legacypipe.oneblob import get_inblob_map
        refstars.radius_pix = np.ceil(refstars.radius * 3600. / targetwcs.pixel_scale()).astype(int)
        refmap = get_inblob_map(targetwcs, refstars)
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

def queue_work(brickname, inqueue, checkpointqueue, opt):
    '''
    Called from the input thread to generate work packets for the given *brickname*.
    '''
    from astrometry.util.file import unpickle_from_file

    pickle_fn = opt.pickle % dict(brick=brickname)
    print('Looking for', pickle_fn)
    if not os.path.exists(pickle_fn):
        raise RuntimeError('Input pickle does not exist: ' + pickle_fn)
    kwargs = unpickle_from_file(pickle_fn)
    print('Unpickled:', kwargs.keys())

    # Total blobs includes checkpointed ones.
    nchk = 0

    # Check for and read existing checkpoint file.
    checkpoint_fn = opt.checkpoint % dict(brick=brickname)
    if os.path.exists(checkpoint_fn):
        print('Reading checkpoint file', checkpoint_fn)
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

        picl = pickle.dumps(arg, -1)

        qitem = PrioritizedItem(priority=priority, item=(br, iblob, (args is None), picl))
        print('Queuing blob', (nq+1), 'for brick', brickname, '- queue size ~', inqueue.qsize())
        nq += 1
        inqueue.put(qitem)

    # Finished queuing all blobs for this brick -- record how many blobs we sent out.
    return nchk + nq

def input_thread(queuename, inqueue, checkpointqueue, blobsizes, opt):

    import qdo
    q = qdo.connect(queuename)

    while True:
        task = q.get(timeout=10)

        if task is not None:
            try:
                brickname = task.task
                print('Brick', brickname)
                # WORK
                nblobs = queue_work(brickname, inqueue, checkpointqueue, opt)
                blobsizes.set(brickname, (nblobs, task.id))
                #
                print('Finished', brickname, 'with', nblobs, 'blobs')
                #task.set_state(qdo.Task.SUCCEEDED)
            except:
                print('Oops')
                import traceback
                traceback.print_exc()
                task.set_state(qdo.Task.FAILED, err=1)
        else:
            #- no more tasks in queue so break
            break

if __name__ == '__main__':
    main()
