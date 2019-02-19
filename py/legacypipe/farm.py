from legacypipe.runbrick import *
from legacypipe.runbrick import _blob_iter, _write_checkpoint
import pickle

import zmq

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queue', help='QDO queue name to get brick names from')
    parser.add_argument('--pickle', default='pickles/runbrick-%(brick)s-srcs.pickle',
                        help='Pickle pattern for "srcs" (source detection) stage pickles')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint-%(brick)s.pickle',
                        help='Checkpoint filename pattern')
    opt = parser.parse_args()

    queuename = opt.queue


    import threading
    import queue

    inqueue = queue.PriorityQueue(maxsize=1000)
    outqueue = queue.Queue()
    # queue directly from the input thread to the output thread, when checkpointed
    # results are read from disk.
    checkpointqueue = queue.Queue()
    
    blobsizes = BlobSizeDict()

    inthread = threading.Thread(target=input_thread, args=(queuename, inqueue, checkpointqueue, blobsizes, opt),
                                daemon=True)
    inthread.start()

    outthread = threading.Thread(target=output_thread, args=(queuename, outqueue, checkpointqueue, blobsizes, opt),
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

    while True:

        print('Network thread: approximate size of input queue:', inqueue.qsize(), 'output queue:', outqueue.qsize())
        if inqueue.empty():
            work = nowork
            iswork = False
            print('Work queue is empty.')
        else:
            arg = inqueue.get()
            work = arg.item
            iswork = True
            print('Next work packet:', len(work), 'bytes')

        print('Waiting for request')
        result = sock.recv()
        print('Request:', len(result), 'bytes')
        try:
            sock.send(work, flags=zmq.NOBLOCK)
            if iswork:
                worksent += 1
        except:
            import traceback
            traceback.print_exc()
            pass

        if result == nowork:
            print('Empty result')
        else:
            print('Non-empty result')
            resultsreceived += 1
        print('Now sent', worksent, 'work packages, received', resultsreceived, 'results; outstanding:', worksent-resultsreceived)

        outqueue.put(result)


def output_thread(queuename, outqueue, checkpointqueue, blobsizes, opt):

    import qdo
    q = qdo.connect(queuename)

    allresults = {}

    # thread-Local cache of 'blobsizes'
    brick_nblobs = {}
    brick_taskids = {}

    while True:
        from collections import Counter

        # Read any checkpointed results
        c = Counter()
        while True:
            try:
                (brick, iblob, res) = checkpointqueue.get(block=False)
            except:
                break
            if not brick in allresults:
                allresults[brick] = []
            allresults[brick].append((iblob,res))
            c.update([brick])
        print('Read checkpointed results:', c)

        msg = outqueue.get()
        result = pickle.loads(msg)
        if result is None:
            continue

        # Worker sent a blob result
        (brick, iblob, res) = result
        if not brick in allresults:
            allresults[brick] = []
        allresults[brick].append((iblob,res))

        nblobs = None
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

        print('Received result number', len(allresults[brick]), 'for brick', brick, 'of total', (nblobs or '(unknown)'))

        if nblobs is not None:
            if len(allresults[brick]) == nblobs:
                # Done this brick!  Set qdo state=Succeeded
                # tasks = q.tasks(id=taskid)
                # print('Found qdo tasks', task)
                # for t in tasks:
                #     qdo.set_task_state(

                ### FIXME -- be robust against duplicate iblobs

                checkpoint_fn = opt.checkpoint % dict(brick=brick)
                R = [r for iblob,r in allresults[brick]]
                print('Writing checkpoint', checkpoint_fn)
                _write_checkpoint(R, checkpoint_fn)

                print('Setting QDO task to Succeeded')
                q.set_task_state(taskid, qdo.Task.SUCCEEDED)

        if len(allresults[brick]) % 1000 == 0:

            ### FIXME!!
            print('Write interim checkpoint for', brick, '????')
                

    # q.set_task_state(taskid, state)
    # q.tasks(id=None)

def get_blob_iter(T=None,
                   brickname=None,
                   brickid=None,
                   brick=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None, blobradec=None, blobid=None,
                   max_blobsize=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   refstars=None,
                   rex=False,
                   bailout=False,
                   record_event=None,
                   custom_brick=False,
                   **kwargs):
    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()
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
    blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                          cat, bands, plots, ps, simul_opt, use_ceres,
                          refmap, brick, rex,
                          max_blobsize=max_blobsize, custom_brick=custom_brick)
    return blobiter


# python3.7 foncy
# from dataclasses import dataclass, field
# from typing import Any
# @dataclass(order=True)
# class PrioritizedItem:
#     priority: int
#     item: Any=field(compare=False)

class PrioritizedItem(object):
    def __init__(self, priority=0, item=None):
        self.priority = priority
        self.item = item
    def __lt__(self, other):
        return self.priority < other.priority


class BlobSizeDict(object):
    def __init__(self):
        import threading
        self.d = dict()
        self.lock = threading.Lock()
    def set(self, brickname, nblobs):
        with self.lock:
            self.d[brickname] = nblobs
    def get(self, brickname):
        with self.lock:
            return self.d.get(brickname)


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

    # Check for and read existing checkpoint file.
    checkpoint_fn = opt.checkpoint % dict(brick=brickname)
    if os.path.exists(checkpoint_fn):
        print('Reading checkpoint file', checkpoint_fn)
        R = unpickle_from_file(checkpoint_fn)
        print('Read', len(R), 'from checkpoint file')

        n = len(R)
        R = [r for r in R if r is not None]
        print('Keeping', len(R), 'of', n, 'non-None checkpointed results')
        skipblobs = []
        for r in R:
            checkpointqueue.put((brickname, r.iblob, r))
            skipblobs.append(r.iblob)
        kwargs.update(skipblobs=skipblobs)

    blobiter = get_blob_iter(**kwargs)

    k = 0
    for arg in blobiter:
        if arg is None:
            continue

        # HACK -- reach into args to get blob id...
        iblob = arg[1]
        blobw = arg[6]
        blobh = arg[7]
        arg = (brickname, iblob, arg)
        priority = -(blobw*blobh)

        picl = pickle.dumps(arg, -1)

        qitem = PrioritizedItem(priority=priority, item=picl)
        print('Queuing blob', (k+1), 'for brick', brickname)
        k += 1
        inqueue.put(qitem)

    # Finished queuing all blobs for this brick -- record how many blobs we sent out.
    return k


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
