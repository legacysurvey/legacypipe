from legacypipe.runbrick import *
from legacypipe.runbrick import _blob_iter, _write_checkpoint
import pickle

import zmq

def main():
    import logging
    parser = get_parser()
    opt = parser.parse_args()
    optdict = vars(opt)
    verbose = optdict.pop('verbose')

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    from astrometry.util.stages import CallGlobalTime, runstage
    stagefunc = CallGlobalTime('stage_%s', globals())

    run_brick(opt.brick, survey, prereqs_update={'farmblobs': 'srcs'}, stagefunc=stagefunc,
              **kwargs)


def main2():
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








def stage_farmblobs(T=None,
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
    from tractor import Catalog
    from legacypipe.survey import IN_BLOB

    tlast = Time()


    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()

    # Were we asked to only run a subset of blobs?
    keepblobs = None
    if blobradec is not None:
        # blobradec is a list like [(ra0,dec0), ...]
        rd = np.array(blobradec)
        ok,x,y = targetwcs.radec2pixelxy(rd[:,0], rd[:,1])
        x = (x - 1).astype(int)
        y = (y - 1).astype(int)
        blobxy = list(zip(x, y))
        print('Blobradec -> blobxy:', len(blobxy), 'points')

    if blobxy is not None:
        # blobxy is a list like [(x0,y0), (x1,y1), ...]
        keepblobs = []
        for x,y in blobxy:
            x,y = int(x), int(y)
            if x < 0 or x >= W or y < 0 or y >= H:
                print('Warning: clipping blob x,y to brick bounds', x,y)
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)
            blob = blobs[y,x]
            if blob >= 0:
                keepblobs.append(blob)
            else:
                print('WARNING: blobxy', x,y, 'is not in a blob!')
        keepblobs = np.unique(keepblobs)

    if blobid is not None:
        # comma-separated list of blob id numbers.
        keepblobs = np.array([int(b) for b in blobid.split(',')])

    if blob0 is not None or (nblobs is not None and nblobs < len(blobslices)):
        if blob0 is None:
            blob0 = 0
        if nblobs is None:
            nblobs = len(blobslices) - blob0
        keepblobs = np.arange(blob0, blob0+nblobs)

    # keepblobs can be None or empty list
    if keepblobs is not None and len(keepblobs):
        # 'blobs' is an image with values -1 for no blob, or the index
        # of the blob.  Create a map from old 'blob number+1' to new
        # 'blob number', keeping only blobs in the 'keepblobs' list.
        # The +1 is so that -1 is a valid index in the mapping.
        NB = len(blobslices)
        blobmap = np.empty(NB+1, int)
        blobmap[:] = -1
        blobmap[keepblobs + 1] = np.arange(len(keepblobs))
        # apply the map!
        blobs = blobmap[blobs + 1]

        # 'blobslices' and 'blobsrcs' are lists where the index corresponds to the
        # value in the 'blobs' map.
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]

        # one more place where blob numbers are recorded...
        T.blob = blobs[T.iby, T.ibx]

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



    import socket
    me = socket.gethostname()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:5555')
    print('Listening on tcp://%s:5555' % me)

    allresults = {}
    allsent = {}
    allsent[brickname] = False
    nsent = {}
    sendwork = True

    nextwork = None
    nextwork_brick = brickname
    finished = False

    while True:

        if sendwork:
            try:
                arg = next(blobiter)
            except StopIteration:
                sendwork = False
                #nextwork = None
                arg = None
                nextwork = pickle.dumps(arg, -1)
                nextwork_brick = None
                allsent[brickname] = True
                continue
            if arg is None:
                continue

            # HACK -- reach into args to get blob id...
            iblob = arg[1]
            arg = (brickname, iblob, arg)
            nextwork = pickle.dumps(arg, -1)
            print('Next arg:', len(nextwork), 'bytes')

        print('Waiting for request')
        msg = sock.recv()
        print('Request:', len(msg), 'bytes')

        m = pickle.loads(msg)
        #print('Unpickled:', m)
        if m is not None:
            # Worker sent the result of a previous computation.
            (br, iblob, res) = m
            if not br in allresults:
                allresults[br] = []
            allresults[br].append((iblob,res))
            print('Received result number', len(allresults[br]), 'for brick', br, 'nsent', nsent.get(br,0), 'all sent:', allsent.get(br,None))
            if allsent[br] and len(allresults[br]) == nsent[br]:
                print('Finished receiving all results for', br)
                finished = True
                R = [r for iblob,r in allresults[br]]
                _write_checkpoint(R, checkpoint_filename)

        print('Sending work...')
        try:
            sock.send(nextwork, flags=zmq.NOBLOCK)
            if nextwork_brick is not None:
                if not nextwork_brick in nsent:
                    nsent[nextwork_brick] = 0
                nsent[nextwork_brick]+= 1
        except:
            import traceback
            traceback.print_exc()
            pass

        #if finished:
        #break

    #print('All done brick', brickname)

    # t0 = Time()
    # args = list(blobiter)
    # t1 = Time()
    # print('Took', t1-t0, 'to compute blob args')
    # 
    # sizes = []
    # for i,a in enumerate(args):
    #     p = pickle.dumps(a, -1)
    #     sizes.append(len(p))
    # t2 = Time()
    # print('Took', t2-t1, 'to pickle; total size', sum(sizes), 'bytes')



if __name__ == '__main__':
    #main()
    main2()
