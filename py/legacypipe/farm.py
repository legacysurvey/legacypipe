'''
The farm.py script is a way of coordinating a bunch of worker
processes (in worker.py) to run "blobs" of work for the Legacy Surveys
data reduction.  In particular, it performs the work of the most
time-consuming phase, "blob-fitting", where we fit forward models of
stars and galaxies to overlapping images.

There is a drawing of the structure of the farm.py app in ../doc/farm.jpg
(https://github.com/legacysurvey/legacypipe/blob/main/doc/farm.jpg).

Farm.py uses QDO to read the list of "bricks" it is going to process
and to keep track of their status (ready / running / finished).

For each brick, the input is a "pickle" file containing the results of
the pipeline up to the blob-fitting stage.  The stage just before the
blob-fitting stage (called "stage_fitblobs" in the runbrick.py code)
is called "stage_srcs", because it determines the set of sources
(stars and galaxies) we are going to fit.  Sources are grouped
together into "blobs" if they overlap on the sky; if they overlap, we
have to fit them together.  Blobs are processed independently of each
other, and are the basic unit of work we are doing.

For each brick, the output of farm.py is a "checkpoint" file, which
contains the fitting results for the blobs in that brick.

(Once we are done running farm.py, we will re-run the runbrick.py
script, which will read the "srcs" pickle, start on the "fitblobs"
stage of processing, find that a "checkpoint" file exists, and that
most or all of its work is done; it will then proceed to the remaining
stages of the pipeline.)

Farm.py and its workers communicate over a ZeroMQ socket, which is a
fancy software layer over top of (in our case) a TCP socket.  Farm.py
acts as a server, and worker.py as clients.  When we launch worker.py
processes, we have to tell them the address of the farm.py server they
will get their work from.  Each worker.py will run independently on a
single core, because the code that processes blobs is single-threaded.

Farm.py is itself a multi-process application:
- one process handles communication on the socket ("network_thread")
- one process handles collecting & writing outputs ("output_thread")
- multiple processes handle reading input pickle files and preparing
  work packets ("input_thread")

These processes communicate with each other over multiprocessing.Queue
objects:

- "inqueue" goes from input_threads to network_thread and contains
  "work packets" that will be sent to the workers.
- "outqueue" goes from network_thread to output_thread and contains
  results received from workers.
- "checkpointqueue" goes from input_threads to output_thread, and is
  used to send results from an existing checkpoint file directly to
  the output_thread.

There is also a "bigqueue", which is like the "inqueue", but for "big
blobs" -- chunks of work that we expect to take a long time to
complete, so we handle them separately.  There's a "big" version of
the network_thread that reads from this queue.

Finally, there are a couple of book-keeping queues:
- the "blobsizes" queue goes from the input_thread to the output_thread
  and tells the output_thread how many blobs exist in each brick (so it
  knows when a brick is complete).
- the "finished_bricks" queue goes from the output_thread to the
  network_thread to report when a brick has been completed, just for
  logging purposes (network_thread is responsible for printing the status).

The overall data flow is:

- one of the input_threads queries the QDO database for the next brick
  is should work on

- it reads the srcs pickle for that brick, and reads an existing
  checkpoint file, if it exists.  (This happens in the "queue_work"
  function.)  It sends checkpointed results over the checkpoint queue
  straight to the output_thread.  It then calls the "get_blob_iter"
  function to produce a series of work packets (one for each blob,
  containing subimages and sources to be fit).  Each work packet gets
  put on the "input_queue".  (Once all work packets are produced for a
  brick, it puts the number of blobs found on the "blobsizes" queue to
  the output_thread.

- a worker.py client sends a message to the farm.py socket.  The first
  time it calls, it sends an empty message.

- the network_thread receives the message, and replies with a work
  packet (that it has pulled off the input_queue).

- the worker.py calls the one_blob() function to perform the work in
  its work packet.  It sends the result to the farm.py socket.  (And
  receives in reply its next work assignment.)

- the network_thread receives the message containing the results (and
  replies with the next work packet for the worker).  It puts the
  result on the output_queue.

- the output_thread pops a result off the output_queue, and if it
  determines that this is the final result for this brick, then it
  will write the output checkpoint file and mark the brick as finished
  in QDO.  (And notify the network_thread over the finished_bricks
  queue.)


At present, we run this by starting the farm.py script on an
interactive-queue node (preferably Haswell), copying the node's socket
address into a script that we will use to launch worker.py processes
via Slurm (on KNL nodes).  This gives the input_thread processes some
time to prepare some work packets before the worker.py processes show
up asking to be fed, and also allows us to monitor the overall state
of the run.

Going forward, it would probably be good to automatically start one
farm.py process per Slurm job, and then start a bunch of worker.py
processes to talk to it.

The worker.py processes should not have to touch the filesystem to
read any data.  If we run them within Shifter containers, then they
should have little I/O overhead and should scale quite well.

Bottlenecks in the current setup (as I understand them) are:
- eventually the network link (which uses the infiniband fabric) will
  saturate
- the input_threads have to do a fair bit of work, so each one can
  feed only a handful of worker.py nodes; eventually we'll fill a
  Haswell node with input_threads.
- last I profiled, it seemed like the network_thread was spending a
  significant fraction of its time popping items from the input_queue
  -- perhaps due to contention for the lock.  It could be that we want
  to use one input_queue per input_thread, rather than having all of
  them share a single queue.  There is no particular reason they need
  to share a queue; we would then have to implement a simple
  round-robin scheme to pull work from the different input_queues.
- currently, the worker.py processes are synchronous: they ask for
  work, wait for the reply, do the work, and send in the result.  We
  could keep a short queue (per worker or even for the set of workers
  sharing a node) to reduce this latency.

Last I checked, I could keep up with about 64 KNL nodes x 68 worker.py
processes with 8 input_thread processes, but efficiency was starting
to drop.

'''
import sys
import os
import pickle
import time
from collections import Counter, OrderedDict

import multiprocessing as mp
import queue
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
    parser.add_argument('--inthreads', type=int, default=1,
                        help='Number of brick-processing processes to start')
    parser.add_argument('--queuesize', type=int, default=1000, help='Number of blobs of work to queue')
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

    # inqueue: for holding blob-work-packets
    #inqueue = queue.PriorityQueue(maxsize=10000)
    # Effectively, run a brick at a time, prioritizing as usual...
    #inqueue = mp.Queue(maxsize=opt.queuesize)
    # We instead are going to use one queue per input thread, below

    # bigqueue: like inqueue, but for big blobs.
    if opt.big == 'queue':
        bigqueue = mp.Queue(maxsize=1000)
    else:
        bigqueue = None

    # outqueue: for holding blob results
    outqueue = mp.Queue()

    # queue directly from the input thread to the output thread, when checkpointed
    # results are read from disk.
    checkpointqueue = mp.Queue()

    # queue from the input thread to output thread, used to communicate the number
    # of blobs in a brick.  This is used by the output thread to decide that it has
    # received all results for a brick.
    blobsizes = mp.Queue()

    # queue from the output thread to the network thread, only for reporting
    # that a brick has been finished.
    finished_bricks = mp.Queue()

    inthreads = []
    inqueues = []
    for i in range(opt.inthreads):
        inqueue = mp.Queue(maxsize=opt.queuesize)
        inqueues.append(inqueue)
        inthread = mp.Process(target=input_thread,
                              args=(queuename, inqueue, bigqueue, checkpointqueue,
                                    blobsizes, opt, i),
                              daemon=True)
        inthreads.append(inthread)
        inthread.start()

    outthread = mp.Process(target=output_thread,
                                 args=(queuename, outqueue, checkpointqueue, blobsizes,
                                       finished_bricks, opt),
                                 daemon=True)
    outthread.start()

    ctx = None
    networkthread = mp.Process(target=network_thread,
                               args=(ctx, opt.port, opt.command_port, inqueues, outqueue,
                                     finished_bricks, 'main'),
                               daemon=True)
    networkthread.start()

    if opt.big == 'queue':
        bignetworkthread = mp.Process(target=network_thread,
                                            args=(ctx, opt.big_port, opt.big_command_port,
                                                  [bigqueue], outqueue, None, 'big'),
                                      daemon=True)
        bignetworkthread.start()
    else:
        bignetworkthread = None

    inthread.join()
    outthread.join()
    networkthread.join()
    if bignetworkthread is not None:
        bignetworkthread.join()


def network_thread(ctx, port, command_port, inqueues, outqueue, finished_bricks, qname):
    # Set my process name
    try:
        import setproctitle
        setproctitle.setproctitle('farm: network')
    except:
        pass

    # Set up ZeroMQ socket for communicating with workers
    import socket
    me = socket.gethostname()
    if ctx is None:
        ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    addr = 'tcp://*:' + str(port)
    sock.bind(addr)
    info('Listening on tcp://%s:%i for work (queue: %s)' % (me, port, qname))

    # Set up ZeroMQ socket for accepting control messages.
    command_sock = None
    if command_port is not None:
        command_sock = ctx.socket(zmq.REP)
        caddr = 'tcp://*:' + str(command_port)
        command_sock.bind(caddr)
        info('Listening on tcp://%s:%i for commands (queue: %s)' % (me, command_port, qname))

    nowork = pickle.dumps(None, -1)
    havework = False
    work = None
    worksent = 0
    resultsreceived = 0

    all_finished_bricks = {}

    outstanding_work = {}

    next_inqueue = 0

    # Worker-reported cpu time in oneblob(), wall time in oneblob(), and overhead.
    # [brick] -> float time
    brick_cputime = Counter()
    brick_nblobs = Counter()
    brick_starttime = OrderedDict()

    # Per-status-report-period counters
    status_cputime  = 0.
    status_walltime = 0.
    status_overhead = 0.
    status_nblobs   = 0

    # Overall
    total_nblobs   = 0
    total_cputime  = 0.
    total_walltime = 0.
    total_overhead = 0.

    tnow = time.time()
    last_printout = tnow
    last_print_workqueue = tnow
    last_check_command = tnow

    nworkpackets = nworkbytes = 0
    nwaitingCounter = Counter()

    # For keeping track of how much time is spent in different parts of my job
    t_in = 0
    t_poll = 0
    t_recv = 0
    t_send = 0
    t_decode = 0
    t_out = 0

    n_noblock = 0
    t_noblock = 0.
    n_block = 0
    t_block = 0.
    n_nowork = 0
    
    while True:
        tnow = time.time()
        t1 = tnow

        if command_sock is not None and (tnow - last_check_command) > 1:
            # Check for any messages on the command socket
            while True:
                try:
                    msg = command_sock.recv(flags=zmq.NOBLOCK)
                    pymsg = pickle.loads(msg)
                    info('Message on command socket:', pymsg)
                    if pymsg[0] == 'reset':
                        ncan = 0
                        for (br,ib) in outstanding_work.keys():
                            worksent -= 1
                            info('Network thread: cancelling', br,ib)
                            outqueue.put((br, ib, 'cancel'))
                            ncan += 1
                        outstanding_work = {}
                        reply = (True, 'cancelled %i blobs' % ncan)
                    else:
                        info('Unrecognized message on command socket:', pymsg)
                        reply = (False, 'unrecognized command')
                    msg = pickle.dumps(reply, -1)
                    command_sock.send(msg)
                except zmq.ZMQError:
                    # no message waiting
                    break
            last_check_command = tnow

        #debug(qname, 'Network thread: work queues:', inqueue.qsize(), 'out queue:', outqueue.qsize(), 'work sent:', worksent, ', received:', resultsreceived, 'outstanding:', worksent-resultsreceived)

        # Retrieve the next work assignment from my input_threads.
        if not havework:

            arg = None
            # Round robin, non-blocking
            for i in range(len(inqueues)):
                try:
                    t_x = time.time()
                    inq = inqueues[(i + next_inqueue) % len(inqueues)]
                    arg = inq.get(block=False)
                    next_inqueue += (i+1)
                    t_y = time.time()
                    n_noblock += 1
                    t_noblock += (t_y - t_x)
                    break
                except queue.Empty:
                    pass

            if arg is None:
                # Round robin, blocking
                try:
                    t_x = time.time()
                    inq = inqueues[next_inqueue % len(inqueues)]
                    next_inqueue += 1
                    arg = inq.get(block=True, timeout=1)
                    t_y = time.time()
                    n_block += 1
                    t_block += (t_y - t_x)
                except queue.Empty:
                    n_nowork += 1
                    work = nowork
                    havework = False

            if arg is not None:
                (work_brick,work_iblob,work) = arg.item
                havework = True

            # try:
            #     t_x = time.time()
            #     arg = inqueue.get(block=False)
            #     (work_brick,work_iblob,work) = arg.item
            #     havework = True
            #     t_y = time.time()
            #     n_noblock += 1
            #     t_noblock += (t_y - t_x)
            # except queue.Empty:
            #     try:
            #         t_x = time.time()
            #         arg = inqueue.get(block=True, timeout=1)
            #         t_y = time.time()
            #         (work_brick,work_iblob,work) = arg.item
            #         havework = True
            #         n_block += 1
            #         t_block += (t_y - t_x)
            #         #debug('Next work packet:', len(work), 'bytes')
            #     except queue.Empty:
            #         n_nowork += 1
            #         work = nowork
            #         havework = False
            #         info('Network thread: Work queue is empty (or timed out).  qsize=', inqueue.qsize())

        t1a = time.time()

        if tnow - last_print_workqueue > 2:
            #print(qname, 'Work queue:', inqueue.qsize(), 'Work packets sent:', nworkpackets, 'bytes:', nworkbytes)
            nw = list(nwaitingCounter.keys())
            nw.sort()
            print(qname, 'Histogram of number of packets waiting in socket:')
            for n in nw:
                print('  ', n, ':', nwaitingCounter[n])
            nwaitingCounter.clear()
            #nworkpackets = nworkbytes = 0
            last_print_workqueue = tnow

        if tnow - last_printout > 15:
            if finished_bricks is not None:
                try:
                    while True:
                        br,nb = finished_bricks.get(block=False)
                        all_finished_bricks[br] = nb
                        info('Notified that brick', br, 'finished.')
                        # Delete outstanding_work entries.
                        todel = []
                        for k,v in outstanding_work.items():
                            (b,i) = k
                            if b == br:
                                print('Brick', b, 'iblob', i, 'found in outstanding_work ?!')
                                todel.append((b,i))
                        for k in todel:
                            del outstanding_work[k]
                        # Delete from our other (smaller) status maps.
                        for L in [brick_cputime, brick_nblobs, brick_starttime]:
                            try:
                                del L[br]
                            except:
                                pass
                except queue.Empty:
                    pass

            info()
            info(qname, 'Work queues:', ', '.join(['%i'%q.qsize() for q in inqueues]), ', Output queue: %i.  Work packets sent: %i, Results received: %i, Outstanding work packets: %i' %
                 (outqueue.qsize(), worksent, resultsreceived, worksent-resultsreceived))

            # Outstanding work tallies
            brick_out_n = Counter()
            brick_out_t = Counter()
            brick_out_tmax = {}
            for k,v in outstanding_work.items():
                (worker,tstart) = v
                (br,ib) = k
                tout = tnow - tstart
                brick_out_n[br] += 1
                brick_out_t[br] += tout
                # max time of any outstanding blob (by brick)
                try:
                    mx = brick_out_tmax[br]
                    if tout > mx:
                        brick_out_tmax[br] = tout
                except KeyError:
                    brick_out_tmax[br] = tout

            info('In this period, %5i finished blobs, total CPU time %8i, wall %8i, overhead %8i sec, %5.1f %% CPU use' %
                 (status_nblobs, int(status_cputime), int(status_walltime), int(status_overhead),
                  100. * (status_cputime / max(1, status_walltime + status_overhead))))
            total_cputime  += status_cputime
            total_walltime += status_walltime
            total_overhead += status_overhead
            total_nblobs   += status_nblobs
            status_cputime  = 0.
            status_walltime = 0.
            status_overhead = 0.
            status_nblobs   = 0


            info('Overall,        %5i finished blobs, total CPU time %8i, wall %8i, overhead %8i sec, %5.1f %% CPU use' %
                 (total_nblobs, int(total_cputime), int(total_walltime), int(total_overhead),
                  100. * (total_cputime / max(1, total_walltime + total_overhead))))


            info('Bricks:')
            for br,t_start in brick_starttime.items():
                info('  %s running for %5i s, done %5i blobs in %5i s CPU time. Waiting for %5i blobs, total elapsed time %5i s, max %5i s.' %
                     (br, int(tnow - t_start),
                      brick_nblobs.get(br, 0),
                      int(brick_cputime.get(br, 0)),
                      brick_out_n.get(br, 0),
                      int(brick_out_t.get(br, 0)),
                      int(brick_out_tmax.get(br, 0))))

            info(qname, 'My time spent in:')
            info('  input:     %.1f' % t_in)
            info('  polling:   %.1f' % t_poll)
            info('  receiving: %.1f' % t_recv)
            info('  sending:   %.1f' % t_send)
            info('  decoding:  %.1f' % t_decode)
            info('  output:    %.1f' % t_out)

            t_in = t_poll = t_recv = t_send = t_decode = t_out = 0

            info('Reading from inqueue work queues:')
            info('  %5i non-blocking reads, taking %5.1f sec' % (n_noblock, t_noblock))
            info('  %5i     blocking reads, taking %5.1f sec' % (n_block, t_block))
            info('  %5i times the blocking read timed out' % (n_nowork))
            n_noblock = n_block = n_nowork = 0
            t_noblock = t_block = 0.

            last_printout = tnow

        debug('Waiting for request')

        #if not havework or command_sock is not None:
        #     events = sock.poll(5000)
        #     if events == 0:
        #         # recv timed out; check work queue again
        #         continue


        events = sock.poll(timeout=0)
        nwaitingCounter[events] += 1

        events = sock.poll(timeout=5000)
        #print('Messages waiting:', events)
        if events == 0:
            # recv timed out; check work queue again
            continue

        t1b = time.time()

        parts = sock.recv_multipart()

        t2 = time.time()

        assert(len(parts) == 3)
        worker = parts[0]
        meta   = parts[1]
        result = parts[2]
        debug('Request: from', worker, ':', len(result), 'bytes')
        try:
            sock.send(work) #, flags=zmq.NOBLOCK)
            if havework:
                nworkpackets += 1
                nworkbytes += len(work)

                worksent += 1
                tnow = time.time()
                havework = False
                outstanding_work[(work_brick,work_iblob)] = (worker, tnow)
                if not work_brick in brick_starttime:
                    brick_starttime[work_brick] = tnow
        except:
            info('Network thread: sock.send(work) failed:')
            import traceback
            traceback.print_exc()

        t3 = time.time()

        if result == nowork:
            debug('Empty result')
            continue
        else:
            debug('Non-empty result')
            resultsreceived += 1
        # Parse metadata of the result.
        (brick,iblob,cpu,wall,overhead) = pickle.loads(meta)

        brick_cputime[brick] += cpu
        brick_nblobs [brick] += 1
        status_cputime  += cpu
        status_walltime += wall
        status_overhead += overhead
        status_nblobs   += 1
        try:
            del outstanding_work[(brick, iblob)]
        except KeyError:
            info('Failed to remove brick', brick, 'blob', iblob, 'from outstanding_work ?!')
            pass
        t4 = time.time()
        outqueue.put((brick, iblob, result))
        t5 = time.time()

        t_in += (t1a - t1)
        t_poll += (t1b - t1a)
        t_recv += (t2 - t1b)
        t_send += (t3 - t2)
        t_decode += (t4 - t3)
        t_out += (t5 - t4)

def output_thread(queuename, outqueue, checkpointqueue, blobsizes,
                  finished_bricks, opt):
    try:
        import setproctitle
        setproctitle.setproctitle('farm: output')
    except:
        pass

    import qdo
    q = qdo.connect(queuename)

    allresults = {}

    # Stored values from the 'blobsizes' queue.
    # brick -> (nblobs, qdo_taskid)
    brick_info = {}

    # Local mapping of brickname -> [set of cancelled blob ids]
    brick_cancelled = {}

    def get_brick_nblobs(brick, defnblobs=None):
        if not brick in brick_info:
            try:
                while True:
                    br, nb, tid = blobsizes.get(block=False)
                    brick_info[br] = (nb,tid)
            except queue.Empty:
                pass
        return brick_info.get(brick, (defnblobs,None))

    last_brick_status = {}
    brick_status_period = 60

    # Bricks that finished since the last time we printed status information
    status_bricks_finished = []
    # Total
    n_bricks_finished = 0

    def check_brick_done(brick):
        nblobs,taskid = get_brick_nblobs(brick)
        ncancelled = len(brick_cancelled.get(brick, []))
        ndone = len(allresults.get(brick, []))
        tnow = time.time()
        if not brick in last_brick_status:
            last_brick_status[brick] = tnow
        if tnow - last_brick_status[brick] > brick_status_period:
            info('Brick', brick, ':', ncancelled, 'cancelled,', ndone, 'done, total',
                  (nblobs if nblobs is not None else '(unknown)'))
            last_brick_status[brick] = tnow

        if nblobs is None:
            return
        if ndone + ncancelled < nblobs:
            return
        # Done this brick!  Set qdo state=Succeeded
        checkpoint_fn = opt.checkpoint % dict(brick=brick, brickpre=brick[:3])
        R = [dict(brickname=brick, iblob=iblob, result=res) for
             iblob,res in allresults[brick].items()]
        info('Writing final checkpoint', checkpoint_fn)
        _write_checkpoint(R, checkpoint_fn)
        info('Setting QDO task to Succeeded:', brick)
        q.set_task_state(taskid, qdo.Task.SUCCEEDED)
        del allresults[brick]
        finished_bricks.put((brick, len(R)))
        status_bricks_finished.append((brick, len(R)))
        nonlocal n_bricks_finished
        n_bricks_finished += 1

    last_checkpoint = time.time()
    last_checkpoint_size = {}

    last_status = time.time()
    status_period = 15

    while True:
        tnow = time.time()
        dt = tnow - last_checkpoint
        if dt > opt.checkpoint_period:
            info('Output thread: checkpoint q:', checkpointqueue.qsize(),
                  ', outputq:', outqueue.qsize())
            for brick,brickresults in allresults.items():
                if brick in last_checkpoint_size:
                    if len(brickresults) == last_checkpoint_size[brick]:
                        #print('Brick', brick, 'has not changed since last checkpoint was written')
                        continue
                checkpoint_fn = opt.checkpoint % dict(brick=brick, brickpre=brick[:3])
                R = [dict(brickname=brick, iblob=iblob, result=res) for
                     iblob,res in brickresults.items()]
                last_checkpoint_size[brick] = len(brickresults)
                nblobs,_ = get_brick_nblobs(brick, '(unknown)')
                info('Writing interim checkpoint', checkpoint_fn, ':', len(brickresults), 'of',
                      nblobs, 'results')
                _write_checkpoint(R, checkpoint_fn)
            last_checkpoint = tnow

        dt = tnow - last_status
        if dt > status_period:
            info('')
            info('Status:')
            info('Total bricks completed:', n_bricks_finished)
            for brick,n in status_bricks_finished:
                info('Brick', brick, 'completed')
            status_bricks_finished = []
            bricks = list(allresults.keys())
            bricks.sort()
            for brick in bricks:
                nblobs,_ = get_brick_nblobs(brick, '(unknown)')
                info('Brick', brick, ': now', len(allresults[brick]), 'of', nblobs, 'blobs done')
            info('')
            last_status = tnow

        # Read any checkpointed results sent by the input thread
        #print('checkpointqueue approx size:', checkpointqueue.qsize())
        if not checkpointqueue.empty():
            info('Checkpoints queued:', checkpointqueue.qsize())
            c = Counter()
            nchk = 0
            while True:
                try:
                    #(brick, iblob, res) = checkpointqueue.get(block=False)
                    (brick, iblob, res) = checkpointqueue.get(timeout=1)
                except:
                    #print('Exception reading checkpointqueue:')
                    #import traceback
                    #traceback.print_exc()
                    break
                if not brick in allresults:
                    allresults[brick] = {}
                allresults[brick][iblob] = res
                c[brick] += 1
                nchk += 1
            #print('Read', nchk, 'from checkpointqueue')
            if len(c):
                info('Read checkpointed results:', c)
            for brick,_ in c.most_common():
                nblobs,_ = get_brick_nblobs(brick, '(unknown)')
                info('Brick', brick, ': now', len(allresults[brick]), 'of', nblobs, 'done')
                check_brick_done(brick)

        try:
            brick,iblob,msg = outqueue.get(timeout=5)
        except:
            # timeout
            continue

        if msg == 'cancel':
            if not brick in brick_cancelled:
                brick_cancelled[brick] = set()
            brick_cancelled[brick].add(iblob)
            debug('Output thread: got cancel for brick', brick, 'blob', iblob)

        else:
            if msg is None:
                # short-cut empty work packet.
                continue
            # Worker sent a blob result
            result = pickle.loads(msg)
            if result is None:
                ### FIXME -- ???
                continue
            if not brick in allresults:
                allresults[brick] = {}
            allresults[brick][iblob] = result

        check_brick_done(brick)


def get_blob_iter(skipblobs=None,
                  brickname=None,
                  brick=None,
                  blobsrcs=None, blobslices=None, blobmap=None,
                  cat=None,
                  targetwcs=None,
                  W=None,H=None,
                  bands=None, ps=None, tims=None,
                  survey=None,
                  plots=False, plots2=False,
                  mp=None,
                  max_blobsize=None,
                  use_ceres=True,
                  reoptimize=False,
                  iterative=True,
                  less_masking=False,
                  large_galaxies_force_pointsource=True,
                  refstars=None,
                  T=None,
                  T_clusters=None,
                  custom_brick=False,
                  **kwargs):
    from legacypipe.runbrick import get_frozen_galaxies, get_blobiter_ref_map
    if skipblobs is None:
        skipblobs = []

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    frozen_galaxies = get_frozen_galaxies(T, blobsrcs, blobmap, targetwcs, cat)
    refmap = get_blobiter_ref_map(refstars, T_clusters, less_masking, targetwcs)

    # Create the iterator over blobs to process
    blobiter = _blob_iter(brickname, blobslices, blobsrcs, blobmap,
                          targetwcs, tims,
                          cat, bands, plots, ps,
                          reoptimize, iterative, use_ceres,
                          refmap,
                          large_galaxies_force_pointsource,
                          less_masking,
                          brick,
                          frozen_galaxies,
                          max_blobsize=max_blobsize, custom_brick=custom_brick,
                          skipblobs=skipblobs)
    return blobiter

class PrioritizedItem(object):
    def __init__(self, priority=0, item=None):
        self.priority = priority
        self.item = item
    def __lt__(self, other):
        return self.priority < other.priority

# we switched to multiprocessing and mp.Queue, so don't need this any more
# class ThreadSafeDict(object):
#     def __init__(self):
#         import threading
#         self.d = dict()
#         self.lock = threading.Lock()
#     def set(self, k, v):
#         with self.lock:
#             self.d[k] = v
#     def get(self, k):
#         with self.lock:
#             return self.d.get(k)
#     def get_and_del(self, k):
#         with self.lock:
#             v = self.d.get(k)
#             del self.d[k]
#             return v
#     def delete(self, k):
#         with self.lock:
#             del self.d[k]
#     def copy(self):
#         with self.lock:
#             return self.d.copy()

def queue_work(brickname, inqueue, bigqueue, checkpointqueue, opt):
    '''
    Called from the input thread to generate work packets for the given *brickname*.
    '''
    from astrometry.util.file import unpickle_from_file

    pickle_fn = opt.pickle % dict(brick=brickname, brickpre=brickname[:3])
    info('Looking for', pickle_fn)
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
        info('Brick', brickname, ': Read', len(R), 'from checkpoint file')

        skipblobs = []
        for r in R:
            br = r['brickname']
            assert(br == brickname)
            iblob = r['iblob']
            result = r['result']
            checkpointqueue.put((brickname, iblob, result))
            skipblobs.append(iblob)
            nchk += 1
        #print('Sent', nchk, 'over checkpointqueue')
        #print('checkpointqueue approx size:', checkpointqueue.qsize())
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
            continue

        # HACK -- reach into args to get blob size, for priority ordering
        blobw = args[6]
        blobh = args[7]
        priority = -(blobw*blobh)

        if opt.big == 'drop' and blobw*blobh > big_npix:
            info('Brick', brickname, ': Dropping a blob of size', blobw, 'x', blobh)
            continue

        dest_queue = inqueue

        if opt.big == 'queue' and blobw*blobh > big_npix:
            info('Blob of size', blobw, 'x', blobh, 'goes on big queue')
            dest_queue = bigqueue

        picl = pickle.dumps(arg, -1)

        qitem = PrioritizedItem(priority=priority, item=(br, iblob, picl))

        #debug('Queuing blob', (nq+1), 'for brick', brickname, '- queue size ~', inqueue.qsize())
        #mypid = os.getpid()
        #print('Input proc', mypid, 'queuing blob', (nq+1), 'for brick', brickname, 'qsize ~', inqueue.qsize())
        nq += 1
        dest_queue.put(qitem)

    # Finished queuing all blobs for this brick -- record how many blobs we sent out.
    return nchk + nq

def input_thread(queuename, inqueue, bigqueue, checkpointqueue, blobsizes, opt, input_num):

    try:
        import setproctitle
        setproctitle.setproctitle('farm: input_%i' % input_num)
    except:
        pass

    #info('Input process', os.getpid(), 'starting')
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
            blobsizes.put((brickname, nblobs, task.id))
            #
            debug('Finished', brickname, 'with', nblobs, 'blobs')
        except:
            info('Oops')
            import traceback
            traceback.print_exc()
            task.set_state(qdo.Task.FAILED, err=1)

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()
