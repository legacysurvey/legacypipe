import os
import argparse
import pickle
import time

import zmq

from legacypipe.oneblob import one_blob

def worker(workq, resultq):
    import socket
    myid = '%s-pid%05i' % (socket.gethostname(), os.getpid())

    req = None
    meta = None
    tprev_wall = time.time()
    while True:
        qsize = workq.qsize()
        ta_wall = time.time()
        work = workq.get()
        tb_wall = time.time()
        work = pickle.loads(work)
        tc_wall = time.time()
        (brickname, iblob, args) = work

        # DEBUG -- unpack "args" to print the following...
        # (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
        #  srcs, bands, plots, ps, reoptimize, iterative, use_ceres, refmap,
        #  large_galaxies_force_pointsource, less_masking, frozen_galaxies) = args
        # (_, iblob, Isrcs, _, _, _, blobw, blobh, _, timargs,
        #  _, _, _, _, _, _, _, _,
        #  _, _, _) = args
        #print('Work: brick', brickname, 'blob', iblob, 'size', blobw, 'x', blobh, 'with',
        #      len(timargs), 'images and', len(Isrcs), 'sources')
        #print('Calling one_blob...')
        t0_wall = time.time()
        t0_cpu  = time.process_time()

        result = one_blob(args)

        t1_cpu  = time.process_time()
        t1_wall = time.time()
        overhead = t0_wall - tprev_wall
        tprev_wall = t1_wall
        # metadata about this blob
        meta = (brickname, iblob, t1_cpu-t0_cpu, t1_wall-t0_wall, overhead)
        # pickle
        t2_wall = time.time()
        msg = pickle.dumps(result, -1)
        meta_msg = pickle.dumps(meta, -1)
        t3_wall = time.time()
        resultq.put((msg, meta_msg, brickname, iblob))
        t4_wall = time.time()

        tunpickle = tc_wall - tb_wall
        tget = tb_wall - ta_wall
        tpickle = t3_wall - t2_wall
        tput = t4_wall - t3_wall
        if max([tget, tpickle, tput, tunpickle]) > 1:
            print('Worker', myid, ': work %5.2f, unpickle %5.2f, get work %5.2f (queue size %i), pickle %5.2f, put results %5.2f' % (t1_wall-t0_wall, tunpickle, tget, qsize, tpickle, tput))

def queue_feeder(server, workq, resultq):
    from queue import Empty

    # Build job id string to identify myself to the farm.py server.
    cluster = os.environ.get('SLURM_CLUSTER_NAME', '')
    jid = os.environ.get('SLURM_JOB_ID', '')
    aid = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    ajid = os.environ.get('SLURM_ARRAY_JOB_ID', '')
    nid = os.environ.get('SLURM_NODEID', '')
    # print('SLURM_CLUSTER_NAME', cluster)
    # print('SLURM_JOB_ID', jid)
    # print('SLURM_ARRAY_TASK_ID', aid)
    # print('SLURM_ARRAY_JOB_ID', ajid)
    # print('SLURM_NODEID', nid)
    import socket
    me = socket.gethostname()
    #print('Hostname', me)
    if len(cluster + jid + aid) == 0:
        jobid = me + '_' + 'pid' + str(os.getpid())
    else:
        if len(aid):
            jobid = '%s_%s_%s_%s_%s' % (cluster, ajid, aid, nid, me)
        else:
            jobid = '%s_%s_%s_%s' % (cluster, jid, nid, me)
    print('Setting jobid "%s"' % jobid)
    jobid = jobid.encode()

    print('Connecting to', server)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server)

    nonemsg = pickle.dumps(None, -1)

    was_full = False

    nassigned = 0
    while True:

        if workq.full():
            #print('Work queue is full.  Waiting.')
            was_full = True
            #s_0 = workq.qsize()
            # FIXME -- dynamic sleep time?
            time.sleep(0.1)
            # s_1 = workq.qsize()
            # if s_1 < s_0:
            #     print('Work queue dropped in size by %i items while I slept' % (s_0-s_1))
            continue

        # if was_full:
        #     print('Work queue is newly not-full.  Work queue size:', workq.qsize())
        # was_full = False

        t_0 = time.time()
        # Check for a result produced by worker processes
        try:
            result,rmeta,brick,iblob = resultq.get_nowait()
            #print('Completed work: brick', brick, 'blob', iblob)
        except Empty:
            result,rmeta = nonemsg,nonemsg
        #print('Work queue contains ~%i items.  Results queue contains ~%i items.' %
        #      (workq.qsize(), resultq.qsize()))

        # Send result (if any) to server (and get back work item)
        t_a = time.time()
        sock.send_multipart([jobid, rmeta, result])
        t_b = time.time()
        work = sock.recv()
        t_c = time.time()
        # only unpickle very short work packets to check for None.
        if len(work) < 10:
            realwork = pickle.loads(work)
            if realwork is None:
                print('No work assigned!')
                time.sleep(1)
                continue
        nassigned += 1
        # We don't unpickle the work packet, we let the worker process do that
        workq.put(work)
        t_e = time.time()
        #print('Queue feeder: read result: %5.2f, send: %5.2f, recv: %5.2f, queue %5.2f, queue size %i' %
        #      (t_a - t_0, t_b - t_a, t_c - t_b, t_e - t_c, workq.qsize()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server', nargs=1, help='Server URL, eg tcp://edison08:5555')
    parser.add_argument('--threads', type=int, help='Number of processes to run')

    opt = parser.parse_args()

    server = opt.server[0]

    from multiprocessing import Process
    from multiprocessing import Queue

    # We have one "feeder" process that talks to the server to
    # fetch work and put it on a local (multi-process) queue --
    # this is to reduce the number of clients contacting the
    # server and to provide a short local buffer of work to reduce
    # overheads.  There is also a "results" queue where the
    # workers place their finished results.
    nqueued = 8
    workq = Queue(nqueued)
    resultq = Queue()

    p_feeder = Process(target=queue_feeder, args=(server, workq, resultq))
    p_feeder.start()

    if opt.threads:
        procs = []
        for i in range(opt.threads):
            #p = Process(target=run, args=(server,))
            p = Process(target=worker, args=(workq, resultq))
            p.start()
            procs.append(p)
        for i,p in enumerate(procs):
            p.join()
            print('Joined process', (i+1), 'of', len(procs))
    else:
        worker(workq, resultq)

    p_feeder.kill()
    p_feeder.close()
    workq.close()
    resultq.close()
    workq.join_thread()
    resultq.join_thread()
    print('All done!')

if __name__ == '__main__':
    main()
