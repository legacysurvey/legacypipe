import os
import argparse
import pickle
import time

import zmq

from legacypipe.oneblob import one_blob

def worker(workq, resultq):
    req = None
    meta = None
    tprev_wall = time.time()
    while True:
        work = workq.get()
        (brickname, iblob, args) = work

        # DEBUG -- unpack "args" to print the following...
        # (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
        #  srcs, bands, plots, ps, reoptimize, iterative, use_ceres, refmap,
        #  large_galaxies_force_pointsource, less_masking, frozen_galaxies) = args
        (_, iblob, Isrcs, _, _, _, blobw, blobh, _, timargs,
         _, _, _, _, _, _, _, _,
         _, _, _) = args
        print('Work: brick', brickname, 'blob', iblob, 'size', blobw, 'x', blobh, 'with',
              len(timargs), 'images and', len(Isrcs), 'sources')
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
        msg = pickle.dumps(result, -1)
        meta_msg = pickle.dumps(meta, -1)
        resultq.put((msg, meta_msg, brickname, iblob))
        
def queue_feeder(server, workq, resultq):
    from queue import Empty

    # Build job id string to identify myself to the farm.py server.
    cluster = os.environ.get('SLURM_CLUSTER_NAME', '')
    jid = os.environ.get('SLURM_JOB_ID', '')
    aid = os.environ.get('SLURM_ARRAY_TASK_ID', '')
    ajid = os.environ.get('SLURM_ARRAY_JOB_ID', '')
    nid = os.environ.get('SLURM_NODEID', '')
    print('SLURM_CLUSTER_NAME', cluster)
    print('SLURM_JOB_ID', jid)
    print('SLURM_ARRAY_TASK_ID', aid)
    print('SLURM_ARRAY_JOB_ID', ajid)
    print('SLURM_NODEID', nid)
    import socket
    me = socket.gethostname()
    print('Hostname', me)
    if len(cluster + jid + aid) == 0:
        jobid = me + '_' + 'pid' + str(os.getpid())
    else:
        if len(aid):
            jobid = '%s_%s_%s_%s_%s' % (cluster, ajid, aid, nid, me)
        else:
            jobid = '%s_%s_%s_%s' % (cluster, jid, nid, me)
    jobid = jobid.encode()
    print('Setting jobid', jobid)

    print('Connecting to', server)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server)

    nonemsg = pickle.dumps(None, -1)

    while True:
        print('Work queue contains ~%i items.  Results queue contains ~%i items.' %
              (workq.qsize(), resultq.qsize()))
        if workq.full():
            print('Work queue is full.')
            time.sleep(5)
            continue

        # Read any results produced by worker processes
        try:
            result,rmeta,brick,iblob = resultq.get_nowait()
            print('Received a result: brick', brick, 'blob', iblob)
        except Empty:
            result,rmeta = nonemsg,nonemsg

        # Send result (if any) to server (and get back work item)
        sock.send_multipart([jobid, rmeta, result])
        work = sock.recv()
        work = pickle.loads(work)
        if work is None:
            print('No work assigned!')
            time.sleep(5)
            continue
        # DEBUG -- peek into work packet
        (brickname, iblob, args) = work
        print('Got work: brick', brickname, 'blob', iblob)

        workq.put(work)

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
    nqueued = 4
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
