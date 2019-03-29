import os
import argparse
import pickle
import time

import zmq

from legacypipe.oneblob import *


def run(server):

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

    if len(cluster + jid + aid) == 0:
        jobid = ''
    else:
        if len(aid):
            jobid = '%s_%s_%s_%s' % (cluster, ajid, aid, nid)
        else:
            jobid = '%s_%s_%s' % (cluster, jid, nid)
    jobid = jobid.encode()
    print('Setting jobid', jobid)

    print('Connecting to', server)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server)

    req = None
    while True:
        msg = pickle.dumps(req, -1)
        print('Sending', len(msg))
        sock.send_multipart([jobid, msg])
        rep = sock.recv()
        print('Received reply:', len(rep), 'bytes')
        rep = pickle.loads(rep)
        #print('Reply:', rep)
        if rep is None:
            print('No work assigned!')
            req = None
            time.sleep(5)
            continue
            #break

        (brickname, iblob, args) = rep

        print('Calling one_blob...')
        result = one_blob(args)
        # send our answer along with our next request for work!
        req = (brickname, iblob, result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server', nargs=1, help='Server URL, eg tcp://edison08:5555')
    parser.add_argument('--threads', type=int, help='Number of processes to run')

    opt = parser.parse_args()

    server = opt.server[0]

    if opt.threads:
        from multiprocessing import Process

        procs = []
        for i in range(opt.threads):
            p = Process(target=run, args=(server,))
            p.start()
            procs.append(p)
        for i,p in enumerate(procs):
            p.join()
            print('Joined process', (i+1), 'of', len(procs))

    else:
        run(server)
    print('All done!')

if __name__ == '__main__':
    main()
