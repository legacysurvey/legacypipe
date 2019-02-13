import zmq
import pickle
from legacypipe.oneblob import *
import argparse


def run(server):
    print('Connecting to', server)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server)

    req = None
    while True:
        msg = pickle.dumps(req, -1)
        print('Sending', len(msg))
        sock.send(msg)
        rep = sock.recv()
        print('Received reply:', len(rep), 'bytes')
        rep = pickle.loads(rep)
        #print('Reply:', rep)
        if rep is None:
            print('No work assigned!')
            break

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
        for p in procs:
            p.join()

    else:
        run(server)


if __name__ == '__main__':
    main()
