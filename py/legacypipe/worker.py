import zmq
import pickle
from legacypipe.oneblob import *
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server', nargs=1, help='Server URL, eg tcp://edison08:5555')
    opt = parser.parse_args()

    server = opt.server[0]

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
        print('Reply:', rep)

        print('Calling one_blob...')
        result = one_blob(rep)
        # send our answer along with our next request for work!
        req = result



if __name__ == '__main__':
    main()
