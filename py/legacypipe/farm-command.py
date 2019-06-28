import os
import argparse
import pickle
import time

import zmq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('server', nargs=1, help='Server URL, eg tcp://edison08:5555')

    opt = parser.parse_args()

    server = opt.server[0]

    print('Connecting to', server)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(server)

    req = ['reset']
    msg = pickle.dumps(req, -1)
    print('Sending...')
    sock.send(msg)
    print('Waiting for reply')
    reply = sock.recv()
    msg = pickle.loads(reply)
    print('Reply:', msg)

if __name__ == '__main__':
    main()
