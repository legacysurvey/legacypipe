from __future__ import division, print_function

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
parser.add_argument('--completed',action='store',help='list of all completed bricks',required=True)
parser.add_argument('--ontape',action='store',help='list of all brick on tape',required=True)
parser.add_argument('--outfn',action='store',help='output fn',required=True)
args = parser.parse_args()

comp= np.loadtxt(args.completed,dtype=str)
tape= np.loadtxt(args.ontape,dtype=str)
new= set(comp).difference(set(tape))
with open(args.outfn,'w') as foo:
    for brick in list(new):
        foo.write('%s\n' % brick)
print('Wrote %s' % args.outfn)
